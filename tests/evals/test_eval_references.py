"""The evaluation's references and hidden-label mapping, on the full generated dataset.

The references are independent of production (pandas on raw extracts; the Phase 2 and Phase 3
reference implementations). They are checked here against the production KPI service, so a
drift on either side shows up as a test failure rather than as a wrong benchmark verdict.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from app.analytics.kpis.service import calculate_kpi
from app.database.base import Database
from evals.reference import kpis
from evals.reference.context import EvalContext
from evals.reference.expectations import resolve
from evals.reference.labels import HiddenLabels
from evals.reference.periods import month_label, month_ranges, parse_period, shift_months
from evals.scenarios.model import EvaluationDataset, ReferenceCheck
from tests import reference_timeseries as ref_ts

pytestmark = pytest.mark.slow


# ------------------------------------------------------------------ periods (no data needed)


@pytest.mark.parametrize(
    ("label", "start", "end"),
    [
        ("2026-08", date(2026, 8, 1), date(2026, 8, 31)),
        ("2024-02", date(2024, 2, 1), date(2024, 2, 29)),
        ("2026-Q2", date(2026, 4, 1), date(2026, 6, 30)),
        ("2025", date(2025, 1, 1), date(2025, 12, 31)),
        ("2026-03:2026-05", date(2026, 3, 1), date(2026, 5, 31)),
    ],
)
def test_period_labels(label: str, start: date, end: date) -> None:
    period = parse_period(label)
    assert (period.label, period.start, period.end) == (label, start, end)


def test_period_helpers() -> None:
    with pytest.raises(ValueError, match="Unsupported period"):
        parse_period("last month")
    assert shift_months("2026-01", -1) == "2025-12" and shift_months("2025-12", 1) == "2026-01"
    assert month_label(date(2026, 3, 17)) == "2026-03"
    months = month_ranges(date(2025, 11, 15), date(2026, 2, 1))
    assert [m.label for m in months] == ["2025-11", "2025-12", "2026-01", "2026-02"]


# ------------------------------------------------------------------ references vs production

PERIODS = ("2026-08", "2026-Q2")


@pytest.mark.parametrize("period_label", PERIODS)
def test_reference_kpis_agree_with_the_production_service(
    eval_ctx: EvalContext, full_db: Database, period_label: str
) -> None:
    period = parse_period(period_label)
    feature = kpis.dimension_members(eval_ctx.reference, "product_feature")[0]
    compared = 0
    for key, tolerance in kpis.KPI_TOLERANCE.items():
        if key == "revenue_growth":
            continue  # a change between periods: covered by the kpi_change check
        filters = {"product_feature": feature} if key == "product_adoption" else {}
        expected = kpis.kpi(eval_ctx.reference, key, period, filters)
        result = calculate_kpi(full_db, key, start_date=period.start, end_date=period.end, **filters)
        assert result.value is not None, key
        assert kpis.within(float(result.value), expected, tolerance), (key, result.value, expected)
        compared += 1
    assert compared == len(kpis.KPI_TOLERANCE) - 1


def test_filtered_reference_values_agree_with_production(eval_ctx: EvalContext, full_db: Database) -> None:
    period = parse_period("2026-08")
    for key, filters in [
        ("revenue", {"country": "Singapore"}),
        ("logo_churn_rate", {"segment": "SMB"}),
        ("support_ticket_volume", {"ticket_category": "Bug"}),
        ("customer_count", {"region": "APAC"}),
    ]:
        expected = kpis.kpi(eval_ctx.reference, key, period, filters)
        result = calculate_kpi(full_db, key, start_date=period.start, end_date=period.end, **filters)
        assert result.value is not None
        assert kpis.within(float(result.value), expected, kpis.KPI_TOLERANCE[key]), (key, filters)


def test_monthly_series_and_time_series_references(eval_ctx: EvalContext, full_db: Database) -> None:
    series = kpis.monthly_series(eval_ctx.reference, "revenue", date(2026, 1, 1), date(2026, 8, 31))
    assert len(series) == 8
    for month, value in zip(month_ranges(date(2026, 1, 1), date(2026, 8, 31)), series, strict=True):
        production = calculate_kpi(full_db, "revenue", start_date=month.start, end_date=month.end).value
        assert production is not None and kpis.within(float(production), value, "money")
    assert ref_ts.naive_forecast(series, 3) == [series[-1]] * 3
    drift = ref_ts.drift_forecast(series, 2)
    slope = (series[-1] - series[0]) / (len(series) - 1)
    assert drift == pytest.approx([series[-1] + slope, series[-1] + 2 * slope])


# ------------------------------------------------------------------ resolving checks


def test_kpi_value_and_change_resolve_from_the_reference(eval_ctx: EvalContext) -> None:
    value = resolve(ReferenceCheck(kind="kpi_value", metric="revenue", period="2026-08"), eval_ctx)
    assert value.values["value"] == kpis.kpi(eval_ctx.reference, "revenue", parse_period("2026-08"))
    assert value.values["tolerance"] == "money"
    change = resolve(
        ReferenceCheck(kind="kpi_change", metric="revenue", period="2026-08", comparison_period="2026-07"), eval_ctx
    ).values
    assert change["change"] == pytest.approx(change["current"] - change["previous"])
    assert change["direction"] == ("decrease" if change["change"] < 0 else "increase")


def test_top_member_resolves_from_the_reference_and_records_the_label(eval_ctx: EvalContext) -> None:
    check = ReferenceCheck(
        kind="top_member", metric="logo_churn_rate", dimension="segment", which="highest", period="2026-08", event="E6"
    )
    expected = resolve(check, eval_ctx)
    by_member: dict[str, float] = expected.values["all"]
    assert expected.values["member"] == max(by_member, key=lambda m: by_member[m])
    assert expected.values["label_member"] == eval_ctx.labels.observable("E6").segment
    assert expected.values["member"] == expected.values["label_member"] and not expected.note


def test_event_discovery_is_confirmed_against_the_data(eval_ctx: EvalContext) -> None:
    e1 = resolve(ReferenceCheck(kind="event_discovery", event="E1"), eval_ctx)
    observable = eval_ctx.labels.observable("E1")
    assert e1.applicable and e1.values["country"] == observable.country and e1.values["segment"] == observable.segment
    assert e1.values["country_change"] < 0
    for event in ("E2", "E5", "E6"):
        assert resolve(ReferenceCheck(kind="event_discovery", event=event), eval_ctx).applicable, event
    anomaly = resolve(ReferenceCheck(kind="anomaly_event", event="E1", metric="revenue"), eval_ctx)
    assert anomaly.values["month"] == observable.month and anomaly.values["direction"] == "negative"
    assert len(anomaly.values["history"]) == len(anomaly.values["months"]) == 24


def test_forecast_reference_uses_only_history_up_to_the_cutoff(eval_ctx: EvalContext) -> None:
    expected = resolve(ReferenceCheck(kind="forecast", metric="revenue", horizon=3), eval_ctx).values
    assert expected["cutoff"] == eval_ctx.as_of.isoformat()
    assert len(expected["history"]) == 24 and len(expected["naive"]) == len(expected["drift"]) == 3
    assert expected["naive"] == [expected["history"][-1]] * 3


def test_every_dataset_check_resolves_and_no_answer_is_written_into_its_scenario(
    eval_ctx: EvalContext, eval_dataset: EvaluationDataset
) -> None:
    resolved = 0
    for scenario in eval_dataset.scenarios:
        text = scenario.model_dump_json()
        for check in scenario.reference_expectations:
            expected = resolve(check, eval_ctx)
            resolved += 1
            if check.kind == "top_member":
                assert str(expected.values["member"]) not in text, (scenario.scenario_id, expected.values["member"])
    assert resolved >= 30


# ------------------------------------------------------------------ hidden labels


def test_labels_map_to_observable_manifestations(eval_ctx: EvalContext) -> None:
    labels: HiddenLabels = eval_ctx.labels
    assert set(labels.events) == {f"E{i}" for i in range(1, 8)}
    observed: dict[str, Any] = {e: labels.observable(e) for e in labels.events}
    assert observed["E1"].month == "2026-08" and observed["E1"].country and observed["E1"].segment
    assert observed["E2"].months[0] == "2026-06" and observed["E2"].ticket_categories
    assert observed["E3"].channel and observed["E3"].campaign_id
    assert observed["E4"].sales_rep and observed["E5"].feature and observed["E6"].segment
    assert observed["E7"].direction == "decrease"
    with pytest.raises(KeyError):
        labels.observable("E9")
    markers = labels.leak_markers()
    assert "injected_events" in markers and all(len(m) >= 8 for m in markers)
    assert all(labels.events[e].name in markers for e in labels.events)
