"""Resolve a scenario's ``ReferenceCheck`` into concrete expected values, at run time.

Every expected number comes from the independent reference implementations
(``evals.reference.kpis``). Every event expectation comes from the hidden labels and is then
*confirmed* against the reference data. An event whose label does not show in the observable
data of this dataset (possible for another seed) makes the check not applicable, and is never
counted as an agent failure.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from evals.reference import kpis
from evals.reference.context import EvalContext
from evals.reference.labels import ObservableEvent
from evals.reference.periods import PeriodRange, month_ranges, parse_period, shift_months
from evals.scenarios.model import ReferenceCheck


class Expected(BaseModel):
    check: ReferenceCheck
    values: dict[str, Any] = Field(default_factory=dict)
    observable: ObservableEvent | None = None
    applicable: bool = True
    note: str = ""


def resolve(check: ReferenceCheck, ctx: EvalContext) -> Expected:
    handler = _HANDLERS[check.kind]
    return handler(check, ctx)


def _period(label: str | None) -> PeriodRange:
    if label is None:
        raise ValueError("This reference check needs a period")
    return parse_period(label)


def _kpi_value(check: ReferenceCheck, ctx: EvalContext) -> Expected:
    assert check.metric is not None
    value = kpis.kpi(ctx.reference, check.metric, _period(check.period), check.filters)
    tolerance = check.tolerance or kpis.KPI_TOLERANCE[check.metric]
    return Expected(check=check, values={"value": value, "tolerance": tolerance})


def _kpi_change(check: ReferenceCheck, ctx: EvalContext) -> Expected:
    assert check.metric is not None
    current = kpis.kpi(ctx.reference, check.metric, _period(check.period), check.filters)
    previous = kpis.kpi(ctx.reference, check.metric, _period(check.comparison_period), check.filters)
    tolerance = check.tolerance or kpis.KPI_TOLERANCE[check.metric]
    return Expected(
        check=check,
        values={
            "current": current,
            "previous": previous,
            "change": current - previous,
            "direction": "increase" if current > previous else "decrease" if current < previous else "none",
            "tolerance": tolerance,
        },
    )


def _eligibility(check: ReferenceCheck, ctx: EvalContext) -> Callable[[str], bool] | None:
    """Minimum-sample rules stated explicitly in the scenario (the same argument the tool receives)."""
    if check.min_sample is None or check.dimension is None:
        return None
    period = _period(check.period)
    minimum = check.min_sample
    ref = ctx.reference
    if check.dimension == "sales_rep":
        return lambda rep: len(ref.closed(period.start, period.end, {"sales_rep": rep})) >= minimum
    if check.dimension in ("campaign", "acquisition_channel"):
        return lambda member: (
            ref.marketing(period.start, period.end, {check.dimension or "": member})["conversions"] >= minimum
        )
    return None


def _top_member(check: ReferenceCheck, ctx: EvalContext) -> Expected:
    assert check.dimension is not None
    period = _period(check.period)
    if check.which == "largest_decline":
        changes = kpis.revenue_changes(ctx.reference, check.dimension, period, _period(check.comparison_period))
        member = min(changes, key=lambda m: changes[m])
        values: dict[str, Any] = {"member": member, "value": changes[member], "all": changes}
    else:
        assert check.metric is not None
        by_member = kpis.member_values(ctx.reference, check.metric, check.dimension, period, _eligibility(check, ctx))
        pick = max if check.which == "highest" else min
        member = pick(by_member, key=lambda m: by_member[m])
        values = {"member": member, "value": by_member[member], "all": by_member}
    expected = Expected(check=check, values=values)
    if check.event:
        observable = ctx.labels.observable(check.event)
        label_member = observable.segment or observable.sales_rep or observable.campaign_id or observable.feature
        expected.observable = observable
        expected.values["label_member"] = label_member
        if label_member is not None and label_member != member:
            expected.note = f"The hidden label names {label_member}, but the reference data shows {member}."
    return expected


def _forecast(check: ReferenceCheck, ctx: EvalContext) -> Expected:
    from tests import reference_timeseries as ref_ts

    assert check.metric is not None and check.horizon is not None
    history = kpis.monthly_series(ctx.reference, check.metric, ctx.source.coverage_start, ctx.as_of, check.filters)
    return Expected(
        check=check,
        values={
            "cutoff": ctx.as_of.isoformat(),
            "horizon": check.horizon,
            "history": history,
            "naive": ref_ts.naive_forecast(history, check.horizon),
            "drift": ref_ts.drift_forecast(history, check.horizon),
        },
    )


def _anomaly_event(check: ReferenceCheck, ctx: EvalContext) -> Expected:
    assert check.event is not None and check.metric is not None
    observable = ctx.labels.observable(check.event)
    history = kpis.monthly_series(ctx.reference, check.metric, ctx.source.coverage_start, ctx.as_of, check.filters)
    months = [p.label for p in month_ranges(ctx.source.coverage_start, ctx.as_of)]
    direction = "negative" if observable.direction == "decrease" else "positive"
    return Expected(
        check=check,
        observable=observable,
        values={"month": observable.month, "direction": direction, "history": history, "months": months},
    )


def _event_discovery(check: ReferenceCheck, ctx: EvalContext) -> Expected:
    assert check.event is not None
    observable = ctx.labels.observable(check.event)
    ref = ctx.reference
    expected = Expected(check=check, observable=observable)
    confirmed = True
    if check.event == "E1":
        current, previous = _period(observable.month), _period(_previous_month(observable.month))
        by_country = kpis.revenue_changes(ref, "country", current, previous)
        top_country = min(by_country, key=lambda m: by_country[m])
        in_country = {
            segment: kpis.kpi(ref, "revenue", current, {"country": top_country, "segment": segment})
            - kpis.kpi(ref, "revenue", previous, {"country": top_country, "segment": segment})
            for segment in kpis.dimension_members(ref, "segment")
        }
        top_segment = min(in_country, key=lambda m: in_country[m])
        confirmed = top_country == observable.country and top_segment == observable.segment
        expected.values = {"country": top_country, "segment": top_segment, "country_change": by_country[top_country]}
    elif check.event == "E2":
        month = check.period or observable.month
        current, previous = _period(month), _period(_previous_month(month))
        now = kpis.kpi(ref, "support_ticket_volume", current)
        before = kpis.kpi(ref, "support_ticket_volume", previous)
        confirmed = now > before
        expected.values = {"month": month, "current": now, "previous": before}
    elif check.event == "E5":
        assert observable.feature is not None
        current = _period(check.period or ctx.as_of.strftime("%Y-%m"))
        previous = _period(check.comparison_period or _previous_month(current.label))
        now = kpis.kpi(ref, "product_adoption", current, {"product_feature": observable.feature})
        before = kpis.kpi(ref, "product_adoption", previous, {"product_feature": observable.feature})
        confirmed = now > before
        expected.values = {"feature": observable.feature, "current": now, "previous": before}
    elif check.event == "E6":
        period = _period(check.period or ctx.as_of.strftime("%Y-%m"))
        churn = kpis.member_values(ref, "logo_churn_rate", "segment", period)
        top = max(churn, key=lambda m: churn[m])
        confirmed = top == observable.segment
        expected.values = {"segment": top, "rate": churn[top]}
    elif check.event == "E7":
        expected.values = {}  # label-only: the association must be surfaced, never stated causally
    else:
        raise ValueError(f"No discovery check for event {check.event}")
    if not confirmed:
        expected.applicable = False
        expected.note = f"{check.event}'s labelled manifestation is not visible in this dataset's reference data."
    return expected


def _previous_month(label: str | None) -> str:
    if label is None:
        raise ValueError("A month label is needed")
    return shift_months(label, -1)


_HANDLERS: dict[str, Callable[[ReferenceCheck, EvalContext], Expected]] = {
    "kpi_value": _kpi_value,
    "kpi_change": _kpi_change,
    "top_member": _top_member,
    "forecast": _forecast,
    "anomaly_event": _anomaly_event,
    "event_discovery": _event_discovery,
}
