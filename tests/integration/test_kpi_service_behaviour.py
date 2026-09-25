"""KPI service behaviour: validation, empty/insufficient data, edge cases and provenance."""

from __future__ import annotations

from datetime import date

import pytest

from app.analytics.errors import (
    InvalidFilterValueError,
    InvalidPeriodError,
    InvalidRequestError,
    UnsupportedDimensionError,
    UnsupportedKPIError,
)
from app.analytics.kpis import KPIService, calculate_kpi, get_kpi_definition
from app.database.base import Database
from tests.integration.reference_kpis import Reference

pytestmark = pytest.mark.slow

AUG_2026 = {"start_date": date(2026, 8, 1), "end_date": date(2026, 8, 31)}


# ---- invalid requests -----------------------------------------------------------------------------------


def test_unknown_kpi(full_db: Database) -> None:
    with pytest.raises(UnsupportedKPIError, match="Registered KPIs"):
        calculate_kpi(full_db, "ebitda")


def test_unknown_dimension(full_db: Database) -> None:
    with pytest.raises(UnsupportedDimensionError, match="Allowed dimensions"):
        calculate_kpi(full_db, "revenue", dimension="favourite_colour")


def test_unknown_parameter_is_rejected(full_db: Database) -> None:
    with pytest.raises(UnsupportedDimensionError, match="continent"):
        calculate_kpi(full_db, "revenue", continent="Asia")


def test_dimension_not_applicable_to_kpi(full_db: Database) -> None:
    with pytest.raises(UnsupportedDimensionError, match="cannot be broken down by 'sales_rep'"):
        calculate_kpi(full_db, "revenue", dimension="sales_rep")
    with pytest.raises(UnsupportedDimensionError, match="cannot be filtered by 'country'"):
        calculate_kpi(full_db, "win_rate", country="Singapore")


def test_invalid_enumerated_filter_value(full_db: Database) -> None:
    with pytest.raises(InvalidFilterValueError, match="Allowed: SMB, Mid-Market, Enterprise"):
        calculate_kpi(full_db, "revenue", segment="Mega-Corp")


def test_enumerated_values_are_case_insensitive(full_db: Database) -> None:
    assert calculate_kpi(full_db, "revenue", segment="enterprise").filters == {"segment": "Enterprise"}


def test_unknown_lookup_value(full_db: Database) -> None:
    with pytest.raises(InvalidFilterValueError, match="No customer"):
        calculate_kpi(full_db, "revenue", customer_id="CUST-999999")
    with pytest.raises(InvalidFilterValueError, match="No sales rep"):
        calculate_kpi(full_db, "win_rate", sales_rep="Nobody Real")


@pytest.mark.parametrize("payload", ["SMB' OR '1'='1", "Enterprise; DROP TABLE customers; --"])
def test_injection_attempt_in_enumerated_filter_is_rejected(full_db: Database, payload: str) -> None:
    with pytest.raises(InvalidFilterValueError):
        calculate_kpi(full_db, "revenue", segment=payload)


def test_injection_attempt_in_lookup_filter_is_bound_not_executed(full_db: Database) -> None:
    with pytest.raises(InvalidFilterValueError):
        calculate_kpi(full_db, "revenue", customer_id="x' OR 1=1 --")
    assert calculate_kpi(full_db, "customer_count").value > 0  # database untouched


def test_invalid_periods(full_db: Database) -> None:
    with pytest.raises(InvalidPeriodError):
        calculate_kpi(full_db, "revenue", period="last_fortnight")
    with pytest.raises(InvalidPeriodError):
        calculate_kpi(full_db, "revenue", start_date=date(2026, 8, 31), end_date=date(2026, 8, 1))
    with pytest.raises(InvalidRequestError):
        calculate_kpi(full_db, "revenue", start_date=date(2026, 8, 1))
    with pytest.raises(InvalidRequestError, match="does not use a comparison period"):
        calculate_kpi(full_db, "revenue", comparison_period="2026-07")


def test_product_adoption_requires_a_feature(full_db: Database) -> None:
    with pytest.raises(InvalidRequestError, match="product_feature"):
        calculate_kpi(full_db, "product_adoption")


# ---- empty and insufficient data ------------------------------------------------------------------------


def test_period_outside_coverage_is_no_data_not_zero(full_db: Database) -> None:
    for key in ("revenue", "mrr", "support_ticket_volume", "customer_count"):
        result = calculate_kpi(full_db, key, period="2027-01")
        assert result.status == "no_data" and result.value is None, key
        assert "coverage" in (result.message or "") or "observed" in (result.message or "")


def test_partial_coverage_is_flagged(full_db: Database, reference: Reference) -> None:
    result = calculate_kpi(full_db, "revenue", period="2026")
    assert result.status == "ok"
    assert any("beyond the data coverage" in item for item in result.limitations)
    assert result.value == pytest.approx(reference.revenue(date(2026, 1, 1), date(2026, 8, 31), {})["revenue"])


def test_churn_before_observable_opening_is_insufficient(full_db: Database) -> None:
    result = calculate_kpi(full_db, "logo_churn_rate", period="2024-08")
    assert result.status == "insufficient_data" and result.value is None


def test_zero_denominator_is_insufficient_not_zero(full_db: Database, reference: Reference) -> None:
    # A customer who signed up inside the period has no opening base.
    newcomer = reference.t["customers"].query("signup_date >= '2026-08-05'").iloc[0]["customer_id"]
    churn = calculate_kpi(full_db, "logo_churn_rate", customer_id=newcomer, **AUG_2026)
    assert churn.status == "insufficient_data" and churn.value is None
    assert churn.components["opening_customers"] == 0


def test_win_rate_without_closed_deals_is_insufficient(full_db: Database, reference: Reference) -> None:
    with_opportunities = set(reference.t["sales_opportunities"]["customer_id"].dropna())
    no_deals = next(c for c in reference.t["customers"]["customer_id"] if c not in with_opportunities)
    for key in ("win_rate", "average_order_value", "sales_cycle"):
        result = calculate_kpi(full_db, key, period="trailing_12_months", customer_id=no_deals)
        assert result.status == "insufficient_data" and result.value is None, key
        assert result.components["closed_opportunities"] == 0


def test_count_kpis_report_true_zero(full_db: Database) -> None:
    result = calculate_kpi(
        full_db,
        "support_ticket_volume",
        ticket_category="Billing",
        start_date=date(2026, 8, 30),
        end_date=date(2026, 8, 30),
        customer_id="CUST-000001",
    )
    assert result.status == "ok" and result.value == 0


def test_feature_before_launch_is_no_data(full_db: Database, reference: Reference) -> None:
    first = reference.t["product_features"].query("feature_name == 'AI Insights'")["date"].min().date()
    before = calculate_kpi(
        full_db,
        "product_adoption",
        product_feature="AI Insights",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
    )
    assert before.status == "no_data" and before.value is None
    after = calculate_kpi(full_db, "product_adoption", product_feature="AI Insights", start_date=first, end_date=first)
    assert after.status == "ok"


def test_cac_without_spend_or_at_unsupported_grain_is_insufficient(full_db: Database) -> None:
    outbound = calculate_kpi(full_db, "cac", acquisition_channel="Outbound Sales")
    assert outbound.status == "insufficient_data" and "no campaigns" in (outbound.message or "")
    by_segment = calculate_kpi(full_db, "cac", segment="Enterprise")
    assert by_segment.status == "insufficient_data" and "cannot be attributed" in (by_segment.message or "")
    adoption_by_region = calculate_kpi(full_db, "product_adoption", product_feature="Dashboards", dimension="region")
    assert adoption_by_region.status == "insufficient_data"


def test_unresolved_tickets_excluded_from_resolution_time(full_db: Database, reference: Reference) -> None:
    result = calculate_kpi(full_db, "average_resolution_time", **AUG_2026)
    tickets = reference.tickets(date(2026, 8, 1), date(2026, 8, 31), {})
    assert result.components["unresolved_tickets"] == int((tickets["status"] == "Open").sum()) > 0
    assert result.components["resolved_tickets"] + result.components["unresolved_tickets"] == len(tickets)
    assert result.value == pytest.approx(tickets.loc[tickets["status"] == "Resolved", "resolution_time"].mean())


def test_open_opportunities_excluded_from_win_rate_and_cycle(full_db: Database, reference: Reference) -> None:
    result = calculate_kpi(full_db, "win_rate", period="trailing_12_months")
    o = reference.t["sales_opportunities"]
    assert (o["stage"].isin(["Lead", "Qualified", "Proposal", "Negotiation"])).sum() > 0  # open deals exist
    closed = result.components["won_opportunities"] + result.components["lost_opportunities"]
    assert closed == result.components["closed_opportunities"]


# ---- identities between KPIs --------------------------------------------------------------------------------


@pytest.mark.parametrize("period", ["last_month", "2026-Q2", "2025"])
def test_kpi_identities(full_db: Database, period: str) -> None:
    get = lambda key: calculate_kpi(full_db, key, period=period).value  # noqa: E731
    assert get("arr") == pytest.approx(12 * get("mrr"))
    assert get("arpu") * get("customer_count") == pytest.approx(get("mrr"))
    assert get("retention_rate") == pytest.approx(1 - get("logo_churn_rate"))


def test_nrr_excludes_new_business(full_db: Database, reference: Reference) -> None:
    """NRR x opening MRR equals the opening cohort's closing MRR, so new customers' MRR is excluded."""
    start, end = date(2025, 1, 1), date(2025, 12, 31)
    nrr = calculate_kpi(full_db, "nrr", start_date=start, end_date=end)
    opening_mrr = float(nrr.components["opening_mrr"])  # type: ignore[arg-type]
    assert nrr.value * opening_mrr == pytest.approx(reference.cohort_closing_mrr(start, end, {}), abs=0.02)
    closing_all = calculate_kpi(full_db, "mrr", start_date=start, end_date=end).value
    assert closing_all > nrr.value * opening_mrr  # total MRR also contains new business


def test_revenue_growth_sign(full_db: Database) -> None:
    growth = calculate_kpi(full_db, "revenue_growth", period="2026-06", comparison_period="2025-06")
    c = growth.components
    assert (growth.value > 0) == (c["current_revenue"] > c["comparison_revenue"])
    reverse = calculate_kpi(full_db, "revenue_growth", period="2025-06", comparison_period="2026-06")
    assert (growth.value > 0) != (reverse.value > 0)


def test_default_comparison_is_previous_period(full_db: Database) -> None:
    result = calculate_kpi(full_db, "revenue_growth", period="2026-Q2")
    assert result.comparison_period is not None and result.comparison_period.label == "2026-Q1"


# ---- evidence-ready results -------------------------------------------------------------------------------


def test_result_is_evidence_ready(full_db: Database) -> None:
    result = calculate_kpi(full_db, "revenue", period="last_month", segment="Enterprise", dimension="region")
    dumped = result.model_dump()
    for field in (
        "key",
        "name",
        "value",
        "unit",
        "period",
        "filters",
        "dimension",
        "calculation",
        "sql",
        "source_tables",
        "query_ids",
        "operation_id",
        "execution_timestamp",
        "interpretation",
        "limitations",
    ):
        assert dumped[field] is not None, field
    assert result.source_tables == ["customers", "daily_revenue"]
    assert result.period.start == date(2026, 8, 1) and result.period.end == date(2026, 8, 31)
    assert len(result.query_ids) == 2  # total + breakdown
    for query in result.provenance.queries:
        assert "Enterprise" not in query.sql  # values are bound, never interpolated
        assert query.parameters["f_segment"] == "Enterprise"
        assert query.lineage.dataset_version == "1.0.0"
        assert query.lineage.tool_run_id == result.operation_id


def test_definitions_are_retrievable(full_db: Database) -> None:
    definition = KPIService(full_db).get_kpi_definition("NRR")
    assert definition.key == "nrr" and "opening MRR" in definition.formula
    assert get_kpi_definition("cac").insufficient_grains
