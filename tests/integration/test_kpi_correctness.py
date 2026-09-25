"""Every KPI vs an independent pandas reference on the full generated dataset.

Production: registered SQL templates + value rules, executed through the Database abstraction.
Reference:  ``tests/integration/reference_kpis.py``: pandas on raw ``SELECT *`` extracts, written
from the KPI definitions without reusing any production code.

Tolerances: monetary values agree within SGD 0.01 or 1e-9 relative (float vs DECIMAL
summation); rates and averages within 1e-9 absolute.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import pytest

from app.analytics.kpis import KPI_KEYS, KPIResult, calculate_kpi
from app.database.base import Database
from tests.integration.reference_kpis import Reference

pytestmark = pytest.mark.slow

PERIODS: dict[str, tuple[date, date]] = {
    "aug_2026": (date(2026, 8, 1), date(2026, 8, 31)),
    "q2_2026": (date(2026, 4, 1), date(2026, 6, 30)),
    "year_2025": (date(2025, 1, 1), date(2025, 12, 31)),
    "custom_range": (date(2025, 10, 10), date(2026, 1, 20)),
}
COVERAGE_START = date(2024, 9, 1)
CUSTOMER_FILTERS: list[dict[str, str]] = [{}, {"segment": "Enterprise"}, {"region": "APAC"}, {"plan": "Professional"}]
SALES_FILTERS: list[dict[str, str]] = [{}, {"segment": "Enterprise"}, {"region": "APAC"}]
MARKETING_FILTERS: list[dict[str, str]] = [{}, {"acquisition_channel": "Paid Search"}]
SUPPORT_FILTERS: list[dict[str, str]] = [{}, {"segment": "SMB"}, {"ticket_category": "Bug"}]


def months(start: date, end: date) -> float:
    """Independent month count: exact for whole calendar months, else days / (365.25 / 12)."""
    if start.day == 1 and _is_month_end(end):
        return float((end.year - start.year) * 12 + end.month - start.month + 1)
    return ((end - start).days + 1) / (365.25 / 12)


def _is_month_end(d: date) -> bool:
    from calendar import monthrange

    return d.day == monthrange(d.year, d.month)[1]


def close(actual: float | None, expected: float, *, money: bool = False) -> bool:
    assert actual is not None
    if money:
        return abs(actual - expected) <= max(0.01, abs(expected) * 1e-9)
    return abs(actual - expected) <= 1e-9 + abs(expected) * 1e-12


Expected = Callable[[Reference, date, date, dict[str, str], KPIResult], float]

CASES: dict[str, tuple[list[dict[str, str]], Expected, bool]] = {
    "revenue": (CUSTOMER_FILTERS, lambda r, s, e, f, res: r.revenue(s, e, f)["revenue"], True),
    "mrr": (CUSTOMER_FILTERS, lambda r, s, e, f, res: r.recurring_state(e, f)["mrr"], True),
    "arr": (CUSTOMER_FILTERS, lambda r, s, e, f, res: 12 * r.recurring_state(e, f)["mrr"], True),
    "arpu": (
        CUSTOMER_FILTERS,
        lambda r, s, e, f, res: r.recurring_state(e, f)["mrr"] / r.recurring_state(e, f)["customers"],
        True,
    ),
    "customer_count": (CUSTOMER_FILTERS, lambda r, s, e, f, res: r.recurring_state(e, f)["customers"], False),
    "revenue_growth": (
        [{}, {"segment": "Enterprise"}],
        lambda r, s, e, f, res: r.revenue_growth(
            s,
            e,
            res.comparison_period.start,
            res.comparison_period.end,
            f,  # type: ignore[union-attr]
        ),
        False,
    ),
    "logo_churn_rate": (CUSTOMER_FILTERS, lambda r, s, e, f, res: r.logo_churn(s, e, f), False),
    "retention_rate": (CUSTOMER_FILTERS, lambda r, s, e, f, res: 1 - r.logo_churn(s, e, f), False),
    "revenue_churn_rate": (
        CUSTOMER_FILTERS,
        lambda r, s, e, f, res: r.cohort(s, e, f)["churned_mrr"] / r.cohort(s, e, f)["opening_mrr"],
        False,
    ),
    "nrr": (CUSTOMER_FILTERS, lambda r, s, e, f, res: r.nrr(s, e, f), False),
    "clv": ([{}, {"segment": "SMB"}], lambda r, s, e, f, res: r.clv(s, e, months(s, e), f), True),
    "cac": (
        MARKETING_FILTERS,
        lambda r, s, e, f, res: r.marketing(s, e, f)["spend"] / r.marketing(s, e, f)["conversions"],
        True,
    ),
    "conversion_rate": (
        MARKETING_FILTERS,
        lambda r, s, e, f, res: r.marketing(s, e, f)["conversions"] / r.marketing(s, e, f)["leads"],
        False,
    ),
    "win_rate": (SALES_FILTERS, lambda r, s, e, f, res: r.win_rate(s, e, f), False),
    "average_order_value": (SALES_FILTERS, lambda r, s, e, f, res: r.average_order_value(s, e, f), True),
    "sales_cycle": (SALES_FILTERS, lambda r, s, e, f, res: r.sales_cycle(s, e, f), False),
    "pipeline_value": (SALES_FILTERS, lambda r, s, e, f, res: r.pipeline(e, f), True),
    "support_ticket_volume": (SUPPORT_FILTERS, lambda r, s, e, f, res: len(r.tickets(s, e, f)), False),
    "average_resolution_time": (
        SUPPORT_FILTERS,
        lambda r, s, e, f, res: float(r.tickets(s, e, f).query("status == 'Resolved'")["resolution_time"].mean()),
        False,
    ),
    "product_adoption": (
        [{"product_feature": "Dashboards"}, {"product_feature": "Collaboration"}],
        lambda r, s, e, f, res: r.adoption(s, e, f["product_feature"]),
        False,
    ),
}


def test_every_registered_kpi_has_a_reference_case() -> None:
    assert sorted(CASES) == sorted(KPI_KEYS)
    assert len(KPI_KEYS) == 20


@pytest.mark.parametrize("period_name", sorted(PERIODS))
@pytest.mark.parametrize("key", sorted(CASES))
def test_kpi_matches_independent_reference(full_db: Database, reference: Reference, key: str, period_name: str) -> None:
    start, end = PERIODS[period_name]
    filter_sets, expected_fn, money = CASES[key]
    for filters in filter_sets:
        result = calculate_kpi(full_db, key, start_date=start, end_date=end, **filters)
        comparison = result.comparison_period
        if comparison is not None and comparison.start < COVERAGE_START:
            # Growth against a comparison period the data does not fully cover is refused, not guessed.
            assert result.status == "insufficient_data" and result.value is None
            continue
        assert result.status == "ok", (key, period_name, filters, result.message)
        expected = expected_fn(reference, start, end, filters, result)
        assert close(result.value, expected, money=money), (key, period_name, filters, result.value, expected)


@pytest.mark.parametrize(
    ("key", "dimension"),
    [
        ("revenue", "segment"),
        ("revenue", "plan"),
        ("mrr", "region"),
        ("customer_count", "plan"),
        ("support_ticket_volume", "ticket_category"),
        ("pipeline_value", "sales_rep"),
    ],
)
def test_additive_breakdowns_sum_to_the_total(full_db: Database, key: str, dimension: str) -> None:
    result = calculate_kpi(full_db, key, period="last_quarter", dimension=dimension)
    assert result.breakdown
    assert sum(r.value or 0 for r in result.breakdown) == pytest.approx(result.value, abs=0.02)


@pytest.mark.parametrize(
    ("key", "dimension", "reference_filter"),
    [
        ("logo_churn_rate", "segment", "segment"),
        ("nrr", "region", "region"),
        ("win_rate", "sales_rep", "sales_rep"),
        ("average_resolution_time", "segment", "segment"),
    ],
)
def test_ratio_breakdowns_match_filtered_reference(
    full_db: Database, reference: Reference, key: str, dimension: str, reference_filter: str
) -> None:
    start, end = PERIODS["year_2025"]
    result = calculate_kpi(full_db, key, start_date=start, end_date=end, dimension=dimension)
    expected_fn = CASES[key][1]
    for row in result.breakdown:
        filters = {reference_filter: row.dimension_value}
        expected = expected_fn(reference, start, end, filters, result)
        assert close(row.value, expected), (key, row.dimension_value, row.value, expected)


def test_dimension_value_equals_filter(full_db: Database) -> None:
    by_value = calculate_kpi(full_db, "revenue", period="2026-Q2", dimension="country", dimension_value="singapore")
    filtered = calculate_kpi(full_db, "revenue", period="2026-Q2", country="Singapore")
    assert by_value.value == filtered.value
    assert by_value.filters == {"country": "Singapore"} and not by_value.breakdown
