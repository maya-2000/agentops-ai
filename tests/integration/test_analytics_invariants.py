"""Business invariants over every month of the generated dataset.

These check broad properties (bounds, identities, reconciliation) rather than fragile exact
values, and do not encode any injected-event ground truth.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from app.analytics import cohorts, revenue
from app.analytics.kpis import KPIService
from app.analytics.periods import month_period
from app.database.base import Database

pytestmark = pytest.mark.slow

MONTHS = [month_period(2024 + (8 + i) // 12, (8 + i) % 12 + 1) for i in range(24)]  # 2024-09 .. 2026-08


@pytest.fixture(scope="module")
def monthly(full_db: Database) -> dict[str, dict[str, float | None]]:
    service = KPIService(full_db)
    keys = (
        "revenue",
        "mrr",
        "arr",
        "customer_count",
        "logo_churn_rate",
        "retention_rate",
        "revenue_churn_rate",
        "nrr",
        "win_rate",
        "conversion_rate",
        "average_resolution_time",
        "arpu",
    )
    return {
        m.label: {k: service.calculate_kpi(k, start_date=m.start, end_date=m.end).value for k in keys} for m in MONTHS
    }


def test_monthly_bounds_and_identities(monthly: dict[str, dict[str, float | None]]) -> None:
    for label, v in monthly.items():
        assert v["revenue"] is not None and v["revenue"] > 0, label
        assert v["mrr"] is not None and v["mrr"] > 0, label
        assert v["customer_count"] is not None and v["customer_count"] > 0, label
        assert v["arr"] == pytest.approx(12 * v["mrr"]), label
        assert v["arpu"] == pytest.approx(v["mrr"] / v["customer_count"]), label
        for rate in ("logo_churn_rate", "retention_rate", "revenue_churn_rate", "win_rate", "conversion_rate"):
            assert v[rate] is not None and 0 <= v[rate] <= 1, (label, rate)
        assert v["retention_rate"] == pytest.approx(1 - v["logo_churn_rate"]), label
        assert v["nrr"] is not None and v["nrr"] > 0, label
        assert v["average_resolution_time"] is not None and v["average_resolution_time"] >= 0, label


def test_revenue_growth_is_consistent_with_revenue(
    full_db: Database, monthly: dict[str, dict[str, float | None]]
) -> None:
    service = KPIService(full_db)
    for previous, current in pairwise(MONTHS):
        growth = service.calculate_kpi("revenue_growth", start_date=current.start, end_date=current.end).value
        expected = monthly[current.label]["revenue"] / monthly[previous.label]["revenue"] - 1  # type: ignore[operator]
        assert growth == pytest.approx(expected), current.label


def test_every_month_bridge_and_decomposition_reconcile(full_db: Database) -> None:
    for month in MONTHS[1:]:
        bridge = revenue.revenue_bridge(full_db, month)
        assert bridge.summary["reconciled"] is True, month.label
        decomposition = revenue.decompose_revenue_change(full_db, "region", month)
        assert decomposition.summary["reconciled"] is True, month.label


def test_cohort_retention_never_exceeds_cohort(full_db: Database) -> None:
    result = cohorts.cohort_retention(full_db)
    sizes: dict[str, int] = {}
    for cell in result.data:
        sizes.setdefault(cell.cohort_month, cell.cohort_size)
        assert cell.cohort_size == sizes[cell.cohort_month]
        assert 0 <= cell.active_customers <= cell.cohort_size
        assert cell.mrr >= 0 and (cell.revenue_retention is None or cell.revenue_retention >= 0)
