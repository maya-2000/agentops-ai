"""Latency budgets on the full dataset (~2.8M rows).

Aggregation happens in DuckDB; Python only applies value rules to aggregated components.
Budgets are generous (several times the observed latency) so they catch regressions such as
accidentally pulling raw rows into Python, not normal machine variance.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from app.analytics import cohorts, customers, revenue, risk
from app.analytics.kpis import KPI_KEYS, KPIService
from app.database.base import Database

pytestmark = pytest.mark.slow


def _elapsed(fn: Callable[[], object]) -> float:
    fn()  # warm-up (first scan of a table)
    started = time.perf_counter()
    fn()
    return time.perf_counter() - started


def test_every_kpi_is_fast(full_db: Database) -> None:
    service = KPIService(full_db)
    for key in KPI_KEYS:
        extra = {"product_feature": "Dashboards"} if key == "product_adoption" else {}
        seconds = _elapsed(
            lambda key=key, extra=extra: service.calculate_kpi(key, period="trailing_12_months", **extra)
        )
        assert seconds < 2.0, (key, seconds)


@pytest.mark.parametrize(
    ("name", "operation", "budget"),
    [
        (
            "revenue by country breakdown",
            lambda db: KPIService(db).calculate_kpi("revenue", period="2025", dimension="country"),
            3.0,
        ),
        ("revenue decomposition", lambda db: revenue.decompose_revenue_change(db, "country"), 3.0),
        ("mrr bridge", lambda db: revenue.revenue_bridge(db, "2025"), 3.0),
        ("cohort matrix", lambda db: cohorts.cohort_retention(db), 6.0),
        ("risk scoring", lambda db: risk.score_customer_risk(db), 6.0),
        ("usage-churn relationship", lambda db: customers.usage_churn_relationship(db), 6.0),
    ],
)
def test_module_operations_are_fast(
    full_db: Database, name: str, operation: Callable[[Database], object], budget: float
) -> None:
    seconds = _elapsed(lambda: operation(full_db))
    assert seconds < budget, (name, seconds)
