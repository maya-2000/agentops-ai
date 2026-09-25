"""Latency budgets for Phase 3 on the full dataset (~2.8M rows).

Aggregation happens in DuckDB (one monthly query per series). Python only sees at most 24 values
per series, so the cost is dominated by backtests and model fitting. Budgets are generous (several
times the observed latency) so they catch regressions such as pulling raw rows into Python, not
machine variance.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from app.anomalies import AnomalyService
from app.database.base import Database
from app.forecasting import ForecastService
from app.timeseries import prepare_monthly_series

pytestmark = pytest.mark.slow


def _elapsed(fn: Callable[[], object]) -> float:
    fn()  # warm-up (first scan of a table, first statsmodels import)
    started = time.perf_counter()
    fn()
    return time.perf_counter() - started


@pytest.mark.parametrize(
    ("name", "operation", "budget"),
    [
        ("revenue series", lambda db: prepare_monthly_series(db, "revenue"), 2.0),
        ("mrr series", lambda db: prepare_monthly_series(db, "mrr"), 2.0),
        ("revenue forecast h=3", lambda db: ForecastService(db).forecast(metric="revenue", horizon=3), 5.0),
        ("mrr forecast h=6", lambda db: ForecastService(db).forecast(metric="mrr", horizon=6), 5.0),
        (
            "ticket forecast h=1",
            lambda db: ForecastService(db).forecast(metric="support_ticket_volume", horizon=1),
            5.0,
        ),
        ("revenue z-score", lambda db: AnomalyService(db).detect("revenue", detector="rolling_zscore"), 3.0),
        ("revenue IQR", lambda db: AnomalyService(db).detect("revenue", detector="iqr"), 3.0),
        ("mrr forecast residual", lambda db: AnomalyService(db).detect("mrr", detector="forecast_residual"), 3.0),
        ("scan of four metrics", lambda db: AnomalyService(db).scan(detector="forecast_residual"), 8.0),
        ("nested selection evaluation", lambda db: ForecastService(db).evaluate_selection("mrr", horizon=1), 10.0),
    ],
)
def test_phase3_operations_are_fast(
    full_db: Database, name: str, operation: Callable[[Database], object], budget: float
) -> None:
    seconds = _elapsed(lambda: operation(full_db))
    assert seconds < budget, (name, seconds)
