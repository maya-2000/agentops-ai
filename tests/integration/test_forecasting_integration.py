"""Forecasting against the generated dataset: every metric and horizon, dimensions, cutoffs and provenance."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from app.analytics.errors import UnsupportedDimensionError
from app.database.base import Database
from app.forecasting import ForecastResult, ForecastService, forecast_metric
from app.timeseries import UnsupportedMetricError

pytestmark = pytest.mark.slow

AS_OF = date(2026, 8, 31)
CASES: list[tuple[str, dict[str, str]]] = [
    ("revenue", {}),
    ("mrr", {}),
    ("customer_count", {}),
    ("support_ticket_volume", {}),
    ("product_adoption", {"product_feature": "Dashboards"}),
]


def comparable(result: ForecastResult) -> dict[str, Any]:
    data = result.model_dump(mode="json")
    data.pop("provenance")
    data["history"].pop("provenance")
    for key in ("operation_id", "execution_timestamp", "query_ids", "query_id"):
        data.pop(key)
    data["history"].pop("query_ids")
    data["history"].pop("query_id")
    return data


@pytest.fixture(scope="module")
def service(full_db: Database) -> ForecastService:
    return ForecastService(full_db)


@pytest.mark.parametrize(("metric", "filters"), CASES)
@pytest.mark.parametrize("horizon", [1, 3, 6])
def test_forecast_every_metric_and_horizon(
    service: ForecastService, metric: str, filters: dict[str, str], horizon: int
) -> None:
    result = service.forecast(metric=metric, horizon=horizon, filters=filters)
    assert result.status == "ok", result.message
    assert result.cutoff_date == AS_OF and result.historical_end == AS_OF
    assert result.historical_start == date(2024, 9, 1) and result.history_observations == 24
    expected_periods = ["2026-09", "2026-10", "2026-11", "2026-12", "2027-01", "2027-02"][:horizon]
    assert [p.period for p in result.forecast_points] == expected_periods
    for point in result.forecast_points:
        assert point.predicted_value >= 0
        assert point.lower_bound is not None and point.upper_bound is not None
        assert 0 <= point.lower_bound <= point.predicted_value <= point.upper_bound
        if metric == "product_adoption":
            assert point.upper_bound <= 1
    assert result.model in {c.model for c in result.candidates if c.eligible}
    assert result.baseline_metrics is not None and result.baseline_metrics.sample_count > 0
    backtest = result.backtest(result.model or "")
    assert backtest.metrics.fold_count >= 3 and backtest.metrics.horizon == horizon
    for fold in backtest.folds:
        assert fold.training_start == "2024-09" and fold.training_end < fold.validation_start
        assert fold.validation_end <= "2026-08"  # backtests never reach beyond the cutoff
    assert result.selected_model_reason and "backtest" in result.selected_model_reason
    assert result.method is not None and result.method.training_end == AS_OF
    assert result.provenance.queries and result.query_id in result.query_ids
    expected_tables = {"product_features"} if metric == "product_adoption" else {"customers"}
    assert expected_tables <= set(result.source_tables)


def test_forecasts_are_deterministic(full_db: Database) -> None:
    first = forecast_metric(full_db, "mrr", 3)
    second = forecast_metric(full_db, "mrr", 3)
    assert comparable(first) == comparable(second)
    assert first.operation_id != second.operation_id  # each run is its own traceable operation


def test_earlier_cutoff(service: ForecastService) -> None:
    result = service.forecast(metric="revenue", horizon=3, cutoff_date=date(2026, 5, 31))
    assert result.status == "ok"
    assert result.historical_end == date(2026, 5, 31)
    assert [p.period for p in result.forecast_points] == ["2026-06", "2026-07", "2026-08"]
    for query in result.provenance.queries:
        assert query.parameters["end_date"] <= "2026-05-31"


def test_mid_month_cutoff_uses_complete_months_only(service: ForecastService) -> None:
    result = service.forecast(metric="support_ticket_volume", horizon=1, cutoff_date=date(2026, 8, 20))
    assert result.historical_end == date(2026, 7, 31)
    assert [p.period for p in result.forecast_points] == ["2026-08"]


@pytest.mark.parametrize(
    ("metric", "filters"),
    [
        ("revenue", {"region": "APAC"}),
        ("revenue", {"segment": "SMB"}),
        ("mrr", {"segment": "Enterprise"}),
        ("support_ticket_volume", {"ticket_category": "Billing"}),
    ],
)
def test_explicit_dimension_forecasts(service: ForecastService, metric: str, filters: dict[str, str]) -> None:
    result = service.forecast(metric=metric, horizon=3, filters=filters)
    assert result.status == "ok" and result.filters == filters
    assert all(f"{k}={v}" in result.calculation for k, v in filters.items())


def test_insufficient_history_and_sparse_groups(service: ForecastService) -> None:
    launched = service.forecast(metric="product_adoption", horizon=3, filters={"product_feature": "AI Insights"})
    assert launched.status == "insufficient_history" and not launched.forecast_points
    assert launched.message and "6 usable month(s)" in launched.message
    sparse = service.forecast(
        metric="support_ticket_volume",
        horizon=1,
        filters={"country": "France", "ticket_category": "Billing", "ticket_priority": "Urgent"},
    )
    assert sparse.status == "insufficient_data" and "intermittent" in (sparse.message or "")
    early = service.forecast(metric="revenue", horizon=6, cutoff_date=date(2025, 9, 30))
    assert early.status == "insufficient_history"


def test_invalid_requests(service: ForecastService) -> None:
    with pytest.raises(UnsupportedMetricError):
        service.forecast(metric="nrr", horizon=3)
    with pytest.raises(UnsupportedDimensionError):
        service.forecast(metric="revenue", horizon=3, filters={"customer_id": "CUST-000001"})


def test_nested_selection_evaluation(service: ForecastService) -> None:
    evaluation = service.evaluate_selection("mrr", horizon=1)
    assert evaluation.outcomes and evaluation.strategy_metrics.sample_count == len(evaluation.outcomes)
    assert evaluation.naive_metrics.sample_count == evaluation.strategy_metrics.sample_count
    assert all(o.origin < "2026-08" for o in evaluation.outcomes)
    assert sum(evaluation.selected_counts.values()) == len(evaluation.outcomes)
