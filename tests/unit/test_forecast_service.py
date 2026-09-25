"""The forecasting service core (``forecast_series``) on synthetic series: statuses, rules and the result contract."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import pytest

from app.analytics.errors import InvalidRequestError
from app.forecasting import ForecastConfig, ForecastRequest, ForecastResult, forecast_series
from app.timeseries import InvalidHorizonError, UnsupportedMethodError
from tests.phase3_support import linear, noisy, synthetic_series

CUTOFF = date(2026, 8, 31)  # 24 synthetic months from 2024-09 end here
CONFIG = ForecastConfig()


def run(
    values: Sequence[float | None], *, cutoff: date = CUTOFF, metric: str = "revenue", **request: Any
) -> ForecastResult:
    series = synthetic_series(values, metric=metric)
    return forecast_series(series, ForecastRequest(metric=metric, **request), CONFIG, cutoff=cutoff)


def comparable(result: ForecastResult) -> dict[str, Any]:
    """Result content without run-specific identifiers and timestamps."""
    data = result.model_dump(mode="json")
    data.pop("provenance")
    data["history"].pop("provenance")
    for key in ("operation_id", "execution_timestamp", "query_ids"):
        data.pop(key)
    return data


def test_linear_series_selects_drift_and_continues_the_line() -> None:
    result = run(linear(24), horizon=3)
    assert result.status == "ok" and result.sufficient_history
    assert result.model == "drift"
    assert [p.period for p in result.forecast_points] == ["2026-09", "2026-10", "2026-11"]
    assert [p.predicted_value for p in result.forecast_points] == pytest.approx([340.0, 350.0, 360.0])
    assert result.historical_start == date(2024, 9, 1) and result.historical_end == date(2026, 8, 31)
    assert result.cutoff_date == CUTOFF and result.history_observations == 24
    assert result.baseline_metrics is not None and result.baseline_metrics.model == "naive"
    assert result.backtest_metrics is not None and result.backtest_metrics.model == "drift"
    assert result.backtest_metrics.mae == pytest.approx(0.0)
    assert result.improvement_over_baseline == pytest.approx(1.0)
    assert result.selected_model_reason and "lower than the naive baseline" in result.selected_model_reason


def test_result_contract_and_provenance() -> None:
    result = run(noisy(24), horizon=6, confidence_level=0.9)
    assert result.confidence_level == 0.9
    assert result.interval is not None and result.interval.kind == "prediction_interval" and result.interval.available
    assert result.lower_bound == [p.lower_bound for p in result.forecast_points]
    assert result.upper_bound == [p.upper_bound for p in result.forecast_points]
    assert {c.model for c in result.candidates} == set(CONFIG.candidates)
    assert result.method is not None
    assert result.method.training_end == date(2026, 8, 31) and result.method.cutoff_date == CUTOFF
    assert result.method.random_seed == CONFIG.random_seed and result.method.parameters
    assert result.provenance.operation == "forecast:revenue" and result.operation_id.startswith("T-")
    assert result.history.points[-1].period == "2026-08"
    assert "prediction interval" in result.calculation
    assert any("prediction intervals" in note for note in result.limitations)
    assert any("backtest" in note and "contained" in note for note in result.limitations)
    for fold in result.backtest(result.model or "").folds:
        assert fold.validation_start > fold.training_end


def test_deterministic() -> None:
    assert comparable(run(noisy(24), horizon=3)) == comparable(run(noisy(24), horizon=3))


@pytest.mark.parametrize("horizon", [0, 7, 12, -1])
def test_invalid_horizons(horizon: int) -> None:
    with pytest.raises(InvalidHorizonError):
        ForecastRequest(metric="revenue", horizon=horizon)


def test_invalid_model_and_confidence() -> None:
    with pytest.raises(UnsupportedMethodError):
        ForecastRequest(metric="revenue", model="lstm")
    with pytest.raises(ValueError):
        ForecastRequest(metric="revenue", confidence_level=1.0)
    with pytest.raises(ValueError):
        ForecastRequest(metric="revenue", unexpected=True)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("months", "horizon", "status"), [(17, 3, "ok"), (16, 3, "insufficient_history"), (15, 1, "ok")]
)
def test_minimum_history_boundary(months: int, horizon: int, status: str) -> None:
    result = run(noisy(months), cutoff=date(2030, 1, 1), horizon=horizon)
    assert result.status == status
    if status != "ok":
        assert not result.sufficient_history and not result.forecast_points and result.model is None
        assert result.message and f"at least {CONFIG.required_history(horizon)}" in result.message


def test_missing_months_use_only_the_gap_free_tail() -> None:
    values: list[float | None] = list(noisy(30))
    values[3] = None
    result = run(values, cutoff=date(2030, 1, 1), horizon=3)
    assert result.status == "ok" and result.history_observations == 26
    assert any("missing" in note for note in result.limitations)
    values[20] = None  # the gap-free tail is now too short
    short = run(values, cutoff=date(2030, 1, 1), horizon=3)
    assert short.status == "insufficient_history"


def test_series_ending_in_a_missing_month() -> None:
    values: list[float | None] = [*noisy(23), None]
    assert run(values, horizon=1).status == "insufficient_history"


def test_zero_values_and_intermittent_series() -> None:
    values = noisy(24)
    values[2] = values[9] = 0.0
    result = run(values, horizon=3)
    assert result.status == "ok"
    assert result.baseline_metrics is not None and result.baseline_metrics.wape is not None
    sparse = [0.0 if i % 3 == 0 else 50.0 + i for i in range(24)]
    rejected = run(sparse, horizon=3, metric="support_ticket_volume")
    assert rejected.status == "insufficient_data" and "intermittent" in (rejected.message or "")


def test_forecasts_are_limited_to_the_metric_range() -> None:
    declining = [240.0 - 20 * i for i in range(12)] + [0.0] * 0
    declining = [max(v, 5.0) for v in declining] + [5.0 - 0.3 * i for i in range(12)]
    result = run(declining, horizon=6)
    assert result.status == "ok"
    assert all(p.predicted_value >= 0 and (p.lower_bound or 0) >= 0 for p in result.forecast_points)
    adoption = run([0.90 + 0.01 * i for i in range(24)], metric="product_adoption", horizon=6)
    assert all(p.predicted_value <= 1 and (p.upper_bound or 0) <= 1 for p in adoption.forecast_points)
    assert any("possible range" in note for note in adoption.limitations)


def test_explicit_model_is_compared_with_the_baseline() -> None:
    result = run(noisy(24), horizon=3, model="moving_average")
    assert result.model == "moving_average"
    assert [c.model for c in result.candidates] == ["naive", "moving_average"]
    assert result.selected_model_reason and "requested explicitly" in result.selected_model_reason
    naive = run(noisy(24), horizon=3, model="naive")
    assert naive.model == "naive" and [c.model for c in naive.candidates] == ["naive"]


def test_seasonality_assessment() -> None:
    full = run(noisy(24), horizon=1)
    assert full.seasonality is not None and full.seasonality.full_cycles == 2 and full.seasonality.estimable
    short = run(noisy(18), cutoff=date(2030, 1, 1), horizon=1)
    assert short.seasonality is not None and not short.seasonality.estimable
    assert short.seasonality.seasonal_naive_available


def test_series_after_the_cutoff_is_refused() -> None:
    with pytest.raises(InvalidRequestError):
        run(noisy(24), cutoff=date(2026, 7, 31), horizon=1)


def test_no_data_series() -> None:
    result = run([None, None, None], horizon=1)
    assert result.status == "no_data" and not result.forecast_points
