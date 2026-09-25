"""Production Phase 3 calculations against the independent reference (``tests/reference_timeseries.py``).

The reference is written from textbook definitions with plain Python and pandas and shares no
code with ``app/forecasting`` or ``app/anomalies``. Each check runs on several seeded random series.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.anomalies import AnomalyConfig, ForecastResidualDetails, IQRDetails
from app.anomalies.detectors import forecast_residual, iqr_detector, rolling_zscore, transform_series
from app.forecasting import build_method, error_metrics, fold_origins, rolling_origin_backtest
from tests import reference_timeseries as ref
from tests.phase3_support import synthetic_series

SEEDS = [1, 2, 3, 4, 5]


def series(seed: int, n: int = 30) -> list[float]:
    rng = np.random.default_rng(seed)
    trend = rng.uniform(-5.0, 25.0)
    return [float(500.0 + trend * i + rng.normal(0.0, 30.0)) for i in range(n)]


@pytest.mark.parametrize("seed", SEEDS)
def test_naive_and_drift_forecasts(seed: int) -> None:
    y = series(seed)
    for horizon in (1, 3, 6):
        assert list(build_method("naive").forecast(np.array(y), horizon, 0.95).mean) == ref.naive_forecast(y, horizon)
        assert list(build_method("drift").forecast(np.array(y), horizon, 0.95).mean) == pytest.approx(
            ref.drift_forecast(y, horizon)
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_error_metrics(seed: int) -> None:
    rng = np.random.default_rng(seed)
    actual = list(rng.uniform(50.0, 150.0, 20))
    forecast = [a + float(e) for a, e in zip(actual, rng.normal(0.0, 10.0, 20), strict=True)]
    m = error_metrics("x", 1, 20, forecast, actual)
    assert m.mae == pytest.approx(ref.mae(forecast, actual))
    assert m.rmse == pytest.approx(ref.rmse(forecast, actual))
    assert m.bias == pytest.approx(ref.bias(forecast, actual))
    assert m.mape == pytest.approx(ref.mape(forecast, actual))
    assert m.wape == pytest.approx(ref.wape(forecast, actual))
    actual[3] = 0.0
    zero = error_metrics("x", 1, 20, forecast, actual)
    assert zero.mape is None and ref.mape(forecast, actual) is None
    assert zero.wape == pytest.approx(ref.wape(forecast, actual))


@pytest.mark.parametrize(
    ("n", "initial", "horizon", "step"), [(24, 12, 1, 1), (24, 12, 3, 1), (24, 12, 6, 1), (30, 10, 3, 2)]
)
def test_rolling_split_boundaries(n: int, initial: int, horizon: int, step: int) -> None:
    expected = ref.rolling_splits(n, initial, horizon, step)
    assert fold_origins(n, initial, horizon, step) == [len(train) for train, _ in expected]
    y = series(1, n)
    labels = synthetic_series(y).labels()
    result = rolling_origin_backtest(
        y, labels, build_method("naive"), horizon=horizon, initial_window=initial, step=step
    )
    for fold, (train, validation) in zip(result.folds, expected, strict=True):
        assert fold.training_start == labels[train[0]] and fold.training_end == labels[train[-1]]
        assert fold.validation_start == labels[validation[0]] and fold.validation_end == labels[validation[-1]]
        assert max(train) < min(validation)
        assert fold.predictions == ref.naive_forecast([y[i] for i in train], horizon)
    pooled_forecast = [p for f in result.folds for p in f.predictions]
    pooled_actual = [a for f in result.folds for a in f.actuals]
    assert result.metrics.mae == pytest.approx(ref.mae(pooled_forecast, pooled_actual))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("transform", ["level", "pct_change"])
def test_rolling_zscores(seed: int, transform: str) -> None:
    y = series(seed)
    config = AnomalyConfig(window=12, min_history=6)
    reference_input = pd.Series(y) if transform == "level" else ref.pct_change(y)
    expected = ref.rolling_zscores(reference_input, 12, 6)
    output = rolling_zscore(y, list(range(len(y))), config, transform)  # type: ignore[arg-type]
    scored = {a.index: a.score for a in output.assessments}
    for index, value in expected.items():
        if math.isnan(value):
            assert index not in scored
        else:
            assert scored[index] == pytest.approx(value)
    assert np.allclose(transform_series(y, transform), reference_input.to_numpy(), equal_nan=True)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", SEEDS)
def test_iqr_bounds(seed: int) -> None:
    y = series(seed)
    config = AnomalyConfig(window=12, min_history=6)
    quartiles = ref.rolling_quartiles(pd.Series(y), 12, 6)
    for a in iqr_detector(y, list(range(len(y))), config, "level").assessments:
        assert isinstance(a.details, IQRDetails)
        q1, q3 = quartiles.loc[a.index, "q1"], quartiles.loc[a.index, "q3"]
        assert a.details.q1 == pytest.approx(q1) and a.details.q3 == pytest.approx(q3)
        assert a.details.lower_fence == pytest.approx(q1 - 1.5 * (q3 - q1))
        assert a.details.upper_fence == pytest.approx(q3 + 1.5 * (q3 - q1))


@pytest.mark.parametrize("seed", SEEDS)
def test_forecast_residuals(seed: int) -> None:
    y = series(seed)
    config = AnomalyConfig(window=12, min_history=6)
    residuals = ref.drift_one_step_residuals(y, 12, 6)
    labels = synthetic_series(y).labels()
    for a in forecast_residual(y, list(range(len(y))), config, labels).assessments:
        assert isinstance(a.details, ForecastResidualDetails)
        assert a.details.residual == pytest.approx(residuals[a.index])
        prior = [residuals[j] for j in range(max(0, a.index - 12), a.index) if j in residuals]
        assert a.details.residual_mean == pytest.approx(sum(prior) / len(prior))
        assert a.details.residual_std == pytest.approx(ref.sample_std(prior))
