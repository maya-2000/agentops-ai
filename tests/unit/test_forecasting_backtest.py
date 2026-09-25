"""Rolling-origin backtesting, model selection and the nested evaluation of the selection rule."""

from __future__ import annotations

import numpy as np
import pytest

from app.analytics.errors import InvalidRequestError
from app.forecasting import ForecastConfig, build_method, fold_origins, rolling_origin_backtest, select_model
from app.forecasting.selection import backtest_candidates, evaluate_selection
from tests.phase3_support import constant, linear, noisy, synthetic_series

LABELS = synthetic_series(noisy(24)).labels()


def test_fold_origins() -> None:
    assert fold_origins(24, 12, 3) == list(range(12, 22))
    assert fold_origins(24, 12, 1) == list(range(12, 24))
    assert fold_origins(24, 12, 3, step=3) == [12, 15, 18, 21]
    assert fold_origins(14, 12, 3) == []
    with pytest.raises(ValueError):
        fold_origins(24, 0, 3)


def test_folds_are_chronological_expanding_and_equal_to_direct_forecasts() -> None:
    y = np.array(noisy(24))
    method = build_method("drift")
    result = rolling_origin_backtest(y, LABELS, method, horizon=3, initial_window=12)
    assert result.eligible and len(result.folds) == 10 and not result.failures
    for fold, origin in zip(result.folds, range(12, 22), strict=True):
        assert fold.training_start == LABELS[0]  # expanding window
        assert fold.training_end == LABELS[origin - 1]
        assert fold.training_observations == origin
        assert fold.validation_start == LABELS[origin] > fold.training_end  # validation strictly after training
        assert fold.validation_end == LABELS[origin + 2]
        assert fold.predictions == list(method.forecast(y[:origin], 3, 0.95).mean)
        assert fold.actuals == pytest.approx(list(y[origin : origin + 3]))
    assert result.metrics.sample_count == 30 and result.metrics.fold_count == 10


def test_methods_that_cannot_fit_every_fold_are_ineligible() -> None:
    y = np.array(noisy(24))
    result = rolling_origin_backtest(y, LABELS, build_method("ets_damped_trend"), horizon=1, initial_window=8)
    assert not result.eligible
    assert len(result.failures) == 2  # origins 8 and 9 have fewer than 10 observations
    assert len(result.folds) == 14


def test_backtest_rejects_misaligned_labels() -> None:
    with pytest.raises(ValueError):
        rolling_origin_backtest([1.0, 2.0, 3.0], ["a"], build_method("naive"), horizon=1, initial_window=1)


def _config(**overrides: object) -> ForecastConfig:
    return ForecastConfig.model_validate(overrides)


def test_selection_prefers_a_candidate_that_beats_naive() -> None:
    y = linear(24)  # drift is exact on a straight line; naive lags
    config = _config()
    selection = select_model(backtest_candidates(y, LABELS, config, 3), config)
    assert selection.model == "drift"
    assert selection.improvement_over_baseline == pytest.approx(1.0)
    assert "lower than the naive baseline" in selection.reason and "rolling-origin folds" in selection.reason
    assert "accurate" not in selection.reason.lower()


def test_selection_keeps_naive_when_nothing_is_strictly_better() -> None:
    y = constant(24)  # every model is exact: ties go to the baseline
    config = _config()
    selection = select_model(backtest_candidates(y, LABELS, config, 3), config)
    assert selection.model == "naive"
    assert "no candidate achieved a lower" in selection.reason


def test_selection_ignores_ineligible_candidates() -> None:
    y = linear(24)
    config = _config(minimum_history=8)
    backtests = backtest_candidates(y, LABELS, config, 1)
    assert not backtests["ets_damped_trend"].eligible and not backtests["seasonal_naive"].eligible
    assert select_model(backtests, config).model == "drift"


def test_selection_by_rmse() -> None:
    config = _config(selection_metric="rmse")
    assert "RMSE" in select_model(backtest_candidates(linear(24), LABELS, config, 1), config).reason


def test_config_validation() -> None:
    assert ForecastConfig().required_history(3) == 12 + 2 + 3
    assert _config(backtest_step=2).required_history(1) == 12 + 4 + 1
    with pytest.raises(InvalidRequestError):
        _config(candidates=("drift",))  # the naive baseline is mandatory
    with pytest.raises(InvalidRequestError):
        _config(candidates=("naive", "naive"))
    with pytest.raises(ValueError):
        _config(confidence_level=1.0)
    with pytest.raises(ValueError):
        _config(minimum_history=2)


def test_nested_selection_evaluation() -> None:
    config = _config()
    evaluation = evaluate_selection(linear(24), LABELS, config, 1)
    required = config.required_history(1)
    assert [o.origin for o in evaluation.outcomes] == LABELS[required - 1 : 23]
    assert evaluation.selected_counts == {"drift": len(evaluation.outcomes)}
    assert evaluation.strategy_metrics.mae == pytest.approx(0.0)
    assert evaluation.naive_metrics.mae == pytest.approx(10.0)
    assert evaluate_selection(linear(24), LABELS, config, 6).message  # too short at h=6
