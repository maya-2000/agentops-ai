"""Leakage regression tests for forecasting: changing the future must not change the past.

Each test perturbs observations from index ``K`` onwards and checks that everything computed
from information before ``K`` is unchanged: backtest predictions, training statistics, model
selection and the forecast itself.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from app.analytics.errors import InvalidRequestError
from app.forecasting import MODEL_NAMES, ForecastConfig, ForecastRequest, build_method, forecast_series
from app.forecasting.selection import backtest_candidates, evaluate_selection, select_model
from tests.phase3_support import noisy, synthetic_series

K = 18
BASE = noisy(24)
FUTURE_CHANGED = BASE[:K] + [v * 3.0 + 500.0 for v in BASE[K:]]
LABELS = synthetic_series(BASE).labels()
CONFIG = ForecastConfig()


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_backtest_predictions_before_the_change_are_identical(model: str) -> None:
    method = build_method(model, CONFIG)
    original = backtest_candidates(BASE, LABELS, CONFIG, 3, [model])[model]
    changed = backtest_candidates(FUTURE_CHANGED, LABELS, CONFIG, 3, [model])[model]
    for before, after in zip(original.folds, changed.folds, strict=True):
        if before.training_observations <= K:
            # the fold's inputs lie before K: its forecast (and bounds) cannot change
            assert before.predictions == after.predictions
            assert before.lower == after.lower and before.upper == after.upper
        if before.training_observations + 3 <= K:
            assert before == after  # validation months are before K too: the whole fold is unchanged
    assert method.forecast(np.array(BASE[:K]), 3, 0.95) == method.forecast(np.array(FUTURE_CHANGED[:K]), 3, 0.95)


def test_backtest_folds_do_depend_on_their_own_training_data() -> None:
    """Control: the check above is meaningful because changing training data does change folds."""
    past_changed = [BASE[0] + 1000.0, *BASE[1:]]
    original = backtest_candidates(BASE, LABELS, CONFIG, 3, ["drift"])["drift"]
    changed = backtest_candidates(past_changed, LABELS, CONFIG, 3, ["drift"])["drift"]
    assert original.folds[0].predictions != changed.folds[0].predictions


def test_selection_on_history_ignores_later_observations() -> None:
    before = select_model(backtest_candidates(BASE[:K], LABELS[:K], CONFIG, 1), CONFIG)
    after = select_model(backtest_candidates(FUTURE_CHANGED[:K], LABELS[:K], CONFIG, 1), CONFIG)
    assert before == after


def test_forecast_at_a_cutoff_ignores_later_observations() -> None:
    cutoff = date(2026, 2, 28)  # month K-1 = 2026-02
    request = ForecastRequest(metric="revenue", horizon=3)
    original = forecast_series(synthetic_series(BASE[:K]), request, CONFIG, cutoff=cutoff)
    changed = forecast_series(synthetic_series(FUTURE_CHANGED[:K]), request, CONFIG, cutoff=cutoff)
    assert original.status == "ok"
    assert original.forecast_points == changed.forecast_points
    assert original.backtests == changed.backtests
    assert original.selected_model_reason == changed.selected_model_reason
    assert [p.period for p in original.forecast_points] == ["2026-03", "2026-04", "2026-05"]


def test_forecast_refuses_a_series_that_extends_past_the_cutoff() -> None:
    with pytest.raises(InvalidRequestError):
        forecast_series(
            synthetic_series(BASE), ForecastRequest(metric="revenue", horizon=1), CONFIG, cutoff=date(2026, 2, 28)
        )


def test_nested_evaluation_origins_only_use_earlier_data() -> None:
    original = evaluate_selection(BASE, LABELS, CONFIG, 1)
    changed = evaluate_selection(FUTURE_CHANGED, LABELS, CONFIG, 1)
    for a, b in zip(original.outcomes, changed.outcomes, strict=True):
        origin = LABELS.index(a.origin) + 1  # training size
        if origin <= K:
            assert a.selected_model == b.selected_model
            assert a.predictions == b.predictions
