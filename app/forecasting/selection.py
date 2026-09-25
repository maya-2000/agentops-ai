"""Model selection from historical backtests only, plus an honest evaluation of that selection rule.

Selection rule (deterministic):

1. Only **eligible** candidates compete: those that produced a forecast on every backtest fold,
   so all are scored on exactly the same months.
2. Rank by the selection metric (MAE by default), then RMSE, then candidate order (the simpler
   baselines come first).
3. The winner must be **strictly better than the naive baseline** on the selection metric;
   otherwise the naive baseline is selected and the result says so.

The backtest metrics of the selected model were also used to select it, so they are optimistic
(selection bias). ``evaluate_selection`` measures the whole procedure without that bias. At each
later origin it re-runs selection using only earlier data, then scores the chosen model on the
months that follow.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel

from app.forecasting.backtest import BacktestResult, rolling_origin_backtest
from app.forecasting.config import BASELINE_MODEL, ForecastConfig
from app.forecasting.evaluation import ErrorMetrics, error_metrics
from app.forecasting.methods import build_method


class Selection(BaseModel):
    model: str
    reason: str
    baseline_model: str
    improvement_over_baseline: float | None  # relative reduction of the selection metric vs the baseline


def backtest_candidates(
    values: Sequence[float] | np.ndarray,
    labels: Sequence[str],
    config: ForecastConfig,
    horizon: int,
    candidates: Sequence[str] | None = None,
) -> dict[str, BacktestResult]:
    return {
        name: rolling_origin_backtest(
            values,
            labels,
            build_method(name, config),
            horizon=horizon,
            initial_window=config.minimum_history,
            step=config.backtest_step,
            level=config.confidence_level,
            near_zero_fraction=config.mape_near_zero_fraction,
        )
        for name in (candidates or config.candidates)
    }


def select_model(backtests: Mapping[str, BacktestResult], config: ForecastConfig) -> Selection:
    metric = config.selection_metric
    baseline = backtests.get(BASELINE_MODEL)
    if baseline is None or not baseline.eligible:
        raise ValueError("The naive baseline must be backtested on every fold before selection")
    order: dict[str, int] = {name: index for index, name in enumerate(config.candidates)}

    def key(result: BacktestResult) -> tuple[float, float, int]:
        m = result.metrics
        return (_metric(m, metric), m.rmse if m.rmse is not None else float("inf"), order.get(result.model, 99))

    eligible = sorted((b for b in backtests.values() if b.eligible), key=key)
    best = eligible[0]
    baseline_value = _metric(baseline.metrics, metric)
    folds = baseline.metrics.fold_count
    label = metric.upper()
    if best.model != BASELINE_MODEL and _metric(best.metrics, metric) < baseline_value:
        best_value = _metric(best.metrics, metric)
        improvement = (baseline_value - best_value) / baseline_value if baseline_value > 0 else None
        pct = f" ({improvement:.1%} lower)" if improvement is not None else ""
        return Selection(
            model=best.model,
            reason=(
                f"Selected {best.model}: lowest historical backtest {label} ({_fmt(best_value)}) across {folds} "
                f"rolling-origin folds, lower than the naive baseline's {label} ({_fmt(baseline_value)}){pct}."
            ),
            baseline_model=BASELINE_MODEL,
            improvement_over_baseline=improvement,
        )
    return Selection(
        model=BASELINE_MODEL,
        reason=(
            f"Selected the naive baseline: no candidate achieved a lower historical backtest {label} than the "
            f"naive baseline ({_fmt(baseline_value)}) across {folds} rolling-origin folds."
        ),
        baseline_model=BASELINE_MODEL,
        improvement_over_baseline=0.0,
    )


class SelectionOutcome(BaseModel):
    origin: str  # last training month
    selected_model: str
    predictions: list[float]
    naive_predictions: list[float]
    actuals: list[float]


class SelectionEvaluation(BaseModel):
    """Out-of-sample errors of the selection *procedure* versus always using the naive baseline."""

    horizon: int
    outcomes: list[SelectionOutcome]
    strategy_metrics: ErrorMetrics
    naive_metrics: ErrorMetrics
    selected_counts: dict[str, int]
    message: str | None = None


def evaluate_selection(
    values: Sequence[float] | np.ndarray, labels: Sequence[str], config: ForecastConfig, horizon: int
) -> SelectionEvaluation:
    """Nested rolling evaluation: select on data before each origin, score on the months after it."""
    y = np.asarray(values, dtype=float)
    naive = build_method(BASELINE_MODEL, config)
    outcomes: list[SelectionOutcome] = []
    for origin in range(config.required_history(horizon), len(y) - horizon + 1, config.backtest_step):
        history, history_labels = y[:origin], list(labels[:origin])
        selection = select_model(backtest_candidates(history, history_labels, config, horizon), config)
        chosen = build_method(selection.model, config).forecast(history, horizon, config.confidence_level)
        baseline = naive.forecast(history, horizon, config.confidence_level)
        outcomes.append(
            SelectionOutcome(
                origin=labels[origin - 1],
                selected_model=selection.model,
                predictions=list(chosen.mean),
                naive_predictions=list(baseline.mean),
                actuals=[float(v) for v in y[origin : origin + horizon]],
            )
        )
    actuals = [a for o in outcomes for a in o.actuals]
    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.selected_model] = counts.get(outcome.selected_model, 0) + 1
    near_zero = config.mape_near_zero_fraction
    return SelectionEvaluation(
        horizon=horizon,
        outcomes=outcomes,
        strategy_metrics=error_metrics(
            "selection_strategy",
            horizon,
            len(outcomes),
            [p for o in outcomes for p in o.predictions],
            actuals,
            near_zero_fraction=near_zero,
        ),
        naive_metrics=error_metrics(
            BASELINE_MODEL,
            horizon,
            len(outcomes),
            [p for o in outcomes for p in o.naive_predictions],
            actuals,
            near_zero_fraction=near_zero,
        ),
        selected_counts=counts,
        message=None if outcomes else "The series is too short for a nested evaluation at this horizon.",
    )


def _metric(metrics: ErrorMetrics, name: str) -> float:
    value = metrics.mae if name == "mae" else metrics.rmse
    return value if value is not None else float("inf")


def _fmt(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:,.2f}"
    return f"{value:.4f}"
