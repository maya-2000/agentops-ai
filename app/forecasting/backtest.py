"""Rolling-origin (expanding-window) backtesting. Folds are always chronological, never random splits.

For a series ``y_1..y_n``, an initial window ``w``, horizon ``h`` and step ``s``, fold ``k``
(origin ``o_k = w + k*s``) trains on ``y_1..y_{o_k}`` and validates on ``y_{o_k+1}..y_{o_k+h}``:

    fold 1: train [1 .. w]       validate [w+1 .. w+h]
    fold 2: train [1 .. w+s]     validate [w+s+1 .. w+s+h]
    ...     (while the validation window ends on or before n)

Each fold's validation months lie strictly after its training months. A fold's forecast is a
function of its training slice only, so observations after a fold's validation window cannot
change that fold (tested by the leakage regression tests).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field

from app.forecasting.base import ForecastMethod, MethodUnavailableError
from app.forecasting.evaluation import ErrorMetrics, error_metrics


class BacktestFold(BaseModel):
    fold: int
    training_start: str
    training_end: str
    training_observations: int
    validation_start: str
    validation_end: str
    predictions: list[float]
    actuals: list[float]
    lower: list[float] | None = None
    upper: list[float] | None = None


class BacktestResult(BaseModel):
    model: str
    horizon: int
    initial_window: int
    step: int
    eligible: bool
    folds: list[BacktestFold] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    metrics: ErrorMetrics


def fold_origins(observations: int, initial_window: int, horizon: int, step: int = 1) -> list[int]:
    """Training-set sizes of every fold (the validation window starts right after each)."""
    if initial_window < 1 or horizon < 1 or step < 1:
        raise ValueError("initial_window, horizon and step must be positive")
    return list(range(initial_window, observations - horizon + 1, step))


def rolling_origin_backtest(
    values: Sequence[float] | np.ndarray,
    labels: Sequence[str],
    method: ForecastMethod,
    *,
    horizon: int,
    initial_window: int,
    step: int = 1,
    level: float = 0.95,
    near_zero_fraction: float = 0.01,
) -> BacktestResult:
    """Backtest ``method`` on every rolling origin of a gap-free series."""
    y = np.asarray(values, dtype=float)
    if len(labels) != len(y):
        raise ValueError("labels and values must have the same length")
    folds: list[BacktestFold] = []
    failures: list[str] = []
    for number, origin in enumerate(fold_origins(len(y), initial_window, horizon, step), start=1):
        try:
            output = method.forecast(y[:origin], horizon, level)
        except MethodUnavailableError as exc:
            failures.append(f"fold {number} (training to {labels[origin - 1]}): {exc}")
            continue
        folds.append(
            BacktestFold(
                fold=number,
                training_start=labels[0],
                training_end=labels[origin - 1],
                training_observations=origin,
                validation_start=labels[origin],
                validation_end=labels[origin + horizon - 1],
                predictions=list(output.mean),
                actuals=[float(v) for v in y[origin : origin + horizon]],
                lower=list(output.lower) if output.lower is not None else None,
                upper=list(output.upper) if output.upper is not None else None,
            )
        )
    predictions = [p for f in folds for p in f.predictions]
    actuals = [a for f in folds for a in f.actuals]
    lower = [lo for f in folds for lo in _bounds(f.lower, len(f.predictions))]
    upper = [hi for f in folds for hi in _bounds(f.upper, len(f.predictions))]
    metrics = error_metrics(
        method.name,
        horizon,
        len(folds),
        predictions,
        actuals,
        lower,
        upper,
        near_zero_fraction=near_zero_fraction,
    )
    return BacktestResult(
        model=method.name,
        horizon=horizon,
        initial_window=initial_window,
        step=step,
        eligible=bool(folds) and not failures,
        folds=folds,
        failures=failures,
        metrics=metrics,
    )


def _bounds(bounds: list[float] | None, size: int) -> list[float | None]:
    return list(bounds) if bounds is not None else [None] * size
