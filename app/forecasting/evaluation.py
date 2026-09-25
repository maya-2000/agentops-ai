"""Forecast error metrics.

Conventions (``e = forecast - actual``):

- **MAE** = mean(|e|), **RMSE** = sqrt(mean(e^2)), in the metric's unit.
- **Bias** = mean(e): positive means the method over-forecast on average.
- **WAPE** = sum(|e|) / sum(|actual|), a fraction (0.05 = 5%). It stays defined when individual
  actuals are zero and is the preferred scale-free measure.
- **MAPE** = mean(|e| / |actual|), a fraction. Zero-denominator policy: MAPE is reported only
  when **every** actual is non-zero and at least ``near_zero_fraction`` of the mean absolute actual.
  Otherwise it is ``None`` with an explanatory note. It is never infinite, and points are never
  silently dropped (MAPE always covers the same sample as MAE and RMSE).
- **Interval coverage** = the share of actuals inside the reported prediction interval, over the
  points where an interval exists.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel


class ErrorMetrics(BaseModel):
    model: str
    horizon: int
    fold_count: int
    sample_count: int
    mae: float | None
    rmse: float | None
    mape: float | None
    wape: float | None
    bias: float | None
    interval_coverage: float | None
    interval_sample_count: int
    mape_note: str | None = None


def error_metrics(
    model: str,
    horizon: int,
    fold_count: int,
    predictions: Sequence[float],
    actuals: Sequence[float],
    lower: Sequence[float | None] | None = None,
    upper: Sequence[float | None] | None = None,
    *,
    near_zero_fraction: float = 0.01,
) -> ErrorMetrics:
    forecast = np.asarray(predictions, dtype=float)
    actual = np.asarray(actuals, dtype=float)
    if forecast.shape != actual.shape:
        raise ValueError("predictions and actuals must have the same length")
    n = len(actual)
    coverage, covered_n = _coverage(actual, lower, upper)
    if n == 0:
        return ErrorMetrics(
            model=model,
            horizon=horizon,
            fold_count=fold_count,
            sample_count=0,
            mae=None,
            rmse=None,
            mape=None,
            wape=None,
            bias=None,
            interval_coverage=coverage,
            interval_sample_count=covered_n,
            mape_note="No backtest predictions.",
        )
    errors = forecast - actual
    absolute = np.abs(errors)
    total_actual = float(np.sum(np.abs(actual)))
    mape, note = _mape(absolute, actual, near_zero_fraction)
    return ErrorMetrics(
        model=model,
        horizon=horizon,
        fold_count=fold_count,
        sample_count=n,
        mae=float(np.mean(absolute)),
        rmse=math.sqrt(float(np.mean(np.square(errors)))),
        mape=mape,
        wape=float(np.sum(absolute)) / total_actual if total_actual > 0 else None,
        bias=float(np.mean(errors)),
        interval_coverage=coverage,
        interval_sample_count=covered_n,
        mape_note=note,
    )


def _mape(absolute: np.ndarray, actual: np.ndarray, near_zero_fraction: float) -> tuple[float | None, str | None]:
    magnitude = np.abs(actual)
    if np.any(magnitude == 0):
        return None, "MAPE not reported: at least one actual is zero (undefined percentage error). Use WAPE."
    floor = near_zero_fraction * float(np.mean(magnitude))
    if np.any(magnitude < floor):
        return None, (
            f"MAPE not reported: at least one actual is below {near_zero_fraction:.0%} of the mean absolute "
            "actual, which would dominate the average. Use WAPE."
        )
    return float(np.mean(absolute / magnitude)), None


def _coverage(
    actual: np.ndarray, lower: Sequence[float | None] | None, upper: Sequence[float | None] | None
) -> tuple[float | None, int]:
    if lower is None or upper is None:
        return None, 0
    inside = [
        lo <= a <= hi
        for a, lo, hi in zip(actual.tolist(), lower, upper, strict=True)
        if lo is not None and hi is not None
    ]
    return (sum(inside) / len(inside), len(inside)) if inside else (None, 0)
