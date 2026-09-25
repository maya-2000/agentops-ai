"""Independent reference implementations of the Phase 3 calculations.

They deliberately share **no code** with ``app/forecasting`` or ``app/anomalies``: plain Python
loops and pandas, written from the textbook definitions. Agreement between the two
implementations is the correctness evidence for the production code.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd


def naive_forecast(history: Sequence[float], horizon: int) -> list[float]:
    return [float(history[-1])] * horizon


def drift_forecast(history: Sequence[float], horizon: int) -> list[float]:
    slope = (history[-1] - history[0]) / (len(history) - 1)
    return [history[-1] + slope * step for step in range(1, horizon + 1)]


def mae(forecast: Sequence[float], actual: Sequence[float]) -> float:
    return sum(abs(f - a) for f, a in zip(forecast, actual, strict=True)) / len(actual)


def rmse(forecast: Sequence[float], actual: Sequence[float]) -> float:
    return math.sqrt(sum((f - a) ** 2 for f, a in zip(forecast, actual, strict=True)) / len(actual))


def bias(forecast: Sequence[float], actual: Sequence[float]) -> float:
    return sum(f - a for f, a in zip(forecast, actual, strict=True)) / len(actual)


def mape(forecast: Sequence[float], actual: Sequence[float]) -> float | None:
    if any(a == 0 for a in actual):
        return None
    return sum(abs(f - a) / abs(a) for f, a in zip(forecast, actual, strict=True)) / len(actual)


def wape(forecast: Sequence[float], actual: Sequence[float]) -> float | None:
    total = sum(abs(a) for a in actual)
    if total == 0:
        return None
    return sum(abs(f - a) for f, a in zip(forecast, actual, strict=True)) / total


def rolling_splits(n: int, initial: int, horizon: int, step: int) -> list[tuple[list[int], list[int]]]:
    """(training indices, validation indices) per fold, expanding window."""
    splits = []
    origin = initial
    while origin + horizon <= n:
        splits.append((list(range(origin)), list(range(origin, origin + horizon))))
        origin += step
    return splits


def pct_change(values: Sequence[float]) -> pd.Series:
    return pd.Series(values, dtype=float).pct_change()


def rolling_zscores(series: pd.Series, window: int, min_history: int) -> pd.Series:
    """z of each point against the mean/std (ddof=1) of the previous ``window`` points (shifted by one)."""
    prior_mean = series.rolling(window, min_periods=min_history).mean().shift(1)
    prior_std = series.rolling(window, min_periods=min_history).std(ddof=1).shift(1)
    return (series - prior_mean) / prior_std


def rolling_quartiles(series: pd.Series, window: int, min_history: int) -> pd.DataFrame:
    """Q1/Q3 (linear interpolation) of the previous ``window`` points, shifted by one."""
    roll = series.rolling(window, min_periods=min_history)
    return pd.DataFrame({"q1": roll.quantile(0.25).shift(1), "q3": roll.quantile(0.75).shift(1)})


def drift_one_step_residuals(values: Sequence[float], window: int, min_history: int) -> dict[int, float]:
    """Residual y_j - drift forecast fitted on the previous ``window`` points, for every j with enough history."""
    residuals = {}
    for j in range(len(values)):
        history = values[max(0, j - window) : j]
        if len(history) >= min_history:
            residuals[j] = values[j] - drift_forecast(history, 1)[0]
    return residuals


def sample_std(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
