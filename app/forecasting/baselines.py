"""Benchmark forecasting methods: naive, seasonal naive, moving average and drift.

Prediction intervals (Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed.,
section 5.5) assume normally distributed, uncorrelated residuals with ``sigma`` estimated from
the method's one-step residuals on the training data:

========================  ==================================  =====================================
method                    h-step standard deviation            residuals
========================  ==================================  =====================================
naive                     sigma * sqrt(h)                      y_t - y_{t-1}
seasonal naive (m)        sigma * sqrt(k + 1), k=floor((h-1)/m) y_t - y_{t-m}
drift                     sigma * sqrt(h * (1 + h / (T - 1)))  y_t - y_{t-1} - b (one parameter, b)
moving average (k)        empirical RMS of the method's own     y_{t+h-1} - mean(y_{t-k}..y_{t-1})
                          h-step errors on the training data
========================  ==================================  =====================================

The drift variance follows from the forecast error ``sum of h future shocks - h * (b_hat - b)``
with ``Var(b_hat) = sigma^2 / (T - 1)``. The moving average has no textbook closed form, so its
interval is residual-based. No interval is reported when fewer than
``MIN_RESIDUALS_FOR_INTERVAL`` residuals exist.
"""

from __future__ import annotations

import numpy as np

from app.forecasting.base import ForecastMethod, MethodOutput, residual_sigma, symmetric_interval
from app.forecasting.config import MIN_RESIDUALS_FOR_INTERVAL

_TOO_FEW_RESIDUALS = "Too few residuals to estimate the prediction interval; point forecast only."


class NaiveMethod(ForecastMethod):
    name = "naive"
    display_name = "Naive (last value)"
    description = "Every future month equals the last observed month."
    interval_method = "analytic: sigma * sqrt(h), sigma from one-step naive residuals (normal errors)"
    min_observations = 1

    def _forecast(self, y: np.ndarray, horizon: int, level: float) -> MethodOutput:
        mean = np.full(horizon, y[-1])
        residuals = np.diff(y)
        sigma = residual_sigma(residuals) if len(residuals) >= MIN_RESIDUALS_FOR_INTERVAL else None
        parameters: dict[str, float | int | str] = {"last_value": float(y[-1]), "residuals": len(residuals)}
        if sigma is None:
            return MethodOutput(tuple(mean.tolist()), None, None, parameters, _TOO_FEW_RESIDUALS)
        parameters["residual_sigma"] = sigma
        steps = np.arange(1, horizon + 1)
        lower, upper = symmetric_interval(mean, sigma * np.sqrt(steps), level)
        return MethodOutput(tuple(mean.tolist()), lower, upper, parameters)


class SeasonalNaiveMethod(ForecastMethod):
    name = "seasonal_naive"
    display_name = "Seasonal naive"
    interval_method = "analytic: sigma * sqrt(k + 1), sigma from seasonal-difference residuals (normal errors)"

    def __init__(self, season_length: int = 12):
        self.season_length = season_length
        self.min_observations = season_length
        self.description = (
            f"Every future month equals the same month one season ({season_length} months) earlier. "
            "Needs at least one full season of history."
        )

    def _forecast(self, y: np.ndarray, horizon: int, level: float) -> MethodOutput:
        m, n = self.season_length, len(y)
        steps = np.arange(1, horizon + 1)
        mean = np.array([y[n - m + ((h - 1) % m)] for h in steps], dtype=float)
        residuals = y[m:] - y[:-m]
        sigma = residual_sigma(residuals) if len(residuals) >= MIN_RESIDUALS_FOR_INTERVAL else None
        parameters: dict[str, float | int | str] = {"season_length": m, "residuals": len(residuals)}
        if sigma is None:
            return MethodOutput(tuple(mean.tolist()), None, None, parameters, _TOO_FEW_RESIDUALS)
        parameters["residual_sigma"] = sigma
        k = (steps - 1) // m
        lower, upper = symmetric_interval(mean, sigma * np.sqrt(k + 1), level)
        return MethodOutput(tuple(mean.tolist()), lower, upper, parameters)


class MovingAverageMethod(ForecastMethod):
    name = "moving_average"
    display_name = "Moving average"
    interval_method = "residual-based: RMS of the method's own h-step errors on the training data (normal errors)"

    def __init__(self, window: int = 3):
        self.window = window
        self.min_observations = window
        self.description = f"Every future month equals the mean of the last {window} observed months."

    def _forecast(self, y: np.ndarray, horizon: int, level: float) -> MethodOutput:
        k, n = self.window, len(y)
        mean = np.full(horizon, float(np.mean(y[-k:])))
        parameters: dict[str, float | int | str] = {"window": k, "last_window_mean": float(mean[0])}
        spreads: list[float] = []
        for step in range(1, horizon + 1):
            errors = np.array([y[t + step - 1] - np.mean(y[t - k : t]) for t in range(k, n - step + 1)], dtype=float)
            sigma = residual_sigma(errors) if len(errors) >= MIN_RESIDUALS_FOR_INTERVAL else None
            if sigma is None:
                return MethodOutput(tuple(mean.tolist()), None, None, parameters, _TOO_FEW_RESIDUALS)
            spreads.append(sigma)
        parameters["one_step_error_rms"] = spreads[0]
        lower, upper = symmetric_interval(mean, np.array(spreads), level)
        return MethodOutput(tuple(mean.tolist()), lower, upper, parameters)


class DriftMethod(ForecastMethod):
    name = "drift"
    display_name = "Drift (linear trend from first to last value)"
    description = (
        "The last value plus the average monthly change over the training history: y_T + h * (y_T - y_1) / (T - 1)."
    )
    interval_method = "analytic: sigma * sqrt(h * (1 + h / (T - 1))), sigma from drift-adjusted residuals"
    min_observations = 2

    def _forecast(self, y: np.ndarray, horizon: int, level: float) -> MethodOutput:
        n = len(y)
        slope = float((y[-1] - y[0]) / (n - 1))
        steps = np.arange(1, horizon + 1)
        mean = y[-1] + steps * slope
        residuals = np.diff(y) - slope
        sigma = residual_sigma(residuals, 1) if len(residuals) >= MIN_RESIDUALS_FOR_INTERVAL else None
        parameters: dict[str, float | int | str] = {"slope_per_month": slope, "last_value": float(y[-1])}
        if sigma is None:
            return MethodOutput(tuple(mean.tolist()), None, None, parameters, _TOO_FEW_RESIDUALS)
        parameters["residual_sigma"] = sigma
        spread = sigma * np.sqrt(steps * (1 + steps / (n - 1)))
        lower, upper = symmetric_interval(mean, spread, level)
        return MethodOutput(tuple(float(v) for v in mean), lower, upper, parameters)
