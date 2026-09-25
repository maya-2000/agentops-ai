"""Shared contract for forecasting methods (pure functions of a numeric history)."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

ParameterValue = float | int | str


class MethodUnavailableError(Exception):
    """The method cannot produce a forecast from this history (too short, invalid values, fit failure)."""


@dataclass(frozen=True)
class MethodOutput:
    """Point forecasts for steps 1..h and, when defensible, prediction-interval bounds."""

    mean: tuple[float, ...]
    lower: tuple[float, ...] | None
    upper: tuple[float, ...] | None
    parameters: dict[str, ParameterValue]
    interval_note: str | None = None
    notes: tuple[str, ...] = field(default=())


def normal_quantile(level: float) -> float:
    """Two-sided standard-normal quantile for a central ``level`` interval (1.96 for 0.95)."""
    return NormalDist().inv_cdf(0.5 + level / 2.0)


def residual_sigma(residuals: np.ndarray, estimated_parameters: int = 0) -> float | None:
    """sqrt(sum(e^2) / (N - K)): the residual standard deviation used by the textbook benchmark methods."""
    dof = len(residuals) - estimated_parameters
    if dof <= 0:
        return None
    return math.sqrt(float(np.sum(np.square(residuals))) / dof)


def symmetric_interval(
    mean: np.ndarray, spread: np.ndarray, level: float
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    z = normal_quantile(level)
    return tuple(float(v) for v in mean - z * spread), tuple(float(v) for v in mean + z * spread)


class ForecastMethod(ABC):
    """A forecasting method. Subclasses implement ``_forecast`` on a validated, finite history."""

    name: str
    display_name: str
    description: str
    interval_method: str
    min_observations: int

    def forecast(self, history: np.ndarray, horizon: int, level: float) -> MethodOutput:
        y = np.asarray(history, dtype=float)
        if y.ndim != 1 or len(y) < self.min_observations:
            raise MethodUnavailableError(
                f"{self.name} needs at least {self.min_observations} observations, got {len(y)}"
            )
        if not np.all(np.isfinite(y)):
            raise MethodUnavailableError(f"{self.name} needs a history without missing values")
        if horizon < 1:
            raise MethodUnavailableError("horizon must be at least 1")
        return self._forecast(y, horizon, level)

    @abstractmethod
    def _forecast(self, y: np.ndarray, horizon: int, level: float) -> MethodOutput: ...
