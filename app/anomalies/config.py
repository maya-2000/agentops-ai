"""Typed anomaly-detection configuration: detector, window, minimum history, transforms and thresholds."""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.analytics.errors import InvalidRequestError
from app.forecasting.config import ModelName
from app.timeseries.metrics import Transform

DetectorName = Literal["rolling_zscore", "iqr", "forecast_residual"]
DETECTOR_NAMES: tuple[DetectorName, ...] = get_args(DetectorName)
Severity = Literal["normal", "watch", "significant", "extreme"]
SEVERITY_ORDER: tuple[Severity, ...] = get_args(Severity)


class StandardizedThresholds(BaseModel):
    """|score| cut-offs for the standardised detectors (rolling z-score, forecast residual).

    Under a normal distribution the two-sided tail probabilities are about 4.6% (2), 0.27% (3)
    and 0.006% (4). Small windows have heavier tails, which is why results also carry a
    t-distribution tail probability.
    """

    model_config = ConfigDict(frozen=True)

    watch: float = 2.0
    significant: float = 3.0
    extreme: float = 4.0

    @model_validator(mode="after")
    def _ascending(self) -> StandardizedThresholds:
        if not 0 < self.watch < self.significant < self.extreme:
            raise InvalidRequestError("Thresholds must satisfy 0 < watch < significant < extreme")
        return self


class IQRFences(BaseModel):
    """Tukey's fences, in IQR units beyond the quartiles: 1.5 (inner, "outlier") and 3.0 (outer, "far out")."""

    model_config = ConfigDict(frozen=True)

    inner: float = 1.5
    outer: float = 3.0

    @model_validator(mode="after")
    def _ordered(self) -> IQRFences:
        if not 0 < self.inner < self.outer:
            raise InvalidRequestError("IQR fences must satisfy 0 < inner < outer")
        return self


class AnomalyConfig(BaseModel):
    """Detector settings. Every statistic for month t uses only the ``window`` months before t.

    - ``min_history``: the fewest prior observations (or prior residuals) needed to score a month.
    - ``transform``: ``None`` uses the metric's default (month-over-month % change for the trending
      business series, difference for adoption rates). It is ignored by ``forecast_residual``,
      which works on levels.
    - ``expectation_model``: the forecasting method that gives the expected value for
      ``forecast_residual``. It is fitted on the ``window`` months before each scored month.
    - ``flag_severity``: the lowest severity counted as an anomaly (``is_anomaly``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    detector: DetectorName = "rolling_zscore"
    window: int = Field(default=12, ge=3, le=36)
    min_history: int = Field(default=6, ge=3, le=36)
    transform: Transform | None = None
    thresholds: StandardizedThresholds = Field(default_factory=StandardizedThresholds)
    iqr_fences: IQRFences = Field(default_factory=IQRFences)
    expectation_model: ModelName = "drift"
    flag_severity: Literal["watch", "significant", "extreme"] = "significant"

    @model_validator(mode="after")
    def _history_fits_window(self) -> AnomalyConfig:
        if self.min_history > self.window:
            raise InvalidRequestError(f"min_history ({self.min_history}) cannot exceed the window ({self.window})")
        return self
