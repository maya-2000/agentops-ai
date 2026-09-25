"""Typed request and result contracts for forecasting (evidence-ready for the Phase 4 layer)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from app.analytics.dimensions import Filters
from app.analytics.models import Provenance
from app.forecasting.backtest import BacktestResult
from app.forecasting.base import ParameterValue
from app.forecasting.config import BASELINE_MODEL, MODEL_NAMES, SUPPORTED_HORIZONS, ForecastConfig
from app.forecasting.evaluation import ErrorMetrics
from app.timeseries.errors import InvalidHorizonError, SeriesStatus, UnsupportedMethodError
from app.timeseries.models import TimeSeries

AUTO_MODEL = "auto"


class ForecastRequest(BaseModel):
    """What to forecast. ``cutoff_date`` defaults to the business as-of date; ``model="auto"`` selects by backtest."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    horizon: int = 3
    cutoff_date: date | None = None
    filters: dict[str, str] = Field(default_factory=dict)
    model: str = AUTO_MODEL
    confidence_level: float | None = Field(default=None, gt=0.5, lt=1.0)

    @model_validator(mode="before")
    @classmethod
    def _filters_from_model(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("filters"), Filters):
            return {**data, "filters": data["filters"].active()}
        return data

    @field_validator("horizon")
    @classmethod
    def _horizon(cls, value: int) -> int:
        if value not in SUPPORTED_HORIZONS:
            raise InvalidHorizonError(
                f"Forecast horizon must be between {SUPPORTED_HORIZONS.start} and {SUPPORTED_HORIZONS.stop - 1} "
                f"months, got {value}"
            )
        return value

    @field_validator("model")
    @classmethod
    def _model(cls, value: str) -> str:
        if value != AUTO_MODEL and value not in MODEL_NAMES:
            raise UnsupportedMethodError(
                f"Unknown forecasting model {value!r}. Use 'auto' or one of: {', '.join(MODEL_NAMES)}"
            )
        return value


class ForecastPoint(BaseModel):
    period: str  # YYYY-MM
    start: date
    end: date
    predicted_value: float
    lower_bound: float | None
    upper_bound: float | None


class PredictionInterval(BaseModel):
    """Describes the bounds on the forecast points. They are *prediction* intervals for future values."""

    kind: Literal["prediction_interval"] = "prediction_interval"
    confidence_level: float
    available: bool
    method: str | None
    note: str | None = None


class SeasonalityAssessment(BaseModel):
    season_length: int
    history_observations: int
    full_cycles: int
    estimable: bool  # at least two full cycles
    seasonal_naive_available: bool  # at least one full cycle
    note: str


class ModelSpecification(BaseModel):
    """How the final forecast was produced: model, fitted parameters, configuration and training window."""

    model: str
    display_name: str
    description: str
    parameters: dict[str, ParameterValue]
    configuration: ForecastConfig
    training_start: date
    training_end: date
    training_observations: int
    cutoff_date: date
    random_seed: int
    notes: list[str] = Field(default_factory=list)


class CandidateSummary(BaseModel):
    model: str
    eligible: bool
    metrics: ErrorMetrics
    failures: list[str] = Field(default_factory=list)


class ForecastResult(BaseModel):
    operation: Literal["forecast"] = "forecast"
    metric: str
    metric_name: str
    unit: str
    status: SeriesStatus
    message: str | None = None
    model: str | None = None
    cutoff_date: date
    horizon: int
    confidence_level: float
    forecast_points: list[ForecastPoint] = Field(default_factory=list)
    interval: PredictionInterval | None = None
    historical_start: date | None = None
    historical_end: date | None = None
    history_observations: int = 0
    sufficient_history: bool
    seasonality: SeasonalityAssessment | None = None
    selected_model_reason: str | None = None
    improvement_over_baseline: float | None = None
    backtests: list[BacktestResult] = Field(default_factory=list)
    candidates: list[CandidateSummary] = Field(default_factory=list)
    method: ModelSpecification | None = None
    filters: dict[str, str] = Field(default_factory=dict)
    calculation: str
    limitations: list[str] = Field(default_factory=list)
    history: TimeSeries
    provenance: Provenance

    @computed_field  # type: ignore[prop-decorator]
    @property
    def backtest_metrics(self) -> ErrorMetrics | None:
        """Historical backtest metrics of the selected model (optimistic: also used for selection)."""
        return self._metrics_of(self.model)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def baseline_metrics(self) -> ErrorMetrics | None:
        """Historical backtest metrics of the naive baseline on the same folds."""
        return self._metrics_of(BASELINE_MODEL)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def lower_bound(self) -> list[float | None]:
        return [p.lower_bound for p in self.forecast_points]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def upper_bound(self) -> list[float | None]:
        return [p.upper_bound for p in self.forecast_points]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def source_tables(self) -> list[str]:
        return self.provenance.source_tables

    @computed_field  # type: ignore[prop-decorator]
    @property
    def query_id(self) -> str | None:
        return self.history.query_id

    @computed_field  # type: ignore[prop-decorator]
    @property
    def query_ids(self) -> list[str]:
        return self.provenance.query_ids

    @computed_field  # type: ignore[prop-decorator]
    @property
    def operation_id(self) -> str:
        return self.provenance.operation_id

    @computed_field  # type: ignore[prop-decorator]
    @property
    def execution_timestamp(self) -> datetime:
        return self.provenance.execution_timestamp

    def _metrics_of(self, model: str | None) -> ErrorMetrics | None:
        for backtest in self.backtests:
            if backtest.model == model:
                return backtest.metrics
        return None

    def backtest(self, model: str) -> BacktestResult:
        for backtest in self.backtests:
            if backtest.model == model:
                return backtest
        raise KeyError(model)
