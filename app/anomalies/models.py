"""Typed anomaly results. Each result explains itself: observed vs expected, deviation, score, threshold and window."""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, computed_field

from app.analytics.models import Provenance
from app.anomalies.config import SEVERITY_ORDER, AnomalyConfig, DetectorName, Severity
from app.timeseries.errors import SeriesStatus
from app.timeseries.metrics import Transform
from app.timeseries.models import TimeSeries

# Statistical direction only: observed above (positive) or below (negative) the expected value.
# It says nothing about whether the movement is favourable (more tickets is "positive" too).
Direction = Literal["positive", "negative", "none"]


class RollingZScoreDetails(BaseModel):
    detector: Literal["rolling_zscore"] = "rolling_zscore"
    transform: Transform
    transformed_value: float
    baseline_mean: float
    baseline_std: float
    tail_probability: float | None  # two-sided, Student t with n-1 df (normal i.i.d. assumption)


class IQRDetails(BaseModel):
    detector: Literal["iqr"] = "iqr"
    transform: Transform
    transformed_value: float
    q1: float
    median: float
    q3: float
    iqr: float
    lower_fence: float  # q1 - inner * iqr (transformed units)
    upper_fence: float  # q3 + inner * iqr
    outer_lower_fence: float  # q1 - outer * iqr
    outer_upper_fence: float  # q3 + outer * iqr
    inner_multiplier: float
    outer_multiplier: float


class ForecastResidualDetails(BaseModel):
    detector: Literal["forecast_residual"] = "forecast_residual"
    expectation_model: str
    training_start: str
    training_end: str
    one_step_forecast: float
    residual: float  # observed - one-step forecast
    residual_mean: float  # mean of prior one-step residuals (bias correction)
    residual_std: float
    standardized_residual: float | None
    tail_probability: float | None


AnomalyDetails = Annotated[RollingZScoreDetails | IQRDetails | ForecastResidualDetails, Field(discriminator="detector")]


class HistoricalWindow(BaseModel):
    """The prior months whose values (or residuals) formed the baseline. The scored month is never inside it."""

    start: str
    end: str
    observations: int
    window_months: int


class AnomalyResult(BaseModel):
    metric: str
    metric_name: str
    unit: str
    period: str
    period_start: date
    period_end: date
    observed_value: float
    expected_value: float
    deviation: float
    deviation_percentage: float | None  # deviation / |expected|, None when expected is 0
    score: float | None  # None: zero historical dispersion with a non-zero deviation (unbounded score)
    threshold: float
    lower_bound: float | None  # observed values below this are flagged (metric units)
    upper_bound: float | None  # observed values above this are flagged (metric units)
    detector: DetectorName
    severity: Severity
    direction: Direction
    is_anomaly: bool
    anomaly_start: str | None = None  # first month of the consecutive same-direction run this anomaly belongs to
    historical_window: HistoricalWindow
    explanation: str
    filters: dict[str, str] = Field(default_factory=dict)
    source_tables: list[str] = Field(default_factory=list)
    calculation: str
    query_id: str | None
    operation_id: str
    execution_timestamp: datetime
    limitations: list[str] = Field(default_factory=list)
    details: AnomalyDetails

    @property
    def rank_key(self) -> tuple[int, float, str]:
        """Ranking: undefined (unbounded) scores first, then |score| descending, then chronological."""
        if self.score is None:
            return (0, 0.0, self.period)
        return (1, -abs(self.score), self.period)


class SkippedPeriod(BaseModel):
    period: str
    reason: str


class DetectorSpecification(BaseModel):
    detector: DetectorName
    description: str
    transform: Transform | None
    window: int
    min_history: int
    threshold: float
    severity_policy: str
    expectation_model: str | None = None
    configuration: AnomalyConfig


class AnomalyReport(BaseModel):
    operation: Literal["detect_anomalies"] = "detect_anomalies"
    metric: str
    metric_name: str
    unit: str
    detector: DetectorName
    status: SeriesStatus
    message: str | None = None
    filters: dict[str, str] = Field(default_factory=dict)
    evaluation_start: date
    evaluation_end: date
    method: DetectorSpecification
    results: list[AnomalyResult] = Field(default_factory=list)  # every scored month, chronological
    skipped: list[SkippedPeriod] = Field(default_factory=list)
    calculation: str
    limitations: list[str] = Field(default_factory=list)
    series: TimeSeries
    provenance: Provenance

    @computed_field  # type: ignore[prop-decorator]
    @property
    def anomalies(self) -> list[AnomalyResult]:
        """Flagged months ranked by statistical severity (``AnomalyResult.rank_key``)."""
        return sorted((r for r in self.results if r.is_anomaly), key=lambda r: r.rank_key)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def severity_counts(self) -> dict[str, int]:
        return {level: sum(1 for r in self.results if r.severity == level) for level in SEVERITY_ORDER}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def source_tables(self) -> list[str]:
        return self.provenance.source_tables

    @computed_field  # type: ignore[prop-decorator]
    @property
    def query_id(self) -> str | None:
        return self.series.query_id

    @computed_field  # type: ignore[prop-decorator]
    @property
    def operation_id(self) -> str:
        return self.provenance.operation_id

    @computed_field  # type: ignore[prop-decorator]
    @property
    def execution_timestamp(self) -> datetime:
        return self.provenance.execution_timestamp

    def result(self, period: str) -> AnomalyResult:
        for result in self.results:
            if result.period == period:
                return result
        raise KeyError(period)


class AnomalyScan(BaseModel):
    """The same detector run over an explicit list of metrics (total level or one filter set)."""

    detector: DetectorName
    evaluation_start: date
    evaluation_end: date
    filters: dict[str, str] = Field(default_factory=dict)
    reports: list[AnomalyReport]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def anomalies(self) -> list[AnomalyResult]:
        return sorted((a for r in self.reports for a in r.anomalies), key=lambda a: (*a.rank_key, a.metric))
