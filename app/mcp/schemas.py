"""Typed MCP tool output: one envelope for every tool, success or failure.

Input schemas are not defined here: every MCP tool uses its Phase 4 Pydantic input model (see
``app/mcp/registry.py``). This module defines only the response envelope ``MCPToolOutput``,
which is also each tool's MCP ``outputSchema``:

- ``status``, ``tool_name``, ``request_id`` (the run ID shared by every audit event of the call)
  and ``query_id``;
- ``result``: the Phase 2/3 service's typed result, serialised unchanged;
- ``forecast`` / ``anomalies``: explicit views of forecast and anomaly results, so their
  labelling and method fields cannot be lost;
- ``evidence``: Phase 4 evidence items (fingerprinted, with full provenance);
- ``provenance``, ``warnings``, ``limitations``;
- ``error``: a sanitised MCP error (category, code, fixed message) when ``status`` is ``error``.

The views only copy fields from the typed result: no number is recomputed here.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.anomalies import AnomalyReport, AnomalyResult
from app.evidence.models import Evidence
from app.forecasting import ForecastResult
from app.forecasting.evaluation import ErrorMetrics

SCHEMA_VERSION = "1.0"

MCPErrorCategory = Literal[
    "INVALID_ARGUMENT",
    "UNAUTHORIZED_TOOL",
    "UNSAFE_QUERY",
    "RESOURCE_LIMIT",
    "TOOL_FAILURE",
    "TIMEOUT",
    "UNSUPPORTED_REQUEST",
    "VALIDATION_FAILURE",
    "INTERNAL_ERROR",
]
MCPStatus = Literal["ok", "no_data", "insufficient_data", "insufficient_history", "error"]
MCPOutputKind = Literal["observed", "risk_score", "forecast", "anomaly", "ad_hoc_query"]


class MCPError(BaseModel):
    """A client-safe error: never exception text, paths, SQL internals or secrets."""

    category: MCPErrorCategory
    code: str
    message: str
    detail: str | None = None  # sanitised, bounded argument feedback (argument errors only)
    retryable: bool = False


class MCPProvenance(BaseModel):
    """How the result was produced (copied from the tool result; nothing is inferred)."""

    tool: str  # the Phase 4 tool that ran
    source_layer: str
    operation: str
    call_id: str
    arguments: dict[str, Any]  # the canonical, validated arguments that ran
    query_ids: list[str] = Field(default_factory=list)
    source_tables: list[str] = Field(default_factory=list)
    calculation: str | None = None
    executed_at: datetime
    execution_time_ms: float
    attempts: int = 1
    dataset_version: str
    as_of: date
    toolset_version: str


class ForecastPointView(BaseModel):
    period: str
    start: date
    end: date
    predicted_value: float
    lower_bound: float | None
    upper_bound: float | None


class ForecastView(BaseModel):
    """The forecast fields a consumer must never lose. This is a forecast, not observed data."""

    kind: Literal["forecast"] = "forecast"
    metric: str
    metric_name: str
    unit: str
    status: str
    message: str | None = None
    model: str | None
    cutoff_date: date
    horizon: int
    forecast_period_start: date | None
    forecast_period_end: date | None
    confidence_level: float
    interval_method: str | None
    interval_available: bool
    history_end: date | None
    history_observations: int
    points: list[ForecastPointView]
    backtest: ErrorMetrics | None  # rolling-origin backtest of the selected model
    baseline: ErrorMetrics | None  # the naive baseline on the same folds
    improvement_over_baseline: float | None
    filters: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_result(cls, r: ForecastResult) -> ForecastView:
        points = [
            ForecastPointView(
                period=p.period,
                start=p.start,
                end=p.end,
                predicted_value=p.predicted_value,
                lower_bound=p.lower_bound,
                upper_bound=p.upper_bound,
            )
            for p in r.forecast_points
        ]
        return cls(
            metric=r.metric,
            metric_name=r.metric_name,
            unit=r.unit,
            status=r.status,
            message=r.message,
            model=r.model,
            cutoff_date=r.cutoff_date,
            horizon=r.horizon,
            forecast_period_start=points[0].start if points else None,
            forecast_period_end=points[-1].end if points else None,
            confidence_level=r.confidence_level,
            interval_method=r.interval.method if r.interval else None,
            interval_available=bool(r.interval and r.interval.available),
            history_end=r.historical_end,
            history_observations=r.history_observations,
            points=points,
            backtest=r.backtest_metrics,
            baseline=r.baseline_metrics,
            improvement_over_baseline=r.improvement_over_baseline,
            filters=r.filters,
        )


class AnomalyView(BaseModel):
    """One scored month. A flag is statistical, not a business judgement or a cause."""

    period: str
    period_start: date
    period_end: date
    observed_value: float
    expected_value: float
    deviation: float
    deviation_percentage: float | None
    score: float | None  # None: zero historical dispersion with a non-zero deviation (unbounded)
    threshold: float
    detector: str
    severity: str
    direction: str
    is_anomaly: bool
    anomaly_start: str | None
    explanation: str

    @classmethod
    def from_result(cls, a: AnomalyResult) -> AnomalyView:
        return cls(
            period=a.period,
            period_start=a.period_start,
            period_end=a.period_end,
            observed_value=a.observed_value,
            expected_value=a.expected_value,
            deviation=a.deviation,
            deviation_percentage=a.deviation_percentage,
            score=a.score,
            threshold=a.threshold,
            detector=a.detector,
            severity=a.severity,
            direction=a.direction,
            is_anomaly=a.is_anomaly,
            anomaly_start=a.anomaly_start,
            explanation=a.explanation,
        )


class AnomalyReportView(BaseModel):
    """The anomaly fields a consumer must never lose: flagged months ranked by statistical severity."""

    kind: Literal["anomaly"] = "anomaly"
    metric: str
    metric_name: str
    unit: str
    detector: str
    status: str
    message: str | None = None
    evaluation_start: date
    evaluation_end: date
    window: int
    threshold: float
    scored_periods: int
    severity_counts: dict[str, int]
    flagged: list[AnomalyView]
    filters: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_report(cls, r: AnomalyReport) -> AnomalyReportView:
        return cls(
            metric=r.metric,
            metric_name=r.metric_name,
            unit=r.unit,
            detector=r.detector,
            status=r.status,
            message=r.message,
            evaluation_start=r.evaluation_start,
            evaluation_end=r.evaluation_end,
            window=r.method.window,
            threshold=r.method.threshold,
            scored_periods=len(r.results),
            severity_counts=r.severity_counts,
            flagged=[AnomalyView.from_result(a) for a in r.anomalies],
            filters=r.filters,
        )


class MCPToolOutput(BaseModel):
    """The structured content of every AgentOps MCP tool response."""

    schema_version: str = SCHEMA_VERSION
    tool_name: str
    request_id: str
    status: MCPStatus
    output_kind: MCPOutputKind
    result_type: str | None = None
    result: dict[str, Any] | None = None
    forecast: ForecastView | None = None
    anomalies: AnomalyReportView | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    provenance: MCPProvenance | None = None
    query_id: str | None = None
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    error: MCPError | None = None
    truncated: bool = False  # parts were dropped to respect the response size limit (see warnings)
