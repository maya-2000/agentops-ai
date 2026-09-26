"""Response models of the API.

``AskResponse`` serialises the agent's existing domain objects unchanged: the validated
``AgentResponse`` (answer, findings, caveats, cited evidence, user-safe tool trace), every ``Claim``
(typed observed_fact / calculated_result / inference / recommendation) and every ``Evidence`` item
(fingerprinted, with provenance), and the ``ValidatedRequest`` (intent, periods, filters). The other
fields are views that select or copy from those objects: KPI values, the analysis trace, the
forecast and anomaly sections, and the chart specs. None of them computes a business number.

Never included: prompts, model reasoning, security-policy internals (screening patterns, event
details), raw tool results, stack traces, file paths, environment values or secrets.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.agent.records import AgentStatus
from app.agent.request import ValidatedRequest
from app.agent.response import AgentResponse
from app.analytics.periods import Period
from app.api import SCHEMA_VERSION
from app.api.schemas.visualization import VisualizationSpec
from app.evidence.models import Claim, ClaimType, Evidence, EvidenceType
from app.tools.views import AnomalyReportView, ForecastView

# What happened, for a client deciding how to present the response (``status`` keeps the agent's own value).
Outcome = Literal["answered", "partial", "refused", "unsupported", "insufficient_evidence", "failed"]

FORECAST_NOTICE = "Forecast: an estimate from historical patterns, not observed data. It cannot anticipate new events."
ANOMALY_NOTICE = (
    "Anomaly: a statistically unusual movement against recent months. It is not necessarily bad, and it "
    "does not explain the cause."
)


class KPIValue(BaseModel):
    """A headline number, copied from one evidence item (never recomputed)."""

    evidence_id: str
    claim_ids: list[str] = Field(default_factory=list)  # the claims that cite this evidence
    claim_type: ClaimType | None = None  # the type of the first citing claim
    primary: bool = False  # cited by a claim that directly answers the question
    metric: str
    label: str
    value: float | int
    display_value: str | None
    unit: str | None
    evidence_type: EvidenceType
    period: str | None
    comparison_period: str | None
    filters: dict[str, str] = Field(default_factory=dict)
    percentage_change: float | None = None  # copied from the evidence attributes when present
    statement: str


class TraceStep(BaseModel):
    """One tool call of the run: what ran, why (the plan step's purpose), how it ended and how long it took."""

    step: int
    call_id: str
    step_id: str
    tool_name: str
    purpose: str
    status: str
    success: bool
    started_at: datetime
    execution_time_ms: float
    attempts: int
    result_summary: str
    query_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    error: str | None = None  # a fixed, user-safe category message; never exception text


class ForecastSection(BaseModel):
    call_id: str
    evidence_ids: list[str]
    forecast: ForecastView  # horizon, points with intervals, model, cutoff, backtest and baseline
    limitations: list[str] = Field(default_factory=list)
    label: Literal["forecast"] = "forecast"
    notice: str = FORECAST_NOTICE


class AnomalySection(BaseModel):
    call_id: str
    evidence_ids: list[str]
    report: AnomalyReportView  # detector, window, threshold, severity counts, flagged months
    limitations: list[str] = Field(default_factory=list)
    notice: str = ANOMALY_NOTICE


class Refusal(BaseModel):
    """Why nothing was analysed. ``message`` is the agent's own fixed, user-facing text."""

    kind: Literal["policy", "out_of_scope", "invalid_input"]
    message: str


class StageTiming(BaseModel):
    """One finished stage of the agent graph: which, how long, and whether the run stopped there."""

    stage: str  # the graph node
    label: str
    duration_ms: float
    ok: bool = True  # False for a stopping stage (declined, insufficient evidence, failure)


class RunSummary(BaseModel):
    run_id: str  # equals the request ID: the key of every agent log line and audit event of this run
    llm_provider: str
    llm_model: str
    tool_calls: int
    retries: int
    agent_time_ms: float
    pipeline: list[str] = Field(default_factory=list)  # the agent graph's nodes, in order (no content)
    stages: list[StageTiming] = Field(default_factory=list)  # the same nodes with their durations


class AskResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    session_id: str | None = None
    status: AgentStatus
    outcome: Outcome
    question: str  # as the agent stored it: secret-like values redacted
    answer: str
    response: AgentResponse
    scope: ValidatedRequest | None = None
    period: Period | None = None
    comparison_period: Period | None = None
    kpis: list[KPIValue] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    forecasts: list[ForecastSection] = Field(default_factory=list)
    anomalies: list[AnomalySection] = Field(default_factory=list)
    visualizations: list[VisualizationSpec] = Field(default_factory=list)
    refusal: Refusal | None = None
    run: RunSummary
    api_time_ms: float = 0.0


# ---------------------------------------------------------------------------------------- errors


class FieldIssue(BaseModel):
    location: str  # e.g. "body.question"
    message: str  # the validator's message; the submitted value is never echoed


class APIErrorDetail(BaseModel):
    code: str
    message: str
    retryable: bool = False
    issues: list[FieldIssue] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    request_id: str
    error: APIErrorDetail


# ---------------------------------------------------------------------------------------- streaming


class ProgressEvent(BaseModel):
    type: Literal["progress"] = "progress"
    request_id: str
    stage: str  # the agent graph node that finished
    label: str  # a short user-facing description of that stage
    elapsed_ms: float


class ResultEvent(BaseModel):
    type: Literal["result"] = "result"
    request_id: str
    data: AskResponse


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    request_id: str
    status_code: int
    error: APIErrorDetail


# ---------------------------------------------------------------------------------------- service


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    version: str
    agent_available: bool
    database_available: bool
    dataset_version: str | None = None
    as_of_date: date | None = None
    llm_provider: str | None = None


class NamedItem(BaseModel):
    key: str
    name: str
    unit: str | None = None
    description: str | None = None


class AnalysisCapability(BaseModel):
    tool: str
    when_to_use: str
    not_for: str


class Limits(BaseModel):
    max_question_chars: int
    request_timeout_seconds: float
    max_tool_calls: int


class CapabilitiesResponse(BaseModel):
    version: str
    schema_version: str = SCHEMA_VERSION
    as_of_date: date
    data_start: date | None
    data_end: date | None
    kpis: list[NamedItem]
    forecast_metrics: list[NamedItem]
    anomaly_metrics: list[NamedItem]
    anomaly_detectors: list[str]
    dimensions: list[NamedItem]
    analyses: list[AnalysisCapability]
    outcomes: list[str]
    limits: Limits
    example_questions: list[str]
    not_supported: list[str]


class MetricsResponse(BaseModel):
    """In-process counters since start-up (reset on restart; nothing is persisted)."""

    uptime_seconds: float
    requests_total: int
    in_flight: int
    by_outcome: dict[str, int]
    by_status_code: dict[str, int]
    agent_time_ms_avg: float | None
    agent_time_ms_max: float | None
    api_overhead_ms_avg: float | None
