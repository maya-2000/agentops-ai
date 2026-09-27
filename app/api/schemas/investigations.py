"""Response models of the investigation endpoints (Phase 10).

``InvestigationResponse`` serialises the investigation's domain objects unchanged: the decision brief,
the findings (validated claims with their identity), the relationships between findings, every ``Claim``
and ``Evidence`` item, and the validated scope. The plan, trace, forecast and anomaly sections and the
chart specs are views that copy from those objects, built by the same presenter functions as ``/ask``.

Never included: prompts, model reasoning, security-policy internals, raw tool results, validation
messages, stack traces, file paths, environment values or secrets.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agent.request import ValidatedRequest
from app.analytics.periods import Period
from app.api import SCHEMA_VERSION
from app.api.schemas.responses import (
    AnomalySection,
    APIErrorDetail,
    ForecastSection,
    KPIValue,
    Outcome,
    Refusal,
    TraceStep,
)
from app.api.schemas.visualization import VisualizationSpec
from app.evidence.models import Claim, Evidence
from app.investigation.models import (
    DecisionBrief,
    Finding,
    FindingRelationship,
    InvestigationBudgetReport,
    InvestigationEfficiency,
    InvestigationStatus,
    InvestigationTimings,
    StepStatus,
    TemplateName,
)


class PlanStepView(BaseModel):
    """One step of the analysis plan and what happened to it (concise titles; never model reasoning)."""

    step_id: str
    title: str
    area: str
    tool_name: str
    depends_on: list[str] = Field(default_factory=list)
    condition: str | None = None
    status: StepStatus
    reason: str | None = None
    duration_ms: float = 0.0
    evidence_ids: list[str] = Field(default_factory=list)
    reused_from: str | None = None


class InvestigationRunSummary(BaseModel):
    investigation_id: str  # equals the request ID: the key of every log line and audit event of this investigation
    llm_provider: str
    llm_model: str
    tool_calls: int
    efficiency: InvestigationEfficiency
    timings: InvestigationTimings
    budget: InvestigationBudgetReport
    validation: dict[str, int] = Field(default_factory=dict)  # removed / downgraded items (counts only)


class InvestigationResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    session_id: str | None = None
    investigation_id: str
    status: InvestigationStatus
    outcome: Outcome
    objective: str  # as stored: secret-like values redacted
    message: str | None = None
    title: str | None = None
    template: TemplateName | None = None
    scope: ValidatedRequest | None = None
    period: Period | None = None
    comparison_period: Period | None = None
    plan: list[PlanStepView] = Field(default_factory=list)
    brief: DecisionBrief | None = None
    findings: list[Finding] = Field(default_factory=list)
    relationships: list[FindingRelationship] = Field(default_factory=list)
    kpis: list[KPIValue] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    forecasts: list[ForecastSection] = Field(default_factory=list)
    anomalies: list[AnomalySection] = Field(default_factory=list)
    visualizations: list[VisualizationSpec] = Field(default_factory=list)
    refusal: Refusal | None = None
    run: InvestigationRunSummary
    api_time_ms: float = 0.0


class InvestigationProgressEvent(BaseModel):
    """A lifecycle stage or a step that started or finished: fixed labels, tool names, status and timing only."""

    type: Literal["progress"] = "progress"
    request_id: str
    stage: str
    label: str
    elapsed_ms: float
    step_id: str | None = None
    title: str | None = None
    area: str | None = None
    tool_name: str | None = None
    status: str | None = None
    duration_ms: float | None = None
    steps: list[dict[str, Any]] | None = None  # the plan, on the "plan" event


class InvestigationResultEvent(BaseModel):
    type: Literal["result"] = "result"
    request_id: str
    data: InvestigationResponse


class InvestigationErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    request_id: str
    status_code: int
    error: APIErrorDetail
