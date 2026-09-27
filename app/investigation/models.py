"""The typed investigation: objective, analysis plan, step records, findings, relationships and the brief.

An investigation (Phase 10) answers a business objective ("Why is revenue growth slowing?") with a
structured, multi-step analysis instead of a single agent run. Everything here is data:

- ``AnalysisPlan`` / ``AnalysisStep``: the analytical steps chosen for the objective. Each step names one
  allow-listed tool, its arguments, the validated intent it is authorised under, and optional
  dependencies. Conditions and evidence-bound arguments come from closed sets (``StepCondition``,
  ``StepBinding``); there are no executable expressions.
- ``StepRecord``: what happened to each step (completed, reused, skipped, failed, not run).
- ``Finding``: a validated claim with its full identity (metric, unit, period, comparison period,
  dimension, member, filters) and the evidence and steps it rests on.
- ``FindingRelationship`` / ``Driver``: rule-based links between findings (``contributes_to``,
  ``supports``, ``correlates_with``, ``contradicts``, ``contextualizes``). None of them is causal.
- ``Recommendation``: a suggested next step that cites the findings it rests on.
- ``DecisionBrief``: the executive summary, key findings, drivers, contradictions, risks,
  recommendations, uncertainty and (for a management brief) sections.

No prompts, model reasoning or raw tool results are stored. The objective is kept redacted.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agent.records import AgentError, ToolCallRecord
from app.agent.request import ValidatedRequest
from app.analytics.periods import Period
from app.evidence.models import Claim, ClaimType, Confidence, Direction, Evidence
from app.llm.schemas import Intent
from app.security.budget import BudgetUsage
from app.security.events import SecurityEvent
from app.tools.base import ToolResult

Area = Literal["revenue", "customers", "sales", "marketing", "product", "support", "anomalies", "forecast"]
TemplateName = Literal["revenue", "customer", "sales", "product_support", "general", "management_brief"]
InvestigationStatus = Literal[
    "planned",
    "running",
    "completed",
    "insufficient_evidence",
    "budget_exhausted",
    "refused",
    "unsupported",
    "failed",
    "cancelled",
]
FINAL_STATUSES: tuple[InvestigationStatus, ...] = (
    "completed",
    "insufficient_evidence",
    "budget_exhausted",
    "refused",
    "unsupported",
    "failed",
    "cancelled",
)
StepStatus = Literal["pending", "completed", "reused", "skipped", "failed", "not_run"]
# Conditions are named, code-defined checks on the evidence of a step's dependencies (never expressions).
StepCondition = Literal[
    "outcome_changed",  # the dependency measured a non-zero change
    "concentrated",  # one member of the dependency's decomposition accounts for most of the change
]
RelationshipType = Literal["supports", "correlates_with", "contributes_to", "contradicts", "contextualizes"]
DriverCategory = Literal[
    "customer_churn",
    "retention",
    "contraction",
    "expansion",
    "new_business",
    "regional_performance",
    "segment_performance",
    "sales_performance",
    "pipeline",
    "conversion",
    "acquisition",
    "product_adoption",
    "support_activity",
    "customer_base",
    "anomaly",
    "forecast",
]
FindingLabel = Literal["observed", "calculated", "inferred", "recommended"]
FINDING_LABELS: dict[ClaimType, FindingLabel] = {
    "observed_fact": "observed",
    "calculated_result": "calculated",
    "inference": "inferred",
    "recommendation": "recommended",
}
STOP_MESSAGE = "Investigation stopped because the analysis budget was reached."


class StepBinding(BaseModel):
    """An argument read from the evidence of an earlier step (a closed set of kinds, resolved in code).

    - ``concentrated_member``: the member of ``source_step``'s decomposition that accounts for most of
      the change; bound as a filter ``{dimension: member}``.
    - ``feature``: the ``index``-th feature of ``source_step``'s adoption result, in the tool's own order;
      bound as the ``product_feature`` filter.
    """

    kind: Literal["concentrated_member", "feature"]
    source_step: str
    index: int = Field(default=1, ge=1, le=10)


class AnalysisStep(BaseModel):
    step_id: str
    title: str  # concise, user-facing ("Compare regional performance"); never model reasoning
    area: Area
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    authorized_as: Intent  # the validated intent the step's tool call is authorised under (fixed by the template)
    depends_on: list[str] = Field(default_factory=list)
    condition: StepCondition | None = None
    binding: StepBinding | None = None


class AnalysisPlan(BaseModel):
    template: TemplateName
    title: str  # "Revenue investigation"
    outcome_metric: str | None = None  # the business outcome the investigation explains (None: a brief)
    period: Period | None = None
    comparison_period: Period | None = None
    period_label: str | None = None
    comparison_label: str | None = None
    steps: list[AnalysisStep] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    def step(self, step_id: str) -> AnalysisStep | None:
        return next((s for s in self.steps if s.step_id == step_id), None)


class StepRecord(BaseModel):
    step_id: str
    title: str
    area: Area
    tool_name: str
    status: StepStatus = "pending"
    reason: str | None = None  # why a step was skipped, not run or failed (a fixed, user-safe text)
    call_id: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)  # the arguments it ran with (after binding)
    evidence_ids: list[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    attempts: int = 0
    reused_from: str | None = None  # the step whose identical call this step reused


class Finding(BaseModel):
    """A validated claim, with the identity of what it is about (copied from its evidence, never re-derived)."""

    finding_id: str
    claim_id: str
    text: str
    claim_type: ClaimType
    label: FindingLabel
    kind: str
    area: Area
    metric: str | None = None
    unit: str | None = None
    period: str | None = None
    comparison_period: str | None = None
    dimension: str | None = None
    breakdown: str | None = None  # the dimension member (e.g. a region name)
    filters: dict[str, str] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    step_ids: list[str] = Field(default_factory=list)
    direction: Direction | None = None
    confidence: Confidence = "high"
    primary: bool = False  # the investigation's outcome
    limitations: list[str] = Field(default_factory=list)


class FindingRelationship(BaseModel):
    """A link between two findings, created only by a documented rule over their evidence."""

    source_finding_id: str
    target_finding_id: str
    relationship: RelationshipType
    rule: str  # which rule created it (e.g. "same_period_co_movement")
    evidence_ids: list[str] = Field(default_factory=list)


class Driver(BaseModel):
    """An observable factor associated with the outcome in the available evidence (never a proven cause)."""

    driver_id: str
    name: str
    category: DriverCategory
    relationship: RelationshipType
    statement: str  # non-causal wording, built from the finding texts
    direction: Direction | None = None
    magnitude: str | None = None  # the change or contribution, as the evidence states it
    share: float | None = None  # share of the gross change, when the evidence reports one
    confidence: Confidence = "medium"
    finding_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    recommendation_id: str
    claim_id: str
    text: str
    rationale: str
    supporting_finding_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    claim_type: Literal["recommendation"] = "recommendation"
    uncertainty: str | None = None


class RiskItem(BaseModel):
    text: str
    finding_ids: list[str] = Field(default_factory=list)


class BriefSection(BaseModel):
    area: Area
    title: str
    finding_ids: list[str] = Field(default_factory=list)


class DecisionBrief(BaseModel):
    title: str
    objective: str
    executive_summary: str
    summary_finding_ids: list[str] = Field(default_factory=list)
    key_finding_ids: list[str] = Field(default_factory=list)
    drivers: list[Driver] = Field(default_factory=list)
    contradictions: list[Driver] = Field(default_factory=list)
    context: list[Driver] = Field(default_factory=list)
    risks: list[RiskItem] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)
    sections: list[BriefSection] = Field(default_factory=list)  # management brief: only supported sections
    complete: bool = True  # False when the investigation stopped before every planned step ran


class ValidationIssue(BaseModel):
    """What cross-finding validation did to a finding, driver or recommendation (never silently repaired)."""

    item_id: str
    action: Literal["removed", "downgraded"]
    reason: str


class InvestigationBudgetReport(BaseModel):
    max_steps: int
    max_tool_calls: int
    max_seconds: float
    max_evidence: int
    max_output_chars: int
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    steps_run: int = 0
    evidence_items: int = 0
    output_chars: int = 0
    exhausted: list[str] = Field(default_factory=list)


class InvestigationEfficiency(BaseModel):
    steps_planned: int = 0
    steps_completed: int = 0
    steps_reused: int = 0
    steps_skipped: int = 0
    steps_failed: int = 0
    steps_not_run: int = 0
    tool_calls: int = 0
    duplicate_tool_calls: int = 0  # identical calls executed twice (reuse keeps this at 0)


class InvestigationTimings(BaseModel):
    understanding_ms: float = 0.0
    planning_ms: float = 0.0
    execution_ms: float = 0.0
    validation_ms: float = 0.0
    synthesis_ms: float = 0.0
    total_ms: float = 0.0


class Investigation(BaseModel):
    investigation_id: str
    objective: str  # as stored: secret-like values redacted
    status: InvestigationStatus = "planned"
    message: str | None = None  # a fixed, user-facing explanation for a status other than completed
    created_at: datetime
    completed_at: datetime | None = None
    template: TemplateName | None = None
    scope: ValidatedRequest | None = None
    plan: AnalysisPlan | None = None
    steps: list[StepRecord] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    relationships: list[FindingRelationship] = Field(default_factory=list)
    brief: DecisionBrief | None = None
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    tool_trace: list[ToolCallRecord] = Field(default_factory=list)
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    budget: InvestigationBudgetReport
    efficiency: InvestigationEfficiency = Field(default_factory=InvestigationEfficiency)
    timings: InvestigationTimings = Field(default_factory=InvestigationTimings)
    llm_provider: str = ""
    llm_model: str = ""
    errors: list[AgentError] = Field(default_factory=list)
    security_events: list[SecurityEvent] = Field(default_factory=list)
    # Typed tool results for in-process consumers (the API's forecast and anomaly views). Excluded from
    # serialisation and repr: a raw result can hold fields the evidence layer withholds.
    tool_results: list[ToolResult] = Field(default_factory=list, exclude=True, repr=False)

    def finding(self, finding_id: str) -> Finding | None:
        return next((f for f in self.findings if f.finding_id == finding_id), None)
