"""The typed, serialisable agent state carried through the LangGraph state machine.

The state holds only data: the question, the structured understanding, the validated request, the
plans, the tool-call trace, tool results, the evidence graph, drafts, validation results, counters
and errors. Database connections, model clients and secrets live in the runtime, never here, and
prompts are not stored.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.agent.records import AgentError, AgentStatus, InvestigationPlan, LLMCallRecord, PlanStep, ToolCallRecord
from app.agent.request import ValidatedRequest
from app.agent.response import AgentResponse
from app.analytics.periods import Period
from app.evidence.models import EvidenceGraph
from app.evidence.validation import EvidenceValidationResult, ResponseValidationResult
from app.llm.schemas import Intent, ResponseDraftOutput, UnderstandingOutput
from app.tools.base import ToolResult


class AgentState(BaseModel):
    run_id: str
    question: str
    started_at: datetime
    normalized_question: str = ""
    understanding: UnderstandingOutput | None = None
    request: ValidatedRequest | None = None
    plans: list[InvestigationPlan] = Field(default_factory=list)
    pending_steps: list[PlanStep] = Field(default_factory=list)
    current_step: int = 0
    planning_iterations: int = 0
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    processed_call_ids: list[str] = Field(default_factory=list)
    evidence_graph: EvidenceGraph = Field(default_factory=EvidenceGraph)
    evidence_validation: EvidenceValidationResult | None = None
    response_draft: ResponseDraftOutput | None = None
    response_validation: ResponseValidationResult | None = None
    response: AgentResponse | None = None
    response_generated_by: str | None = None
    retry_count: int = 0
    tool_retries: int = 0
    llm_retries: int = 0
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    limit_reached: bool = False
    status: AgentStatus = "running"
    status_message: str | None = None
    errors: list[AgentError] = Field(default_factory=list)
    transitions: list[str] = Field(default_factory=list)
    route: str = ""  # the next node chosen by the current node (read by the conditional edges)

    # ------------------------------------------------------------------ convenient views
    @property
    def intent(self) -> Intent | None:
        return self.request.intent if self.request else self.understanding.intent if self.understanding else None

    @property
    def date_range(self) -> Period | None:
        return self.request.period if self.request else None

    @property
    def filters(self) -> dict[str, str]:
        return dict(self.request.filters) if self.request else {}
