"""The typed, serialisable agent state carried through the LangGraph state machine.

The state holds only data: the question, the structured understanding, the validated request, the
plans, the tool-call trace, tool results, the evidence graph, drafts, validation results, counters
and errors. Phase 5 adds the input-screening verdict, the run's SQL privilege, the security audit
events, the budget usage and the retry records. Database connections, model clients and secrets
live in the runtime, never here, and prompts are not stored. The question is kept redacted.
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
from app.security.budget import BudgetUsage
from app.security.events import SecurityEvent
from app.security.injection import InjectionScan
from app.security.retry import RetryRecord
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

    # ------------------------------------------------------------------ Phase 5 security
    question_is_text: bool = True  # the runner received a string (anything else is rejected)
    question_redacted: bool = False  # secrets were removed from the question before it entered the state
    input_screen: InjectionScan | None = None  # prompt-injection screening verdict (pattern names only)
    sql_permitted: bool = True  # revoked for the whole run when the input screen restricts the request
    security_events: list[SecurityEvent] = Field(default_factory=list)
    budget_usage: BudgetUsage = Field(default_factory=BudgetUsage)
    retries: list[RetryRecord] = Field(default_factory=list)
    response_trimmed: bool = False  # lower-priority items dropped to meet the response length limit

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
