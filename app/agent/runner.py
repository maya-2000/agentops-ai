"""Run the agent: one question in, one serialisable ``AgentRunResult`` out."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field

from app.agent.config import AgentConfig
from app.agent.graph import AgentRuntime, build_graph
from app.agent.observability import is_valid_run_id, new_run_id
from app.agent.records import AgentError, AgentStatus, InvestigationPlan, LLMCallRecord, ToolCallRecord
from app.agent.request import ValidatedRequest
from app.agent.response import LIMIT_MESSAGE, AgentResponse, failure_response
from app.agent.state import AgentState
from app.config import get_settings
from app.database.base import Database
from app.evidence.models import Claim, Evidence
from app.llm.base import LLMClient
from app.llm.factory import create_llm_client
from app.llm.schemas import UnderstandingOutput
from app.security.budget import BudgetUsage
from app.security.events import SecurityEvent, Severity, security_event
from app.security.injection import InjectionScan
from app.security.redaction import redact
from app.security.retry import RetryRecord
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry

# Called with a node name each time a graph node finishes (a progress signal; never data).
ProgressCallback = Callable[[str], None]


class AgentRunResult(BaseModel):
    run_id: str
    question: str
    status: AgentStatus
    response: AgentResponse
    understanding: UnderstandingOutput | None = None
    request: ValidatedRequest | None = None
    plans: list[InvestigationPlan] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    tool_trace: list[ToolCallRecord] = Field(default_factory=list)
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    total_tool_calls: int = 0
    total_retries: int = 0
    errors: list[AgentError] = Field(default_factory=list)
    transitions: list[str] = Field(default_factory=list)
    execution_time_ms: float
    llm_provider: str
    llm_model: str
    security_events: list[SecurityEvent] = Field(default_factory=list)
    budget_usage: BudgetUsage = Field(default_factory=BudgetUsage)
    retries: list[RetryRecord] = Field(default_factory=list)
    input_screen: InjectionScan | None = None
    # The typed tool results, for in-process consumers only (the API's forecast and anomaly views).
    # Excluded from serialisation and repr: a raw result can hold fields the evidence layer withholds.
    tool_results: list[ToolResult] = Field(default_factory=list, exclude=True, repr=False)


class AgentRunner:
    """Owns the runtime (database, tools, model client) and the compiled graph; runs are independent."""

    def __init__(
        self,
        db: Database,
        *,
        llm: LLMClient | None = None,
        config: AgentConfig | None = None,
        registry: ToolRegistry | None = None,
        as_of: date | None = None,
    ):
        self.config = config or AgentConfig.from_settings()
        self.llm = llm or create_llm_client()
        self.runtime = AgentRuntime(
            db=db,
            llm=self.llm,
            config=self.config,
            registry=registry or ToolRegistry(),
            as_of=as_of or get_settings().as_of_date,
        )
        self.graph = build_graph(self.runtime)

    def run(
        self, question: Any, *, run_id: str | None = None, on_progress: ProgressCallback | None = None
    ) -> AgentRunResult:
        """Answer one question. The question is untrusted: it is redacted before it enters the state.

        ``run_id`` lets a caller correlate the run with its own request ID (for example the API's);
        it must be a short token of letters, digits and ``._:-``. ``on_progress`` receives each
        finished graph node's name; without it the graph runs exactly as before (``invoke``).
        """
        if run_id is not None and not is_valid_run_id(run_id):
            raise ValueError("run_id must be 1-64 characters: letters, digits and ._:- (starting alphanumeric)")
        run_id = run_id or new_run_id()
        clock = time.perf_counter()
        is_text = isinstance(question, str)
        text = question if isinstance(question, str) else ""
        clean = redact(text)
        initial = AgentState(
            run_id=run_id,
            question=clean,
            question_is_text=is_text,
            question_redacted=clean != text,
            started_at=datetime.now(UTC),
        )
        try:
            raw = self._execute(initial, on_progress)
            state = raw if isinstance(raw, AgentState) else AgentState.model_validate(raw)
        except GraphRecursionError:
            stop = security_event(
                run_id,
                "budget_exceeded",
                Severity.HIGH,
                component="graph",
                action="recursion_limit",
                decision="stop",
                reason="The graph step limit was reached.",
            )
            state = initial.model_copy(
                update={
                    "status": "insufficient_evidence",
                    "response": failure_response("insufficient_evidence", LIMIT_MESSAGE),
                    "errors": [AgentError(stage="graph", code="recursion_limit", message=LIMIT_MESSAGE)],
                    "security_events": [stop],
                }
            )
        assert state.response is not None, "every terminal node sets a response"
        return AgentRunResult(
            run_id=run_id,
            question=state.question,  # the redacted question; the raw input is never stored
            status=state.status,
            response=state.response,
            understanding=state.understanding,
            request=state.request,
            plans=state.plans,
            claims=list(state.evidence_graph.claims.values()),
            evidence=list(state.evidence_graph.evidence.values()),
            tool_trace=state.tool_calls,
            llm_calls=state.llm_calls,
            total_tool_calls=len(state.tool_calls),
            total_retries=state.tool_retries + state.llm_retries + state.retry_count,
            errors=state.errors,
            transitions=state.transitions,
            execution_time_ms=round((time.perf_counter() - clock) * 1000, 1),
            llm_provider=self.llm.provider,
            llm_model=self.llm.model,
            security_events=state.security_events,
            budget_usage=state.budget_usage,
            retries=state.retries,
            input_screen=state.input_screen,
            tool_results=state.tool_results,
        )

    def _execute(self, initial: AgentState, on_progress: ProgressCallback | None) -> Any:
        """Run the graph to its end state; stream node updates only when a progress callback is given."""
        config: Any = {"recursion_limit": self.config.recursion_limit}
        if on_progress is None:
            return self.graph.invoke(initial, config=config)
        final: Any = initial
        reporting = True
        for mode, chunk in self.graph.stream(initial, config=config, stream_mode=["updates", "values"]):
            if mode == "values":
                final = chunk
            elif reporting:
                try:
                    for node in chunk:
                        on_progress(node)
                except Exception:  # progress is advisory: a failing observer never changes the run
                    reporting = False
        return final


def run_agent(db: Database, question: str, **kwargs: Any) -> AgentRunResult:
    """Convenience wrapper: build a runner and answer one question."""
    return AgentRunner(db, **kwargs).run(question)
