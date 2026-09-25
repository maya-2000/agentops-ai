"""Run the agent: one question in, one serialisable ``AgentRunResult`` out."""

from __future__ import annotations

import time
import uuid
from datetime import UTC, date, datetime
from typing import Any

from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field

from app.agent.config import AgentConfig
from app.agent.graph import AgentRuntime, build_graph
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
from app.tools.registry import ToolRegistry


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

    def run(self, question: str) -> AgentRunResult:
        run_id = f"R-{uuid.uuid4().hex[:12]}"
        clock = time.perf_counter()
        initial = AgentState(run_id=run_id, question=question, started_at=datetime.now(UTC))
        try:
            raw: Any = self.graph.invoke(initial, config={"recursion_limit": self.config.recursion_limit})
            state = raw if isinstance(raw, AgentState) else AgentState.model_validate(raw)
        except GraphRecursionError:
            state = initial.model_copy(
                update={
                    "status": "insufficient_evidence",
                    "response": failure_response("insufficient_evidence", LIMIT_MESSAGE),
                    "errors": [AgentError(stage="graph", code="recursion_limit", message=LIMIT_MESSAGE)],
                }
            )
        assert state.response is not None, "every terminal node sets a response"
        return AgentRunResult(
            run_id=run_id,
            question=question,
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
        )


def run_agent(db: Database, question: str, **kwargs: Any) -> AgentRunResult:
    """Convenience wrapper: build a runner and answer one question."""
    return AgentRunner(db, **kwargs).run(question)
