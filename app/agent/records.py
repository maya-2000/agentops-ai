"""Typed records shared by the agent state and the agent response (plans, trace entries, errors)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.tools.base import ToolError

AgentStatus = Literal[
    "running",
    "completed",
    "insufficient_evidence",
    "unsupported_request",
    "tool_error",
    "validation_failure",
    "planning_failure",
]


class PlanStep(BaseModel):
    step_id: str
    tool_name: str
    arguments: dict[str, Any]
    purpose: str
    iteration: int


class InvestigationPlan(BaseModel):
    iteration: int
    steps: list[PlanStep]
    rationale: str = ""


class ToolCallRecord(BaseModel):
    call_id: str
    step_id: str
    tool_name: str
    input: dict[str, Any]
    start_time: datetime
    end_time: datetime
    execution_time_ms: float
    success: bool
    status: str
    attempts: int
    result_summary: str
    query_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    error: ToolError | None = None

    @property
    def query_id(self) -> str | None:
        return self.query_ids[-1] if self.query_ids else None


class LLMCallRecord(BaseModel):
    task: str
    provider: str
    model: str
    attempts: int
    success: bool
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


class AgentError(BaseModel):
    stage: str
    code: str
    message: str
