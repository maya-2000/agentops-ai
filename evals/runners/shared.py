"""Shared-execution probe: show that the agent and MCP both run tools through ``app/security/execution.py``.

For the duration of the probe, ``SecuredToolExecutor.execute`` and the ``execution_deadline``
used inside it are wrapped with recorders. The wrappers call the originals unchanged, and they
are removed afterwards. Then one agent question and the scenario's MCP calls run. Each record
says which entry point triggered the executor, whether authorization allowed the call, whether
a deadline was applied, whether the budget was charged and which security events were emitted.

This is instrumentation of a real run, not a second execution path: removing the shared
executor from either entry point makes the probe report that path as missing.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import app.security.execution as execution_module
from app.security.execution import SecuredCall, SecuredToolExecutor
from evals.reference.context import EvalContext
from evals.runners.agent import LLMFactory, run_agent
from evals.runners.tools import run_mcp
from evals.scenarios.model import EvaluationScenario


@dataclass
class ExecutorRecord:
    entry_point: str  # "agent" or "mcp"
    tool: str
    allowed: bool
    code: str | None
    deadline_applied: bool
    budget_charged: bool
    attempts: int = 0
    events: list[str] = field(default_factory=list)


@dataclass
class SharedObservation:
    records: list[ExecutorRecord]
    agent_status: str | None
    mcp_outcomes: list[tuple[str, bool]]  # (tool, is_error)
    latency_ms: float


@contextmanager
def instrument_executor(records: list[ExecutorRecord], entry_point: list[str]) -> Iterator[None]:
    original_execute = SecuredToolExecutor.execute
    original_deadline = execution_module.execution_deadline
    deadlines: list[float | None] = []

    def recording_deadline(seconds: float | None) -> Any:
        deadlines.append(seconds)
        return original_deadline(seconds)

    def recording_execute(self: SecuredToolExecutor, **kwargs: Any) -> SecuredCall:
        before = len(deadlines)
        call: SecuredCall = original_execute(self, **kwargs)
        records.append(
            ExecutorRecord(
                entry_point=entry_point[0],
                tool=str(kwargs.get("tool_name")),
                allowed=call.decision.allowed,
                code=call.result.error.code if call.result.error else None,
                deadline_applied=len(deadlines) > before,
                budget_charged=call.usage.tool_calls > kwargs["usage"].tool_calls,
                attempts=call.attempts,
                events=[e.event_type for e in call.events],
            )
        )
        return call

    with (
        patch.object(SecuredToolExecutor, "execute", recording_execute),
        patch.object(execution_module, "execution_deadline", recording_deadline),
    ):
        yield


def run_shared(ctx: EvalContext, scenario: EvaluationScenario, llm_factory: LLMFactory) -> SharedObservation:
    records: list[ExecutorRecord] = []
    entry_point = ["agent"]
    with instrument_executor(records, entry_point):
        agent = run_agent(ctx, scenario, llm_factory)
        entry_point[0] = "mcp"
        mcp = run_mcp(ctx, scenario)
    return SharedObservation(
        records=records,
        agent_status=agent.result.status if agent.result else None,
        mcp_outcomes=[(c.call.tool, c.is_error) for c in mcp.calls],
        latency_ms=agent.latency_ms + sum(c.latency_ms for c in mcp.calls),
    )
