"""Secured execution of one tool call: the single path from a requested call to a validated result.

Two entry points share it, so their security behaviour cannot drift apart: the LangGraph agent
(``execute_tools``) and the MCP server (``app/mcp``). One call runs these steps in order:

1. ``ToolAuthorizationPolicy.authorize`` checks the allowlist, enabled state, intent, SQL
   privilege, typed arguments, argument and data-exposure policy, budget and prerequisites. A
   denied call never reaches a handler.
2. The registry runs the canonical (validated) arguments under the tool deadline. SQL runs
   under its own statement deadline.
3. A result that finished after the deadline is discarded (post-hoc timeout).
4. A transient failure is retried only when the retry policy allows it.
5. ``ToolOutputValidator`` checks the output before it can become evidence. A rejected output
   is discarded, never partly used.
6. The call is charged to the caller's budget.

Every decision is recorded as a ``SecurityEvent`` under the caller's run ID. The caller owns the
budget, the call ID and what happens next (the agent's trace or the MCP response).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.database.deadline import execution_deadline
from app.security.authorization import (
    SQL_TOOL,
    AuthorizationContext,
    AuthorizationDecision,
    ToolAuthorizationPolicy,
)
from app.security.budget import BudgetUsage, RunBudget, charge_tool_call, mark_exhausted
from app.security.events import SecurityEvent, SecurityEventType, Severity, security_event
from app.security.output_guard import ToolOutputValidator
from app.security.retry import RetryPolicy, RetryRecord
from app.tools.base import ToolContext, ToolError, ToolRequest, ToolResult
from app.tools.registry import ToolRegistry
from app.tools.results import SQLResult

DENIAL_EVENTS: dict[str, SecurityEventType] = {
    "unsafe_sql": "sql_rejected",
    "budget_exceeded": "budget_exceeded",
    "unknown_tool": "tool_denied",
    "tool_disabled": "tool_denied",
    "tool_not_permitted": "tool_denied",
    "sql_not_permitted": "tool_denied",
    "prerequisite_missing": "tool_denied",
}


def denial_event(run_id: str, decision: AuthorizationDecision, component: str) -> SecurityEvent:
    """The audit event for a denied authorization decision."""
    return security_event(
        run_id,
        DENIAL_EVENTS.get(decision.code or "", "argument_rejected"),
        decision.severity,
        component=component,
        action=decision.tool_name,
        decision="deny",
        reason=f"{decision.code}: {decision.reason}",
        code=decision.code,
    )


def failed_result(result: ToolResult, code: str, message: str) -> ToolResult:
    """Turn a result into a controlled failure: the output is discarded, never partially used."""
    return result.model_copy(
        update={
            "success": False,
            "status": "error",
            "result": None,
            "result_type": None,
            "error": ToolError(code=code, message=message, retryable=False),
            "message": message,
            "query_ids": [],
            "source_tables": [],
        }
    )


@dataclass(frozen=True)
class SecuredCall:
    """The outcome of one secured call: the (possibly failed) result and everything that was decided."""

    result: ToolResult
    decision: AuthorizationDecision
    usage: BudgetUsage
    attempts: int
    retries: list[RetryRecord] = field(default_factory=list)
    events: list[SecurityEvent] = field(default_factory=list)


class SecuredToolExecutor:
    """Runs tool calls through authorization, deadline, retry, output validation and budget charging."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: ToolAuthorizationPolicy,
        output_validator: ToolOutputValidator,
        retry_policy: RetryPolicy,
        tool_context: ToolContext,
        *,
        tool_timeout_seconds: float,
    ):
        self.registry = registry
        self.policy = policy
        self.output_validator = output_validator
        self.retry_policy = retry_policy
        self.tool_context = tool_context
        self.tool_timeout_seconds = tool_timeout_seconds

    def execute(
        self,
        *,
        run_id: str,
        call_id: str,
        tool_name: str,
        arguments: Any,
        purpose: str,
        context: AuthorizationContext,
        budget: RunBudget,
        usage: BudgetUsage,
    ) -> SecuredCall:
        events: list[SecurityEvent] = []
        retries: list[RetryRecord] = []
        attempts = 0
        decision = self.policy.authorize(tool_name, arguments, context=context, budget=budget, usage=usage)
        if not decision.allowed:
            events.append(denial_event(run_id, decision, "tool_authorization"))
            now = datetime.now(UTC)
            result = ToolResult(
                call_id=call_id,
                tool_name=str(tool_name)[:80],
                arguments=dict(arguments) if isinstance(arguments, Mapping) else {},
                success=False,
                status="error",
                error=ToolError(code=decision.code or "unauthorized_tool", message=decision.reason, retryable=False),
                message=decision.reason,
                started_at=now,
                finished_at=now,
                execution_time_ms=0.0,
            )
            if decision.code == "budget_exceeded":
                usage = mark_exhausted(usage, "tool_budget")
            return SecuredCall(result.model_copy(update={"attempts": 1}), decision, usage, 0, retries, events)

        events.append(
            security_event(
                run_id,
                "tool_authorized",
                Severity.INFO,
                component="tool_authorization",
                action=tool_name,
                decision="allow",
                reason="All authorization checks passed.",
                checks=decision.checks_passed,
            )
        )
        canonical = decision.arguments if decision.arguments is not None else dict(arguments)
        limit = self.tool_timeout_seconds
        while True:
            attempts += 1
            clock = time.perf_counter()
            with execution_deadline(limit):
                result = self.registry.execute(
                    ToolRequest(call_id=call_id, tool_name=tool_name, arguments=canonical, purpose=purpose),
                    self.tool_context,
                )
            took = time.perf_counter() - clock
            if result.success and took > limit:
                result = failed_result(result, "timeout", f"The tool exceeded its {limit:g}s limit.")
            code = result.error.code if result.error else None
            if code == "timeout":
                events.append(
                    security_event(
                        run_id,
                        "timeout",
                        Severity.WARNING,
                        component="tool_execution",
                        action=tool_name,
                        decision="stop",
                        reason="The tool call exceeded its time limit; no result was used.",
                    )
                )
            if result.success:
                break
            retry = self.retry_policy.should_retry(code, attempts)
            retries.append(
                RetryRecord(
                    stage=f"execute_tools:{tool_name}",
                    attempt=attempts,
                    code=code or "unknown",
                    decision="retry" if retry else "stop",
                )
            )
            if not retry:
                break
            events.append(
                security_event(
                    run_id,
                    "retry",
                    Severity.WARNING,
                    component="tool_execution",
                    action=tool_name,
                    decision="retry",
                    reason=f"Transient failure ({code}); attempt {attempts + 1}.",
                )
            )
        violations = self.output_validator.validate(result)
        if violations:
            events.append(
                security_event(
                    run_id,
                    "tool_output_rejected",
                    Severity.HIGH,
                    component="output_guard",
                    action=tool_name,
                    decision="deny",
                    reason="; ".join(f"{v.field}: {v.message}" for v in violations[:3]),
                )
            )
            data_policy = any(v.code == "data_policy" for v in violations)
            rejection = "data_policy" if data_policy else "invalid_tool_output"
            result = failed_result(result, rejection, violations[0].message)
        sql_rows = result.result.row_count if isinstance(result.result, SQLResult) else 0
        usage = charge_tool_call(usage, sql=tool_name == SQL_TOOL, sql_rows=sql_rows, retries=max(0, attempts - 1))
        return SecuredCall(result.model_copy(update={"attempts": attempts}), decision, usage, attempts, retries, events)
