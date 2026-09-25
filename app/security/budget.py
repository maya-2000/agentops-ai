"""Per-run resource budget: what one agent run may consume, and what it has consumed.

``RunBudget`` holds the caps, taken from ``SecurityLimits``, and ``BudgetUsage`` the running
totals. The usage lives in the agent state and is carried through the graph. Every tool call,
SQL execution, returned SQL row, retry and model call is charged before or as it happens.
When a cap is reached, the next charge is refused and the run stops that activity. Nothing
resets the usage within a run, and nothing the model or the user says can raise a cap.

Tracked: tool calls, SQL executions, SQL rows, retries (tool, model and response
regeneration), model calls, context items and characters sent to the model, runtime and
response size.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.security.limits import SecurityLimits


class RunBudget(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_tool_calls: int
    max_sql_calls: int
    max_sql_rows: int
    max_retries_total: int
    max_model_calls: int
    max_runtime_seconds: float
    max_context_items: int
    max_context_chars: int
    max_response_chars: int

    @classmethod
    def from_limits(cls, limits: SecurityLimits) -> RunBudget:
        attempts = limits.max_retries + 1
        steps = 2 + limits.max_planning_iterations + (limits.max_retries + 1)  # understand, plans, (re)writes
        return cls(
            max_tool_calls=limits.max_tool_calls,
            max_sql_calls=limits.max_sql_calls,
            max_sql_rows=limits.max_sql_rows_total,
            # Retries per LLM step, per tool call and for regeneration, but never unbounded in total.
            max_retries_total=limits.max_retries * (steps + limits.max_tool_calls),
            max_model_calls=steps * attempts,
            max_runtime_seconds=limits.max_run_seconds,
            max_context_items=limits.max_context_items,
            max_context_chars=limits.max_context_chars,
            max_response_chars=limits.max_response_chars,
        )


class BudgetUsage(BaseModel):
    tool_calls: int = 0
    sql_calls: int = 0
    sql_rows: int = 0
    retries: int = 0
    model_calls: int = 0
    context_items: int = 0  # evidence/claim items sent to the model, summed over calls
    context_chars: int = 0  # prompt characters sent to the model, summed over calls
    max_prompt_chars: int = 0  # the largest single prompt
    response_chars: int = 0
    exhausted: list[str] = Field(default_factory=list)  # names of budgets that refused a charge


class BudgetExceededError(Exception):
    def __init__(self, resource: str, limit: float):
        super().__init__(f"{resource} budget exhausted (limit {limit:g})")
        self.resource = resource
        self.limit = limit


def remaining_tool_calls(budget: RunBudget, usage: BudgetUsage) -> int:
    return max(0, budget.max_tool_calls - usage.tool_calls)


def check_tool_call(budget: RunBudget, usage: BudgetUsage, *, sql: bool) -> str | None:
    """Name of the budget that would be exceeded by one more tool call, or ``None``."""
    if usage.tool_calls >= budget.max_tool_calls:
        return "tool_calls"
    if sql and usage.sql_calls >= budget.max_sql_calls:
        return "sql_calls"
    if sql and usage.sql_rows >= budget.max_sql_rows:
        return "sql_rows"
    return None


def charge_tool_call(usage: BudgetUsage, *, sql: bool, sql_rows: int = 0, retries: int = 0) -> BudgetUsage:
    return usage.model_copy(
        update={
            "tool_calls": usage.tool_calls + 1,
            "sql_calls": usage.sql_calls + (1 if sql else 0),
            "sql_rows": usage.sql_rows + max(0, sql_rows),
            "retries": usage.retries + max(0, retries),
        }
    )


def charge_model_call(usage: BudgetUsage, *, attempts: int, context_items: int, prompt_chars: int) -> BudgetUsage:
    return usage.model_copy(
        update={
            "model_calls": usage.model_calls + attempts,
            "retries": usage.retries + max(0, attempts - 1),
            "context_items": usage.context_items + context_items * attempts,
            "context_chars": usage.context_chars + prompt_chars * attempts,
            "max_prompt_chars": max(usage.max_prompt_chars, prompt_chars),
        }
    )


def mark_exhausted(usage: BudgetUsage, resource: str) -> BudgetUsage:
    if resource in usage.exhausted:
        return usage
    return usage.model_copy(update={"exhausted": [*usage.exhausted, resource]})


def elapsed_seconds(started_at: datetime, now: datetime) -> float:
    return (now - started_at).total_seconds()
