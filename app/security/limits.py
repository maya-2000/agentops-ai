"""Every numeric security limit of an agent run, in one typed, immutable object.

``SecurityLimits`` is the single source of the limits that the input guard, plan validator, tool
authorization, SQL validator, run budget, context budget and output guard enforce. Values come
from settings (``AGENT_*`` environment variables, see ``.env.example``); nothing else in the
code base hard-codes them. The model and the user cannot change them: they are read once when
the runner is created and never taken from a question, a plan or a tool result.

| Brief name            | Field                  | Environment variable            |
|-----------------------|------------------------|---------------------------------|
| MAX_QUESTION_LENGTH   | max_question_chars     | AGENT_MAX_QUESTION_CHARS        |
| MAX_FILTERS           | max_filters            | AGENT_MAX_FILTERS               |
| MAX_DIMENSIONS        | max_dimensions         | AGENT_MAX_DIMENSIONS            |
| MAX_PLAN_STEPS        | max_plan_steps         | AGENT_MAX_PLAN_STEPS            |
| MAX_TOOL_CALLS        | max_tool_calls         | AGENT_MAX_TOOL_CALLS            |
| MAX_RETRIES           | max_retries            | AGENT_MAX_RETRIES               |
| MAX_SQL_ROWS          | sql_row_limit          | AGENT_SQL_ROW_LIMIT             |
| MAX_SQL_LENGTH        | max_sql_length         | AGENT_MAX_SQL_LENGTH            |
| MAX_RESPONSE_LENGTH   | max_response_chars     | AGENT_MAX_RESPONSE_CHARS        |
| MAX_CONTEXT_ITEMS     | max_context_items      | AGENT_MAX_CONTEXT_ITEMS         |
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SecurityLimits(BaseModel):
    model_config = ConfigDict(frozen=True)

    # ---- input
    max_question_chars: int = Field(default=1000, ge=20, le=10000)
    max_filters: int = Field(default=5, ge=0, le=20)
    max_dimensions: int = Field(default=3, ge=0, le=10)
    max_text_field_chars: int = Field(default=500, ge=50, le=5000)  # any free-text field in model output or tool input
    max_argument_chars: int = Field(default=8000, ge=500, le=100000)  # one tool call's arguments, as JSON

    # ---- planning and execution
    max_plan_steps: int = Field(default=8, ge=1, le=50)
    max_tool_calls: int = Field(default=12, ge=1, le=50)
    max_retries: int = Field(default=2, ge=0, le=5)  # per LLM step, per retryable tool error, per response regeneration
    max_planning_iterations: int = Field(default=2, ge=1, le=5)
    max_run_seconds: float = Field(default=120.0, gt=0)
    tool_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)

    # ---- SQL
    sql_row_limit: int = Field(default=200, ge=1, le=5000)  # rows returned by one query
    max_sql_calls: int = Field(default=3, ge=0, le=20)  # run_safe_sql executions per run
    max_sql_rows_total: int = Field(default=1000, ge=1, le=100000)  # rows returned by all queries of a run
    max_sql_length: int = Field(default=4000, ge=100, le=100000)
    max_sql_joins: int = Field(default=4, ge=0, le=20)
    max_sql_nesting_depth: int = Field(default=3, ge=0, le=10)  # nested subqueries
    max_sql_ctes: int = Field(default=4, ge=0, le=20)
    max_sql_parameters: int = Field(default=20, ge=0, le=200)
    sql_timeout_seconds: float = Field(default=10.0, gt=0)

    # ---- data exposure
    max_customer_rows: int = Field(default=25, ge=1, le=200)  # customer-level rows one tool call may return

    # ---- context and output
    max_context_items: int = Field(default=40, ge=5, le=500)  # evidence/claim items sent to the model per call
    max_context_chars: int = Field(default=60000, ge=2000, le=1000000)  # rendered prompt size per model call
    max_response_chars: int = Field(default=4000, ge=200, le=20000)

    @model_validator(mode="before")
    @classmethod
    def _consistent(cls, data: Any) -> Any:
        """Keep dependent limits consistent: plan steps never exceed tool calls, the per-run SQL row
        budget is never below one query's row limit."""
        if not isinstance(data, dict):
            return data
        fields = cls.model_fields
        tools = data.get("max_tool_calls", fields["max_tool_calls"].default)
        plan = data.get("max_plan_steps", fields["max_plan_steps"].default)
        rows = data.get("sql_row_limit", fields["sql_row_limit"].default)
        total = data.get("max_sql_rows_total", fields["max_sql_rows_total"].default)
        return {**data, "max_plan_steps": min(plan, tools), "max_sql_rows_total": max(total, rows)}
