"""Configuration of one agent run: every security limit plus the model-call settings.

``AgentConfig`` extends ``SecurityLimits`` (the single typed source of all limits, see
``app/security/limits.py``) with the model-call settings and the disabled-tool list. It is built
from settings once per runner. Nothing in a question, plan or tool result can change it.
"""

from __future__ import annotations

from pydantic import Field

from app.config import Settings, get_settings
from app.security.limits import SecurityLimits


class AgentConfig(SecurityLimits):
    llm_max_tokens: int = Field(default=16000, ge=256, le=64000)
    llm_temperature: float | None = None
    disabled_tools: frozenset[str] = frozenset()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> AgentConfig:
        s = settings or get_settings()
        disabled = frozenset(t.strip() for t in s.agent_disabled_tools.split(",") if t.strip())
        return cls(
            max_tool_calls=s.agent_max_tool_calls,
            max_retries=s.agent_max_retries,
            max_planning_iterations=s.agent_max_planning_iterations,
            sql_row_limit=s.agent_sql_row_limit,
            max_run_seconds=s.agent_max_run_seconds,
            max_response_chars=s.agent_max_response_chars,
            max_question_chars=s.agent_max_question_chars,
            max_filters=s.agent_max_filters,
            max_dimensions=s.agent_max_dimensions,
            max_plan_steps=s.agent_max_plan_steps,
            max_sql_calls=s.agent_max_sql_calls,
            max_sql_rows_total=s.agent_max_sql_rows_total,
            max_sql_length=s.agent_max_sql_length,
            max_sql_joins=s.agent_max_sql_joins,
            max_sql_nesting_depth=s.agent_max_sql_nesting_depth,
            max_customer_rows=s.agent_max_customer_rows,
            max_context_items=s.agent_max_context_items,
            max_context_chars=s.agent_max_context_chars,
            tool_timeout_seconds=s.agent_tool_timeout_seconds,
            sql_timeout_seconds=s.agent_sql_timeout_seconds,
            llm_timeout_seconds=s.llm_timeout_seconds,
            llm_max_tokens=s.llm_max_tokens,
            llm_temperature=s.llm_temperature,
            disabled_tools=disabled,
        )

    @property
    def recursion_limit(self) -> int:
        """Upper bound on graph super-steps for one run (LangGraph aborts beyond it)."""
        attempts = self.max_retries + 1
        return 6 + 2 * self.max_planning_iterations + self.max_tool_calls + 2 * attempts + 5
