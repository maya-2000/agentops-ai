"""Hard limits for one agent run. Every loop in the graph is bounded by these values."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings


class AgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_tool_calls: int = Field(default=12, ge=1, le=50)
    max_retries: int = Field(default=2, ge=0, le=5)  # per LLM step, per tool call, and for response regeneration
    max_planning_iterations: int = Field(default=2, ge=1, le=5)
    sql_row_limit: int = Field(default=200, ge=1, le=5000)
    max_run_seconds: float = Field(default=120.0, gt=0)
    max_response_chars: int = Field(default=4000, ge=200, le=20000)
    max_question_chars: int = Field(default=1000, ge=20, le=10000)
    llm_max_tokens: int = Field(default=16000, ge=256, le=64000)
    llm_temperature: float | None = None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> AgentConfig:
        s = settings or get_settings()
        return cls(
            max_tool_calls=s.agent_max_tool_calls,
            max_retries=s.agent_max_retries,
            max_planning_iterations=s.agent_max_planning_iterations,
            sql_row_limit=s.agent_sql_row_limit,
            max_run_seconds=s.agent_max_run_seconds,
            max_response_chars=s.agent_max_response_chars,
            llm_max_tokens=s.llm_max_tokens,
            llm_temperature=s.llm_temperature,
        )

    @property
    def recursion_limit(self) -> int:
        """Upper bound on graph super-steps for one run (LangGraph aborts beyond it)."""
        attempts = self.max_retries + 1
        return 6 + 2 * self.max_planning_iterations + self.max_tool_calls + 2 * attempts + 5
