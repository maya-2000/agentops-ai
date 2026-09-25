"""Application settings loaded from environment variables (and an optional ``.env`` file)."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LLMProviderName = Literal["deterministic", "offline", "anthropic"]
DEFAULT_LLM_MODEL = "claude-opus-5"


class Settings(BaseSettings):
    """Runtime configuration. Secrets use ``SecretStr`` and are never logged or placed in agent state."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "duckdb:///database/northwind_cloud.duckdb"
    dataset_manifest_path: Path = Path("data/metadata/dataset_manifest.json")
    as_of_date: date = date(2026, 8, 31)
    data_seed: int = 42
    log_level: str = "INFO"

    # ---- Phase 4: LLM provider (the deterministic provider needs no key and makes no network call) ----
    llm_provider: LLMProviderName = "deterministic"
    llm_model: str = DEFAULT_LLM_MODEL
    llm_temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=16000, ge=256, le=64000)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    anthropic_api_key: SecretStr | None = None

    # ---- Phase 4: agent limits (hard bounds on every run) ----
    agent_max_tool_calls: int = Field(default=12, ge=1, le=50)
    agent_max_retries: int = Field(default=2, ge=0, le=5)
    agent_max_planning_iterations: int = Field(default=2, ge=1, le=5)
    agent_sql_row_limit: int = Field(default=200, ge=1, le=5000)
    agent_max_run_seconds: float = Field(default=120.0, gt=0)
    agent_max_response_chars: int = Field(default=4000, ge=200, le=20000)

    @field_validator("llm_model", mode="before")
    @classmethod
    def _default_model(cls, value: object) -> object:
        return DEFAULT_LLM_MODEL if value in (None, "") else value

    @field_validator("llm_temperature", mode="before")
    @classmethod
    def _optional_temperature(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("anthropic_api_key", mode="before")
    @classmethod
    def _empty_key_is_none(cls, value: object) -> object:
        return None if value in (None, "") else value

    def resolve_path(self, path: Path) -> Path:
        """Resolve a project-relative path against the repository root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
