"""Application settings loaded from environment variables (and an optional ``.env`` file)."""

from __future__ import annotations

import re
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LLMProviderName = Literal["deterministic", "offline", "anthropic"]
MCPTransport = Literal["stdio"]
MCPLogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
AppEnvironment = Literal["development", "test", "production"]
AuthMode = Literal["token", "disabled"]
LogFormat = Literal["json", "text"]
DEFAULT_LLM_MODEL = "claude-opus-5"
MIN_API_TOKEN_CHARS = 32
_RATE_LIMIT = re.compile(r"^\s*(\d{1,6})\s*/\s*(second|minute|hour)\s*$", re.IGNORECASE)
_ORIGIN = re.compile(r"^https?://[A-Za-z0-9.\-]+(:\d{1,5})?$")
_PERIOD_SECONDS = {"second": 1.0, "minute": 60.0, "hour": 3600.0}


class Settings(BaseSettings):
    """Runtime configuration. Secrets use ``SecretStr`` and are never logged or placed in agent state."""

    # hide_input_in_errors: a rejected value (a token, a key) is never echoed in a start-up error.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", hide_input_in_errors=True
    )

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

    # ---- Phase 4/5: agent limits (hard bounds on every run; see app/security/limits.py) ----
    agent_max_tool_calls: int = Field(default=12, ge=1, le=50)
    agent_max_retries: int = Field(default=2, ge=0, le=5)
    agent_max_planning_iterations: int = Field(default=2, ge=1, le=5)
    agent_sql_row_limit: int = Field(default=200, ge=1, le=5000)
    agent_max_run_seconds: float = Field(default=120.0, gt=0)
    agent_max_response_chars: int = Field(default=4000, ge=200, le=20000)
    agent_max_question_chars: int = Field(default=1000, ge=20, le=10000)
    agent_max_filters: int = Field(default=5, ge=0, le=20)
    agent_max_dimensions: int = Field(default=3, ge=0, le=10)
    agent_max_plan_steps: int = Field(default=8, ge=1, le=50)
    agent_max_sql_calls: int = Field(default=3, ge=0, le=20)
    agent_max_sql_rows_total: int = Field(default=1000, ge=1, le=100000)
    agent_max_sql_length: int = Field(default=4000, ge=100, le=100000)
    agent_max_sql_joins: int = Field(default=4, ge=0, le=20)
    agent_max_sql_nesting_depth: int = Field(default=3, ge=0, le=10)
    agent_max_customer_rows: int = Field(default=25, ge=1, le=200)
    agent_max_context_items: int = Field(default=40, ge=5, le=500)
    agent_max_context_chars: int = Field(default=60000, ge=2000, le=1000000)
    agent_tool_timeout_seconds: float = Field(default=30.0, gt=0)
    agent_sql_timeout_seconds: float = Field(default=10.0, gt=0)
    agent_disabled_tools: str = ""  # comma-separated tool names switched off for the agent

    # ---- Phase 6: MCP server (local stdio transport; see docs/mcp-architecture.md) ----
    # The MCP server uses the agent limits above; these settings only add what is MCP-specific.
    mcp_server_name: str = Field(default="agentops-ai", pattern=r"^[a-z][a-z0-9_.-]{0,63}$")
    mcp_server_version: str = Field(default="0.6.0", pattern=r"^\d+\.\d+\.\d+$")
    mcp_enabled_tools: str = ""  # comma-separated agentops_* tool names; empty means all twelve
    mcp_transport: MCPTransport = "stdio"
    mcp_log_level: MCPLogLevel = "WARNING"
    mcp_max_request_bytes: int = Field(default=16384, ge=1024, le=1_048_576)
    mcp_max_response_bytes: int = Field(default=262144, ge=16384, le=8_388_608)
    mcp_sql_enabled: bool = True  # False unlists agentops_run_safe_sql and revokes the SQL privilege

    # ---- Phase 8: HTTP API and web UI (local development; see docs/api.md and docs/ui.md) ----
    # The API runs the Phase 4/5 agent with the AGENT_* limits above; it adds only transport settings.
    api_host: str = Field(default="127.0.0.1", pattern=r"^[A-Za-z0-9.:\[\]-]{1,253}$")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_request_timeout_seconds: float = Field(default=150.0, gt=0, le=3600)  # above AGENT_MAX_RUN_SECONDS
    api_max_request_bytes: int = Field(default=16384, ge=1024, le=1_048_576)
    api_max_pending_requests: int = Field(default=4, ge=1, le=64)  # agent runs are serialised; more waiting -> 503
    ui_api_url: str = Field(default="http://127.0.0.1:8000", pattern=r"^https?://[^\s/?#]+(:\d{1,5})?/?$")
    ui_request_timeout_seconds: float = Field(default=180.0, gt=0, le=3600)  # above API_REQUEST_TIMEOUT_SECONDS

    # ---- Phase 9: environment and production hardening (see docs/deployment.md, docs/security.md) ----
    # APP_ENV=production turns on strict start-up checks for the API (app/api/config.py): a token, an
    # explicit database location, no wildcard CORS, rate limiting on, and ordered timeouts.
    app_env: AppEnvironment = "development"
    log_format: LogFormat = "json"  # json: one JSON object per line (timestamp, level, logger, event fields)
    api_auth_mode: AuthMode = "token"  # "disabled" is refused when APP_ENV=production
    api_auth_token: SecretStr | None = None  # bearer token (at least 32 characters); never logged or shown
    api_rate_limit: str = "20/minute"  # per client, on /ask and /ask/stream; "off" is refused in production
    api_rate_limit_max_clients: int = Field(default=10000, ge=10, le=1_000_000)  # tracked clients (memory bound)
    api_cors_origins: str = ""  # comma-separated browser origins; empty: no cross-origin access
    api_docs_enabled: bool | None = None  # OpenAPI docs; None: on, except in production
    api_metrics_enabled: bool = True  # GET /api/v1/metrics (authenticated)
    api_shutdown_grace_seconds: float = Field(default=10.0, ge=0, le=300)  # in-flight runs finish or are cancelled
    ui_history_limit: int = Field(default=20, ge=0, le=200)  # questions kept per browser session (memory only)
    ui_host: str = Field(default="127.0.0.1", pattern=r"^[A-Za-z0-9.:\[\]-]{1,253}$")  # python -m app.ui
    ui_port: int = Field(default=8501, ge=1, le=65535)
    # The hostname users browse to (behind a reverse proxy); also stops Streamlit's public-IP lookup.
    ui_public_address: str = Field(default="localhost", pattern=r"^[A-Za-z0-9.-]{1,253}$")

    @field_validator("llm_model", mode="before")
    @classmethod
    def _default_model(cls, value: object) -> object:
        return DEFAULT_LLM_MODEL if value in (None, "") else value

    @field_validator("llm_temperature", mode="before")
    @classmethod
    def _optional_temperature(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("anthropic_api_key", "api_auth_token", "api_docs_enabled", mode="before")
    @classmethod
    def _empty_key_is_none(cls, value: object) -> object:
        return None if value in (None, "") else value

    @field_validator("api_auth_token")
    @classmethod
    def _token_shape(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None:
            token = value.get_secret_value()
            if len(token) < MIN_API_TOKEN_CHARS or any(ch.isspace() for ch in token):
                # The message never contains the value.
                raise ValueError(f"API_AUTH_TOKEN must be at least {MIN_API_TOKEN_CHARS} characters without spaces")
        return value

    @field_validator("api_rate_limit")
    @classmethod
    def _rate_limit_shape(cls, value: str) -> str:
        if value.strip().lower() != "off" and not _RATE_LIMIT.match(value):
            raise ValueError("API_RATE_LIMIT must look like '20/minute' (per second, minute or hour) or 'off'")
        return value.strip()

    @field_validator("api_cors_origins")
    @classmethod
    def _origins_shape(cls, value: str) -> str:
        for origin in (o.strip() for o in value.split(",") if o.strip()):
            if origin != "*" and not _ORIGIN.match(origin):
                raise ValueError(f"API_CORS_ORIGINS entries must be scheme://host[:port] origins, not {origin[:80]!r}")
        return value

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.api_cors_origins.split(",") if o.strip()]

    @property
    def rate_limit(self) -> tuple[int, float] | None:
        """(requests, window seconds), or None when rate limiting is off."""
        match = _RATE_LIMIT.match(self.api_rate_limit)
        if match is None:
            return None
        return int(match.group(1)), _PERIOD_SECONDS[match.group(2).lower()]

    @property
    def docs_enabled(self) -> bool:
        return self.api_docs_enabled if self.api_docs_enabled is not None else self.app_env != "production"

    def resolve_path(self, path: Path) -> Path:
        """Resolve a project-relative path against the repository root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def settings_errors(error: ValidationError) -> list[str]:
    """One line per invalid setting: the variable name and the reason, never the submitted value."""
    return [
        f"{'.'.join(str(part) for part in e['loc']).upper() or 'SETTINGS'}: {e['msg']}"
        for e in error.errors(include_url=False, include_context=False, include_input=False)
    ]


def settings_or_exit() -> Settings:
    """``get_settings()`` for command-line entry points: invalid settings end the process with exit code 2
    and one line per setting on stderr (never a value or a traceback)."""
    try:
        return get_settings()
    except ValidationError as error:
        print("The configuration is invalid:", *(f"- {e}" for e in settings_errors(error)), sep="\n", file=sys.stderr)
        raise SystemExit(2) from None
