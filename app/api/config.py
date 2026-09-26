"""API configuration, read once from the application settings, and the API's start-up checks.

``APIConfig.startup_problems`` lists what would make the API unsafe to serve. The API refuses to
start while any remain. The messages name settings, never their values.

- Every environment: token authentication needs ``API_AUTH_TOKEN``.
- ``APP_ENV=production`` also requires:
  - token authentication (``disabled`` is refused);
  - rate limiting on;
  - no wildcard and no plain-HTTP CORS origin;
  - an explicitly configured ``DATABASE_URL``;
  - timeouts that nest: SQL ≤ tool ≤ agent run ≤ API request ≤ UI request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

from pydantic import SecretStr

from app.api import API_VERSION
from app.config import AppEnvironment, AuthMode, Settings, get_settings


class ConfigurationError(RuntimeError):
    """The API configuration is unsafe or incomplete; the message lists the settings to fix."""


@dataclass(frozen=True)
class RateLimit:
    requests: int
    window_seconds: float
    max_clients: int = 10000


@dataclass(frozen=True)
class APIConfig:
    host: str
    port: int
    request_timeout_seconds: float
    max_request_bytes: int
    max_pending_requests: int
    max_question_chars: int  # the agent's own limit (AGENT_MAX_QUESTION_CHARS), checked before a run starts
    version: str = API_VERSION
    # ---- Phase 9 (secure defaults: token authentication, and rate limiting on) ----
    environment: AppEnvironment = "development"
    auth_mode: AuthMode = "token"
    auth_token: SecretStr | None = field(default=None, repr=False)
    rate_limit: RateLimit | None = RateLimit(requests=20, window_seconds=60.0)
    cors_origins: tuple[str, ...] = ()
    docs_enabled: bool = True
    metrics_enabled: bool = True
    shutdown_grace_seconds: float = 10.0
    problems: tuple[str, ...] = ()  # settings-level problems found by from_settings (production rules)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> APIConfig:
        s = settings or get_settings()
        limit = s.rate_limit
        return cls(
            host=s.api_host,
            port=s.api_port,
            request_timeout_seconds=s.api_request_timeout_seconds,
            max_request_bytes=s.api_max_request_bytes,
            max_pending_requests=s.api_max_pending_requests,
            max_question_chars=s.agent_max_question_chars,
            environment=s.app_env,
            auth_mode=s.api_auth_mode,
            auth_token=s.api_auth_token,
            rate_limit=RateLimit(limit[0], limit[1], s.api_rate_limit_max_clients) if limit else None,
            cors_origins=tuple(s.cors_origins),
            docs_enabled=s.docs_enabled,
            metrics_enabled=s.api_metrics_enabled,
            shutdown_grace_seconds=s.api_shutdown_grace_seconds,
            problems=tuple(_production_problems(s)) if s.app_env == "production" else (),
        )

    @property
    def auth_enabled(self) -> bool:
        return self.auth_mode == "token"

    def startup_problems(self) -> list[str]:
        problems = list(self.problems)
        if self.auth_enabled and self.auth_token is None:
            problems.append(
                "API_AUTH_TOKEN is not set. Set a token of at least 32 characters (for example the output of "
                '`python -c "import secrets; print(secrets.token_urlsafe(32))"`), or, for local development '
                "only, set API_AUTH_MODE=disabled."
            )
        if self.environment == "production":
            if not self.auth_enabled:
                problems.append("API_AUTH_MODE=disabled is not allowed when APP_ENV=production.")
            if self.rate_limit is None:
                problems.append("API_RATE_LIMIT=off is not allowed when APP_ENV=production.")
            if "*" in self.cors_origins:
                problems.append("API_CORS_ORIGINS may not contain '*' when APP_ENV=production.")
            if any(o.startswith("http://") for o in self.cors_origins):
                problems.append("API_CORS_ORIGINS must use https:// origins when APP_ENV=production.")
        return list(dict.fromkeys(problems))

    def check(self) -> None:
        """Raise ``ConfigurationError`` listing every start-up problem (none: the API may serve)."""
        problems = self.startup_problems()
        if problems:
            raise ConfigurationError("The API configuration is not safe to serve:\n- " + "\n- ".join(problems))


def _production_problems(s: Settings) -> list[str]:
    """Rules that need the full settings (not only the API's own fields)."""
    problems = []
    if "database_url" not in s.model_fields_set:
        problems.append("DATABASE_URL must be set explicitly when APP_ENV=production (no default location).")
    chain = [
        ("AGENT_SQL_TIMEOUT_SECONDS", s.agent_sql_timeout_seconds),
        ("AGENT_TOOL_TIMEOUT_SECONDS", s.agent_tool_timeout_seconds),
        ("AGENT_MAX_RUN_SECONDS", s.agent_max_run_seconds),
        ("API_REQUEST_TIMEOUT_SECONDS", s.api_request_timeout_seconds),
        ("UI_REQUEST_TIMEOUT_SECONDS", s.ui_request_timeout_seconds),
    ]
    for (inner, inner_value), (outer, outer_value) in pairwise(chain):
        if inner_value > outer_value:
            problems.append(f"{inner} must not exceed {outer} (timeouts nest from the inside out).")
    return problems
