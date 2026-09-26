"""API configuration, read once from the application settings (``API_*`` environment variables)."""

from __future__ import annotations

from dataclasses import dataclass

from app.api import API_VERSION
from app.config import Settings, get_settings


@dataclass(frozen=True)
class APIConfig:
    host: str
    port: int
    request_timeout_seconds: float
    max_request_bytes: int
    max_pending_requests: int
    max_question_chars: int  # the agent's own limit (AGENT_MAX_QUESTION_CHARS), checked before a run starts
    version: str = API_VERSION

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> APIConfig:
        s = settings or get_settings()
        return cls(
            host=s.api_host,
            port=s.api_port,
            request_timeout_seconds=s.api_request_timeout_seconds,
            max_request_bytes=s.api_max_request_bytes,
            max_pending_requests=s.api_max_pending_requests,
            max_question_chars=s.agent_max_question_chars,
        )
