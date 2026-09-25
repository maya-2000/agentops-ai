"""Create the configured ``LLMClient``. The default is the deterministic offline model."""

from __future__ import annotations

from app.config import Settings, get_settings
from app.llm.base import LLMClient, LLMConfigurationError
from app.security.redaction import register_secret


def create_llm_client(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    if settings.llm_provider in ("deterministic", "offline"):
        from app.llm.deterministic import DeterministicLLM

        return DeterministicLLM()
    if settings.llm_provider == "anthropic":
        from app.llm.anthropic_provider import AnthropicLLM

        key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        register_secret(key)  # redacted wherever it might appear (errors, logs, traces)
        return AnthropicLLM(settings.llm_model, api_key=key, timeout_seconds=settings.llm_timeout_seconds)
    raise LLMConfigurationError(f"Unknown LLM provider {settings.llm_provider!r}")  # pragma: no cover
