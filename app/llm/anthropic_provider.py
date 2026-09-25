"""Claude via the official Anthropic SDK (optional extra: ``pip install -e ".[anthropic]"``).

- Structured outputs: ``output_config.format`` with the task's strict JSON schema, so the first text
  block is valid JSON for that schema (the agent still validates it).
- Refusals: ``stop_reason == "refusal"`` is handled before reading content. Server-side fallbacks
  (``fallbacks="default"``, beta ``server-side-fallback-2026-07-01``) are enabled by default, so a
  declined request is re-run on Anthropic's recommended fallback model.
- Credentials: the SDK resolves them from the environment (``ANTHROPIC_API_KEY`` or a profile); a key
  passed from settings is used when present. Keys are never logged or placed in agent state.
- Sampling: current Claude Opus models reject ``temperature``, so it is only sent when configured.

The SDK is imported lazily, so the rest of the system (and every test) runs without it. Tests
inject a fake client and never reach the network.
"""

from __future__ import annotations

from typing import Any

from app.llm.base import LLMConfigurationError, LLMError, LLMRequest, LLMResponse, LLMUsage

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM:
    provider = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        client: Any | None = None,
        enable_fallbacks: bool = True,
    ):
        self.model = model
        self.enable_fallbacks = enable_fallbacks
        self._client = client if client is not None else self._create_client(api_key, timeout_seconds)

    @staticmethod
    def _create_client(api_key: str | None, timeout_seconds: float) -> Any:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
            raise LLMConfigurationError(
                'The Anthropic SDK is not installed. Install the extra with: pip install -e ".[anthropic]"'
            ) from exc
        kwargs: dict[str, Any] = {"timeout": timeout_seconds}
        if api_key:
            kwargs["api_key"] = api_key
        return anthropic.Anthropic(**kwargs)

    def generate(self, request: LLMRequest) -> LLMResponse:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": [{"role": "user", "content": request.prompt}],
            "output_config": {"format": {"type": "json_schema", "schema": request.output_schema}},
        }
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if self.enable_fallbacks:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        try:
            response = self._client.beta.messages.create(**params)
        except Exception as exc:  # SDK errors are normalised; the agent decides whether to retry
            raise LLMError(f"Anthropic request failed: {type(exc).__name__}", retryable=_retryable(exc)) from exc
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            raise LLMError("The model declined the request (stop_reason=refusal).", retryable=False)
        if stop_reason == "max_tokens":
            raise LLMError("The model response was cut off at max_tokens.", retryable=False)
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        if not text.strip():
            raise LLMError("The model returned no text content.", retryable=True)
        usage = getattr(response, "usage", None)
        return LLMResponse(
            task=request.task,
            content=text,
            provider=self.provider,
            model=str(getattr(response, "model", self.model)),
            stop_reason=stop_reason,
            usage=LLMUsage(
                input_tokens=getattr(usage, "input_tokens", None), output_tokens=getattr(usage, "output_tokens", None)
            ),
        )


def _retryable(exc: Exception) -> bool:
    """Connection errors, rate limits and server errors are retryable; request errors are not."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return type(exc).__name__ in ("APIConnectionError", "APITimeoutError")
