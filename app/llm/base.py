"""The provider-neutral LLM interface used by the agent.

The agent asks a model for exactly three things, each a structured (JSON) answer:

- ``understand_question``: turn a question into a typed intent.
- ``plan_investigation``: choose allow-listed tools and their arguments.
- ``generate_response``: word an answer from already-validated claims and evidence.

Every provider (the deterministic offline model, the scripted test double and the Anthropic
model) implements the same ``LLMClient`` protocol. A request carries the rendered prompt *and*
the structured context it was rendered from. A network model reads the prompt; the deterministic
model reads the context. The agent validates every output against a Pydantic schema, so output
from any provider is treated as untrusted input.

An LLM is never asked for a business number. Numbers come from deterministic tools.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class LLMTask(StrEnum):
    UNDERSTAND = "understand_question"
    PLAN = "plan_investigation"
    RESPOND = "generate_response"


class LLMRequest(BaseModel):
    task: LLMTask
    system: str
    prompt: str
    context: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any]
    max_tokens: int
    temperature: float | None = None


class LLMUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMResponse(BaseModel):
    task: LLMTask
    content: str  # the raw JSON text; parsed and validated by the agent
    provider: str
    model: str
    stop_reason: str | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)


class LLMError(Exception):
    """The provider could not produce a usable response."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class LLMConfigurationError(LLMError):
    """The provider is not configured (missing SDK or credentials)."""


@runtime_checkable
class LLMClient(Protocol):
    provider: str
    model: str

    def generate(self, request: LLMRequest) -> LLMResponse: ...
