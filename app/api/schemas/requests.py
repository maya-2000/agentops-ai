"""The request body of ``POST /api/v1/ask`` (and ``/ask/stream``)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictStr

# A request or session ID is a short token: safe to echo in headers and to write into log lines.
# The same rule as ``app.agent.observability.is_valid_run_id`` (the request ID becomes the run ID).
ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$"
# A transport ceiling only. The configured agent limit (AGENT_MAX_QUESTION_CHARS, at most 10,000)
# is checked by the route, and the agent's input guard checks it again.
MAX_QUESTION_CHARS = 10_000


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: StrictStr = Field(
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
        description="A business question about the Northwind Cloud dataset.",
        examples=["What was revenue in July compared with June?"],
    )
    request_id: StrictStr | None = Field(
        default=None,
        pattern=ID_PATTERN,
        description="Optional client correlation ID (also accepted as the X-Request-ID header). "
        "It becomes the agent run ID; one is generated when absent.",
    )
    session_id: StrictStr | None = Field(
        default=None,
        pattern=ID_PATTERN,
        description="Optional client session ID, echoed back and logged. Nothing is stored per session.",
    )
