"""Structured audit log of the MCP server: one JSON object per event on the ``agentops.mcp`` logger.

Every event of a tool call carries its run ID (``request_id``), which is also the ``run_id`` of
the call's Phase 5 security events. One ID therefore correlates the MCP request, the
authorization decision, the tool execution and the response. The JSON-RPC request ID is
recorded too.

Logged: the server lifecycle and, for each call, the tool name, the validation, authorization and
execution status, the error category and code, sizes, duration, query IDs, evidence IDs and the
number of security events. Never logged: arguments, SQL text, result rows, API keys or other
secrets, environment variables or filesystem paths. Unknown keys are dropped, and every value is
redacted. Logs go to stderr, because stdout carries the stdio protocol.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.security.redaction import redact_value

LOGGER_NAME = "agentops.mcp"
logger = logging.getLogger(LOGGER_NAME)

_ALLOWED_KEYS = frozenset(
    {
        "server_name",
        "server_version",
        "transport",
        "tools",
        "tool_name",
        "mcp_request_id",
        "status",
        "is_error",
        "validation",
        "authorization",
        "execution",
        "evidence_ids",
        "error_category",
        "error_code",
        "request_bytes",
        "response_bytes",
        "truncated",
        "execution_time_ms",
        "query_ids",
        "security_events",
        "highest_severity",
    }
)


def audit(event: str, request_id: str | None = None, **fields: Any) -> dict[str, Any]:
    """Emit one audit record and return it (so callers and tests can inspect exactly what was logged)."""
    record: dict[str, Any] = {"event": event}
    if request_id is not None:
        record["request_id"] = request_id
    record.update({k: redact_value(v) for k, v in fields.items() if k in _ALLOWED_KEYS})
    if logger.isEnabledFor(logging.INFO):
        logger.info(json.dumps(record, default=str, sort_keys=True))
    return record
