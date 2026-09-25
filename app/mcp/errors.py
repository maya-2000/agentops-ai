"""The MCP error model: every internal error code becomes one client-safe ``MCPError``.

The mapping is derived from the Phase 5 error categories (``app.security.errors``), with a few
codes refined into the more specific MCP categories. Messages are fixed strings. Exception text,
stack traces, filesystem paths, SQL internals and secrets never reach the client. Only argument
errors (INVALID_ARGUMENT, UNSUPPORTED_REQUEST) carry a ``detail``, so that a client can correct
its call. The detail is sanitised (first line only, secrets redacted, paths replaced), stripped of
control characters and bounded. Unknown codes map to INTERNAL_ERROR (fail closed).
"""

from __future__ import annotations

import re

from app.mcp.schemas import MCPError, MCPErrorCategory
from app.security.errors import ErrorCategory, error_category, sanitize_detail

MESSAGES: dict[MCPErrorCategory, str] = {
    "INVALID_ARGUMENT": "The request arguments are invalid for this tool.",
    "UNAUTHORIZED_TOOL": "The requested operation was rejected by the data-access policy.",
    "UNSAFE_QUERY": "The requested operation was rejected by the data-access policy.",
    "RESOURCE_LIMIT": "The analysis exceeded the configured execution limit.",
    "TOOL_FAILURE": "The analysis could not be completed because the analytics tool failed.",
    "TIMEOUT": "The analysis exceeded the configured execution limit.",
    "UNSUPPORTED_REQUEST": "The requested analysis is not supported.",
    "VALIDATION_FAILURE": "The analysis could not be completed.",
    "INTERNAL_ERROR": "The analysis could not be completed.",
}
UNSUPPORTED_KPI_MESSAGE = "The requested KPI is not supported."

_FROM_PHASE5: dict[ErrorCategory, MCPErrorCategory] = {
    "rejected_by_policy": "UNSAFE_QUERY",
    "not_permitted": "UNAUTHORIZED_TOOL",
    "invalid_request": "INVALID_ARGUMENT",
    "data_unavailable": "TOOL_FAILURE",
    "resource_limit": "RESOURCE_LIMIT",
    "timeout": "TIMEOUT",
    "service_unavailable": "TOOL_FAILURE",
    "internal_error": "INTERNAL_ERROR",
}
# Codes that the Phase 5 categories group more coarsely than MCP does.
_REFINED: dict[str, MCPErrorCategory] = {
    "unsupported_kpi": "UNSUPPORTED_REQUEST",
    "unsupported_metric": "UNSUPPORTED_REQUEST",
    "unsupported_method": "UNSUPPORTED_REQUEST",
    "unsupported_request": "UNSUPPORTED_REQUEST",
    "request_too_large": "RESOURCE_LIMIT",
    "response_too_large": "RESOURCE_LIMIT",
    "calculation_error": "TOOL_FAILURE",
    "analytics_error": "TOOL_FAILURE",
    "invalid_tool_output": "VALIDATION_FAILURE",
    "evidence_integrity_failed": "VALIDATION_FAILURE",
    "output_validation_failed": "VALIDATION_FAILURE",
}
_WITH_DETAIL: frozenset[MCPErrorCategory] = frozenset({"INVALID_ARGUMENT", "UNSUPPORTED_REQUEST"})
_RETRYABLE: frozenset[str] = frozenset({"database_error"})
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
MAX_DETAIL_CHARS = 240


def mcp_category(code: str | None) -> MCPErrorCategory:
    if code in _REFINED:
        return _REFINED[code]
    return _FROM_PHASE5[error_category(code)]


def mcp_error(code: str | None, detail: str | None = None) -> MCPError:
    """The client-safe error for an internal code. ``detail`` is kept only for argument errors."""
    category = mcp_category(code)
    message = UNSUPPORTED_KPI_MESSAGE if code == "unsupported_kpi" else MESSAGES[category]
    safe_detail = None
    if detail and category in _WITH_DETAIL:
        safe_detail = _CONTROL.sub(" ", sanitize_detail(detail))[:MAX_DETAIL_CHARS] or None
    return MCPError(
        category=category,
        code=code or "internal_error",
        message=message,
        detail=safe_detail,
        retryable=(code or "") in _RETRYABLE,
    )
