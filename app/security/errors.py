"""Error sanitisation: users see a safe category, and details stay in the (redacted) trace.

Raw exception text can reveal file paths, SQL, table internals, driver versions or secrets.
``safe_error`` maps every internal error code to a short, fixed, user-facing message.
``sanitize_detail`` makes an internal detail message safe to keep in the developer trace: secrets
redacted, paths replaced, tracebacks and multi-line driver output cut to the first line, and
the length bounded. Unknown codes map to the generic internal-error message (fail closed).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.security.redaction import redact, redact_paths

ErrorCategory = Literal[
    "rejected_by_policy",
    "not_permitted",
    "invalid_request",
    "data_unavailable",
    "resource_limit",
    "timeout",
    "service_unavailable",
    "internal_error",
]

_CATEGORY_MESSAGES: dict[ErrorCategory, str] = {
    "rejected_by_policy": "The requested query was rejected by the data-access policy.",
    "not_permitted": "The requested operation is not permitted for this request.",
    "invalid_request": "The analysis request had invalid or unsupported parameters.",
    "data_unavailable": "The data needed for this analysis is not available.",
    "resource_limit": "The analysis reached its resource limit.",
    "timeout": "The analysis exceeded its time limit.",
    "service_unavailable": "A required service was temporarily unavailable.",
    "internal_error": "An internal error prevented the analysis from completing.",
}

_CODE_CATEGORIES: dict[str, ErrorCategory] = {
    "unsafe_sql": "rejected_by_policy",
    "sql_rejected": "rejected_by_policy",
    "data_policy": "rejected_by_policy",
    "unknown_tool": "not_permitted",
    "unauthorized_tool": "not_permitted",
    "tool_disabled": "not_permitted",
    "tool_not_permitted": "not_permitted",
    "sql_not_permitted": "not_permitted",
    "prerequisite_missing": "not_permitted",
    "invalid_arguments": "invalid_request",
    "invalid_request": "invalid_request",
    "unsupported_kpi": "invalid_request",
    "unsupported_dimension": "invalid_request",
    "invalid_filter_value": "invalid_request",
    "invalid_period": "invalid_request",
    "invalid_date_range": "invalid_request",
    "unsupported_metric": "invalid_request",
    "invalid_horizon": "invalid_request",
    "unsupported_method": "invalid_request",
    "oversized_input": "invalid_request",
    "no_data": "data_unavailable",
    "insufficient_data": "data_unavailable",
    "insufficient_history": "data_unavailable",
    "budget_exceeded": "resource_limit",
    "timeout": "timeout",
    "database_error": "service_unavailable",
    "provider_error": "service_unavailable",
    "calculation_error": "internal_error",
    "invalid_tool_output": "internal_error",
    "internal_error": "internal_error",
}

MAX_DETAIL_CHARS = 300


class SafeError(BaseModel):
    code: str
    category: ErrorCategory
    message: str


def error_category(code: str | None) -> ErrorCategory:
    return _CODE_CATEGORIES.get(code or "", "internal_error")


def safe_error(code: str | None) -> SafeError:
    """The user-facing form of an error code: never includes exception text."""
    category = error_category(code)
    return SafeError(code=code or "internal_error", category=category, message=_CATEGORY_MESSAGES[category])


def safe_message(code: str | None) -> str:
    return safe_error(code).message


def sanitize_detail(message: str | None) -> str:
    """Make an internal error detail safe for the developer trace (never for the user response)."""
    if not message:
        return ""
    first_line = message.strip().splitlines()[0] if message.strip() else ""
    if first_line.startswith("Traceback"):
        first_line = "internal exception (traceback withheld)"
    return redact_paths(redact(first_line))[:MAX_DETAIL_CHARS]
