"""API errors: a fixed code, HTTP status and client-safe message for every failure of the API itself.

Agent outcomes are not API errors. A refusal, an unsupported or ambiguous question, insufficient
evidence, a failed tool or an answer that failed validation is a controlled agent response: HTTP
200 with ``status``/``outcome`` in the body. API errors are failures to produce such a response:
a malformed or invalid request, an oversized body, a busy or unavailable agent, a timeout, or an
internal error. Their messages are fixed strings: never exception text, SQL, paths, environment
values or secrets.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from app.api.schemas.responses import APIErrorDetail, ErrorResponse, FieldIssue

ErrorCode = Literal[
    "malformed_request",
    "invalid_request",
    "empty_question",
    "question_too_long",
    "request_too_large",
    "not_found",
    "method_not_allowed",
    "busy",
    "agent_unavailable",
    "timeout",
    "internal_error",
]

# code -> (HTTP status, fixed message, retryable)
_ERRORS: dict[str, tuple[int, str, bool]] = {
    "malformed_request": (400, "The request body is not valid JSON.", False),
    "invalid_request": (422, "The request does not match the expected schema.", False),
    "empty_question": (422, "The question is empty. Ask a question about the business data.", False),
    "question_too_long": (422, "The question is longer than the allowed maximum.", False),
    "request_too_large": (413, "The request body is too large.", False),
    "not_found": (404, "The requested resource does not exist.", False),
    "method_not_allowed": (405, "The HTTP method is not allowed for this resource.", False),
    "busy": (503, "The agent is busy with other requests. Please retry shortly.", True),
    "agent_unavailable": (503, "The analysis service is not available.", True),
    "timeout": (504, "The analysis did not finish within the request time limit.", True),
    "internal_error": (500, "An internal error prevented the request from completing.", False),
}
RETRY_AFTER_SECONDS = 5


class APIError(Exception):
    """Raised by the service and routes; rendered by the application's exception handler."""

    def __init__(self, code: ErrorCode, *, issues: Sequence[FieldIssue] = ()):
        super().__init__(code)
        self.code: ErrorCode = code
        self.issues = list(issues)

    @property
    def status_code(self) -> int:
        return _ERRORS[self.code][0]

    @property
    def detail(self) -> APIErrorDetail:
        _, message, retryable = _ERRORS[self.code]
        return APIErrorDetail(code=self.code, message=message, retryable=retryable, issues=self.issues)

    def response(self, request_id: str) -> ErrorResponse:
        return ErrorResponse(request_id=request_id, error=self.detail)

    @property
    def headers(self) -> dict[str, str]:
        return {"Retry-After": str(RETRY_AFTER_SECONDS)} if self.status_code == 503 else {}


def error_status(code: str) -> int:
    return _ERRORS[code][0]


def error_codes() -> list[str]:
    return list(_ERRORS)
