"""Request context for every HTTP request (pure ASGI, so streamed responses pass straight through).

- **Request ID.** A valid ``X-Request-ID`` header (1 to 64 characters from ``A-Za-z0-9._:-``) is kept;
  anything else, including an oversized value, is replaced by a generated one. A route may replace
  it with the body's ``request_id``. The final ID is returned in the ``X-Request-ID`` response
  header and in every error body, and it becomes the agent run ID.
- **Body limit.** A declared or streamed body above ``API_MAX_REQUEST_BYTES`` is refused with 413.
- **Security headers** on every response:
  - ``X-Content-Type-Options: nosniff``;
  - ``Cache-Control: no-store`` (responses carry business data);
  - ``Referrer-Policy: no-referrer``;
  - ``X-Frame-Options: DENY``;
  - ``Content-Security-Policy: default-src 'none'; frame-ancestors 'none'``. The interactive docs
    page is the exception, because it loads its own scripts; it is off in production.
- **One structured log line** and the request counters once the response is complete. The log
  names the matched route template (``/api/v1/ask``), never the raw path.
"""

from __future__ import annotations

import time
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.agent.observability import is_valid_run_id, new_run_id
from app.api.observability import RequestMetrics, log_request
from app.api.security import RATE_LIMITED_PATHS

DOCS_PATHS = frozenset({"/docs", "/docs/oauth2-redirect"})
STRICT_CSP = "default-src 'none'; frame-ancestors 'none'"
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}


class BodyTooLarge(HTTPException):
    """Raised while the body is read; FastAPI re-raises HTTP exceptions, so the 413 handler renders it."""

    def __init__(self) -> None:
        super().__init__(status_code=413)


def _endpoint(scope: Scope) -> str:
    """The matched route template, or a fixed label (attacker-chosen paths are never logged)."""
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else "unmatched"


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, *, max_body_bytes: int, metrics: RequestMetrics):
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        headers = Headers(scope=scope)
        supplied = headers.get("x-request-id")
        state: dict[str, Any] = scope.setdefault("state", {})
        state["request_id"] = supplied if supplied and is_valid_run_id(supplied) else new_run_id()
        status = {"code": 500}
        path = str(scope.get("path", ""))
        self.metrics.started()

        declared = headers.get("content-length")
        too_large = declared is not None and (not declared.isdigit() or int(declared) > self.max_body_bytes)
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            if too_large:
                raise BodyTooLarge()
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise BodyTooLarge()
            return message

        async def send_with_context(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                response_headers = MutableHeaders(scope=message)
                response_headers["X-Request-ID"] = state["request_id"]
                for name, value in SECURITY_HEADERS.items():
                    response_headers[name] = value
                if path not in DOCS_PATHS:
                    response_headers["Content-Security-Policy"] = STRICT_CSP
            await send(message)

        try:
            await self.app(scope, limited_receive, send_with_context)
        finally:
            duration = round((time.perf_counter() - started) * 1000, 1)
            error_code = state.get("error_code")
            self.metrics.finished(
                status["code"], error_code=error_code, duration_ms=duration, ask=path in RATE_LIMITED_PATHS
            )
            log_request(
                state["request_id"],
                "http_request",
                method=scope.get("method"),
                endpoint=_endpoint(scope),
                status_code=status["code"],
                duration_ms=duration,
                session_id=state.get("session_id"),
                agent_status=state.get("agent_status"),
                outcome=state.get("outcome"),
                error_code=error_code,
                error_type=state.get("error_type"),
                tool_calls=state.get("tool_calls"),
                evidence_count=state.get("evidence_count"),
                claim_count=state.get("claim_count"),
                agent_time_ms=state.get("agent_time_ms"),
                streamed=state.get("streamed"),
            )
