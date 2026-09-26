"""Request context for every HTTP request (pure ASGI, so streamed responses pass straight through).

- Request ID: a valid ``X-Request-ID`` header is kept, anything else is replaced by a generated one.
  A route may replace it with the body's ``request_id``. The final ID is returned in the
  ``X-Request-ID`` response header, in every error body, and becomes the agent run ID.
- Body limit: a declared or streamed body above ``API_MAX_REQUEST_BYTES`` is refused with 413.
- Response headers: ``X-Content-Type-Options: nosniff`` and ``Cache-Control: no-store`` (responses
  carry business data).
- One structured log line and the request counters when the response is complete.
"""

from __future__ import annotations

import time
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.agent.observability import is_valid_run_id, new_run_id
from app.api.observability import RequestMetrics, log_request

MAX_LOGGED_PATH_CHARS = 200


class BodyTooLarge(HTTPException):
    """Raised while the body is read; FastAPI re-raises HTTP exceptions, so the 413 handler renders it."""

    def __init__(self) -> None:
        super().__init__(status_code=413)


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
                response_headers["X-Content-Type-Options"] = "nosniff"
                response_headers["Cache-Control"] = "no-store"
            await send(message)

        try:
            await self.app(scope, limited_receive, send_with_context)
        finally:
            self.metrics.finished(status["code"])
            log_request(
                state["request_id"],
                "http_request",
                method=scope.get("method"),
                path=str(scope.get("path", ""))[:MAX_LOGGED_PATH_CHARS],
                status_code=status["code"],
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
                session_id=state.get("session_id"),
                agent_status=state.get("agent_status"),
                outcome=state.get("outcome"),
                error_code=state.get("error_code"),
                error_type=state.get("error_type"),
                tool_calls=state.get("tool_calls"),
                evidence_count=state.get("evidence_count"),
                claim_count=state.get("claim_count"),
                agent_time_ms=state.get("agent_time_ms"),
                streamed=state.get("streamed"),
            )
