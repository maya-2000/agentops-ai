"""The UI's HTTP client for the AgentOps API: the UI's only way to reach the agent.

The UI never opens the database, never imports the agent, tools or analytics, and never reads data
files. It sends the question to the API and renders the JSON it gets back. Failures become an
``APIFailure`` with a short, user-facing message; response bodies of failed requests are only
read for the API's own error envelope (code, message, request ID).

Authentication (Phase 9): the client sends ``Authorization: Bearer <API_AUTH_TOKEN>`` when a token
is configured. The token lives only in this object's private attribute: it is not shown on the
page, not stored in session state, and not part of ``repr``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

API_PREFIX = "/api/v1"


class APIFailure(Exception):
    """A request that produced no usable API response."""

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        request_id: str | None = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.kind = kind  # "connection", "timeout", "http" or "protocol"
        self.message = message
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        self.retryable = retryable


def _failure_from(body: Any, status_code: int, request_id: str | None) -> APIFailure:
    """The API's error envelope as an ``APIFailure`` (a generic message when the body is not one)."""
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        supplied = body.get("request_id") if isinstance(body, dict) else None
        return APIFailure(
            "http",
            error["message"],
            status_code=status_code,
            code=str(error.get("code") or "error"),
            request_id=supplied if isinstance(supplied, str) else request_id,
            retryable=bool(error.get("retryable")),
        )
    message = f"The API answered with HTTP {status_code}."
    return APIFailure("http", message, status_code=status_code, request_id=request_id)


class AgentOpsClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float,
        token: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._transport = transport  # tests pass an httpx.MockTransport

    def __repr__(self) -> str:
        return f"AgentOpsClient(base_url={self.base_url!r}, authenticated={bool(self._headers)})"

    def _client(self, timeout: float | None = None) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url, timeout=timeout or self.timeout, transport=self._transport, headers=self._headers
        )

    def _request(self, method: str, path: str, *, timeout: float | None = None, **kwargs: Any) -> dict[str, Any]:
        try:
            with self._client(timeout) as client:
                response = client.request(method, API_PREFIX + path, **kwargs)
        except httpx.TimeoutException:
            raise APIFailure("timeout", "The API did not answer in time.", retryable=True) from None
        except httpx.HTTPError:
            raise APIFailure(
                "connection", f"The AgentOps API is not reachable at {self.base_url}.", retryable=True
            ) from None
        try:
            body = response.json()
        except ValueError:
            body = None
        # Readiness answers 503 with a regular body when not ready: that is a status, not a failure.
        if response.status_code >= 400 and not (path == "/readiness" and isinstance(body, dict) and "status" in body):
            raise _failure_from(body, response.status_code, response.headers.get("x-request-id"))
        if not isinstance(body, dict):
            raise APIFailure("protocol", "The API returned an unexpected response.", status_code=response.status_code)
        return body

    # ------------------------------------------------------------------ endpoints
    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health", timeout=5.0)

    def readiness(self) -> dict[str, Any]:
        return self._request("GET", "/readiness", timeout=5.0)

    def capabilities(self) -> dict[str, Any]:
        return self._request("GET", "/capabilities", timeout=10.0)

    def ask(self, question: str, *, session_id: str | None = None) -> dict[str, Any]:
        return self._request("POST", "/ask", json={"question": question, "session_id": session_id})

    def ask_stream(
        self, question: str, *, session_id: str | None = None, on_progress: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any]:
        """Like ``ask``, calling ``on_progress`` with each progress event before the result arrives."""
        payload = {"question": question, "session_id": session_id}
        try:
            with self._client() as client, client.stream("POST", API_PREFIX + "/ask/stream", json=payload) as response:
                if response.status_code >= 400:
                    response.read()
                    try:
                        body = response.json()
                    except ValueError:
                        body = None
                    raise _failure_from(body, response.status_code, response.headers.get("x-request-id"))
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        raise APIFailure("protocol", "The API returned an unexpected response.") from None
                    kind = event.get("type") if isinstance(event, dict) else None
                    if kind == "progress":
                        on_progress(event)
                    elif kind == "result" and isinstance(event.get("data"), dict):
                        result: dict[str, Any] = event["data"]
                        return result
                    elif kind == "error":
                        status = event.get("status_code")
                        request_id = response.headers.get("x-request-id")
                        raise _failure_from(event, status if isinstance(status, int) else 500, request_id)
        except httpx.TimeoutException:
            raise APIFailure("timeout", "The API did not answer in time.", retryable=True) from None
        except httpx.HTTPError:
            raise APIFailure(
                "connection", f"The AgentOps API is not reachable at {self.base_url}.", retryable=True
            ) from None
        raise APIFailure("protocol", "The API closed the stream without a result.")
