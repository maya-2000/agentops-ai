"""The API's access boundary: bearer-token authentication, rate limiting and content-type checks.

``AccessControlMiddleware`` runs before routing and before the request body is read, so an
unauthenticated, throttled or wrongly typed request never reaches validation or the agent.

- **Authentication.** ``Authorization: Bearer <token>`` is compared in constant time with
  ``API_AUTH_TOKEN``. Every ``/api/v1`` path requires it except liveness (``/health``) and
  readiness (``/readiness``). Unknown ``/api/v1`` paths also answer 401 first, so routes cannot be
  enumerated without a token. Missing, malformed and wrong credentials all get the same 401. The
  token is never logged, echoed or stored beyond the configuration; it is registered with the
  redaction utility so that it is removed from anything logged.
- **Rate limiting.** ``POST /api/v1/ask`` and ``/ask/stream`` are limited per client (the peer
  address) with a sliding window: at most N requests in any window. Excess requests get 429 with
  ``Retry-After``. The state is in memory, bounded by ``API_RATE_LIMIT_MAX_CLIENTS`` (least-recently
  seen clients are forgotten first), and per process.
- **Content type.** The ask endpoints accept ``application/json`` only (415 otherwise).
"""

from __future__ import annotations

import hmac
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.api import API_PREFIX
from app.api.config import APIConfig, RateLimit
from app.api.errors import APIError
from app.security.redaction import register_secret

PUBLIC_PATHS = frozenset({f"{API_PREFIX}/health", f"{API_PREFIX}/readiness"})
RATE_LIMITED_PATHS = frozenset({f"{API_PREFIX}/ask", f"{API_PREFIX}/ask/stream"})
JSON_TYPES = frozenset({"application/json"})


def bearer_token(header: str | None) -> str | None:
    """The token of a well-formed ``Bearer <token>`` header, else ``None``."""
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token or " " in token.strip() or token != token.strip():
        return None
    return token


class TokenAuthenticator:
    def __init__(self, token: str):
        self._token = token.encode("utf-8")
        register_secret(token)

    def verify(self, header: str | None) -> bool:
        supplied = bearer_token(header)
        # compare_digest runs in time independent of where the values differ.
        return supplied is not None and hmac.compare_digest(supplied.encode("utf-8"), self._token)


@dataclass(frozen=True)
class RateDecision:
    allowed: bool
    remaining: int
    retry_after: int  # whole seconds until a request would be allowed (0 when allowed)


class SlidingWindowRateLimiter:
    """At most ``requests`` per ``window_seconds`` per key; memory bounded by ``max_clients`` keys."""

    def __init__(self, limit: RateLimit, *, clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self._clock = clock
        self._lock = threading.Lock()
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()

    def check(self, key: str) -> RateDecision:
        now = self._clock()
        window = self.limit.window_seconds
        with self._lock:
            hits = self._hits.pop(key, None) or deque()
            while hits and hits[0] <= now - window:
                hits.popleft()
            if len(hits) >= self.limit.requests:
                retry = max(1, int(hits[0] + window - now + 0.999))
                self._remember(key, hits)
                return RateDecision(allowed=False, remaining=0, retry_after=retry)
            hits.append(now)
            self._remember(key, hits)
            return RateDecision(allowed=True, remaining=self.limit.requests - len(hits), retry_after=0)

    def _remember(self, key: str, hits: deque[float]) -> None:
        self._hits[key] = hits  # most recently seen last
        while len(self._hits) > self.limit.max_clients:
            self._hits.popitem(last=False)

    @property
    def tracked_clients(self) -> int:
        with self._lock:
            return len(self._hits)


def client_key(scope: Scope) -> str:
    client = scope.get("client")
    return str(client[0]) if client else "unknown"


class AccessControlMiddleware:
    def __init__(self, app: ASGIApp, *, config: APIConfig, limiter: SlidingWindowRateLimiter | None):
        self.app = app
        self.config = config
        self.limiter = limiter
        token = config.auth_token.get_secret_value() if config.auth_token else None
        self.authenticator = TokenAuthenticator(token) if config.auth_enabled and token else None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path: str = scope.get("path", "")
        method: str = scope.get("method", "GET")
        state: dict[str, Any] = scope.setdefault("state", {})
        protected = path.startswith(API_PREFIX) and path not in PUBLIC_PATHS
        if protected and self.config.auth_enabled:
            header = Headers(scope=scope).get("authorization")
            if self.authenticator is None or not self.authenticator.verify(header):
                await _reject(scope, send, APIError("unauthorized"))
                return
            state["authenticated"] = True
        if method == "POST" and path in RATE_LIMITED_PATHS:
            content_type = Headers(scope=scope).get("content-type", "")
            if content_type.split(";")[0].strip().lower() not in JSON_TYPES:
                await _reject(scope, send, APIError("unsupported_media_type"))
                return
            if self.limiter is not None:
                decision = self.limiter.check(client_key(scope))
                if not decision.allowed:
                    await _reject(scope, send, APIError("rate_limited", retry_after=decision.retry_after))
                    return
        await self.app(scope, receive, send)


async def _reject(scope: Scope, send: Send, error: APIError) -> None:
    """Answer with the API's error envelope without reading the request body."""
    state: dict[str, Any] = scope.setdefault("state", {})
    state["error_code"] = error.code
    request_id = str(state.get("request_id", ""))
    response = JSONResponse(
        error.response(request_id).model_dump(mode="json"), status_code=error.status_code, headers=error.headers
    )

    async def no_body() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    await response(scope, no_body, send)
