"""The FastAPI application: ``create_app`` and the module-level ``app`` for ``uvicorn app.api.main:app``.

Middleware, outermost first:

1. ``RequestContextMiddleware``: request ID, body limit, security headers, request log and metrics.
2. ``CORSMiddleware``: only when ``API_CORS_ORIGINS`` names origins. There is no wildcard in
   production.
3. ``AccessControlMiddleware``: bearer-token authentication, rate limiting and the content type of
   the ask endpoints, all before the body is read.

Start-up refuses an unsafe configuration (``ConfigurationError``). With ``APP_ENV=production`` it
also refuses to serve without its database and agent, so a missing data volume fails the process
loudly instead of leaving it up and unready.

Error handling: every failure becomes an ``ErrorResponse`` (request ID, fixed code and message,
optional field issues). Validation messages never echo the submitted value; unexpected exceptions
are reduced to ``internal_error``. No stack trace, SQL, path, class name or setting ever reaches a
client.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.api import API_PREFIX, API_VERSION
from app.api.config import APIConfig
from app.api.errors import APIError, ErrorCode
from app.api.middleware import RequestContextMiddleware
from app.api.observability import RequestMetrics, log_service
from app.api.routes import ask_router, system_router
from app.api.schemas.responses import FieldIssue
from app.api.security import AccessControlMiddleware, SlidingWindowRateLimiter
from app.api.service import AgentService

MAX_ISSUES = 10
MAX_ISSUE_CHARS = 200
_HTTP_CODES: dict[int, ErrorCode] = {
    400: "malformed_request",
    404: "not_found",
    405: "method_not_allowed",
    413: "request_too_large",
}

DESCRIPTION = (
    "Evidence-backed business intelligence over the Northwind Cloud dataset. Every answer is produced by the "
    "AgentOps agent through its secured, read-only tools; numbers come with typed claims, evidence and provenance. "
    "All /api/v1 endpoints except /health and /readiness need `Authorization: Bearer <token>`."
)


def _error_response(request: Request, error: APIError) -> JSONResponse:
    request.state.error_code = error.code
    body = error.response(request.state.request_id).model_dump(mode="json")
    return JSONResponse(body, status_code=error.status_code, headers=error.headers)


def _issues(errors: list[Any]) -> list[FieldIssue]:
    """Location and message of each validation error; never the submitted input."""
    issues = []
    for error in errors[:MAX_ISSUES]:
        location = ".".join(str(part) for part in error.get("loc", ()))
        message = str(error.get("msg", ""))
        issues.append(FieldIssue(location=location[:MAX_ISSUE_CHARS], message=message[:MAX_ISSUE_CHARS]))
    return issues


def _validation_error(errors: list[Any]) -> APIError:
    if any(e.get("type") == "json_invalid" for e in errors):
        return APIError("malformed_request")
    if any(tuple(e.get("loc", ())) == ("body", "question") and e.get("type") == "string_too_short" for e in errors):
        return APIError("empty_question")
    return APIError("invalid_request", issues=_issues(errors))


def create_app(service: AgentService | None = None, *, config: APIConfig | None = None) -> FastAPI:
    """Build the application. A given ``service`` is used as is (and not closed); otherwise the lifespan
    checks the configuration, opens the configured database, builds the agent and closes both on
    shutdown (graceful: live runs finish or are cancelled within ``API_SHUTDOWN_GRACE_SECONDS``)."""
    config = config or (service.config if service is not None else APIConfig.from_settings())
    metrics = RequestMetrics()
    limiter = SlidingWindowRateLimiter(config.rate_limit) if config.rate_limit else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        problems = config.startup_problems()
        if problems:
            log_service("configuration_rejected", problems=problems, environment=config.environment)
            config.check()  # raises ConfigurationError: the process does not serve
        owned = service is None
        app.state.service = service if service is not None else AgentService.from_settings(config)
        if config.environment == "production" and not app.state.service.available:
            app.state.service.close(0)
            raise RuntimeError("The database or agent is not available; see the service_unavailable log event.")
        limit = config.rate_limit
        log_service(
            "service_started",
            version=API_VERSION,
            environment=config.environment,
            auth=config.auth_mode,
            rate_limit=f"{limit.requests}/{limit.window_seconds:g}s" if limit else "off",
        )
        try:
            yield
        finally:
            if owned:
                app.state.service.close()
            log_service("service_stopped", version=API_VERSION)

    app = FastAPI(
        title="AgentOps AI API",
        version=API_VERSION,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs" if config.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if config.docs_enabled else None,
    )
    app.state.metrics = metrics
    app.state.config = config
    app.state.rate_limiter = limiter
    if service is not None:
        app.state.service = service
    # add_middleware wraps: the last one added runs first.
    app.add_middleware(AccessControlMiddleware, config=config, limiter=limiter)
    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.cors_origins),
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID", "Retry-After"],
            allow_credentials=False,
            max_age=600,
        )
    app.add_middleware(RequestContextMiddleware, max_body_bytes=config.max_request_bytes, metrics=metrics)

    @app.exception_handler(APIError)
    async def api_error(request: Request, exc: APIError) -> JSONResponse:
        return _error_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(request, _validation_error(list(exc.errors())))

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return _error_response(request, APIError(_HTTP_CODES.get(exc.status_code, "invalid_request")))

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        request.state.error_type = type(exc).__name__
        return _error_response(request, APIError("internal_error"))

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"name": "AgentOps AI API", "version": API_VERSION, "api": API_PREFIX}

    app.include_router(ask_router)
    app.include_router(system_router)
    return app


app = create_app()
