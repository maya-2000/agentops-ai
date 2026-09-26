"""The FastAPI application: ``create_app`` and the module-level ``app`` for ``uvicorn app.api.main:app``.

Error handling: every failure becomes an ``ErrorResponse`` (request ID, fixed code and message,
optional field issues). Validation messages never echo the submitted value; unexpected exceptions
are reduced to ``internal_error``. No stack trace, SQL, path or setting ever reaches a client.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.api import API_PREFIX, API_VERSION
from app.api.config import APIConfig
from app.api.errors import APIError, ErrorCode
from app.api.middleware import RequestContextMiddleware
from app.api.observability import RequestMetrics
from app.api.routes import ask_router, system_router
from app.api.schemas.responses import FieldIssue
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
    "Local development service: no authentication."
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
    opens the configured database, builds the agent and closes both on shutdown."""
    config = config or (service.config if service is not None else APIConfig.from_settings())
    metrics = RequestMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = service is None
        app.state.service = service if service is not None else AgentService.from_settings(config)
        try:
            yield
        finally:
            if owned:
                app.state.service.close()

    app = FastAPI(
        title="AgentOps AI API",
        version=API_VERSION,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.state.metrics = metrics
    if service is not None:
        app.state.service = service
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
        return {"name": "AgentOps AI API", "version": API_VERSION, "docs": "/docs", "api": API_PREFIX}

    app.include_router(ask_router)
    app.include_router(system_router)
    return app


app = create_app()
