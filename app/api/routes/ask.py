"""``POST /api/v1/ask`` and ``POST /api/v1/ask/stream``: one question, one agent run, one typed response.

Both routes validate the request, run the agent through ``AgentService`` (the only path to the
agent) and build the response with ``presenter``. The stream route sends newline-delimited JSON:
``progress`` events as agent stages finish, then exactly one ``result`` or ``error`` event.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api import API_PREFIX
from app.api.errors import APIError
from app.api.observability import RequestMetrics
from app.api.presenter import build_response, stage_label
from app.api.schemas.requests import AskRequest
from app.api.schemas.responses import AskResponse, ErrorEvent, ErrorResponse, FieldIssue, ProgressEvent, ResultEvent
from app.api.service import AgentService, ServiceRun

router = APIRouter(prefix=API_PREFIX, tags=["agent"])

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse, "description": description}
    for code, description in {
        400: "Malformed JSON.",
        413: "Request body too large.",
        422: "Invalid request (schema, empty question, question too long).",
        500: "Internal error.",
        503: "Agent unavailable or busy (Retry-After).",
        504: "The analysis exceeded the request time limit.",
    }.items()
}


def service_of(request: Request) -> AgentService:
    service: AgentService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise APIError("agent_unavailable")
    return service


def _prepare(body: AskRequest, request: Request, service: AgentService) -> str:
    """Settle the request ID (body > valid header > generated) and check the question before any run."""
    if body.request_id:
        request.state.request_id = body.request_id
    request.state.session_id = body.session_id
    if not body.question.strip():
        raise APIError("empty_question")
    limit = service.config.max_question_chars
    if len(body.question) > limit:
        issue = FieldIssue(location="body.question", message=f"The question must be at most {limit} characters.")
        raise APIError("question_too_long", issues=[issue])
    request_id: str = request.state.request_id
    return request_id


def _present(request: Request, run: ServiceRun, session_id: str | None, started: float) -> AskResponse:
    response = build_response(run.result, session_id=session_id, stages=run.stages)
    response.api_time_ms = round((time.perf_counter() - started) * 1000, 1)
    state = request.state
    state.agent_status, state.outcome = response.status, response.outcome
    state.tool_calls, state.agent_time_ms = response.run.tool_calls, response.run.agent_time_ms
    state.evidence_count, state.claim_count = len(response.evidence), len(response.claims)
    metrics: RequestMetrics = request.app.state.metrics
    metrics.record_run(response.outcome, response.run.agent_time_ms, response.api_time_ms)
    return response


def _internal(request: Request, exc: Exception) -> APIError:
    """An unexpected failure: the class name goes to the log, nothing about it goes to the client."""
    request.state.error_type = type(exc).__name__
    return APIError("internal_error")


@router.post(
    "/ask",
    response_model=AskResponse,
    responses=_ERROR_RESPONSES,
    summary="Ask a business question",
    description="Runs the evidence-backed agent. Refusals, unsupported questions and insufficient evidence are "
    "controlled responses (HTTP 200; see `status` and `outcome`).",
)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    started = time.perf_counter()
    service = service_of(request)
    request_id = _prepare(body, request, service)
    try:
        run = await service.run(body.question, request_id)
        return _present(request, run, body.session_id, started)
    except APIError:
        raise
    except Exception as exc:
        raise _internal(request, exc) from None


def _line(event: BaseModel) -> bytes:
    return (event.model_dump_json() + "\n").encode("utf-8")


@router.post(
    "/ask/stream",
    responses={200: {"content": {"application/x-ndjson": {}}}, **_ERROR_RESPONSES},
    summary="Ask a business question, with progress events",
    description="Newline-delimited JSON: `progress` events as agent stages finish, then one `result` event (the "
    "same body as /ask) or one `error` event. Request errors are returned before the stream starts.",
    response_class=StreamingResponse,
)
async def ask_stream(body: AskRequest, request: Request) -> StreamingResponse:
    started = time.perf_counter()
    service = service_of(request)
    request_id = _prepare(body, request, service)
    service.check_capacity()  # busy/unavailable are HTTP errors, not stream events
    request.state.streamed = True
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[ProgressEvent] = asyncio.Queue()

    def on_progress(node: str) -> None:  # runs on the agent's worker thread
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        event = ProgressEvent(request_id=request_id, stage=node, label=stage_label(node), elapsed_ms=elapsed)
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def events() -> AsyncIterator[bytes]:
        task = asyncio.ensure_future(service.run(body.question, request_id, on_progress=on_progress))
        try:
            while not task.done():
                getter = asyncio.ensure_future(queue.get())
                await asyncio.wait({task, getter}, return_when=asyncio.FIRST_COMPLETED)
                if getter.done():
                    yield _line(getter.result())
                else:
                    getter.cancel()
            while not queue.empty():
                yield _line(queue.get_nowait())
            try:
                response = _present(request, task.result(), body.session_id, started)
                yield _line(ResultEvent(request_id=request_id, data=response))
            except APIError as exc:
                request.state.error_code = exc.code
                yield _line(ErrorEvent(request_id=request_id, status_code=exc.status_code, error=exc.detail))
            except Exception as exc:
                error = _internal(request, exc)
                request.state.error_code = error.code
                yield _line(ErrorEvent(request_id=request_id, status_code=error.status_code, error=error.detail))
        finally:
            if not task.done():  # the client went away: stop waiting (a started run ends under its own limits)
                task.cancel()

    return StreamingResponse(events(), media_type="application/x-ndjson")
