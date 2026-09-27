"""``POST /api/v1/investigations`` and ``POST /api/v1/investigations/stream`` (Phase 10).

``/ask`` answers one question. ``/investigations`` runs a multi-step investigation of a business issue
(or a management brief) and returns a decision brief with its findings, drivers, recommendations,
uncertainty and evidence. Both run through ``AgentService`` on the same worker, with the same
authentication, rate limit, request limits, timeout and cancellation.

Investigations are synchronous and bounded by the request timeout, so there is no status endpoint to
poll: the stream route reports progress (stage, step title, tool name, status and duration) as it
happens, then exactly one ``result`` or ``error`` event.
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
from app.api.investigation_presenter import STAGE_LABELS, build_investigation_response
from app.api.observability import RequestMetrics
from app.api.routes.ask import service_of
from app.api.schemas.investigations import (
    InvestigationErrorEvent,
    InvestigationProgressEvent,
    InvestigationResponse,
    InvestigationResultEvent,
)
from app.api.schemas.requests import InvestigationRequest
from app.api.schemas.responses import ErrorResponse, FieldIssue
from app.api.service import AgentService
from app.investigation import Investigation

router = APIRouter(prefix=API_PREFIX, tags=["investigations"])

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse, "description": description}
    for code, description in {
        400: "Malformed JSON.",
        401: "Missing or invalid credentials.",
        413: "Request body too large.",
        415: "The body must be application/json.",
        422: "Invalid request (schema, empty objective, objective too long).",
        429: "Too many requests (Retry-After).",
        500: "Internal error.",
        503: "Agent unavailable, busy or shutting down (Retry-After).",
        504: "The investigation exceeded the request time limit.",
    }.items()
}
# The fields a progress event may carry from the investigation engine (anything else is dropped).
_PROGRESS_FIELDS = ("step_id", "title", "area", "tool_name", "status", "duration_ms", "steps")


def _prepare(body: InvestigationRequest, request: Request, service: AgentService) -> str:
    if body.request_id:
        request.state.request_id = body.request_id
    request.state.session_id = body.session_id
    if not body.objective.strip():
        raise APIError("empty_objective")
    limit = service.config.max_question_chars
    if len(body.objective) > limit:
        issue = FieldIssue(location="body.objective", message=f"The objective must be at most {limit} characters.")
        raise APIError("objective_too_long", issues=[issue])
    request_id: str = request.state.request_id
    return request_id


def _present(
    request: Request, investigation: Investigation, session_id: str | None, started: float
) -> InvestigationResponse:
    response = build_investigation_response(investigation, session_id=session_id)
    response.api_time_ms = round((time.perf_counter() - started) * 1000, 1)
    state = request.state
    state.agent_status, state.outcome = response.status, response.outcome
    state.tool_calls, state.agent_time_ms = response.run.tool_calls, response.run.timings.total_ms
    state.evidence_count, state.claim_count = len(response.evidence), len(response.claims)
    metrics: RequestMetrics = request.app.state.metrics
    metrics.record_run(response.outcome, response.run.timings.total_ms, response.api_time_ms)
    return response


def _internal(request: Request, exc: Exception) -> APIError:
    request.state.error_type = type(exc).__name__
    return APIError("internal_error")


@router.post(
    "/investigations",
    response_model=InvestigationResponse,
    responses=_ERROR_RESPONSES,
    summary="Investigate a business issue",
    description="Runs a bounded, multi-step investigation and returns an evidence-backed decision brief. "
    "Refusals, unsupported objectives, insufficient evidence and budget stops are controlled responses "
    "(HTTP 200; see `status` and `outcome`).",
)
async def investigate(body: InvestigationRequest, request: Request) -> InvestigationResponse:
    started = time.perf_counter()
    service = service_of(request)
    request_id = _prepare(body, request, service)
    try:
        investigation = await service.investigate(body.objective, request_id)
        return _present(request, investigation, body.session_id, started)
    except APIError:
        raise
    except Exception as exc:
        raise _internal(request, exc) from None


def _line(event: BaseModel) -> bytes:
    return (event.model_dump_json() + "\n").encode("utf-8")


@router.post(
    "/investigations/stream",
    responses={200: {"content": {"application/x-ndjson": {}}}, **_ERROR_RESPONSES},
    summary="Investigate a business issue, with progress events",
    description="Newline-delimited JSON: `progress` events (stage, step, tool, status, duration), then one "
    "`result` event (the same body as /investigations) or one `error` event. Request errors are returned before "
    "the stream starts.",
    response_class=StreamingResponse,
)
async def investigate_stream(body: InvestigationRequest, request: Request) -> StreamingResponse:
    started = time.perf_counter()
    service = service_of(request)
    request_id = _prepare(body, request, service)
    service.check_capacity()  # busy/unavailable are HTTP errors, not stream events
    request.state.streamed = True
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[InvestigationProgressEvent] = asyncio.Queue()

    def on_progress(event: dict[str, Any]) -> None:  # runs on the agent's worker thread
        stage = str(event.get("stage", ""))
        progress = InvestigationProgressEvent(
            request_id=request_id,
            stage=stage,
            label=str(event.get("label") or STAGE_LABELS.get(stage, "Working")),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            **{k: event[k] for k in _PROGRESS_FIELDS if k in event},
        )
        loop.call_soon_threadsafe(queue.put_nowait, progress)

    async def events() -> AsyncIterator[bytes]:
        task = asyncio.ensure_future(service.investigate(body.objective, request_id, on_progress=on_progress))
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
                yield _line(InvestigationResultEvent(request_id=request_id, data=response))
            except APIError as exc:
                request.state.error_code = exc.code
                yield _line(
                    InvestigationErrorEvent(request_id=request_id, status_code=exc.status_code, error=exc.detail)
                )
            except Exception as exc:
                error = _internal(request, exc)
                request.state.error_code = error.code
                yield _line(
                    InvestigationErrorEvent(request_id=request_id, status_code=error.status_code, error=error.detail)
                )
        finally:
            if not task.done():  # the client went away: the investigation is cancelled before its next step
                task.cancel()

    return StreamingResponse(events(), media_type="application/x-ndjson")
