"""``GET /api/v1/health``, ``/readiness``, ``/capabilities`` and ``/metrics``.

- ``/health`` (public, liveness): the process is up and serving HTTP. It checks nothing else, so a
  restart is never triggered by a dependency problem.
- ``/readiness`` (public): whether requests can be served now. The checks are the configuration
  (the API does not start without a safe one), the database (a metadata query on the shared
  connection, skipped while a run holds it), the agent, and whether the service is accepting
  requests (not shutting down). 200 when ready, 503 when not. Only named booleans are returned:
  never paths, settings, versions of dependencies or error text.
- ``/capabilities`` (authenticated): what the agent can analyse, from the registries (names and
  descriptions only: no SQL, schemas or policy internals), plus the dataset version and as-of date.
- ``/metrics`` (authenticated): request, outcome, error and run counters, and latency percentiles.
"""

from __future__ import annotations

from typing import get_args

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.analytics.dimensions import DIMENSIONS
from app.analytics.kpis import KPI_REGISTRY
from app.anomalies.config import DetectorName
from app.api import API_PREFIX, API_VERSION
from app.api.errors import APIError
from app.api.observability import RequestMetrics
from app.api.routes.ask import service_of
from app.api.schemas.responses import (
    AnalysisCapability,
    CapabilitiesResponse,
    HealthResponse,
    Limits,
    MetricsResponse,
    NamedItem,
    Outcome,
    ReadinessResponse,
)
from app.api.service import AgentService
from app.timeseries import SERIES_METRICS
from app.tools.registry import TOOL_DEFINITIONS

router = APIRouter(prefix=API_PREFIX, tags=["service"])

EXAMPLE_QUESTIONS = (
    "What was revenue in July compared with June?",
    "Which region had the largest revenue decline?",
    "Why did support tickets increase?",
    "What is our 3-month revenue forecast?",
    "Which acquisition channel has the highest CAC?",
    "Are there any unusual customer or product trends?",
    "Which segment had the highest churn last month?",
    "Which customers are at risk?",
)
NOT_SUPPORTED = (
    "Questions outside the Northwind Cloud business dataset (news, weather, general knowledge).",
    "Changing data, running code, reading files, or revealing credentials, prompts or hidden evaluation data.",
    "Proven causes: analyses show associations and contributions, not causation.",
    "Customer names and other withheld personal fields.",
)


@router.get("/health", response_model=HealthResponse, summary="Liveness (public)")
async def health() -> HealthResponse:
    return HealthResponse(version=API_VERSION)


@router.get(
    "/readiness",
    response_model=ReadinessResponse,
    summary="Readiness (public)",
    responses={503: {"model": ReadinessResponse, "description": "Not ready: a named check failed."}},
)
async def readiness(request: Request) -> JSONResponse:
    service: AgentService | None = getattr(request.app.state, "service", None)
    checks = (
        service.readiness()
        if service is not None
        else {"configuration": True, "database": False, "agent": False, "accepting_requests": False}
    )
    ready = all(checks.values())
    body = ReadinessResponse(status="ready" if ready else "not_ready", version=API_VERSION, checks=checks)
    return JSONResponse(body.model_dump(mode="json"), status_code=200 if ready else 503)


@router.get("/capabilities", response_model=CapabilitiesResponse, summary="What the agent can analyse")
async def capabilities(request: Request) -> CapabilitiesResponse:
    service = service_of(request)
    snapshot = service.snapshot
    if snapshot.as_of_date is None:
        raise APIError("agent_unavailable")
    series = [NamedItem(key=m.key, name=m.name, unit=m.unit) for m in SERIES_METRICS.values()]
    return CapabilitiesResponse(
        version=API_VERSION,
        dataset_version=snapshot.dataset_version,
        llm_provider=snapshot.llm_provider,
        as_of_date=snapshot.as_of_date,
        data_start=snapshot.data_start,
        data_end=snapshot.data_end,
        kpis=[NamedItem(key=k.key, name=k.name, unit=k.unit, description=k.definition) for k in KPI_REGISTRY.values()],
        forecast_metrics=series,
        anomaly_metrics=series,
        anomaly_detectors=list(get_args(DetectorName)),
        dimensions=[NamedItem(key=d.key, name=d.display_name) for d in DIMENSIONS.values()],
        analyses=[
            AnalysisCapability(tool=t.name, when_to_use=t.when_to_use, not_for=t.not_for)
            for t in TOOL_DEFINITIONS
            if t.name not in snapshot.disabled_tools
        ],
        outcomes=list(get_args(Outcome)),
        limits=Limits(
            max_question_chars=service.config.max_question_chars,
            request_timeout_seconds=service.config.request_timeout_seconds,
            max_tool_calls=snapshot.max_tool_calls,
        ),
        example_questions=list(EXAMPLE_QUESTIONS),
        not_supported=list(NOT_SUPPORTED),
    )


@router.get("/metrics", response_model=MetricsResponse, summary="Request counters since start-up")
async def metrics(request: Request) -> MetricsResponse:
    config = getattr(request.app.state, "config", None)
    if config is not None and not config.metrics_enabled:
        raise APIError("not_found")
    counters: RequestMetrics = request.app.state.metrics
    service: AgentService | None = getattr(request.app.state, "service", None)
    return counters.snapshot(service.runs.snapshot() if service is not None else None)
