"""``GET /api/v1/health``, ``/api/v1/capabilities`` and ``/api/v1/metrics``.

Health reports readiness from the start-up snapshot plus one metadata query on the shared
connection (skipped while an agent run holds it); it never runs a business query and never reports
paths, settings, versions of dependencies or error text. Capabilities list what the agent can
analyse, from the registries (names and descriptions only: no SQL, schemas or policy internals).
"""

from __future__ import annotations

from typing import Literal, get_args

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


@router.get("/health", response_model=HealthResponse, summary="Service readiness")
async def health(request: Request) -> JSONResponse:
    service: AgentService | None = getattr(request.app.state, "service", None)
    if service is None:
        body = HealthResponse(
            status="unavailable", version=API_VERSION, agent_available=False, database_available=False
        )
        return JSONResponse(body.model_dump(mode="json"), status_code=503)
    snapshot = service.snapshot
    database = service.probe_database()
    agent = service.available
    status: Literal["ok", "degraded", "unavailable"] = (
        "ok" if agent and database else "unavailable" if not agent else "degraded"
    )
    body = HealthResponse(
        status=status,
        version=API_VERSION,
        agent_available=agent,
        database_available=database,
        dataset_version=snapshot.dataset_version,
        as_of_date=snapshot.as_of_date,
        llm_provider=snapshot.llm_provider,
    )
    return JSONResponse(body.model_dump(mode="json"), status_code=503 if status == "unavailable" else 200)


@router.get("/capabilities", response_model=CapabilitiesResponse, summary="What the agent can analyse")
async def capabilities(request: Request) -> CapabilitiesResponse:
    service = service_of(request)
    snapshot = service.snapshot
    if snapshot.as_of_date is None:
        raise APIError("agent_unavailable")
    series = [NamedItem(key=m.key, name=m.name, unit=m.unit) for m in SERIES_METRICS.values()]
    return CapabilitiesResponse(
        version=API_VERSION,
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
    counters: RequestMetrics = request.app.state.metrics
    return counters.snapshot()
