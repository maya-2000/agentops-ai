"""Run investigation scenarios (eval_v2) through the production paths and record what happened.

- ``investigation`` / ``cancellation``: ``Investigator`` on the agent's own runtime, built exactly as the
  API builds it (``AgentRunner(...).runtime``). Progress events are recorded; a cancellation scenario
  sets the cancel event after some finished steps, or passes a deadline.
- ``api``: the real FastAPI application (``create_app``) around an ``AgentService``, called in process
  through ``/api/v1/investigations`` and, when asked, ``/investigations/stream`` and ``/ask``.
- ``ui``: the API response passed through the Streamlit page's view models (what the page renders).
- ``mcp_parity``: after the investigation, every successful step whose tool the MCP server exposes is
  called again through the real MCP server (in-process SDK client) with the same arguments.

Production receives only the read-only database and the as-of date. Nothing here reimplements a tool,
a policy or a calculation; production latency is timed apart from grading.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import anyio
from fastapi.testclient import TestClient
from mcp import Client

from app.agent import AgentConfig, AgentRunner
from app.api.config import APIConfig
from app.api.main import create_app
from app.api.service import AgentService
from app.investigation import Investigation, Investigator
from app.mcp.config import MCPServerConfig
from app.mcp.registry import MCP_TOOL_SPECS
from app.mcp.server import create_server
from app.ui import view_models as vm
from evals.reference.context import EvalContext
from evals.runners.agent import LLMFactory
from evals.runners.capture import RecordingLLM, capture_logs
from evals.scenarios.investigation import InvestigationMode, InvestigationScenario

INVESTIGATE = "/api/v1/investigations"
INVESTIGATE_STREAM = "/api/v1/investigations/stream"
ASK = "/api/v1/ask"


@dataclass
class APIObservation:
    status_code: int
    body: dict[str, Any]
    stream_events: list[dict[str, Any]] = field(default_factory=list)
    ask_status: int | None = None
    ask_body: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ParityCall:
    call_id: str
    tool: str
    mcp_tool: str
    arguments: dict[str, Any]
    is_error: bool
    payload: dict[str, Any]


@dataclass
class InvestigationObservation:
    investigation: Investigation | None
    latency_ms: float
    llm: RecordingLLM | None = None
    logs: list[dict[str, Any]] = field(default_factory=list)
    progress: list[dict[str, Any]] = field(default_factory=list)
    api: APIObservation | None = None
    ui: dict[str, Any] | None = None
    parity: list[ParityCall] = field(default_factory=list)
    mcp_tools: list[str] = field(default_factory=list)
    error: str | None = None


def investigation_config(scenario: InvestigationScenario) -> AgentConfig:
    return AgentConfig(**scenario.limits)


def run_scenario(
    ctx: EvalContext, scenario: InvestigationScenario, llm_factory: LLMFactory
) -> InvestigationObservation:
    if scenario.mode in (InvestigationMode.API, InvestigationMode.UI):
        obs = run_api(ctx, scenario, llm_factory)
        if scenario.mode == InvestigationMode.UI and obs.api is not None and obs.api.status_code == 200:
            obs.ui = ui_views(scenario.objective, obs.api.body)
        return obs
    obs = run_investigation(ctx, scenario, llm_factory)
    if scenario.mode == InvestigationMode.MCP_PARITY and obs.investigation is not None:
        obs.parity, obs.mcp_tools = run_mcp_parity(ctx, scenario, obs.investigation)
    return obs


def run_investigation(
    ctx: EvalContext, scenario: InvestigationScenario, llm_factory: LLMFactory
) -> InvestigationObservation:
    llm = RecordingLLM(llm_factory())
    runner = AgentRunner(ctx.db, llm=llm, config=investigation_config(scenario), as_of=ctx.as_of)
    investigator = Investigator(runner.runtime)
    cancel = threading.Event() if scenario.cancel_after_steps is not None else None
    if cancel is not None and scenario.cancel_after_steps == 0:
        cancel.set()
    progress: list[dict[str, Any]] = []

    def on_progress(event: dict[str, Any]) -> None:
        progress.append(json.loads(json.dumps(event, default=str)))
        finished = sum(1 for e in progress if e.get("stage") == "step_finished")
        if cancel is not None and finished >= (scenario.cancel_after_steps or 0):
            cancel.set()

    with capture_logs() as logs:
        clock = time.perf_counter()
        try:
            investigation = investigator.investigate(
                scenario.objective, cancel=cancel, deadline_seconds=scenario.deadline_seconds, on_progress=on_progress
            )
            error = None
        except Exception as exc:  # an investigation must never raise: recorded as a failure, not hidden
            investigation, error = None, f"{type(exc).__name__}: {str(exc)[:200]}"
        latency = (time.perf_counter() - clock) * 1000
    return InvestigationObservation(
        investigation=investigation, latency_ms=latency, llm=llm, logs=list(logs), progress=progress, error=error
    )


def _api_config() -> APIConfig:
    # In process, without a network listener: authentication and rate limiting are covered by the API tests.
    return APIConfig(
        host="127.0.0.1",
        port=8000,
        request_timeout_seconds=120.0,
        max_request_bytes=16384,
        max_pending_requests=4,
        max_question_chars=1000,
        auth_mode="disabled",
        rate_limit=None,
    )


def run_api(ctx: EvalContext, scenario: InvestigationScenario, llm_factory: LLMFactory) -> InvestigationObservation:
    runner = AgentRunner(ctx.db, llm=llm_factory(), config=investigation_config(scenario), as_of=ctx.as_of)
    service = AgentService(runner, ctx.db, _api_config())
    request_id = f"eval-{scenario.scenario_id}"[:64]
    try:
        with (
            capture_logs("agentops.api", "agentops.security", "agentops.agent") as logs,
            TestClient(create_app(service), raise_server_exceptions=False) as client,
        ):
            clock = time.perf_counter()
            response = client.post(INVESTIGATE, json={"objective": scenario.objective, "request_id": request_id})
            latency = (time.perf_counter() - clock) * 1000
            is_json = response.headers.get("content-type", "").startswith("application/json")
            api = APIObservation(status_code=response.status_code, body=response.json() if is_json else {})
            if scenario.stream:
                with client.stream("POST", INVESTIGATE_STREAM, json={"objective": scenario.objective}) as stream:
                    api.stream_events = [json.loads(line) for line in stream.iter_lines() if line.strip()]
            if "ask_compat" in scenario.tags:
                asked = client.post(ASK, json={"question": scenario.objective})
                api.ask_status, api.ask_body = asked.status_code, asked.json()
        api.logs = list(logs)
    finally:
        service.close()
    return InvestigationObservation(investigation=None, latency_ms=latency, api=api)


def ui_views(objective: str, response: dict[str, Any]) -> dict[str, Any]:
    """What the Investigation Mode page renders from one API response (its view models)."""
    brief = response.get("brief") or {}
    return {
        "view": asdict(vm.investigation_view(response)),
        "plan": [asdict(r) for r in vm.plan_rows(response.get("plan", []))],
        "findings": [asdict(f) for f in vm.finding_items(response)],
        "drivers": [asdict(d) for d in vm.driver_items(brief.get("drivers", []))],
        "contradictions": [asdict(d) for d in vm.driver_items(brief.get("contradictions", []))],
        "recommendations": [asdict(r) for r in vm.recommendation_items(response)],
        "sections": [asdict(s) for s in vm.section_items(response)],
        "history": asdict(vm.investigation_history_entry(objective, response=response)),
    }


def run_mcp_parity(
    ctx: EvalContext, scenario: InvestigationScenario, investigation: Investigation
) -> tuple[list[ParityCall], list[str]]:
    """Repeat every successful, MCP-exposed step through the real MCP server; also list its tools."""
    specs = {spec.tool: spec.name for spec in MCP_TOOL_SPECS}
    calls = [c for c in investigation.tool_trace if c.success and c.tool_name in specs]
    server = create_server(MCPServerConfig(limits=investigation_config(scenario)), database=ctx.db, as_of=ctx.as_of)
    parity: list[ParityCall] = []
    tools: list[str] = []

    async def session() -> None:
        async with Client(server) as client:
            tools.extend(t.name for t in (await client.list_tools()).tools)
            for call in calls:
                result = await client.call_tool(specs[call.tool_name], dict(call.input))
                payload = result.structured_content if isinstance(result.structured_content, dict) else {}
                parity.append(
                    ParityCall(
                        call_id=call.call_id,
                        tool=call.tool_name,
                        mcp_tool=specs[call.tool_name],
                        arguments=dict(call.input),
                        is_error=bool(result.is_error),
                        payload=payload,
                    )
                )

    with capture_logs():
        anyio.run(session)
    return parity, tools
