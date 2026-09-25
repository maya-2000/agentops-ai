"""Run tool calls through the two production tool paths and record the outcomes.

- **Direct**: the agent's own ``SecuredToolExecutor`` (``AgentRuntime.executor``). The call is
  authorized under the intent the MCP registry declares for the tool, so both paths are asked
  exactly the same thing.
- **MCP**: the real Phase 6 server (``create_server``) over the MCP SDK's in-process client,
  i.e. the JSON-RPC handlers, adapters, schemas and error model.

Neither path is reimplemented here: the runner only calls them and times them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import anyio
from mcp import Client

from app.agent.graph import AgentRuntime
from app.evidence.builder import build_evidence
from app.evidence.models import Evidence, EvidenceGraph
from app.llm.deterministic import DeterministicLLM
from app.llm.schemas import Intent
from app.mcp.config import MCPServerConfig
from app.mcp.registry import MCP_TOOL_SPECS, TOOL_PREFIX
from app.mcp.server import create_server
from app.security.authorization import AuthorizationContext
from app.security.budget import BudgetUsage
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry
from evals.reference.context import EvalContext
from evals.runners.agent import agent_config
from evals.runners.capture import capture_logs
from evals.scenarios.model import EvaluationScenario, ToolCall

_SPECS = {s.name: s for s in MCP_TOOL_SPECS}


@dataclass
class DirectCall:
    call: ToolCall
    tool: str  # the Phase 4 tool the direct path was asked to run
    result: ToolResult
    allowed: bool
    code: str | None
    evidence: list[Evidence]
    events: list[str]
    usage: BudgetUsage
    latency_ms: float


@dataclass
class MCPCall:
    call: ToolCall
    is_error: bool
    payload: dict[str, Any]
    latency_ms: float

    @property
    def category(self) -> str | None:
        error = self.payload.get("error") or {}
        return error.get("category")

    @property
    def code(self) -> str | None:
        error = self.payload.get("error") or {}
        return error.get("code")


@dataclass
class MCPObservation:
    calls: list[MCPCall] = field(default_factory=list)
    tools: list[Any] = field(default_factory=list)  # discovery: mcp.types.Tool
    logs: list[dict[str, Any]] = field(default_factory=list)
    server_name: str | None = None
    error: str | None = None


def internal_tool(mcp_name: str) -> tuple[str, Intent]:
    """The Phase 4 tool and the intent the MCP registry authorizes it under (unknown names pass through)."""
    spec = _SPECS.get(mcp_name)
    if spec is not None:
        return spec.tool, spec.intent
    return mcp_name.removeprefix(TOOL_PREFIX), Intent.MIXED_INVESTIGATION


def run_direct(ctx: EvalContext, scenario: EvaluationScenario) -> list[DirectCall]:
    """Each call through the agent's secured executor, with a fresh per-call budget (as MCP has)."""
    runtime = AgentRuntime(ctx.db, DeterministicLLM(), agent_config(scenario), ToolRegistry(), ctx.as_of)
    calls: list[DirectCall] = []
    for index, call in enumerate(scenario.calls, start=1):
        tool, intent = internal_tool(call.tool)
        clock = time.perf_counter()
        with capture_logs():  # security events are read from the SecuredCall; keep stderr quiet
            secured = runtime.executor.execute(
                run_id=f"R-eval-direct-{index}",
                call_id="T1",
                tool_name=tool,
                arguments=call.arguments,
                purpose="evaluation: direct path",
                context=AuthorizationContext(intent=intent),
                budget=runtime.run_budget,
                usage=BudgetUsage(),
            )
        latency = (time.perf_counter() - clock) * 1000
        evidence = build_evidence(secured.result, EvidenceGraph()) if secured.result.success else []
        calls.append(
            DirectCall(
                call=call,
                tool=tool,
                result=secured.result,
                allowed=secured.decision.allowed,
                code=secured.result.error.code if secured.result.error else None,
                evidence=evidence,
                events=[e.event_type for e in secured.events],
                usage=secured.usage,
                latency_ms=latency,
            )
        )
    return calls


def mcp_config(scenario: EvaluationScenario) -> MCPServerConfig:
    return MCPServerConfig(limits=agent_config(scenario))


def run_mcp(ctx: EvalContext, scenario: EvaluationScenario, *, discover: bool = False) -> MCPObservation:
    """The calls through the real MCP server and protocol (in-process SDK client)."""
    server = create_server(mcp_config(scenario), database=ctx.db, as_of=ctx.as_of)
    observation = MCPObservation()

    async def session() -> None:
        async with Client(server) as client:
            info = client.server_info
            observation.server_name = info.name if info else None
            if discover:
                observation.tools = list((await client.list_tools()).tools)
            for call in scenario.calls:
                clock = time.perf_counter()
                result = await client.call_tool(call.tool, call.arguments)
                latency = (time.perf_counter() - clock) * 1000
                payload = result.structured_content if isinstance(result.structured_content, dict) else {}
                observation.calls.append(MCPCall(call, bool(result.is_error), payload, latency))

    with capture_logs() as logs:
        try:
            anyio.run(session)
        except Exception as exc:  # a protocol failure is an MCP_ERROR, recorded rather than raised
            observation.error = f"{type(exc).__name__}: {str(exc)[:200]}"
    observation.logs = list(logs)
    return observation
