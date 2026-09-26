"""Helpers for the Phase 6 MCP tests: servers, adapter services and an in-process MCP client."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import anyio
import mcp.types as types
from mcp import Client
from mcp.server import Server

from app.agent.config import AgentConfig
from app.database.base import Database
from app.mcp.adapters import MCPToolService, authorization_policy
from app.mcp.config import MCPServerConfig
from app.mcp.registry import MCPToolRegistry
from app.mcp.server import MCPServerState, create_server
from app.tools.registry import ToolRegistry
from tests.phase4_support import AS_OF

T = TypeVar("T")

EXPECTED_TOOLS = (
    "agentops_get_kpi",
    "agentops_analyze_revenue",
    "agentops_analyze_customers",
    "agentops_analyze_sales",
    "agentops_analyze_marketing",
    "agentops_analyze_support",
    "agentops_analyze_product",
    "agentops_get_cohort_analysis",
    "agentops_get_customer_risk",
    "agentops_forecast_metric",
    "agentops_detect_anomalies",
    "agentops_run_safe_sql",
)
PROHIBITED_TOOLS = (
    "read_file",
    "list_directory",
    "execute_python",
    "run_shell",
    "read_environment",
    "read_database_file",
)
# One valid call per tool (full dataset: 24 months to the 2026-08-31 as-of date).
VALID_CALLS: dict[str, dict[str, Any]] = {
    "agentops_get_kpi": {"kpi": "revenue", "start_date": "2026-08-01", "end_date": "2026-08-31"},
    "agentops_analyze_revenue": {"operation": "revenue_change", "period": "2026-08", "comparison_period": "2026-07"},
    "agentops_analyze_customers": {"operation": "churn_summary", "period": "2026-08"},
    "agentops_analyze_sales": {"operation": "pipeline_summary"},
    "agentops_analyze_marketing": {"operation": "channel_performance", "period": "2026-Q2"},
    "agentops_analyze_support": {"operation": "support_summary", "period": "2026-08"},
    "agentops_analyze_product": {"operation": "feature_adoption", "period": "2026-08"},
    "agentops_get_cohort_analysis": {"max_months": 3},
    "agentops_get_customer_risk": {"limit": 5},
    "agentops_forecast_metric": {"metric": "revenue", "horizon": 3},
    "agentops_detect_anomalies": {"metric": "support_ticket_volume"},
    # ORDER BY: without it DuckDB may return the groups in a different order on each execution.
    "agentops_run_safe_sql": {
        "sql": "SELECT segment, COUNT(*) AS customers FROM customers GROUP BY segment ORDER BY 1"
    },
}


def mcp_config(limits: AgentConfig | None = None, **overrides: Any) -> MCPServerConfig:
    return MCPServerConfig(limits=limits or AgentConfig(), **overrides)


def service(
    db: Database, config: MCPServerConfig | None = None, registry: ToolRegistry | None = None
) -> MCPToolService:
    """The adapter service exactly as the server builds it."""
    cfg = config or mcp_config()
    tool_registry = registry or ToolRegistry()
    tools = MCPToolRegistry(tool_registry, cfg.limits, enabled=cfg.listed_tools)
    policy = authorization_policy(tool_registry, cfg, tools)
    return MCPToolService(db, cfg, tools, policy, tool_registry=tool_registry, as_of=AS_OF)


def server(
    db: Database, config: MCPServerConfig | None = None, registry: ToolRegistry | None = None
) -> Server[MCPServerState]:
    return create_server(config or mcp_config(), database=db, tool_registry=registry, as_of=AS_OF)


def with_client(target: Server[MCPServerState], fn: Callable[[Client], Awaitable[T]]) -> T:
    """Connect an in-process MCP client (real protocol handlers and lifespan) and run ``fn``."""

    async def main() -> T:
        async with Client(target) as client:
            return await fn(client)

    return anyio.run(main)


def call(target: Server[MCPServerState], name: str, arguments: dict[str, Any] | None = None) -> types.CallToolResult:
    async def fn(client: Client) -> types.CallToolResult:
        return await client.call_tool(name, arguments)

    return with_client(target, fn)


def structured(result: types.CallToolResult) -> dict[str, Any]:
    content = result.structured_content
    assert isinstance(content, dict)
    return content
