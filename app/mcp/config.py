"""MCP server configuration: built once from settings at startup and immutable afterwards.

The security limits are the Phase 5 ``AgentConfig`` (the same ``AGENT_*`` settings the agent
uses). There is no second, MCP-only set of limits. The MCP settings add only what is specific
to serving tools over MCP:

- server name and version;
- the enabled tools;
- the transport (stdio: local development);
- the log level;
- request and response size limits;
- whether the ad-hoc SQL tool is offered.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agent.config import AgentConfig
from app.config import MCPLogLevel, MCPTransport, Settings, get_settings
from app.mcp.registry import MCP_TOOL_NAMES, MCP_TOOL_SPECS

SQL_MCP_TOOL = "agentops_run_safe_sql"


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    server_name: str = "agentops-ai"
    server_version: str = "0.6.0"
    enabled_tools: frozenset[str] = frozenset(MCP_TOOL_NAMES)
    transport: MCPTransport = "stdio"
    log_level: MCPLogLevel = "WARNING"
    max_request_bytes: int = Field(default=16384, ge=1024, le=1_048_576)
    max_response_bytes: int = Field(default=262144, ge=16384, le=8_388_608)
    sql_enabled: bool = True
    limits: AgentConfig = Field(default_factory=AgentConfig)

    @field_validator("enabled_tools")
    @classmethod
    def _known_tools(cls, value: frozenset[str]) -> frozenset[str]:
        unknown = sorted(value - set(MCP_TOOL_NAMES))
        if unknown:
            raise ValueError(f"Unknown MCP tools: {', '.join(unknown)}")
        return value

    @property
    def listed_tools(self) -> frozenset[str]:
        """Tools the server lists and may run.

        A tool is listed only when it is enabled for MCP, is not disabled for the agent
        (``AGENT_DISABLED_TOOLS``) and, for SQL, is allowed by the SQL setting.
        """
        agent_disabled = self.limits.disabled_tools
        return frozenset(
            s.name
            for s in MCP_TOOL_SPECS
            if s.name in self.enabled_tools
            and s.tool not in agent_disabled
            and (self.sql_enabled or s.name != SQL_MCP_TOOL)
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> MCPServerConfig:
        s = settings or get_settings()
        names = [t.strip() for t in s.mcp_enabled_tools.split(",") if t.strip()]
        return cls(
            server_name=s.mcp_server_name,
            server_version=s.mcp_server_version,
            enabled_tools=frozenset(names or MCP_TOOL_NAMES),
            transport=s.mcp_transport,
            log_level=s.mcp_log_level,
            max_request_bytes=s.mcp_max_request_bytes,
            max_response_bytes=s.mcp_max_response_bytes,
            sql_enabled=s.mcp_sql_enabled,
            limits=AgentConfig.from_settings(s),
        )
