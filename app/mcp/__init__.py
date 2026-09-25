"""Phase 6: the AgentOps MCP server, a thin adapter over the secured tool layer.

    MCP client -> MCP server (``server``) -> MCP tool adapters (``adapters``)
        -> Phase 5 secured execution (authorization, validation, budget, deadline, output checks)
        -> Phase 4 tool registry -> Phase 2/3 services and safe SQL -> Phase 4 evidence builder
        -> MCP result (``schemas``)

Modules:

- ``registry``: the twelve ``agentops_*`` tools, their descriptions and schemas, and version
  metadata.
- ``schemas``: the ``MCPToolOutput`` envelope, including the forecast and anomaly views.
- ``adapters``: one call in, one redacted, size-bounded output out; request-scoped state.
- ``errors``: the MCP error categories and fixed, client-safe messages.
- ``config``: server configuration from settings.
- ``audit``: the structured audit log, correlated by run ID.
- ``server`` / ``__main__``: the low-level MCP ``Server``, its lifespan, and the stdio entry point
  ``python -m app.mcp``.

Architecture: ``docs/mcp-architecture.md``.
"""

from app.mcp.config import MCPServerConfig
from app.mcp.registry import MCP_TOOL_NAMES, TOOLSET_VERSION
from app.mcp.server import create_server, serve_stdio

__all__ = ["MCP_TOOL_NAMES", "TOOLSET_VERSION", "MCPServerConfig", "create_server", "serve_stdio"]
