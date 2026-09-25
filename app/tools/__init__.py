"""The allow-listed tool layer: typed wrappers around Phase 1-3 capabilities, shared by the agent (and, later, MCP)."""

from app.tools.base import ToolContext, ToolDefinition, ToolError, ToolInput, ToolRequest, ToolResult
from app.tools.registry import TOOL_DEFINITIONS, ToolRegistry
from app.tools.results import KPIComparison, SQLResult
from app.tools.sql_safety import UnsafeSQLError, validate_sql

__all__ = [
    "TOOL_DEFINITIONS",
    "KPIComparison",
    "SQLResult",
    "ToolContext",
    "ToolDefinition",
    "ToolError",
    "ToolInput",
    "ToolRegistry",
    "ToolRequest",
    "ToolResult",
    "UnsafeSQLError",
    "validate_sql",
]
