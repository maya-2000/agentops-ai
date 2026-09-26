"""Typed MCP tool output: one envelope for every tool, success or failure.

Input schemas are not defined here: every MCP tool uses its Phase 4 Pydantic input model (see
``app/mcp/registry.py``). This module defines only the response envelope ``MCPToolOutput``,
which is also each tool's MCP ``outputSchema``:

- ``status``, ``tool_name``, ``request_id`` (the run ID shared by every audit event of the call)
  and ``query_id``;
- ``result``: the Phase 2/3 service's typed result, serialised unchanged;
- ``forecast`` / ``anomalies``: explicit views of forecast and anomaly results (shared with the
  API in ``app/tools/views.py``), so their labelling and method fields cannot be lost;
- ``evidence``: Phase 4 evidence items (fingerprinted, with full provenance);
- ``provenance``, ``warnings``, ``limitations``;
- ``error``: a sanitised MCP error (category, code, fixed message) when ``status`` is ``error``.

The views only copy fields from the typed result: no number is recomputed here.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.evidence.models import Evidence
from app.tools.views import AnomalyReportView, ForecastPointView, ForecastView

__all__ = [
    "SCHEMA_VERSION",
    "AnomalyReportView",
    "ForecastPointView",
    "ForecastView",
    "MCPError",
    "MCPProvenance",
    "MCPToolOutput",
]

SCHEMA_VERSION = "1.0"

MCPErrorCategory = Literal[
    "INVALID_ARGUMENT",
    "UNAUTHORIZED_TOOL",
    "UNSAFE_QUERY",
    "RESOURCE_LIMIT",
    "TOOL_FAILURE",
    "TIMEOUT",
    "UNSUPPORTED_REQUEST",
    "VALIDATION_FAILURE",
    "INTERNAL_ERROR",
]
MCPStatus = Literal["ok", "no_data", "insufficient_data", "insufficient_history", "error"]
MCPOutputKind = Literal["observed", "risk_score", "forecast", "anomaly", "ad_hoc_query"]


class MCPError(BaseModel):
    """A client-safe error: never exception text, paths, SQL internals or secrets."""

    category: MCPErrorCategory
    code: str
    message: str
    detail: str | None = None  # sanitised, bounded argument feedback (argument errors only)
    retryable: bool = False


class MCPProvenance(BaseModel):
    """How the result was produced (copied from the tool result; nothing is inferred)."""

    tool: str  # the Phase 4 tool that ran
    source_layer: str
    operation: str
    call_id: str
    arguments: dict[str, Any]  # the canonical, validated arguments that ran
    query_ids: list[str] = Field(default_factory=list)
    source_tables: list[str] = Field(default_factory=list)
    calculation: str | None = None
    executed_at: datetime
    execution_time_ms: float
    attempts: int = 1
    dataset_version: str
    as_of: date
    toolset_version: str


class MCPToolOutput(BaseModel):
    """The structured content of every AgentOps MCP tool response."""

    schema_version: str = SCHEMA_VERSION
    tool_name: str
    request_id: str
    status: MCPStatus
    output_kind: MCPOutputKind
    result_type: str | None = None
    result: dict[str, Any] | None = None
    forecast: ForecastView | None = None
    anomalies: AnomalyReportView | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    provenance: MCPProvenance | None = None
    query_id: str | None = None
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    error: MCPError | None = None
    truncated: bool = False  # parts were dropped to respect the response size limit (see warnings)
