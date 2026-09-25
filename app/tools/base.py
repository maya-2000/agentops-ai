"""The shared tool contract: definitions, requests and typed results.

A tool is a thin, deterministic wrapper around an existing Phase 1-3 capability. It validates
typed input, calls the underlying service, and returns a ``ToolResult`` that carries the service's
own typed result together with its provenance (query IDs, source tables, calculation, limitations).
Phase 6 (MCP) can wrap the same ``ToolDefinition`` handlers without change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny

from app.analytics.kpis import KPIService
from app.database.base import Database

SourceLayer = Literal[
    "phase2_kpi",
    "phase2_analytics",
    "phase3_forecasting",
    "phase3_anomalies",
    "phase1_database",
]
ToolStatus = Literal["ok", "no_data", "insufficient_data", "insufficient_history", "error"]


class ToolInput(BaseModel):
    """Base class for tool inputs: unknown arguments are rejected, never ignored."""

    model_config = ConfigDict(extra="forbid")


@dataclass
class ToolContext:
    """Runtime dependencies of a tool call. Never stored in agent state."""

    db: Database
    as_of: date
    sql_row_limit: int
    kpi_service: KPIService = field(init=False)

    def __post_init__(self) -> None:
        self.kpi_service = KPIService(self.db, as_of=self.as_of)


@dataclass(frozen=True)
class ToolOutput:
    """What a handler returns; the registry turns it into a ``ToolResult``."""

    result: BaseModel
    status: ToolStatus
    source_tables: list[str]
    query_ids: list[str]
    calculation: str
    limitations: list[str] = field(default_factory=list)
    message: str | None = None


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    when_to_use: str
    not_for: str
    output_description: str
    limitations: str
    source_layer: SourceLayer
    input_model: type[ToolInput]
    handler: Callable[[ToolContext, Any], ToolOutput]
    deterministic: bool = True

    @property
    def allowed_use(self) -> str:
        return self.when_to_use

    @property
    def input_schema(self) -> dict[str, Any]:
        return self.input_model.model_json_schema()

    @property
    def output_schema(self) -> dict[str, Any]:
        return ToolResult.model_json_schema()

    def catalog_entry(self) -> dict[str, Any]:
        """Compact description for planning prompts (no handler, no internals)."""
        schema = self.input_schema
        return {
            "tool_name": self.name,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "must_not_be_used_for": self.not_for,
            "output": self.output_description,
            "limitations": self.limitations,
            "input_schema": {"properties": schema.get("properties", {}), "required": schema.get("required", [])},
        }


class ToolRequest(BaseModel):
    call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    purpose: str = ""


class ToolError(BaseModel):
    code: str
    message: str
    retryable: bool = False


class ToolResult(BaseModel):
    """Typed outcome of one tool call (successful or not)."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    success: bool
    status: ToolStatus
    result: SerializeAsAny[BaseModel] | None = None
    result_type: str | None = None
    error: ToolError | None = None
    message: str | None = None
    started_at: datetime
    finished_at: datetime
    execution_time_ms: float
    source_tables: list[str] = Field(default_factory=list)
    query_ids: list[str] = Field(default_factory=list)
    calculation: str | None = None
    limitations: list[str] = Field(default_factory=list)
    attempts: int = 1

    @property
    def query_id(self) -> str | None:
        return self.query_ids[-1] if self.query_ids else None

    @property
    def execution_timestamp(self) -> datetime:
        return self.finished_at
