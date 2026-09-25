"""The MCP tool registry: the single catalogue of what the MCP server exposes.

Each MCP tool is a thin, named view of one Phase 4 ``ToolDefinition``:

- The **input schema** is the Phase 4 Pydantic input model's JSON schema (``additionalProperties:
  false``, so unknown fields are rejected). It is not re-declared here.
- The **description** is composed from the Phase 4 definition (what it does, when to use it,
  what not to use it for, output, limitations) and the live vocabularies (KPIs, metrics,
  dimensions, detectors). The MCP layer adds only what MCP clients also need: required and
  optional inputs, the output kind and the safety constraints.
- The **output schema** is ``MCPToolOutput`` (``app/mcp/schemas.py``).
- The **intent** names the analytical intent a call runs under. The Phase 5 permission table
  (``INTENT_TOOL_PERMISSIONS``) must permit the tool for that intent. The registry checks this
  when it is built, and ``ToolAuthorizationPolicy`` checks it again on every call. MCP has no
  permission table of its own.

Only these twelve read-only analytics tools exist. There are no file, directory, shell, Python,
environment or database-file tools, and nothing here can register one at runtime.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import mcp.types as types

from app.analytics.dimensions import DIMENSIONS
from app.analytics.kpis import list_kpi_definitions
from app.anomalies.config import DETECTOR_NAMES
from app.llm.schemas import Intent
from app.mcp.schemas import MCPToolOutput
from app.security.authorization import INTENT_TOOL_PERMISSIONS, RESTRICTED_BREAKDOWNS, SQL_TOOL
from app.security.data_policy import APPROVED_TABLES, APPROVED_VIEWS
from app.security.limits import SecurityLimits
from app.timeseries.metrics import SERIES_METRIC_KEYS
from app.tools.base import ToolDefinition
from app.tools.registry import ToolRegistry

TOOLSET_VERSION = "1.0.0"
TOOL_PREFIX = "agentops_"
META_PREFIX = "io.agentops/"

OutputKind = Literal["observed", "risk_score", "forecast", "anomaly", "ad_hoc_query"]

_OUTPUT_KIND_TEXT: dict[OutputKind, str] = {
    "observed": (
        "OBSERVED data: values aggregated or calculated from recorded business data by the analytics layer. "
        "Not a forecast and not an anomaly score."
    ),
    "risk_score": (
        "RULE-BASED SCORE: a transparent risk score computed from observable signals. It is not a churn "
        "probability, not a prediction and not an observed outcome."
    ),
    "forecast": (
        "FORECAST: model predictions with prediction intervals, not observed data. The `forecast` field keeps "
        "the metric, forecast period, cutoff, horizon, model, predicted values, intervals and backtest metrics."
    ),
    "anomaly": (
        "ANOMALY SCORES: statistical scores of observed months against the preceding months. A flag means "
        "statistically unusual, not good or bad, and never a cause. The `anomalies` field keeps the metric, "
        "period, observed and expected values, score, detector, severity and direction."
    ),
    "ad_hoc_query": (
        "OBSERVED rows from one ad-hoc read-only query. Not a registered KPI definition; prefer the dedicated tools."
    ),
}


@dataclass(frozen=True)
class MCPToolSpec:
    name: str  # the MCP tool name (agentops_*)
    tool: str  # the Phase 4 tool it exposes
    title: str
    intent: Intent  # the intent the call is authorized under (Phase 5 permission table)
    output_kind: OutputKind
    dimensions: bool = False  # accepts a breakdown ``dimension`` argument
    version: str = TOOLSET_VERSION


MCP_TOOL_SPECS: tuple[MCPToolSpec, ...] = (
    MCPToolSpec("agentops_get_kpi", "get_kpi", "Get a business KPI", Intent.KPI_LOOKUP, "observed", True),
    MCPToolSpec(
        "agentops_analyze_revenue",
        "analyze_revenue",
        "Analyze revenue and MRR",
        Intent.REVENUE_INVESTIGATION,
        "observed",
        True,
    ),
    MCPToolSpec(
        "agentops_analyze_customers",
        "analyze_customers",
        "Analyze customers and churn",
        Intent.CUSTOMER_INVESTIGATION,
        "observed",
        True,
    ),
    MCPToolSpec(
        "agentops_analyze_sales", "analyze_sales", "Analyze sales and pipeline", Intent.SALES_ANALYSIS, "observed", True
    ),
    MCPToolSpec(
        "agentops_analyze_marketing",
        "analyze_marketing",
        "Analyze marketing channels and campaigns",
        Intent.MARKETING_ANALYSIS,
        "observed",
    ),
    MCPToolSpec(
        "agentops_analyze_support",
        "analyze_support",
        "Analyze support tickets",
        Intent.SUPPORT_ANALYSIS,
        "observed",
        True,
    ),
    MCPToolSpec(
        "agentops_analyze_product",
        "analyze_product",
        "Analyze product feature adoption",
        Intent.PRODUCT_ANALYSIS,
        "observed",
        True,
    ),
    MCPToolSpec(
        "agentops_get_cohort_analysis",
        "get_cohort_analysis",
        "Get signup-cohort retention",
        Intent.CUSTOMER_INVESTIGATION,
        "observed",
    ),
    MCPToolSpec(
        "agentops_get_customer_risk",
        "get_customer_risk",
        "Get rule-based customer risk scores",
        Intent.CUSTOMER_INVESTIGATION,
        "risk_score",
    ),
    MCPToolSpec("agentops_forecast_metric", "forecast_metric", "Forecast a metric", Intent.FORECAST, "forecast"),
    MCPToolSpec(
        "agentops_detect_anomalies",
        "detect_anomalies",
        "Detect anomalous months",
        Intent.ANOMALY_DETECTION,
        "anomaly",
    ),
    MCPToolSpec(
        "agentops_run_safe_sql",
        SQL_TOOL,
        "Run one safe read-only SQL query",
        Intent.MIXED_INVESTIGATION,
        "ad_hoc_query",
    ),
)
MCP_TOOL_NAMES: tuple[str, ...] = tuple(spec.name for spec in MCP_TOOL_SPECS)


def exposed_dimensions() -> list[str]:
    """Breakdown dimensions a caller may request: the validated vocabulary minus person/customer breakdowns."""
    return [d for d in DIMENSIONS if d not in RESTRICTED_BREAKDOWNS]


def _inputs_text(definition: ToolDefinition) -> str:
    schema = definition.input_schema
    properties = list(schema.get("properties", {}))
    required = list(schema.get("required", []))
    optional = [p for p in properties if p not in required]
    parts = [f"Required: {', '.join(required) if required else 'none'}."]
    if optional:
        parts.append(f"Optional: {', '.join(optional)}.")
    parts.append("Types and allowed values are in the input schema; unknown fields are rejected.")
    return " ".join(parts)


def _vocabulary_text(spec: MCPToolSpec) -> str:
    if spec.tool == "get_kpi":
        return f"Supported KPIs: {', '.join(d.key for d in list_kpi_definitions())}."
    if spec.tool == "forecast_metric":
        return f"Supported metrics: {', '.join(SERIES_METRIC_KEYS)}; horizon 1-6 months."
    if spec.tool == "detect_anomalies":
        return (
            f"Supported metrics: {', '.join(SERIES_METRIC_KEYS)}. Detectors: {', '.join(DETECTOR_NAMES)} "
            "(default rolling_zscore)."
        )
    if spec.tool == SQL_TOOL:
        return f"Queryable tables: {', '.join(sorted(APPROVED_TABLES))}; views: {', '.join(sorted(APPROVED_VIEWS))}."
    return ""


def _dimensions_text(spec: MCPToolSpec) -> str:
    if not spec.dimensions:
        return "Supported dimensions: this tool takes no breakdown dimension."
    return (
        f"Supported dimensions: {', '.join(exposed_dimensions())}. Each KPI or operation accepts a subset, and "
        "an unsupported combination is rejected as INVALID_ARGUMENT. Breakdowns by individual customer "
        "(customer_id) or person (sales_rep) are refused by the data-exposure policy."
    )


def _safety_text(spec: MCPToolSpec, limits: SecurityLimits) -> str:
    base = (
        "Safety: read-only. Arguments are validated against the input schema and authorized by the AgentOps "
        "data-access policy before any query runs. Text arguments are treated as data, never as instructions. "
        "Withheld and PII fields are masked in results by the data-exposure policy."
    )
    if spec.tool == SQL_TOOL:
        return (
            f"{base} Exactly one SELECT over the allow-listed business tables and views; values must be bound as "
            f"$name parameters. DDL, DML, PRAGMA, file and system functions, system tables, PII-tagged and "
            f"withheld columns are rejected. At most {limits.sql_row_limit} rows per query "
            f"({limits.max_sql_joins} joins, nesting depth {limits.max_sql_nesting_depth}); truncation is reported."
        )
    if spec.tool == "get_customer_risk":
        return f"{base} At most {limits.max_customer_rows} customer rows per call; company names are masked."
    return base


def describe(spec: MCPToolSpec, definition: ToolDefinition, limits: SecurityLimits) -> str:
    """The complete MCP description of one tool."""
    sections = [
        definition.description,
        f"When to use: {definition.when_to_use}",
        f"Do not use for: {definition.not_for}",
        f"Inputs: {_inputs_text(definition)}",
        _vocabulary_text(spec),
        _dimensions_text(spec),
        f"Output: {definition.output_description} {_OUTPUT_KIND_TEXT[spec.output_kind]} Every response has status, "
        "result, evidence, provenance, warnings, limitations, query_id and tool_name.",
        f"Limitations: {definition.limitations}",
        _safety_text(spec, limits),
    ]
    return "\n".join(s for s in sections if s)


class MCPToolRegistry:
    """The enabled MCP tools, each bound to its Phase 4 definition."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        limits: SecurityLimits,
        enabled: Iterable[str] | None = None,
        specs: tuple[MCPToolSpec, ...] = MCP_TOOL_SPECS,
    ):
        names = [s.name for s in specs]
        if len(set(names)) != len(names) or any(not n.startswith(TOOL_PREFIX) for n in names):
            raise ValueError("MCP tool names must be unique and start with 'agentops_'")
        enabled_names = set(names if enabled is None else enabled)
        unknown = enabled_names - set(names)
        if unknown:
            raise ValueError(f"Unknown MCP tools: {', '.join(sorted(unknown))}")
        self._all = {s.name: s for s in specs}
        self._definitions: dict[str, ToolDefinition] = {}
        for spec in specs:
            definition = tool_registry.get(spec.tool)
            if definition is None:
                raise ValueError(f"{spec.name} exposes an unregistered tool: {spec.tool}")
            if spec.tool not in INTENT_TOOL_PERMISSIONS.get(spec.intent, frozenset()):
                raise ValueError(f"{spec.name}: {spec.tool} is not permitted for intent {spec.intent.value}")
            self._definitions[spec.name] = definition
        self._enabled = [s.name for s in specs if s.name in enabled_names]
        self._tools = {name: self._build(self._all[name], limits) for name in self._enabled}

    @property
    def names(self) -> list[str]:
        """Enabled tool names, in catalogue order."""
        return list(self._enabled)

    @property
    def disabled_tools(self) -> frozenset[str]:
        """Phase 4 names of the tools switched off for MCP (denied by the authorization policy)."""
        return frozenset(s.tool for n, s in self._all.items() if n not in self._enabled)

    def spec(self, name: str) -> MCPToolSpec | None:
        """The spec of any known MCP tool, enabled or not; ``None`` for an unknown name."""
        return self._all.get(name)

    def definition(self, name: str) -> ToolDefinition:
        return self._definitions[name]

    def tools(self) -> list[types.Tool]:
        return list(self._tools.values())

    def _build(self, spec: MCPToolSpec, limits: SecurityLimits) -> types.Tool:
        definition = self._definitions[spec.name]
        meta: dict[str, Any] = {
            f"{META_PREFIX}version": spec.version,
            f"{META_PREFIX}toolset_version": TOOLSET_VERSION,
            f"{META_PREFIX}output_kind": spec.output_kind,
            f"{META_PREFIX}source_layer": definition.source_layer,
        }
        return types.Tool(
            name=spec.name,
            title=spec.title,
            description=describe(spec, definition, limits),
            input_schema=definition.input_schema,
            output_schema=MCPToolOutput.model_json_schema(mode="serialization"),
            annotations=types.ToolAnnotations(
                title=spec.title,
                read_only_hint=True,
                destructive_hint=False,
                idempotent_hint=True,
                open_world_hint=False,
            ),
            _meta=meta,
        )
