"""MCP tool adapters: one MCP tool call in, one validated ``MCPToolOutput`` out.

The adapter is a thin translation layer. It contains no business logic, no SQL and no
authorization rules of its own:

    MCP call -> name and size checks -> SecuredToolExecutor (Phase 5: authorization, typed
    argument validation, argument and data-exposure policy, budget, deadline, retry, output
    validation) -> Phase 4 tool registry -> Phase 2/3 services / safe SQL
    -> Phase 4 evidence builder -> MCPToolOutput (redacted, size-bounded) -> MCP result

Everything is request-scoped: each call gets a new run ID, a fresh budget, a fresh evidence
graph and its own audit events. The service holds only immutable configuration, the policies
and the database handle. The server runs calls one at a time (the DuckDB connection is not
safe for concurrent use).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel

from app.agent.observability import new_run_id
from app.anomalies import AnomalyReport
from app.database.base import Database
from app.evidence.builder import build_evidence
from app.evidence.models import EvidenceGraph
from app.evidence.validation import validate_evidence
from app.forecasting import ForecastResult
from app.mcp.audit import audit
from app.mcp.config import MCPServerConfig
from app.mcp.errors import mcp_error
from app.mcp.registry import MCPToolRegistry, MCPToolSpec
from app.mcp.schemas import AnomalyReportView, ForecastView, MCPOutputKind, MCPProvenance, MCPToolOutput
from app.security.authorization import AuthorizationContext, ToolAuthorizationPolicy, ToolPermissions
from app.security.budget import BudgetUsage, RunBudget
from app.security.data_policy import default_exposure_policy, mask_withheld_fields
from app.security.events import SecurityEvent, SecurityEventType, Severity, highest_severity, security_event
from app.security.execution import SecuredToolExecutor
from app.security.injection import InjectionScan, PromptInjectionDetector
from app.security.output_guard import ToolOutputValidator
from app.security.redaction import redact_value
from app.security.retry import RetryPolicy
from app.tools.base import ToolContext, ToolResult
from app.tools.registry import ToolRegistry
from app.tools.results import SQLResult

_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
UNKNOWN_TOOL_NAME = "unknown"
_KIND_WARNINGS: dict[MCPOutputKind, str] = {
    "forecast": "Forecast values are model predictions with uncertainty, not observed data.",
    "anomaly": "Anomaly flags are statistical scores; they are not business judgements and do not establish causes.",
    "risk_score": "Risk scores are rule-based signals from observable data, not churn probabilities.",
    "ad_hoc_query": "Ad-hoc query results are not registered KPI definitions.",
    "observed": "",
}


# Denial codes that mean the arguments themselves were malformed or out of vocabulary.
_ARGUMENT_CODES = frozenset(
    {
        "invalid_arguments",
        "invalid_request",
        "oversized_input",
        "unsupported_kpi",
        "unsupported_dimension",
        "unsupported_metric",
        "unsupported_method",
        "invalid_filter_value",
        "invalid_period",
        "invalid_date_range",
        "invalid_horizon",
    }
)
Stage = Literal["passed", "failed", "allowed", "denied", "succeeded", "not_reached", "not_run"]


@dataclass
class CallTrace:
    """How far one call got: validation, authorization and execution (for the audit record)."""

    validation: Stage = "not_reached"
    authorization: Stage = "not_reached"
    execution: Stage = "not_run"
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class MCPCallOutcome:
    """The result of one MCP tool call and everything recorded about it."""

    output: MCPToolOutput
    payload: dict[str, Any]  # the redacted JSON form of ``output`` (what is sent)
    events: list[SecurityEvent] = field(default_factory=list)
    usage: BudgetUsage = field(default_factory=BudgetUsage)
    trace: CallTrace = field(default_factory=CallTrace)
    audit_record: dict[str, Any] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.output.status == "error"


def authorization_policy(
    tool_registry: ToolRegistry, config: MCPServerConfig, tools: MCPToolRegistry
) -> ToolAuthorizationPolicy:
    """The Phase 5 policy for MCP calls: the one permission table, plus MCP's switched-off tools as disabled."""
    permissions = ToolPermissions(disabled=tools.disabled_tools | config.limits.disabled_tools)
    return ToolAuthorizationPolicy(tool_registry, config.limits, permissions)


class MCPToolService:
    """Runs MCP tool calls through the shared Phase 5 execution path."""

    def __init__(
        self,
        db: Database,
        config: MCPServerConfig,
        tools: MCPToolRegistry,
        policy: ToolAuthorizationPolicy,
        *,
        tool_registry: ToolRegistry,
        as_of: date,
    ):
        limits = config.limits
        self.db = db
        self.config = config
        self.tools = tools
        self.as_of = as_of
        self.policy = policy
        self.budget = RunBudget.from_limits(limits)
        self.exposure = default_exposure_policy()
        tool_context = ToolContext(
            db=db,
            as_of=as_of,
            sql_row_limit=limits.sql_row_limit,
            sql_timeout_seconds=limits.sql_timeout_seconds,
            sql_limits=self.policy.sql_limits,
        )
        self.executor = SecuredToolExecutor(
            tool_registry,
            self.policy,
            ToolOutputValidator(limits, self.exposure),
            RetryPolicy(limits.max_retries),
            tool_context,
            tool_timeout_seconds=limits.tool_timeout_seconds,
        )
        self.detector = PromptInjectionDetector()

    # ------------------------------------------------------------------------------------------ call
    def call(self, name: Any, arguments: Any, *, mcp_request_id: Any = None) -> MCPCallOutcome:
        """Run one tool call. Never raises: every failure becomes a sanitised error output."""
        run_id = new_run_id()
        clock = time.perf_counter()
        events: list[SecurityEvent] = []
        usage = BudgetUsage()
        trace = CallTrace()
        tool_name = name if isinstance(name, str) and _TOOL_NAME.match(name) else UNKNOWN_TOOL_NAME
        request_bytes = 0
        try:
            args = {} if arguments is None else arguments
            request_bytes = len(json.dumps(args, default=str).encode("utf-8"))
            spec = self.tools.spec(tool_name)
            if request_bytes > self.config.max_request_bytes:
                events.append(
                    self._event(
                        run_id,
                        "input_rejected",
                        Severity.WARNING,
                        tool_name,
                        "deny",
                        f"Request arguments are {request_bytes} bytes (limit {self.config.max_request_bytes}).",
                    )
                )
                trace.validation = "failed"
                output = self._error(run_id, tool_name, spec, "request_too_large")
            elif spec is None:
                events.append(
                    self._event(
                        run_id,
                        "tool_denied",
                        Severity.HIGH,
                        tool_name,
                        "deny",
                        "unknown_tool: the name is not in the MCP tool registry.",
                        code="unknown_tool",
                    )
                )
                trace.authorization = "denied"
                output = self._error(run_id, tool_name, None, "unknown_tool")
            else:
                output, usage = self._run(run_id, spec, args, events, trace)
        except Exception as exc:  # fail closed: nothing internal reaches the client
            events.append(
                self._event(
                    run_id,
                    "output_validation_failed",
                    Severity.HIGH,
                    tool_name,
                    "stop",
                    f"Unexpected {type(exc).__name__} in the MCP adapter.",
                )
            )
            trace.execution = "failed"
            output = self._error(run_id, tool_name, None, "internal_error")
        try:
            output, payload = self._bounded(run_id, output, events)
        except Exception:  # fail closed here too: a minimal error envelope, never an exception
            output = self._error(run_id, tool_name, None, "internal_error")
            payload = output.model_dump(mode="json")
        trace.evidence_ids = [e.evidence_id for e in output.evidence]
        response_bytes = len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        severity = highest_severity(events)
        record = audit(
            "tool_call",
            run_id,
            tool_name=tool_name,
            mcp_request_id=mcp_request_id,
            status=output.status,
            is_error=output.status == "error",
            validation=trace.validation,
            authorization=trace.authorization,
            execution=trace.execution,
            evidence_ids=trace.evidence_ids,
            error_category=output.error.category if output.error else None,
            error_code=output.error.code if output.error else None,
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            truncated=output.truncated,
            execution_time_ms=round((time.perf_counter() - clock) * 1000, 3),
            query_ids=output.provenance.query_ids if output.provenance else [],
            security_events=len(events),
            highest_severity=severity.value if severity else None,
        )
        return MCPCallOutcome(output, payload, events, usage, trace, record)

    def _run(
        self, run_id: str, spec: MCPToolSpec, arguments: Any, events: list[SecurityEvent], trace: CallTrace
    ) -> tuple[MCPToolOutput, BudgetUsage]:
        flagged = self._screen(arguments)
        if flagged.suspicious:
            # Audit only: parameters are data. They are validated like any other value, never obeyed.
            events.append(
                self._event(
                    run_id,
                    "suspicious_prompt",
                    Severity.WARNING,
                    spec.name,
                    "flag",
                    "Instruction-like text in tool parameters; treated as data.",
                    categories=flagged.categories,
                )
            )
        context = AuthorizationContext(intent=spec.intent, sql_permitted=self.config.sql_enabled)
        call = self.executor.execute(
            run_id=run_id,
            call_id="T1",
            tool_name=spec.tool,
            arguments=arguments,
            purpose=f"mcp:{spec.name}",
            context=context,
            budget=self.budget,
            usage=BudgetUsage(),
        )
        events.extend(call.events)
        decision = call.decision
        if decision.code in _ARGUMENT_CODES:
            trace.validation = "failed"
        elif "typed_arguments" in decision.checks_passed:
            trace.validation = "passed"
        trace.authorization = "allowed" if decision.allowed else "denied"
        result = call.result
        if decision.allowed:
            trace.execution = "succeeded" if result.success else "failed"
        if not result.success or result.result is None:
            code = result.error.code if result.error else "internal_error"
            detail = result.error.message if result.error else None
            return self._error(run_id, spec.name, spec, code, detail), call.usage

        graph = EvidenceGraph()
        evidence = build_evidence(result, graph)
        check = validate_evidence(graph, successful_call_ids=[result.call_id], require_primary=False)
        if not check.valid:
            events.append(
                self._event(
                    run_id, "evidence_integrity_failed", Severity.HIGH, spec.name, "deny", check.errors[0][:200]
                )
            )
            trace.execution = "failed"
            return self._error(run_id, spec.name, spec, "evidence_integrity_failed"), call.usage
        payload = result.result
        operation = f"{spec.tool}.{result.arguments['operation']}" if "operation" in result.arguments else spec.tool
        public, masked = mask_withheld_fields(_public_result(payload), self.exposure, operation)
        if masked:
            events.append(
                self._event(
                    run_id,
                    "secret_redacted",
                    Severity.INFO,
                    spec.name,
                    "redact",
                    f"{masked} withheld field value(s) masked by the data-exposure policy.",
                )
            )
        output = MCPToolOutput(
            tool_name=spec.name,
            request_id=run_id,
            status=result.status,
            output_kind=spec.output_kind,
            result_type=result.result_type,
            result=public,
            forecast=ForecastView.from_result(payload) if isinstance(payload, ForecastResult) else None,
            anomalies=AnomalyReportView.from_report(payload) if isinstance(payload, AnomalyReport) else None,
            evidence=evidence,
            provenance=self._provenance(spec, result),
            query_id=result.query_id,
            warnings=self._warnings(spec, result, check.warnings),
            limitations=list(result.limitations),
        )
        return output, call.usage

    # ------------------------------------------------------------------------------------------ helpers
    def _screen(self, arguments: Any) -> InjectionScan:
        """Screen text parameters (not SQL, which the SQL validator parses) with the Phase 5 detector."""
        texts = [v for k, v in _text_values(arguments) if k != "sql"]
        return self.detector.scan("\n".join(texts)) if texts else InjectionScan()

    def _provenance(self, spec: MCPToolSpec, result: ToolResult) -> MCPProvenance:
        return MCPProvenance(
            tool=spec.tool,
            source_layer=self.tools.definition(spec.name).source_layer,
            operation=str(result.arguments.get("operation", spec.tool)),
            call_id=result.call_id,
            arguments=result.arguments,
            query_ids=result.query_ids,
            source_tables=result.source_tables,
            calculation=result.calculation,
            executed_at=result.finished_at,
            execution_time_ms=result.execution_time_ms,
            attempts=result.attempts,
            dataset_version=self.db.dataset_version,
            as_of=self.as_of,
            toolset_version=spec.version,
        )

    @staticmethod
    def _warnings(spec: MCPToolSpec, result: ToolResult, evidence_warnings: list[str]) -> list[str]:
        warnings = [w for w in [_KIND_WARNINGS[spec.output_kind], *evidence_warnings] if w]
        if result.status != "ok" and not evidence_warnings:
            warnings.append(f"The tool returned status '{result.status}': the result may be empty or partial.")
        if result.attempts > 1:
            warnings.append(f"The tool succeeded after {result.attempts} attempts (transient failure retried).")
        return warnings

    def _error(
        self,
        run_id: str,
        tool_name: str,
        spec: MCPToolSpec | None,
        code: str,
        detail: str | None = None,
    ) -> MCPToolOutput:
        return MCPToolOutput(
            tool_name=tool_name,
            request_id=run_id,
            status="error",
            output_kind=spec.output_kind if spec else "observed",
            error=mcp_error(code, detail),
        )

    def _bounded(
        self, run_id: str, output: MCPToolOutput, events: list[SecurityEvent]
    ) -> tuple[MCPToolOutput, dict[str, Any]]:
        """Redact the output and keep it within the response size limit.

        Bulk data goes first: the full raw result (the evidence, provenance and forecast/anomaly
        views stay), then evidence items from the end. If it still does not fit, the call fails
        with RESOURCE_LIMIT rather than returning a silently incomplete answer.
        """
        limit = self.config.max_response_bytes
        payload = _redacted(output)
        if _size(payload) <= limit:
            return output, payload
        note = f"Response size limit ({limit} bytes): the full result was omitted; evidence and provenance are kept."
        trimmed = output.model_copy(update={"result": None, "truncated": True, "warnings": [*output.warnings, note]})
        payload = _redacted(trimmed)
        while _size(payload) > limit and len(trimmed.evidence) > 1:
            keep = len(trimmed.evidence) // 2
            warning = f"Evidence trimmed to the first {keep} items to respect the response size limit."
            trimmed = trimmed.model_copy(
                update={
                    "evidence": trimmed.evidence[:keep],
                    "warnings": [*(w for w in trimmed.warnings if not w.startswith("Evidence trimmed")), warning],
                }
            )
            payload = _redacted(trimmed)
        fits = _size(payload) <= limit
        events.append(
            self._event(
                run_id,
                "response_truncated",
                Severity.INFO if fits else Severity.WARNING,
                output.tool_name,
                "trim" if fits else "stop",
                f"Response trimmed to fit {limit} bytes." if fits else f"The response exceeds {limit} bytes.",
            )
        )
        if fits:
            return trimmed, payload
        error = self._error(run_id, output.tool_name, self.tools.spec(output.tool_name), "response_too_large")
        return error, _redacted(error)

    @staticmethod
    def _event(
        run_id: str,
        event_type: SecurityEventType,
        severity: Severity,
        action: str,
        decision: Any,
        reason: str,
        **details: Any,
    ) -> SecurityEvent:
        return security_event(
            run_id,
            event_type,
            severity,
            component="mcp_adapter",
            action=action,
            decision=decision,
            reason=reason,
            **details,
        )


def _without_query_text(value: Any) -> Any:
    """Drop the internal SQL text of the analytics layer; query IDs, row counts and lineage stay."""
    if isinstance(value, dict):
        return {k: _without_query_text(v) for k, v in value.items() if k != "sql"}
    if isinstance(value, list):
        return [_without_query_text(v) for v in value]
    return value


def _public_result(payload: BaseModel) -> dict[str, Any]:
    """The typed result as JSON without internal SQL text.

    A ``run_safe_sql`` result keeps its top-level ``sql``: that is the caller's own query, as
    validated and limited.
    """
    dumped: dict[str, Any] = payload.model_dump(mode="json")
    public: dict[str, Any] = _without_query_text(dumped)
    if isinstance(payload, SQLResult):
        public["sql"] = dumped["sql"]
    return public


def _text_values(value: Any, key: str = "") -> list[tuple[str, str]]:
    """Every string in the arguments (dict keys included) with the name of the field it belongs to."""
    if isinstance(value, str):
        return [(key, value)]
    if isinstance(value, dict):
        found: list[tuple[str, str]] = []
        for k, v in value.items():
            found += [(key or str(k), str(k))] if isinstance(k, str) else []
            found += _text_values(v, key or str(k))
        return found
    if isinstance(value, (list, tuple)):
        return [item for v in value for item in _text_values(v, key)]
    return []


def _redacted(output: MCPToolOutput) -> dict[str, Any]:
    payload: dict[str, Any] = redact_value(output.model_dump(mode="json"))
    return payload


def _size(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
