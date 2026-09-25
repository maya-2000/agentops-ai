"""The LangGraph state machine of the analytics agent, with the Phase 5 security boundaries.

    question_received -> understand_question -> validate_request -> plan_investigation
        -> execute_tools (loops once per tool call) -> collect_evidence
        -> [plan_investigation again for evidence-driven follow-up, bounded]
        -> validate_evidence -> generate_response -> validate_response -> done

Failure states: unsupported_request, insufficient_evidence, tool_error, validation_failure and
planning_failure. Each node records its transition and chooses the next node explicitly
(``state.route``); conditional edges only allow the listed targets.

Security boundaries (Phase 5). Each one records its decisions as ``SecurityEvent`` objects:

- ``question_received``: input guard (type, size, control characters, secret redaction) and
  prompt-injection screening (``block`` -> unsupported_request; ``restrict`` -> SQL privilege
  revoked for the run).
- ``understand_question`` / ``plan_investigation`` / ``generate_response``: the model's output is
  untrusted. It is parsed, schema-validated and checked, and retried only within the retry
  budget. The context shown to the model is prioritised and size-bounded. Model calls have a
  timeout.
- ``validate_request``: input-guard limits on the understanding, then the vocabulary and coverage
  validation.
- ``plan_investigation``: the plan validator authorizes every proposed step
  (``ToolAuthorizationPolicy``). Security denials are not retried and the run fails closed.
- ``execute_tools``:
  - each call is authorized again immediately before it runs and charged to the run budget;
  - it runs under a tool deadline, with SQL under its own deadline;
  - retries follow the retry policy;
  - its output is validated before it can become evidence.
- ``validate_evidence``: evidence integrity (fingerprints) and the claim checks.
- ``validate_response``:
  - the output guardrails (numbers, causality, forecast and anomaly wording, recommendations);
  - bounded regeneration, then safe shortening for length only;
  - otherwise the run fails closed.
- ``done`` / failure states: user-facing text is redacted, and errors are shown only as safe
  categories.

Bounded execution: the run budget (tool calls, SQL calls and rows, retries, model calls,
context), a wall clock checked before every tool call, and a LangGraph recursion limit.

The runtime (database, tools, model client, policies, configuration) is closed over by the node
functions and never enters the state.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from functools import partial
from typing import Any, TypeVar

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from app.agent.config import AgentConfig
from app.agent.findings import build_claims
from app.agent.observability import log_event
from app.agent.records import AgentError, AgentStatus, InvestigationPlan, LLMCallRecord, PlanStep, ToolCallRecord
from app.agent.request import validate_understanding
from app.agent.response import LIMIT_MESSAGE, build_caveats, build_response, failure_response, redact_response
from app.agent.state import AgentState
from app.analytics.dimensions import DIMENSIONS
from app.analytics.errors import InvalidRequestError
from app.analytics.executor import QueryRunner
from app.analytics.kpis import list_kpi_definitions
from app.database.base import Database
from app.database.deadline import execution_deadline
from app.evidence.builder import build_evidence, evidence_summary
from app.evidence.models import EvidenceGraph
from app.evidence.validation import ResponseValidationResult, validate_evidence, validate_response
from app.llm.base import LLMClient, LLMError, LLMRequest, LLMTask
from app.llm.deterministic.composition import compose
from app.llm.prompts import SYSTEM_PROMPTS, render_prompt
from app.llm.schemas import (
    PLAN_SCHEMA,
    RESPONSE_SCHEMA,
    UNDERSTANDING_SCHEMA,
    DraftItemOutput,
    Intent,
    PlanOutput,
    ResponseDraftOutput,
    UnderstandingOutput,
)
from app.security.authorization import (
    SQL_TOOL,
    AuthorizationContext,
    AuthorizationDecision,
    ToolAuthorizationPolicy,
    ToolPermissions,
)
from app.security.budget import (
    RunBudget,
    charge_model_call,
    charge_tool_call,
    mark_exhausted,
    remaining_tool_calls,
)
from app.security.context import ContextTooLargeError, fit_context, prioritised_claims, prioritised_evidence_ids
from app.security.data_policy import default_exposure_policy
from app.security.errors import safe_message, sanitize_detail
from app.security.events import SecurityEvent, SecurityEventType, Severity, security_event
from app.security.input_guard import InputGuard, understanding_output_problems
from app.security.output_guard import ToolOutputValidator, shrink_draft
from app.security.plan_validator import PlanValidation, PlanValidator
from app.security.retry import RetryPolicy, RetryRecord
from app.security.timeouts import CallTimeoutError, call_with_timeout
from app.timeseries.metrics import SERIES_METRIC_KEYS
from app.tools.base import ToolContext, ToolError, ToolRequest, ToolResult
from app.tools.registry import ToolRegistry
from app.tools.results import SQLResult

NODES = (
    "question_received",
    "understand_question",
    "validate_request",
    "plan_investigation",
    "execute_tools",
    "collect_evidence",
    "validate_evidence",
    "generate_response",
    "validate_response",
    "done",
)
FAILURE_NODES: tuple[AgentStatus, ...] = (
    "unsupported_request",
    "insufficient_evidence",
    "tool_error",
    "validation_failure",
    "planning_failure",
)
FOLLOW_UP_INTENTS = (
    Intent.REVENUE_INVESTIGATION,
    Intent.CUSTOMER_INVESTIGATION,
    Intent.SUPPORT_ANALYSIS,
    Intent.MIXED_INVESTIGATION,
)
BLOCKED_MESSAGE = (
    "The request asks for something the agent is not permitted to do (such as revealing secrets, "
    "instructions, hidden or ground-truth data, reading files, running code or changing its own rules), "
    "so it was not processed."
)
TRIMMABLE_CONTEXT: dict[LLMTask, tuple[str, ...]] = {
    LLMTask.UNDERSTAND: (),
    LLMTask.PLAN: ("evidence", "executed_steps"),
    LLMTask.RESPOND: ("evidence", "claims"),
}
_DENIAL_EVENTS: dict[str, SecurityEventType] = {
    "unsafe_sql": "sql_rejected",
    "budget_exceeded": "budget_exceeded",
    "unknown_tool": "tool_denied",
    "tool_disabled": "tool_denied",
    "tool_not_permitted": "tool_denied",
    "sql_not_permitted": "tool_denied",
    "prerequisite_missing": "tool_denied",
}

ModelT = TypeVar("ModelT", bound=BaseModel)


class NonRetryableOutputError(Exception):
    """A model output was rejected for a reason that another attempt must not be allowed to retry."""


def business_context(product_features: list[str]) -> dict[str, Any]:
    """The vocabulary the understanding step may use (schema-level; no business values or answers)."""
    values: dict[str, list[str]] = {
        key: list(spec.allowed_values) for key, spec in DIMENSIONS.items() if spec.allowed_values
    }
    values["product_feature"] = list(product_features)
    return {
        "intents": [i.value for i in Intent],
        "metrics": [{"key": d.key, "name": d.name, "unit": d.unit} for d in list_kpi_definitions()],
        "series_metrics": list(SERIES_METRIC_KEYS),
        "dimensions": list(DIMENSIONS),
        "dimension_values": values,
        "period_specs": [
            "last_month",
            "previous_month",
            "last_quarter",
            "previous_quarter",
            "last_year",
            "ytd",
            "trailing_N_months",
            "YYYY-MM",
            "YYYY-Qn",
            "YYYY",
        ],
    }


@dataclass
class LLMOutcome:
    """The result of one bounded model step (all attempts)."""

    parsed: Any
    record: LLMCallRecord
    errors: list[str]
    retries: list[RetryRecord] = field(default_factory=list)
    events: list[SecurityEvent] = field(default_factory=list)
    context_items: int = 0
    prompt_chars: int = 0
    fatal: bool = False  # stopped by a non-retryable rejection


class AgentRuntime:
    """Everything a node needs that must not live in the state, including the security policies."""

    def __init__(self, db: Database, llm: LLMClient, config: AgentConfig, registry: ToolRegistry, as_of: date):
        self.db = db
        self.llm = llm
        self.config = config
        self.registry = registry
        self.as_of = as_of
        self.exposure = default_exposure_policy()
        self.permissions = ToolPermissions(disabled=config.disabled_tools)
        self.authorization = ToolAuthorizationPolicy(registry, config, self.permissions)
        self.plan_validator = PlanValidator(self.authorization)
        self.input_guard = InputGuard(config)
        self.output_validator = ToolOutputValidator(config, self.exposure)
        self.retry_policy = RetryPolicy(config.max_retries)
        self.run_budget = RunBudget.from_limits(config)
        self.tool_context = ToolContext(
            db=db,
            as_of=as_of,
            sql_row_limit=config.sql_row_limit,
            sql_timeout_seconds=config.sql_timeout_seconds,
            sql_limits=self.authorization.sql_limits,
        )
        self._business_context: dict[str, Any] | None = None

    def business_context(self) -> dict[str, Any]:
        """Observable vocabulary for question understanding: KPIs, dimensions and their values."""
        if self._business_context is None:
            rows = QueryRunner(self.db, "business_context").run(
                "SELECT DISTINCT feature_name FROM product_features ORDER BY feature_name"
            )
            self._business_context = business_context([row[0] for row in rows.rows])
        return self._business_context

    def call_llm(
        self,
        run_id: str,
        task: LLMTask,
        context: dict[str, Any],
        model: type[ModelT],
        schema: dict[str, Any],
        check: Callable[[ModelT], list[str]] | None = None,
    ) -> LLMOutcome:
        """Ask the model, parse and validate its JSON; retry (bounded, by policy) with the errors as feedback."""
        errors: list[str] = []
        retries: list[RetryRecord] = []
        events: list[SecurityEvent] = []
        attempts = 0
        usage_in = usage_out = None
        items = chars = 0
        fatal = False
        for attempt in range(1, self.config.max_retries + 2):
            attempts = attempt
            request_context = dict(context)
            if errors:
                request_context["feedback"] = errors[-3:]
            try:
                fitted, usage = fit_context(
                    request_context,
                    lambda c: render_prompt(task, c),
                    trimmable=TRIMMABLE_CONTEXT[task],
                    max_items=self.config.max_context_items,
                    max_chars=self.config.max_context_chars,
                )
            except ContextTooLargeError as exc:
                errors.append(f"context too large: {exc}")
                events.append(self._event(run_id, "context_truncated", Severity.HIGH, task, "stop", str(exc)))
                fatal = True
                break
            items, chars = usage.items, usage.chars
            if usage.dropped:
                events.append(
                    self._event(
                        run_id,
                        "context_truncated",
                        Severity.INFO,
                        task,
                        "trim",
                        f"{usage.dropped} lower-priority context items dropped",
                        dropped=usage.dropped,
                    )
                )
            request = LLMRequest(
                task=task,
                system=SYSTEM_PROMPTS[task],
                prompt=usage.prompt,
                context=fitted,
                output_schema=schema,
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
            )
            code: str
            try:
                response = call_with_timeout(partial(self.llm.generate, request), self.config.llm_timeout_seconds)
            except CallTimeoutError as exc:
                errors.append(f"model call timed out: {exc}")
                code = "llm_timeout"
                events.append(self._event(run_id, "timeout", Severity.WARNING, task, "retry", str(exc)))
            except LLMError as exc:
                errors.append(f"provider error: {sanitize_detail(str(exc))}")
                code = "provider_error_retryable" if exc.retryable else "provider_error"
            else:
                usage_in, usage_out = response.usage.input_tokens, response.usage.output_tokens
                try:
                    parsed = model.model_validate(json.loads(response.content))
                    problems = check(parsed) if check else []
                except (json.JSONDecodeError, ValidationError, InvalidRequestError, TypeError) as exc:
                    problems = [f"invalid {task.value} output: {str(exc).splitlines()[0][:300]}"]
                except NonRetryableOutputError as exc:
                    errors.append(str(exc))
                    fatal = True
                    break
                if not problems:
                    record = LLMCallRecord(
                        task=task.value,
                        provider=self.llm.provider,
                        model=self.llm.model,
                        attempts=attempts,
                        success=True,
                        input_tokens=usage_in,
                        output_tokens=usage_out,
                    )
                    return LLMOutcome(parsed, record, errors, retries, events, items, chars)
                errors.extend(problems)
                code = "model_output_invalid"
                events.append(
                    self._event(
                        run_id,
                        "model_output_rejected",
                        Severity.WARNING,
                        task,
                        "retry",
                        problems[0],
                        problem_count=len(problems),
                    )
                )
            retry = self.retry_policy.should_retry(code, attempt)
            retries.append(
                RetryRecord(stage=task.value, attempt=attempt, code=code, decision="retry" if retry else "stop")
            )
            if not retry:
                break
        record = LLMCallRecord(
            task=task.value,
            provider=self.llm.provider,
            model=self.llm.model,
            attempts=attempts,
            success=False,
            input_tokens=usage_in,
            output_tokens=usage_out,
            error=sanitize_detail(errors[-1]) if errors else "unknown error",
        )
        return LLMOutcome(None, record, errors, retries, events, items, chars, fatal)

    @staticmethod
    def _event(
        run_id: str,
        event_type: SecurityEventType,
        severity: Severity,
        task: LLMTask,
        decision: Any,
        reason: str,
        **details: Any,
    ) -> SecurityEvent:
        return security_event(
            run_id,
            event_type,
            severity,
            component="llm",
            action=task.value,
            decision=decision,
            reason=reason,
            **details,
        )


def build_graph(runtime: AgentRuntime) -> Any:
    cfg = runtime.config

    def go(state: AgentState, node: str, route: str, **updates: Any) -> dict[str, Any]:
        log_event(state.run_id, "transition", node=node, route=route)
        return {**updates, "route": route, "transitions": [*state.transitions, node]}

    def llm_updates(state: AgentState, outcome: LLMOutcome) -> dict[str, Any]:
        usage = charge_model_call(
            state.budget_usage,
            attempts=outcome.record.attempts,
            context_items=outcome.context_items,
            prompt_chars=outcome.prompt_chars,
        )
        return {
            "llm_calls": [*state.llm_calls, outcome.record],
            "llm_retries": state.llm_retries + outcome.record.attempts - 1,
            "retries": [*state.retries, *outcome.retries],
            "security_events": [*state.security_events, *outcome.events],
            "budget_usage": usage,
        }

    def error(state: AgentState, stage: str, code: str, message: str) -> list[AgentError]:
        return [*state.errors, AgentError(stage=stage, code=code, message=sanitize_detail(message))]

    def event(
        state: AgentState,
        event_type: SecurityEventType,
        severity: Severity,
        *,
        component: str,
        action: str,
        decision: Any,
        reason: str,
        **details: Any,
    ) -> SecurityEvent:
        return security_event(
            state.run_id,
            event_type,
            severity,
            component=component,
            action=action,
            decision=decision,
            reason=reason,
            **details,
        )

    def auth_context(state: AgentState, iteration: int) -> AuthorizationContext:
        return AuthorizationContext(
            intent=state.request.intent if state.request else None,
            sql_permitted=state.sql_permitted,
            iteration=iteration,
            has_prior_evidence=bool(state.evidence_graph.evidence),
        )

    def denial_event(state: AgentState, decision: AuthorizationDecision, component: str) -> SecurityEvent:
        event_type = _DENIAL_EVENTS.get(decision.code or "", "argument_rejected")
        return event(
            state,
            event_type,
            decision.severity,
            component=component,
            action=decision.tool_name,
            decision="deny",
            reason=f"{decision.code}: {decision.reason}",
            code=decision.code,
        )

    # ------------------------------------------------------------------ main path
    def question_received(state: AgentState) -> dict[str, Any]:
        events: list[SecurityEvent] = []
        if state.question_redacted:
            events.append(
                event(
                    state,
                    "secret_redacted",
                    Severity.HIGH,
                    component="input_guard",
                    action="question",
                    decision="redact",
                    reason="A secret-like value was removed from the question before processing.",
                )
            )
        if not state.question_is_text:
            events.append(
                event(
                    state,
                    "input_rejected",
                    Severity.WARNING,
                    component="input_guard",
                    action="question",
                    decision="deny",
                    reason="The question is not text.",
                )
            )
            return go(
                state,
                "question_received",
                "unsupported_request",
                status_message="The question must be text.",
                security_events=[*state.security_events, *events],
            )
        check = runtime.input_guard.validate_question(state.question)
        if check.outcome == "rejected":
            events.append(
                event(
                    state,
                    "input_rejected",
                    Severity.WARNING,
                    component="input_guard",
                    action="question",
                    decision="deny",
                    reason=check.reason or "rejected",
                    code=check.code,
                )
            )
            return go(
                state,
                "question_received",
                "unsupported_request",
                normalized_question=check.question,
                status_message=check.reason,
                security_events=[*state.security_events, *events],
            )
        scan = check.scan
        if check.outcome == "blocked":
            events.append(
                event(
                    state,
                    "suspicious_prompt",
                    scan.severity or Severity.HIGH,
                    component="prompt_injection",
                    action="screen",
                    decision="deny",
                    reason="Blocked request categories: " + ", ".join(scan.categories),
                    categories=scan.categories,
                    signals=scan.signals,
                )
            )
            events.append(
                event(
                    state,
                    "unsupported_request",
                    scan.severity or Severity.HIGH,
                    component="input_guard",
                    action="question",
                    decision="deny",
                    reason="Refused before any model or tool call.",
                )
            )
            return go(
                state,
                "question_received",
                "unsupported_request",
                normalized_question=check.question,
                status_message=BLOCKED_MESSAGE,
                input_screen=scan,
                sql_permitted=False,
                security_events=[*state.security_events, *events],
            )
        sql_permitted = True
        if scan.verdict == "restrict":
            sql_permitted = False
            events.append(
                event(
                    state,
                    "suspicious_prompt",
                    scan.severity or Severity.WARNING,
                    component="prompt_injection",
                    action="screen",
                    decision="restrict",
                    reason="Suspicious instruction patterns: " + ", ".join(scan.categories),
                    categories=scan.categories,
                    signals=scan.signals,
                )
            )
            events.append(
                event(
                    state,
                    "privileges_reduced",
                    Severity.WARNING,
                    component="tool_authorization",
                    action=SQL_TOOL,
                    decision="restrict",
                    reason="Ad-hoc SQL disabled for this run; the question is handled as data.",
                )
            )
        return go(
            state,
            "question_received",
            "understand_question",
            normalized_question=check.question,
            input_screen=scan,
            sql_permitted=sql_permitted,
            security_events=[*state.security_events, *events],
        )

    def understand_question(state: AgentState) -> dict[str, Any]:
        context = {
            "question": state.normalized_question,
            "as_of": runtime.as_of.isoformat(),
            **runtime.business_context(),
        }

        def check(u: UnderstandingOutput) -> list[str]:
            return understanding_output_problems(u, cfg)

        outcome = runtime.call_llm(
            state.run_id, LLMTask.UNDERSTAND, context, UnderstandingOutput, UNDERSTANDING_SCHEMA, check
        )
        updates = llm_updates(state, outcome)
        if outcome.parsed is None:
            return go(
                state,
                "understand_question",
                "planning_failure",
                status_message="The question could not be interpreted.",
                errors=error(state, "understand_question", "llm_output_invalid", "; ".join(outcome.errors[-2:])),
                **updates,
            )
        return go(state, "understand_question", "validate_request", understanding=outcome.parsed, **updates)

    def validate_request_node(state: AgentState) -> dict[str, Any]:
        assert state.understanding is not None
        violations = runtime.input_guard.validate_understanding_output(state.understanding)
        if violations:
            first = violations[0]
            route = "insufficient_evidence" if first.code == "invalid_period" else "unsupported_request"
            message = f"The request exceeds the supported input limits ({first.field}: {first.message})."
            if first.code == "invalid_period":
                message = f"The period could not be interpreted ({first.message})."
            flagged = event(
                state,
                "input_rejected",
                Severity.WARNING,
                component="input_guard",
                action="understanding",
                decision="deny",
                reason="; ".join(f"{v.field}: {v.message}" for v in violations[:3]),
                code=first.code,
            )
            return go(
                state,
                "validate_request",
                route,
                status_message=message,
                security_events=[*state.security_events, flagged],
            )
        validation = validate_understanding(
            state.understanding, as_of=runtime.as_of, coverage=runtime.tool_context.kpi_service.coverage()
        )
        route = {
            "valid": "plan_investigation",
            "unsupported": "unsupported_request",
            "clarify": "insufficient_evidence",
            "insufficient": "insufficient_evidence",
        }[validation.outcome]
        updates: dict[str, Any] = {}
        if validation.outcome == "unsupported":
            updates["security_events"] = [
                *state.security_events,
                event(
                    state,
                    "unsupported_request",
                    Severity.INFO,
                    component="request_validation",
                    action="understanding",
                    decision="deny",
                    reason=validation.message or "unsupported",
                ),
            ]
        return go(
            state, "validate_request", route, request=validation.request, status_message=validation.message, **updates
        )

    def plan_investigation(state: AgentState) -> dict[str, Any]:
        assert state.request is not None
        iteration = state.planning_iterations + 1
        remaining = remaining_tool_calls(runtime.run_budget, state.budget_usage)
        if remaining <= 0:
            route = "validate_evidence" if state.tool_calls else "insufficient_evidence"
            return go(
                state,
                "plan_investigation",
                route,
                limit_reached=True,
                status_message=LIMIT_MESSAGE,
                budget_usage=mark_exhausted(state.budget_usage, "tool_calls"),
            )
        executed = {(c.tool_name, json.dumps(c.input, sort_keys=True)) for c in state.tool_calls}
        context_for_auth = auth_context(state, iteration)
        attempts: list[PlanValidation] = []

        def check(plan: PlanOutput) -> list[str]:
            validation = runtime.plan_validator.validate(
                plan,
                context=context_for_auth,
                budget=runtime.run_budget,
                usage=state.budget_usage,
                executed=executed,
            )
            attempts.append(validation)
            denial = validation.security_denial
            if denial is not None:
                raise NonRetryableOutputError(f"step denied ({denial.code}): {denial.reason}")
            return validation.problems

        permitted = runtime.permissions.permitted(state.request.intent)
        tools = [
            entry
            for entry in runtime.registry.catalog()
            if entry["tool_name"] in permitted and (state.sql_permitted or entry["tool_name"] != SQL_TOOL)
        ]
        evidence_order = prioritised_evidence_ids(state.evidence_graph)
        context = {
            "question": state.normalized_question,
            "request": state.request.model_dump(mode="json"),
            "iteration": iteration,
            "remaining_tool_calls": remaining,
            "tools": tools,
            "executed_steps": [
                {"tool_name": c.tool_name, "arguments": c.input, "success": c.success} for c in state.tool_calls
            ],
            "evidence": [
                {
                    **evidence_summary(state.evidence_graph.evidence[e]),
                    "operation": state.evidence_graph.evidence[e].operation,
                    "attributes": state.evidence_graph.evidence[e].attributes,
                }
                for e in evidence_order
            ],
        }
        outcome = runtime.call_llm(state.run_id, LLMTask.PLAN, context, PlanOutput, PLAN_SCHEMA, check)
        updates = {**llm_updates(state, outcome), "planning_iterations": iteration}
        denial_events = [denial_event(state, d, "plan_validator") for v in attempts for d in v.denials]
        updates["security_events"] = [*updates["security_events"], *denial_events]
        budget_exceeded = bool(attempts) and attempts[-1].budget_exceeded
        if outcome.parsed is None:
            if budget_exceeded:
                updates["security_events"].append(
                    event(
                        state,
                        "budget_exceeded",
                        Severity.WARNING,
                        component="plan_validator",
                        action="plan",
                        decision="stop",
                        reason="The proposed plan exceeds the step or tool-call budget.",
                    )
                )
                updates["budget_usage"] = mark_exhausted(updates["budget_usage"], "plan_steps")
            if iteration > 1:  # follow-up planning is optional: continue with the evidence already collected
                return go(
                    state,
                    "plan_investigation",
                    "validate_evidence",
                    limit_reached=state.limit_reached or budget_exceeded,
                    errors=error(state, "plan_investigation", "follow_up_plan_invalid", "; ".join(outcome.errors[-2:])),
                    **updates,
                )
            if budget_exceeded and not outcome.fatal:
                return go(
                    state,
                    "plan_investigation",
                    "insufficient_evidence",
                    limit_reached=True,
                    status_message=LIMIT_MESSAGE,
                    **updates,
                )
            updates["security_events"].append(
                event(
                    state,
                    "plan_rejected",
                    Severity.HIGH if outcome.fatal else Severity.WARNING,
                    component="plan_validator",
                    action="plan",
                    decision="deny",
                    reason="; ".join(outcome.errors[-2:]) or "no valid plan",
                )
            )
            message = (
                "The proposed investigation used an operation that is not permitted, so no tools were run."
                if outcome.fatal
                else "No valid investigation plan could be produced from the allowed tools."
            )
            return go(
                state,
                "plan_investigation",
                "planning_failure",
                status_message=message,
                errors=error(state, "plan_investigation", "plan_invalid", "; ".join(outcome.errors[-2:])),
                **updates,
            )
        accepted = attempts[-1].steps
        steps = [
            PlanStep(
                step_id=f"S{iteration}.{index}",
                tool_name=s.tool_name,
                arguments=s.arguments,
                purpose=s.purpose,
                iteration=iteration,
            )
            for index, s in enumerate(accepted, start=1)
        ]
        log_event(state.run_id, "plan", iteration=iteration, steps=[s.tool_name for s in steps])
        if not steps:
            return go(state, "plan_investigation", "validate_evidence", **updates)
        plan = InvestigationPlan(iteration=iteration, steps=steps, rationale=outcome.parsed.rationale[:500])
        return go(
            state,
            "plan_investigation",
            "execute_tools",
            plans=[*state.plans, plan],
            pending_steps=steps,
            **updates,
        )

    def execute_tools(state: AgentState) -> dict[str, Any]:
        pending = list(state.pending_steps)
        events: list[SecurityEvent] = []
        usage = state.budget_usage
        elapsed = (datetime.now(UTC) - state.started_at).total_seconds()
        if len(state.tool_calls) >= cfg.max_tool_calls or elapsed > cfg.max_run_seconds:
            resource = "runtime" if elapsed > cfg.max_run_seconds else "tool_calls"
            events.append(
                event(
                    state,
                    "budget_exceeded",
                    Severity.WARNING,
                    component="run_budget",
                    action="execute_tools",
                    decision="stop",
                    reason=f"The {resource} budget is exhausted; remaining steps were not run.",
                    pending_steps=len(pending),
                )
            )
            return go(
                state,
                "execute_tools",
                "collect_evidence",
                pending_steps=[],
                limit_reached=True,
                budget_usage=mark_exhausted(usage, resource),
                security_events=[*state.security_events, *events],
            )
        if not pending:
            return go(state, "execute_tools", "collect_evidence")
        step = pending.pop(0)
        call_id = f"T{len(state.tool_calls) + 1}"
        is_sql = step.tool_name == SQL_TOOL
        decision = runtime.authorization.authorize(
            step.tool_name,
            step.arguments,
            context=auth_context(state, step.iteration),
            budget=runtime.run_budget,
            usage=usage,
        )
        retries: list[RetryRecord] = []
        attempts = 0
        if not decision.allowed:
            events.append(denial_event(state, decision, "tool_authorization"))
            now = datetime.now(UTC)
            result = ToolResult(
                call_id=call_id,
                tool_name=step.tool_name,
                arguments=step.arguments,
                success=False,
                status="error",
                error=ToolError(code=decision.code or "unauthorized_tool", message=decision.reason, retryable=False),
                message=decision.reason,
                started_at=now,
                finished_at=now,
                execution_time_ms=0.0,
            )
            if decision.code == "budget_exceeded":
                usage = mark_exhausted(usage, "tool_budget")
                pending = []  # stop: no further step can be charged either
        else:
            events.append(
                event(
                    state,
                    "tool_authorized",
                    Severity.INFO,
                    component="tool_authorization",
                    action=step.tool_name,
                    decision="allow",
                    reason="All authorization checks passed.",
                    checks=decision.checks_passed,
                )
            )
            arguments = decision.arguments or step.arguments
            while True:
                attempts += 1
                clock = time.perf_counter()
                with execution_deadline(cfg.tool_timeout_seconds):
                    result = runtime.registry.execute(
                        ToolRequest(
                            call_id=call_id, tool_name=step.tool_name, arguments=arguments, purpose=step.purpose
                        ),
                        runtime.tool_context,
                    )
                took = time.perf_counter() - clock
                if result.success and took > cfg.tool_timeout_seconds:
                    result = _failed(result, "timeout", f"The tool exceeded its {cfg.tool_timeout_seconds:g}s limit.")
                code = result.error.code if result.error else None
                if code == "timeout":
                    events.append(
                        event(
                            state,
                            "timeout",
                            Severity.WARNING,
                            component="tool_execution",
                            action=step.tool_name,
                            decision="stop",
                            reason="The tool call exceeded its time limit; no result was used.",
                        )
                    )
                if result.success:
                    break
                retry = runtime.retry_policy.should_retry(code, attempts)
                retries.append(
                    RetryRecord(
                        stage=f"execute_tools:{step.tool_name}",
                        attempt=attempts,
                        code=code or "unknown",
                        decision="retry" if retry else "stop",
                    )
                )
                if not retry:
                    break
                events.append(
                    event(
                        state,
                        "retry",
                        Severity.WARNING,
                        component="tool_execution",
                        action=step.tool_name,
                        decision="retry",
                        reason=f"Transient failure ({code}); attempt {attempts + 1}.",
                    )
                )
            violations = runtime.output_validator.validate(result)
            if violations:
                events.append(
                    event(
                        state,
                        "tool_output_rejected",
                        Severity.HIGH,
                        component="output_guard",
                        action=step.tool_name,
                        decision="deny",
                        reason="; ".join(f"{v.field}: {v.message}" for v in violations[:3]),
                    )
                )
                data_policy = any(v.code == "data_policy" for v in violations)
                result = _failed(result, "data_policy" if data_policy else "invalid_tool_output", violations[0].message)
            sql_rows = result.result.row_count if isinstance(result.result, SQLResult) else 0
            usage = charge_tool_call(usage, sql=is_sql, sql_rows=sql_rows, retries=max(0, attempts - 1))
        result = result.model_copy(update={"attempts": max(attempts, 1)})
        safe_error = (
            ToolError(
                code=result.error.code,
                message=sanitize_detail(result.error.message),
                retryable=result.error.retryable,
            )
            if result.error
            else None
        )
        record = ToolCallRecord(
            call_id=call_id,
            step_id=step.step_id,
            tool_name=step.tool_name,
            input=step.arguments,
            start_time=result.started_at,
            end_time=result.finished_at,
            execution_time_ms=result.execution_time_ms,
            success=result.success,
            status=result.status,
            attempts=max(attempts, 1),
            result_summary=_summary(result),
            query_ids=result.query_ids,
            error=safe_error,
        )
        log_event(
            state.run_id,
            "tool_call",
            tool_name=step.tool_name,
            call_id=call_id,
            success=result.success,
            status=result.status,
            attempts=max(attempts, 1),
            error_code=result.error.code if result.error else None,
            execution_time_ms=result.execution_time_ms,
            query_ids=result.query_ids,
        )
        calls = [*state.tool_calls, record]
        more = bool(pending) and len(calls) < cfg.max_tool_calls
        limit = bool(pending) and not more
        return go(
            state,
            "execute_tools",
            "execute_tools" if more else "collect_evidence",
            pending_steps=pending if more else [],
            tool_calls=calls,
            tool_results=[*state.tool_results, result],
            current_step=state.current_step + 1,
            tool_retries=state.tool_retries + max(attempts, 1) - 1,
            limit_reached=state.limit_reached or limit or (decision.code == "budget_exceeded"),
            budget_usage=usage,
            retries=[*state.retries, *retries],
            security_events=[*state.security_events, *events],
        )

    def collect_evidence(state: AgentState) -> dict[str, Any]:
        assert state.request is not None
        graph = state.evidence_graph.model_copy(deep=True)
        processed = set(state.processed_call_ids)
        new_ids: dict[str, list[str]] = {}
        for result in state.tool_results:
            if result.call_id in processed:
                continue
            new_ids[result.call_id] = [e.evidence_id for e in build_evidence(result, graph)]
            processed.add(result.call_id)
        calls = [
            c.model_copy(update={"evidence_ids": c.evidence_ids + new_ids.get(c.call_id, [])}) for c in state.tool_calls
        ]
        build_claims(graph, state.request)
        log_event(state.run_id, "evidence", evidence_count=len(graph.evidence), claim_count=len(graph.claims))
        updates = {"evidence_graph": graph, "tool_calls": calls, "processed_call_ids": sorted(processed)}
        if state.tool_calls and not any(c.success for c in state.tool_calls):
            failed = ", ".join(f"{c.tool_name} ({c.error.code if c.error else 'error'})" for c in state.tool_calls)
            return go(
                state, "collect_evidence", "tool_error", status_message=f"Every tool call failed: {failed}.", **updates
            )
        can_follow_up = (
            state.request.intent in FOLLOW_UP_INTENTS
            and state.planning_iterations < cfg.max_planning_iterations
            and not state.limit_reached
        )
        if can_follow_up and len(state.tool_calls) >= cfg.max_tool_calls:
            # Follow-up planning was still allowed, but the tool budget is spent: say so rather than stop silently.
            return go(state, "collect_evidence", "validate_evidence", limit_reached=True, **updates)
        return go(state, "collect_evidence", "plan_investigation" if can_follow_up else "validate_evidence", **updates)

    def validate_evidence_node(state: AgentState) -> dict[str, Any]:
        graph = state.evidence_graph.model_copy(deep=True)
        successful = [c.call_id for c in state.tool_calls if c.success]
        failed = [
            f"{c.tool_name}: {safe_message(c.error.code if c.error else c.status)}"
            for c in state.tool_calls
            if not c.success
        ]
        events: list[SecurityEvent] = []
        tampered = graph.verify_integrity()
        if tampered:
            events.append(
                event(
                    state,
                    "evidence_integrity_failed",
                    Severity.CRITICAL,
                    component="evidence",
                    action="verify",
                    decision="deny",
                    reason=f"{len(tampered)} evidence items no longer match their fingerprint",
                    evidence_ids=tampered[:10],
                )
            )
        result = validate_evidence(graph, successful_call_ids=successful, failed_tools=failed)
        if not result.valid and result.unsupported_claim_ids:
            removed = list(result.unsupported_claim_ids)
            for claim_id in removed:
                graph.claims.pop(claim_id, None)
            events.append(
                event(
                    state,
                    "evidence_validation_failed",
                    Severity.WARNING,
                    component="evidence",
                    action="claims",
                    decision="deny",
                    reason=f"{len(removed)} unsupported claims removed before writing",
                    claim_ids=removed[:10],
                )
            )
            pruned = validate_evidence(graph, successful_call_ids=successful, failed_tools=failed)
            result = pruned.model_copy(
                update={"warnings": [*pruned.warnings, *(f"Removed unsupported claim: {e}" for e in result.errors)]}
            )
        log_event(
            state.run_id,
            "evidence_validation",
            valid=result.valid,
            error_count=len(result.errors),
            warning_count=len(result.warnings),
        )
        updates: dict[str, Any] = {"evidence_graph": graph, "evidence_validation": result}
        if result.valid:
            return go(
                state,
                "validate_evidence",
                "generate_response",
                security_events=[*state.security_events, *events],
                **updates,
            )
        events.append(
            event(
                state,
                "evidence_validation_failed",
                Severity.WARNING,
                component="evidence",
                action="answer",
                decision="deny",
                reason="No supported claim answers the question.",
            )
        )
        route = "tool_error" if failed and not successful else "insufficient_evidence"
        message = "No supported finding answers the question." + (
            f" Failed tools: {'; '.join(failed)}." if failed else ""
        )
        return go(
            state,
            "validate_evidence",
            route,
            status_message=message,
            security_events=[*state.security_events, *events],
            **updates,
        )

    def generate_response(state: AgentState) -> dict[str, Any]:
        assert state.request is not None
        graph = state.evidence_graph
        context: dict[str, Any] = {
            "question": state.normalized_question,
            "intent": state.request.intent.value,
            "max_response_chars": cfg.max_response_chars,
            "claims": prioritised_claims(_claims_context(graph)),
            "evidence": [evidence_summary(graph.evidence[e]) for e in prioritised_evidence_ids(graph)],
        }
        if state.response_validation is not None and not state.response_validation.valid:
            context["previous_errors"] = state.response_validation.errors[:5]
        outcome = runtime.call_llm(state.run_id, LLMTask.RESPOND, context, ResponseDraftOutput, RESPONSE_SCHEMA)
        updates: dict[str, Any] = llm_updates(state, outcome)
        parsed = outcome.parsed
        if parsed is None:
            parsed = ResponseDraftOutput.model_validate(compose(context))
            updates["errors"] = error(state, "generate_response", "llm_failed", "; ".join(outcome.errors[-2:]))
            updates["response_generated_by"] = "deterministic-fallback"
        else:
            updates["response_generated_by"] = f"{runtime.llm.provider}:{runtime.llm.model}"
        return go(state, "generate_response", "validate_response", response_draft=parsed, **updates)

    def validate_response_node(state: AgentState) -> dict[str, Any]:
        assert state.response_draft is not None
        result = validate_response(state.response_draft, state.evidence_graph, max_chars=cfg.max_response_chars)
        log_event(state.run_id, "response_validation", valid=result.valid, error_count=len(result.errors))
        if result.valid:
            return go(state, "validate_response", "done", response_validation=result)
        rejected = event(
            state,
            "output_validation_failed",
            Severity.WARNING,
            component="output_validation",
            action="response",
            decision="retry" if state.retry_count < cfg.max_retries else "deny",
            reason=_failed_checks(result),
            error_count=len(result.errors),
        )
        if state.retry_count < cfg.max_retries:
            return go(
                state,
                "validate_response",
                "generate_response",
                response_validation=result,
                retry_count=state.retry_count + 1,
                retries=[
                    *state.retries,
                    RetryRecord(
                        stage="generate_response",
                        attempt=state.retry_count + 1,
                        code="model_output_invalid",
                        decision="retry",
                    ),
                ],
                security_events=[*state.security_events, rejected],
            )
        if all("characters (limit" in e for e in result.errors):
            shrunk = shrink_draft(state.response_draft, cfg.max_response_chars)
            if shrunk is not None:
                recheck = validate_response(shrunk, state.evidence_graph, max_chars=cfg.max_response_chars)
                if recheck.valid:
                    trimmed = event(
                        state,
                        "response_truncated",
                        Severity.INFO,
                        component="output_validation",
                        action="response",
                        decision="trim",
                        reason="Lower-priority items were dropped to meet the response length limit.",
                    )
                    return go(
                        state,
                        "validate_response",
                        "done",
                        response_draft=shrunk,
                        response_validation=recheck,
                        response_trimmed=True,
                        security_events=[*state.security_events, rejected, trimmed],
                    )
        # The validator's messages quote the rejected text; they go to the trace, never into the response.
        return go(
            state,
            "validate_response",
            "validation_failure",
            response_validation=result,
            status_message=_failed_checks(result),
            errors=error(state, "validate_response", "response_invalid", "; ".join(result.errors[:5])),
            security_events=[
                *state.security_events,
                rejected.model_copy(update={"severity": Severity.HIGH, "decision": "deny"}),
            ],
        )

    def done(state: AgentState) -> dict[str, Any]:
        assert state.request is not None and state.response_draft is not None
        draft = state.response_draft
        status: AgentStatus = "insufficient_evidence" if state.request.causal_question else "completed"
        cited = [*draft.answer_claim_ids, *(c for item in draft.key_findings for c in item.claim_ids)]
        cited += [c for item in [*draft.interpretation, *draft.recommendations] for c in item.claim_ids]
        caveats = build_caveats(
            state.evidence_graph,
            cited,
            warnings=state.evidence_validation.warnings if state.evidence_validation else [],
            limit_reached=state.limit_reached,
            causal_question=state.request.causal_question,
            response_trimmed=state.response_trimmed,
        )
        response, redacted = redact_response(
            build_response(
                draft,
                state.evidence_graph,
                status=status,
                calls=state.tool_calls,
                assumptions=state.request.assumptions,
                caveats=caveats,
                generated_by=state.response_generated_by or "unknown",
            )
        )
        return finish(state, "done", response, status, redacted)

    def finish(state: AgentState, node: str, response: Any, status: AgentStatus, redacted: bool) -> dict[str, Any]:
        events = list(state.security_events)
        if redacted:
            events.append(
                event(
                    state,
                    "secret_redacted",
                    Severity.HIGH,
                    component="output_validation",
                    action="response",
                    decision="redact",
                    reason="Secret-like values were removed from the response.",
                )
            )
        usage = state.budget_usage.model_copy(update={"response_chars": _response_chars(response)})
        log_event(state.run_id, "final", status=status, tool_calls=len(state.tool_calls))
        return go(state, node, "end", response=response, status=status, security_events=events, budget_usage=usage)

    # ------------------------------------------------------------------ failure states
    class Terminal:
        """A failure state: builds a deterministic response for its status."""

        def __init__(self, status: AgentStatus):
            self.status = status

        def __call__(self, state: AgentState) -> dict[str, Any]:
            graph = state.evidence_graph
            findings: list[DraftItemOutput] = []
            if self.status in ("insufficient_evidence", "validation_failure") and graph.claims:
                draft = ResponseDraftOutput.model_validate(compose({"claims": _claims_context(graph)}))
                findings = (
                    [DraftItemOutput(text=draft.answer, claim_ids=draft.answer_claim_ids)] if draft.answer else []
                )
                findings += draft.key_findings[:4]
            message = state.status_message or ""
            response, redacted = redact_response(
                failure_response(
                    self.status,
                    message,
                    calls=state.tool_calls,
                    assumptions=state.request.assumptions if state.request else [],
                    graph=graph,
                    findings=findings,
                    caveats=[LIMIT_MESSAGE] if state.limit_reached and LIMIT_MESSAGE not in message else [],
                    coverage=runtime.tool_context.kpi_service.coverage(),
                )
            )
            return finish(state, self.status, response, self.status, redacted)

    graph = StateGraph(AgentState)
    graph.add_node("question_received", question_received)
    graph.add_node("understand_question", understand_question)
    graph.add_node("validate_request", validate_request_node)
    graph.add_node("plan_investigation", plan_investigation)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("collect_evidence", collect_evidence)
    graph.add_node("validate_evidence", validate_evidence_node)
    graph.add_node("generate_response", generate_response)
    graph.add_node("validate_response", validate_response_node)
    graph.add_node("done", done)
    for name in FAILURE_NODES:
        graph.add_node(name, Terminal(name))

    def route(state: AgentState) -> str:
        return state.route

    graph.add_edge(START, "question_received")
    graph.add_conditional_edges("question_received", route, ["understand_question", "unsupported_request"])
    graph.add_conditional_edges("understand_question", route, ["validate_request", "planning_failure"])
    graph.add_conditional_edges(
        "validate_request", route, ["plan_investigation", "unsupported_request", "insufficient_evidence"]
    )
    graph.add_conditional_edges(
        "plan_investigation",
        route,
        ["execute_tools", "validate_evidence", "planning_failure", "insufficient_evidence"],
    )
    graph.add_conditional_edges("execute_tools", route, ["execute_tools", "collect_evidence"])
    graph.add_conditional_edges("collect_evidence", route, ["plan_investigation", "validate_evidence", "tool_error"])
    graph.add_conditional_edges(
        "validate_evidence", route, ["generate_response", "insufficient_evidence", "tool_error"]
    )
    graph.add_conditional_edges("generate_response", route, ["validate_response"])
    graph.add_conditional_edges("validate_response", route, ["done", "generate_response", "validation_failure"])
    graph.add_edge("done", END)
    for name in FAILURE_NODES:
        graph.add_edge(name, END)
    return graph.compile()


def _failed(result: ToolResult, code: str, message: str) -> ToolResult:
    """Turn a result into a controlled failure: the output is discarded, never partially used."""
    return result.model_copy(
        update={
            "success": False,
            "status": "error",
            "result": None,
            "result_type": None,
            "error": ToolError(code=code, message=message, retryable=False),
            "message": message,
            "query_ids": [],
            "source_tables": [],
        }
    )


def _response_chars(response: Any) -> int:
    items = [*response.key_findings, *response.interpretation, *response.recommendations]
    return len(response.answer) + sum(len(i.text) for i in items)


def _failed_checks(result: ResponseValidationResult) -> str:
    """Name the failed checks without repeating the rejected content."""
    checks = {
        "numbers not found in the cited evidence": bool(result.unsupported_numbers),
        "unknown or unsupported claim references": bool(result.invalid_claim_refs),
        "causal wording the evidence does not establish": any("causal" in e for e in result.errors),
        "unlabelled forecast or anomaly": any("without being labelled" in e for e in result.errors),
        "forecast presented with certainty": any("with certainty" in e for e in result.errors),
        "anomaly presented as a business judgement": any("business judgement" in e for e in result.errors),
        "direction contradicts the evidence": any("cited evidence shows" in e for e in result.errors),
        "recommendation wording": any(e.startswith("recommendations:") for e in result.errors),
        "response length": any("characters (limit" in e for e in result.errors),
    }
    failed = [name for name, hit in checks.items() if hit] or ["response structure"]
    return f"Failed checks: {', '.join(failed)}."


def _claims_context(graph: EvidenceGraph) -> list[dict[str, Any]]:
    return [
        {
            "claim_id": c.claim_id,
            "type": c.claim_type,
            "kind": c.kind,
            "text": c.text,
            "primary": c.primary,
            "support_status": c.support_status,
            "evidence_ids": c.evidence_ids,
        }
        for c in graph.claims.values()
        if c.support_status != "unsupported"
    ]


def _summary(result: ToolResult) -> str:
    if not result.success:
        detail = sanitize_detail(result.message)
        return f"failed ({result.error.code if result.error else 'error'}): {detail}"[:200]
    queries = f"{len(result.query_ids)} {'query' if len(result.query_ids) == 1 else 'queries'}"
    base = f"{result.status}; {result.result_type}; {queries}"
    return f"{base}; {sanitize_detail(result.message)}"[:200] if result.message else base
