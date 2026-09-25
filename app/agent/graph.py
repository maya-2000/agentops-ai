"""The LangGraph state machine of the analytics agent.

    question_received -> understand_question -> validate_request -> plan_investigation
        -> execute_tools (loops once per tool call) -> collect_evidence
        -> [plan_investigation again for evidence-driven follow-up, bounded]
        -> validate_evidence -> generate_response -> validate_response -> done

Failure states: unsupported_request, insufficient_evidence, tool_error, validation_failure and
planning_failure. Each node records its transition and chooses the next node explicitly
(``state.route``); conditional edges only allow the listed targets.

Bounded execution: at most ``max_tool_calls`` tool calls, ``max_planning_iterations`` planning
rounds, ``max_retries`` retries per LLM step, per tool call and for response regeneration, a wall
clock budget checked before every tool call, and a LangGraph recursion limit derived from these.

The runtime (database, tools, model client, configuration) is closed over by the node functions and
never enters the state.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any, TypeVar

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from app.agent.config import AgentConfig
from app.agent.findings import build_claims
from app.agent.observability import log_event
from app.agent.records import AgentError, AgentStatus, InvestigationPlan, LLMCallRecord, PlanStep, ToolCallRecord
from app.agent.request import validate_understanding
from app.agent.response import LIMIT_MESSAGE, build_caveats, build_response, failure_response
from app.agent.state import AgentState
from app.analytics.dimensions import DIMENSIONS
from app.analytics.errors import InvalidRequestError
from app.analytics.executor import QueryRunner
from app.analytics.kpis import list_kpi_definitions
from app.database.base import Database
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
from app.timeseries.metrics import SERIES_METRIC_KEYS
from app.tools.base import ToolContext, ToolRequest, ToolResult
from app.tools.registry import ToolRegistry

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
MAX_EVIDENCE_IN_PROMPT = 40

ModelT = TypeVar("ModelT", bound=BaseModel)


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


class AgentRuntime:
    """Everything a node needs that must not live in the state."""

    def __init__(self, db: Database, llm: LLMClient, config: AgentConfig, registry: ToolRegistry, as_of: date):
        self.db = db
        self.llm = llm
        self.config = config
        self.registry = registry
        self.as_of = as_of
        self.tool_context = ToolContext(db=db, as_of=as_of, sql_row_limit=config.sql_row_limit)
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
        task: LLMTask,
        context: dict[str, Any],
        model: type[ModelT],
        schema: dict[str, Any],
        check: Callable[[ModelT], list[str]] | None = None,
    ) -> tuple[ModelT | None, LLMCallRecord, list[str]]:
        """Ask the model, parse and validate its JSON; retry (bounded) with the errors as feedback."""
        errors: list[str] = []
        attempts = 0
        usage_in = usage_out = None
        for attempt in range(1, self.config.max_retries + 2):
            attempts = attempt
            request_context = dict(context)
            if errors:
                request_context["feedback"] = errors[-3:]
            request = LLMRequest(
                task=task,
                system=SYSTEM_PROMPTS[task],
                prompt=render_prompt(task, request_context),
                context=request_context,
                output_schema=schema,
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
            )
            try:
                response = self.llm.generate(request)
            except LLMError as exc:
                errors.append(f"provider error: {exc}")
                if not exc.retryable:
                    break
                continue
            usage_in, usage_out = response.usage.input_tokens, response.usage.output_tokens
            try:
                parsed = model.model_validate(json.loads(response.content))
            except (json.JSONDecodeError, ValidationError, InvalidRequestError, TypeError) as exc:
                errors.append(f"invalid {task.value} output: {str(exc).splitlines()[0][:300]}")
                continue
            problems = check(parsed) if check else []
            if problems:
                errors.extend(problems)
                continue
            record = LLMCallRecord(
                task=task.value,
                provider=self.llm.provider,
                model=self.llm.model,
                attempts=attempts,
                success=True,
                input_tokens=usage_in,
                output_tokens=usage_out,
            )
            return parsed, record, errors
        record = LLMCallRecord(
            task=task.value,
            provider=self.llm.provider,
            model=self.llm.model,
            attempts=attempts,
            success=False,
            input_tokens=usage_in,
            output_tokens=usage_out,
            error=errors[-1] if errors else "unknown error",
        )
        return None, record, errors


def build_graph(runtime: AgentRuntime) -> Any:
    cfg = runtime.config

    def go(state: AgentState, node: str, route: str, **updates: Any) -> dict[str, Any]:
        log_event(state.run_id, "transition", node=node, route=route)
        return {**updates, "route": route, "transitions": [*state.transitions, node]}

    def llm_updates(state: AgentState, record: LLMCallRecord) -> dict[str, Any]:
        return {"llm_calls": [*state.llm_calls, record], "llm_retries": state.llm_retries + record.attempts - 1}

    def error(state: AgentState, stage: str, code: str, message: str) -> list[AgentError]:
        return [*state.errors, AgentError(stage=stage, code=code, message=message)]

    # ------------------------------------------------------------------ main path
    def question_received(state: AgentState) -> dict[str, Any]:
        normalized = " ".join(state.question.split())
        if not normalized:
            return go(state, "question_received", "unsupported_request", status_message="The question is empty.")
        if len(normalized) > cfg.max_question_chars:
            return go(
                state,
                "question_received",
                "unsupported_request",
                normalized_question=normalized[: cfg.max_question_chars],
                status_message=f"The question is longer than {cfg.max_question_chars} characters.",
            )
        return go(state, "question_received", "understand_question", normalized_question=normalized)

    def understand_question(state: AgentState) -> dict[str, Any]:
        context = {
            "question": state.normalized_question,
            "as_of": runtime.as_of.isoformat(),
            **runtime.business_context(),
        }
        parsed, record, errors = runtime.call_llm(
            LLMTask.UNDERSTAND, context, UnderstandingOutput, UNDERSTANDING_SCHEMA
        )
        updates = llm_updates(state, record)
        if parsed is None:
            return go(
                state,
                "understand_question",
                "planning_failure",
                status_message="The question could not be interpreted.",
                errors=error(state, "understand_question", "llm_output_invalid", "; ".join(errors[-2:])),
                **updates,
            )
        return go(state, "understand_question", "validate_request", understanding=parsed, **updates)

    def validate_request_node(state: AgentState) -> dict[str, Any]:
        assert state.understanding is not None
        validation = validate_understanding(
            state.understanding, as_of=runtime.as_of, coverage=runtime.tool_context.kpi_service.coverage()
        )
        route = {
            "valid": "plan_investigation",
            "unsupported": "unsupported_request",
            "clarify": "insufficient_evidence",
            "insufficient": "insufficient_evidence",
        }[validation.outcome]
        return go(state, "validate_request", route, request=validation.request, status_message=validation.message)

    def plan_investigation(state: AgentState) -> dict[str, Any]:
        assert state.request is not None
        iteration = state.planning_iterations + 1
        remaining = cfg.max_tool_calls - len(state.tool_calls)
        if remaining <= 0:
            route = "validate_evidence" if state.tool_calls else "insufficient_evidence"
            return go(state, "plan_investigation", route, limit_reached=True, status_message=LIMIT_MESSAGE)
        executed = {(c.tool_name, json.dumps(c.input, sort_keys=True)) for c in state.tool_calls}
        accepted: list[PlanStep] = []
        budget_exceeded = False

        def check(plan: PlanOutput) -> list[str]:
            nonlocal budget_exceeded
            accepted.clear()
            problems: list[str] = []
            seen = set(executed)
            for index, step in enumerate(plan.steps, start=1):
                try:
                    arguments = json.loads(step.arguments_json or "{}")
                    if not isinstance(arguments, dict):
                        raise InvalidRequestError("arguments_json must encode a JSON object")
                    canonical = runtime.registry.validate_arguments(step.tool_name, arguments)
                except (json.JSONDecodeError, InvalidRequestError) as exc:
                    problems.append(f"step {index} ({step.tool_name}): {exc}")
                    continue
                key = (step.tool_name, json.dumps(canonical, sort_keys=True))
                if key in seen:
                    continue  # a repeated call adds no evidence
                seen.add(key)
                accepted.append(
                    PlanStep(
                        step_id=f"S{iteration}.{len(accepted) + 1}",
                        tool_name=step.tool_name,
                        arguments=canonical,
                        purpose=step.purpose,
                        iteration=iteration,
                    )
                )
            if iteration == 1 and not plan.steps:
                problems.append("The plan has no steps.")
            budget_exceeded = len(accepted) > remaining
            if budget_exceeded:
                problems.append(f"The plan has {len(accepted)} tool calls but only {remaining} remain.")
            return problems

        context = {
            "question": state.normalized_question,
            "request": state.request.model_dump(mode="json"),
            "iteration": iteration,
            "remaining_tool_calls": remaining,
            "tools": runtime.registry.catalog(),
            "executed_steps": [
                {"tool_name": c.tool_name, "arguments": c.input, "success": c.success} for c in state.tool_calls
            ],
            "evidence": [
                {**evidence_summary(e), "operation": e.operation, "attributes": e.attributes}
                for e in list(state.evidence_graph.evidence.values())[:MAX_EVIDENCE_IN_PROMPT]
            ],
        }
        parsed, record, errors = runtime.call_llm(LLMTask.PLAN, context, PlanOutput, PLAN_SCHEMA, check)
        updates = {**llm_updates(state, record), "planning_iterations": iteration}
        if parsed is None:
            if iteration > 1:  # follow-up planning is optional: continue with the evidence already collected
                return go(
                    state,
                    "plan_investigation",
                    "validate_evidence",
                    limit_reached=state.limit_reached or budget_exceeded,
                    errors=error(state, "plan_investigation", "follow_up_plan_invalid", "; ".join(errors[-2:])),
                    **updates,
                )
            if budget_exceeded:
                return go(
                    state,
                    "plan_investigation",
                    "insufficient_evidence",
                    limit_reached=True,
                    status_message=LIMIT_MESSAGE,
                    **updates,
                )
            return go(
                state,
                "plan_investigation",
                "planning_failure",
                status_message="No valid investigation plan could be produced from the allowed tools.",
                errors=error(state, "plan_investigation", "plan_invalid", "; ".join(errors[-2:])),
                **updates,
            )
        steps = list(accepted)
        log_event(state.run_id, "plan", iteration=iteration, steps=[s.tool_name for s in steps])
        if not steps:
            return go(state, "plan_investigation", "validate_evidence", **updates)
        plan = InvestigationPlan(iteration=iteration, steps=steps, rationale=parsed.rationale)
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
        elapsed = (datetime.now(UTC) - state.started_at).total_seconds()
        if len(state.tool_calls) >= cfg.max_tool_calls or elapsed > cfg.max_run_seconds:
            return go(state, "execute_tools", "collect_evidence", pending_steps=[], limit_reached=True)
        if not pending:
            return go(state, "execute_tools", "collect_evidence")
        step = pending.pop(0)
        call_id = f"T{len(state.tool_calls) + 1}"
        attempts = 0
        result: ToolResult
        while True:
            attempts += 1
            result = runtime.registry.execute(
                ToolRequest(call_id=call_id, tool_name=step.tool_name, arguments=step.arguments, purpose=step.purpose),
                runtime.tool_context,
            )
            retryable = result.error is not None and result.error.retryable
            if result.success or not retryable or attempts > cfg.max_retries:
                break
        result = result.model_copy(update={"attempts": attempts})
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
            attempts=attempts,
            result_summary=_summary(result),
            query_ids=result.query_ids,
            error=result.error,
        )
        log_event(
            state.run_id,
            "tool_call",
            tool_name=step.tool_name,
            call_id=call_id,
            success=result.success,
            status=result.status,
            attempts=attempts,
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
            tool_retries=state.tool_retries + attempts - 1,
            limit_reached=state.limit_reached or limit,
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
            f"{c.tool_name}: {c.error.message if c.error else c.status}" for c in state.tool_calls if not c.success
        ]
        result = validate_evidence(graph, successful_call_ids=successful, failed_tools=failed)
        removed: list[str] = []
        if not result.valid and result.unsupported_claim_ids:
            removed = list(result.unsupported_claim_ids)
            for claim_id in removed:
                graph.claims.pop(claim_id, None)
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
        updates = {"evidence_graph": graph, "evidence_validation": result}
        if result.valid:
            return go(state, "validate_evidence", "generate_response", **updates)
        route = "tool_error" if failed and not successful else "insufficient_evidence"
        message = "No supported finding answers the question." + (
            f" Failed tools: {'; '.join(failed)}." if failed else ""
        )
        return go(state, "validate_evidence", route, status_message=message, **updates)

    def generate_response(state: AgentState) -> dict[str, Any]:
        assert state.request is not None
        graph = state.evidence_graph
        context: dict[str, Any] = {
            "question": state.normalized_question,
            "intent": state.request.intent.value,
            "claims": _claims_context(graph),
            "evidence": [evidence_summary(e) for e in list(graph.evidence.values())[:MAX_EVIDENCE_IN_PROMPT]],
        }
        if state.response_validation is not None and not state.response_validation.valid:
            context["previous_errors"] = state.response_validation.errors[:5]
        parsed, record, errors = runtime.call_llm(LLMTask.RESPOND, context, ResponseDraftOutput, RESPONSE_SCHEMA)
        updates: dict[str, Any] = llm_updates(state, record)
        if parsed is None:
            parsed = ResponseDraftOutput.model_validate(compose(context))
            updates["errors"] = error(state, "generate_response", "llm_failed", "; ".join(errors[-2:]))
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
        if state.retry_count < cfg.max_retries:
            return go(
                state,
                "validate_response",
                "generate_response",
                response_validation=result,
                retry_count=state.retry_count + 1,
            )
        # The validator's messages quote the rejected text; they go to the trace, never into the response.
        return go(
            state,
            "validate_response",
            "validation_failure",
            response_validation=result,
            status_message=_failed_checks(result),
            errors=error(state, "validate_response", "response_invalid", "; ".join(result.errors[:5])),
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
        )
        response = build_response(
            draft,
            state.evidence_graph,
            status=status,
            calls=state.tool_calls,
            assumptions=state.request.assumptions,
            caveats=caveats,
            generated_by=state.response_generated_by or "unknown",
        )
        log_event(state.run_id, "final", status=status, tool_calls=len(state.tool_calls))
        return go(state, "done", "end", response=response, status=status)

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
            response = failure_response(
                self.status,
                message,
                calls=state.tool_calls,
                assumptions=state.request.assumptions if state.request else [],
                graph=graph,
                findings=findings,
                caveats=[LIMIT_MESSAGE] if state.limit_reached and LIMIT_MESSAGE not in message else [],
                coverage=runtime.tool_context.kpi_service.coverage(),
            )
            log_event(state.run_id, "final", status=self.status, tool_calls=len(state.tool_calls))
            return go(state, self.status, "end", response=response, status=self.status)

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


def _failed_checks(result: ResponseValidationResult) -> str:
    """Name the failed checks without repeating the rejected content."""
    checks = {
        "numbers not found in the cited evidence": bool(result.unsupported_numbers),
        "unknown or unsupported claim references": bool(result.invalid_claim_refs),
        "causal wording the evidence does not establish": any("causal" in e for e in result.errors),
        "unlabelled forecast or anomaly": any("without being labelled" in e for e in result.errors),
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
        return f"failed ({result.error.code if result.error else 'error'}): {result.message or ''}"[:200]
    queries = f"{len(result.query_ids)} {'query' if len(result.query_ids) == 1 else 'queries'}"
    base = f"{result.status}; {result.result_type}; {queries}"
    return f"{base}; {result.message}"[:200] if result.message else base
