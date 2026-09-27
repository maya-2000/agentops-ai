"""The investigation engine: plan -> execute -> validate -> synthesize (Phase 10).

``Investigator`` runs one investigation on the agent's own runtime (``AgentRuntime``): the same input
guard and prompt-injection screen, the same understanding step and request validation, the same
``SecuredToolExecutor`` for every tool call, the same evidence builder, claim builders and evidence
validator. It adds no analytics and no SQL, and it never lets the model choose a tool call.

Lifecycle:

1. **Screen** the objective (input guard, prompt-injection screen). A blocked objective is refused before
   any model or tool call.
2. **Understand** it (the agent's understanding step, then the central request validation).
3. **Plan**: choose a template and build the analysis plan (``app/investigation/planner.py``).
4. **Execute** the steps in plan order, one at a time (one read-only database connection; results are
   deterministic). Each step: dependencies and conditions, evidence-bound arguments, reuse of an identical
   earlier call, the step/tool-call/runtime/evidence budgets, then ``SecuredToolExecutor.execute``
   (authorize -> deadline -> execute -> retry -> output validation -> budget charge) under the validated
   intent the template assigns, without the SQL privilege.
5. **Validate**: claims are built and checked by the evidence validator; findings, relationships, drivers
   and recommendations are checked across findings (``app/investigation/validation.py``).
6. **Synthesize** the decision brief (``app/investigation/brief.py``).

Bounded: the investigation budget (``RunBudget.for_investigation``) caps tool calls, retries and model
calls; steps, runtime and evidence are checked before every step; the brief's size is capped. A budget
stop ends the investigation as ``budget_exhausted``, never ``completed``. Cancellation (a client that
went away, shutdown) and the caller's deadline stop it before its next step; a cancelled investigation
keeps its step records and presents no findings.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.agent.findings import build_claims
from app.agent.graph import AgentRuntime
from app.agent.observability import is_valid_run_id, log_event
from app.agent.records import AgentError, ToolCallRecord
from app.agent.request import ValidatedRequest, validate_understanding
from app.database.deadline import execution_deadline
from app.evidence.builder import build_evidence
from app.evidence.models import EvidenceGraph
from app.evidence.validation import validate_evidence
from app.investigation.brief import brief_chars, compose_brief
from app.investigation.drivers import DriverAnalysis, analyze_drivers
from app.investigation.findings import build_findings, outcome_finding
from app.investigation.models import (
    STOP_MESSAGE,
    AnalysisPlan,
    AnalysisStep,
    Investigation,
    InvestigationBudgetReport,
    InvestigationEfficiency,
    InvestigationStatus,
    InvestigationTimings,
    StepRecord,
    TemplateName,
    ValidationIssue,
)
from app.investigation.planner import asks_for_brief, plan_investigation, select_template
from app.investigation.recommendations import recommend
from app.investigation.steps import SKIP_REASONS, bind, condition_met
from app.investigation.validation import (
    validate_drivers,
    validate_findings,
    validate_recommendations,
    validate_relationships,
)
from app.llm.base import LLMTask
from app.llm.schemas import UNDERSTANDING_SCHEMA, Intent, UnderstandingOutput
from app.security.authorization import AuthorizationContext
from app.security.budget import BudgetUsage, RunBudget, charge_model_call, mark_exhausted
from app.security.errors import safe_message, sanitize_detail
from app.security.events import SecurityEvent, Severity, security_event
from app.security.input_guard import understanding_output_problems
from app.security.redaction import redact
from app.tools.base import ToolError, ToolResult

# What one progress event carries: stage names and fixed labels, never data or reasoning.
ProgressCallback = Callable[[dict[str, Any]], None]
AREA_LABELS = {
    "revenue": "Analyzing revenue",
    "customers": "Analyzing customers",
    "sales": "Analyzing sales",
    "marketing": "Analyzing marketing",
    "product": "Analyzing product",
    "support": "Analyzing support",
    "anomalies": "Checking for anomalies",
    "forecast": "Forecasting",
}
BLOCKED_MESSAGE = (
    "The objective asks for something the agent is not permitted to do (such as revealing secrets, instructions, "
    "hidden or ground-truth data, reading files, running code or changing its own rules), so it was not processed."
)
CANCELLED_MESSAGE = "The investigation was cancelled before it finished; no findings are presented."
DEADLINE_MESSAGE = "The investigation reached its time limit before it finished; no findings are presented."
# The intent the claim builders use for each template (it sets which claims answer the objective).
CLAIM_INTENTS: dict[TemplateName, Intent] = {
    "revenue": Intent.REVENUE_INVESTIGATION,
    "customer": Intent.CUSTOMER_INVESTIGATION,
    "sales": Intent.SALES_ANALYSIS,
    "product_support": Intent.SUPPORT_ANALYSIS,
    "general": Intent.MIXED_INVESTIGATION,
    "management_brief": Intent.MIXED_INVESTIGATION,
}


def new_investigation_id() -> str:
    return f"I-{uuid.uuid4().hex[:12]}"


class _Stopped(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class _Run:
    """The working state of one investigation (never shared between investigations)."""

    investigation_id: str
    created_at: datetime
    cancel: threading.Event | None
    deadline_at: float | None
    on_progress: ProgressCallback | None
    clock: float = field(default_factory=time.perf_counter)
    objective: str = ""
    status: InvestigationStatus = "running"
    message: str | None = None
    request: ValidatedRequest | None = None
    plan: AnalysisPlan | None = None
    records: list[StepRecord] = field(default_factory=list)
    graph: EvidenceGraph = field(default_factory=EvidenceGraph)
    trace: list[ToolCallRecord] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)
    usage: BudgetUsage = field(default_factory=BudgetUsage)
    events: list[SecurityEvent] = field(default_factory=list)
    errors: list[AgentError] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    timings: InvestigationTimings = field(default_factory=InvestigationTimings)
    budget_stop: str | None = None  # the budget that stopped execution

    def elapsed(self) -> float:
        return time.perf_counter() - self.clock

    def progress(self, stage: str, label: str, **fields: Any) -> None:
        if self.on_progress is None:
            return
        try:
            self.on_progress({"stage": stage, "label": label, **fields})
        except Exception:  # an observer can never change an investigation
            self.on_progress = None

    def record(self, step_id: str) -> StepRecord:
        return next(r for r in self.records if r.step_id == step_id)


class Investigator:
    """Runs investigations on an agent runtime; investigations are independent of each other."""

    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime
        self.config = runtime.config
        self.budget = RunBudget.for_investigation(runtime.config)

    def investigate(
        self,
        objective: Any,
        *,
        investigation_id: str | None = None,
        cancel: threading.Event | None = None,
        deadline_seconds: float | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> Investigation:
        """Investigate one objective. The objective is untrusted: it is screened and redacted first.

        ``cancel`` and ``deadline_seconds`` bound the investigation from outside (the API's timeout, a client
        that went away, shutdown): it stops before its next step, database queries are interrupted at the
        deadline, and the model call waits no longer than the time left.
        """
        if investigation_id is not None and not is_valid_run_id(investigation_id):
            raise ValueError(
                "investigation_id must be 1-64 characters: letters, digits and ._:- (starting alphanumeric)"
            )
        run = _Run(
            investigation_id=investigation_id or new_investigation_id(),
            created_at=datetime.now(UTC),
            cancel=cancel,
            deadline_at=time.monotonic() + max(0.0, deadline_seconds) if deadline_seconds is not None else None,
            on_progress=on_progress,
        )
        run.progress("started", "Investigation started")
        try:
            with execution_deadline(deadline_seconds):
                self._investigate(run, objective)
        except _Stopped as stopped:
            run.status = "cancelled"
            run.message = DEADLINE_MESSAGE if stopped.reason == "deadline_exceeded" else CANCELLED_MESSAGE
            run.errors.append(AgentError(stage="investigation", code=stopped.reason, message=run.message))
            for record in run.records:
                if record.status == "pending":
                    record.status, record.reason = "not_run", "Not run: the investigation was stopped."
            return self._finish(run, findings_allowed=False)
        return self._finish(run, findings_allowed=True)

    # ------------------------------------------------------------------ lifecycle
    def _check(self, run: _Run) -> None:
        if run.cancel is not None and run.cancel.is_set():
            raise _Stopped("cancelled")
        if run.deadline_at is not None and time.monotonic() >= run.deadline_at:
            raise _Stopped("deadline_exceeded")

    def _investigate(self, run: _Run, objective: Any) -> None:
        clock = time.perf_counter()
        request, template = self._understand(run, objective)
        run.timings.understanding_ms = round((time.perf_counter() - clock) * 1000, 1)
        if template is None:
            return
        self._check(run)
        clock = time.perf_counter()
        run.progress("planning", "Planning the analysis")
        run.request = request
        run.plan = plan_investigation(template, request, as_of=self.runtime.as_of)
        run.records = [
            StepRecord(step_id=s.step_id, title=s.title, area=s.area, tool_name=s.tool_name) for s in run.plan.steps
        ]
        run.timings.planning_ms = round((time.perf_counter() - clock) * 1000, 1)
        log_event(run.investigation_id, "investigation_plan", template=template, steps=len(run.plan.steps))
        run.progress(
            "plan",
            f"Analysis plan ready: {len(run.plan.steps)} steps",
            steps=[
                {
                    "step_id": s.step_id,
                    "title": s.title,
                    "area": s.area,
                    "tool_name": s.tool_name,
                    "depends_on": s.depends_on,
                }
                for s in run.plan.steps
            ],
        )
        clock = time.perf_counter()
        self._execute(run)
        run.timings.execution_ms = round((time.perf_counter() - clock) * 1000, 1)

    def _understand(self, run: _Run, objective: Any) -> tuple[ValidatedRequest | None, TemplateName | None]:
        runtime = self.runtime
        check = runtime.input_guard.validate_question(objective)
        run.objective = check.question if check.question else (redact(objective) if isinstance(objective, str) else "")
        if check.redacted:
            run.events.append(
                security_event(
                    run.investigation_id,
                    "secret_redacted",
                    Severity.HIGH,
                    component="input_guard",
                    action="objective",
                    decision="redact",
                    reason="A secret-like value was removed from the objective before processing.",
                )
            )
        if check.outcome == "rejected":
            run.status, run.message = "refused", check.reason or "The objective could not be accepted."
            run.events.append(
                security_event(
                    run.investigation_id,
                    "input_rejected",
                    Severity.WARNING,
                    component="input_guard",
                    action="objective",
                    decision="deny",
                    reason=run.message,
                    code=check.code,
                )
            )
            return None, None
        if check.outcome == "blocked":
            run.status, run.message = "refused", BLOCKED_MESSAGE
            run.events.append(
                security_event(
                    run.investigation_id,
                    "suspicious_prompt",
                    check.scan.severity or Severity.HIGH,
                    component="prompt_injection",
                    action="screen",
                    decision="deny",
                    reason="Blocked request categories: " + ", ".join(check.scan.categories),
                    categories=check.scan.categories,
                    signals=check.scan.signals,
                )
            )
            return None, None
        if check.restricted:  # investigations never run ad-hoc SQL; the verdict is recorded for the audit
            run.events.append(
                security_event(
                    run.investigation_id,
                    "suspicious_prompt",
                    check.scan.severity or Severity.WARNING,
                    component="prompt_injection",
                    action="screen",
                    decision="restrict",
                    reason="Suspicious instruction patterns: " + ", ".join(check.scan.categories),
                    categories=check.scan.categories,
                    signals=check.scan.signals,
                )
            )
        self._check(run)
        run.progress("understanding", "Understanding the objective")
        context = {"question": check.question, "as_of": runtime.as_of.isoformat(), **runtime.business_context()}

        def problems(u: UnderstandingOutput) -> list[str]:
            return understanding_output_problems(u, self.config)

        outcome = runtime.call_llm(
            run.investigation_id, LLMTask.UNDERSTAND, context, UnderstandingOutput, UNDERSTANDING_SCHEMA, problems
        )
        run.usage = charge_model_call(
            run.usage,
            attempts=outcome.record.attempts,
            context_items=outcome.context_items,
            prompt_chars=outcome.prompt_chars,
        )
        run.events.extend(outcome.events)
        self._check(run)
        brief = asks_for_brief(check.question)
        if outcome.parsed is None:
            run.status, run.message = "failed", "The objective could not be interpreted."
            run.errors.append(
                AgentError(
                    stage="understand_objective",
                    code="llm_output_invalid",
                    message=sanitize_detail("; ".join(outcome.errors[-2:])),
                )
            )
            return None, None
        understanding: UnderstandingOutput = outcome.parsed
        violations = runtime.input_guard.validate_understanding_output(understanding)
        if violations:
            first = violations[0]
            run.status = "insufficient_evidence" if first.code == "invalid_period" else "unsupported"
            run.message = (
                f"The period could not be interpreted ({first.message})."
                if first.code == "invalid_period"
                else f"The objective exceeds the supported input limits ({first.field}: {first.message})."
            )
            return None, None
        validation = validate_understanding(
            understanding, as_of=runtime.as_of, coverage=runtime.tool_context.kpi_service.coverage()
        )
        if validation.outcome in ("clarify", "insufficient"):
            run.status, run.message = "insufficient_evidence", validation.message
            return None, None
        if validation.outcome == "unsupported" and not brief:
            run.status = "unsupported"
            run.message = validation.message or "The objective is outside the business dataset and its analyses."
            run.events.append(
                security_event(
                    run.investigation_id,
                    "unsupported_request",
                    Severity.INFO,
                    component="request_validation",
                    action="objective",
                    decision="deny",
                    reason=run.message,
                )
            )
            return None, None
        request = validation.request if validation.outcome == "valid" else None
        template = select_template(check.question, request)
        if template is None:
            run.status, run.message = "unsupported", "The objective does not match a supported investigation."
        return request, template

    # ------------------------------------------------------------------ execution
    def _stop_for_budget(self, run: _Run, resource: str) -> None:
        run.budget_stop = run.budget_stop or resource
        run.usage = mark_exhausted(run.usage, resource)
        run.events.append(
            security_event(
                run.investigation_id,
                "budget_exceeded",
                Severity.WARNING,
                component="investigation_budget",
                action="execute_steps",
                decision="stop",
                reason=(
                    f"The investigation {resource.replace('_', ' ')} budget is exhausted; remaining steps were not run."
                ),
            )
        )

    def _budget_exhausted(self, run: _Run, steps_run: int) -> str | None:
        cfg = self.config
        if steps_run >= cfg.max_investigation_steps:
            return "steps"
        if run.usage.tool_calls >= self.budget.max_tool_calls:
            return "tool_calls"
        if run.elapsed() >= self.budget.max_runtime_seconds:
            return "runtime"
        if len(run.graph.evidence) >= cfg.max_investigation_evidence:
            return "evidence"
        return None

    def _execute(self, run: _Run) -> None:
        assert run.plan is not None
        executed: dict[str, StepRecord] = {}  # identical call -> the step that ran it
        steps_run = 0
        for step in run.plan.steps:
            record = run.record(step.step_id)
            self._check(run)
            if run.budget_stop is not None:
                record.status, record.reason = (
                    "not_run",
                    f"Not run: the investigation's {run.budget_stop.replace('_', ' ')} budget was reached.",
                )
                continue
            dependencies = [run.record(d) for d in step.depends_on]
            if any(d.status not in ("completed", "reused") for d in dependencies):
                record.status, record.reason = "skipped", SKIP_REASONS["dependency"]
                continue
            evidence = [run.graph.evidence[e] for d in dependencies for e in d.evidence_ids if e in run.graph.evidence]
            if step.condition is not None and not condition_met(step.condition, evidence):
                record.status, record.reason = "skipped", SKIP_REASONS[step.condition]
                continue
            arguments = dict(step.arguments)
            if step.binding is not None:
                bound = bind(step.binding, evidence, arguments)
                if bound is None:
                    record.status, record.reason = "skipped", SKIP_REASONS["binding"]
                    continue
                arguments = bound
            record.arguments = arguments
            key = f"{step.tool_name}:{json.dumps(arguments, sort_keys=True, default=str)}"
            if key in executed:  # evidence reuse: an identical call is never run twice
                source = executed[key]
                record.status, record.reused_from = "reused", source.step_id
                record.evidence_ids, record.call_id = list(source.evidence_ids), source.call_id
                continue
            exhausted = self._budget_exhausted(run, steps_run)
            if exhausted is not None:
                self._stop_for_budget(run, exhausted)
                record.status, record.reason = (
                    "not_run",
                    f"Not run: the investigation's {exhausted.replace('_', ' ')} budget was reached.",
                )
                continue
            steps_run += 1
            self._run_step(run, step, record, arguments, bool(evidence))
            if record.status == "completed":
                executed[key] = record

    def _run_step(
        self, run: _Run, step: AnalysisStep, record: StepRecord, arguments: dict[str, Any], prior: bool
    ) -> None:
        call_id = f"T{len(run.trace) + 1}"
        run.progress(
            "step_started",
            f"{AREA_LABELS.get(step.area, 'Analyzing')}: {step.title}",
            step_id=step.step_id,
            title=step.title,
            area=step.area,
            tool_name=step.tool_name,
        )
        call = self.runtime.executor.execute(
            run_id=run.investigation_id,
            call_id=call_id,
            tool_name=step.tool_name,
            arguments=arguments,
            purpose=step.title,
            context=AuthorizationContext(
                intent=step.authorized_as,
                sql_permitted=False,  # investigations never run ad-hoc SQL
                iteration=2 if step.depends_on else 1,
                has_prior_evidence=prior,
            ),
            budget=self.budget,
            usage=run.usage,
        )
        run.events.extend(call.events)
        result = call.result
        if call.decision.code == "budget_exceeded":
            run.usage = call.usage
            self._stop_for_budget(run, "tool_calls")
            record.status, record.reason = "not_run", "Not run: the investigation's tool calls budget was reached."
            run.progress(
                "step_finished", "Step not run", step_id=step.step_id, status=record.status, tool_name=step.tool_name
            )
            return
        run.usage = call.usage
        evidence_ids = [e.evidence_id for e in build_evidence(result, run.graph)] if result.success else []
        safe_error = (
            ToolError(
                code=result.error.code, message=sanitize_detail(result.error.message), retryable=result.error.retryable
            )
            if result.error
            else None
        )
        run.trace.append(
            ToolCallRecord(
                call_id=call_id,
                step_id=step.step_id,
                tool_name=step.tool_name,
                input=arguments,
                start_time=result.started_at,
                end_time=result.finished_at,
                execution_time_ms=result.execution_time_ms,
                success=result.success,
                status=result.status,
                attempts=max(call.attempts, 1),
                result_summary=f"{len(evidence_ids)} evidence items"
                if result.success
                else f"failed ({safe_error.code if safe_error else 'error'})",
                query_ids=result.query_ids,
                evidence_ids=evidence_ids,
                error=safe_error,
            )
        )
        run.results.append(result)
        record.call_id = call_id
        record.evidence_ids = evidence_ids
        record.execution_time_ms = result.execution_time_ms
        record.attempts = max(call.attempts, 1)
        if result.success:
            record.status = "completed"
        else:
            record.status = "failed"
            record.reason = safe_message(result.error.code if result.error else "internal_error")
        log_event(
            run.investigation_id,
            "investigation_step",
            step_id=step.step_id,
            tool_name=step.tool_name,
            call_id=call_id,
            success=result.success,
            error_code=result.error.code if result.error else None,
            execution_time_ms=result.execution_time_ms,
        )
        run.progress(
            "step_finished",
            f"{step.title}: {record.status}",
            step_id=step.step_id,
            status=record.status,
            tool_name=step.tool_name,
            duration_ms=round(result.execution_time_ms, 1),
        )

    # ------------------------------------------------------------------ validation and synthesis
    def _finish(self, run: _Run, *, findings_allowed: bool) -> Investigation:
        plan = run.plan
        findings: list = []
        relationships: list = []
        brief = None
        if findings_allowed and plan is not None:
            findings, relationships, brief = self._synthesize(run, plan)
        if not findings_allowed:
            run.graph = EvidenceGraph()  # stopped before validation: nothing unvalidated is presented
            run.results = []
        run.timings.total_ms = round(run.elapsed() * 1000, 1)
        usage = run.usage.model_copy(
            update={"response_chars": brief_chars(brief, {f.finding_id: f for f in findings}) if brief else 0}
        )
        statuses = [r.status for r in run.records]
        investigation = Investigation(
            investigation_id=run.investigation_id,
            objective=run.objective,
            status=run.status,
            message=redact(run.message) if run.message else None,
            created_at=run.created_at,
            completed_at=datetime.now(UTC),
            template=plan.template if plan else None,
            scope=run.request,
            plan=plan,
            steps=run.records,
            findings=findings,
            relationships=relationships,
            brief=brief,
            claims=list(run.graph.claims.values()),
            evidence=list(run.graph.evidence.values()),
            tool_trace=run.trace,
            validation_issues=run.issues,
            budget=InvestigationBudgetReport(
                max_steps=self.config.max_investigation_steps,
                max_tool_calls=self.budget.max_tool_calls,
                max_seconds=self.budget.max_runtime_seconds,
                max_evidence=self.config.max_investigation_evidence,
                max_output_chars=self.config.max_investigation_output_chars,
                usage=usage,
                steps_run=sum(1 for r in run.records if r.status in ("completed", "failed")),
                evidence_items=len(run.graph.evidence),
                output_chars=usage.response_chars,
                exhausted=list(usage.exhausted),
            ),
            efficiency=InvestigationEfficiency(
                steps_planned=len(run.records),
                steps_completed=statuses.count("completed"),
                steps_reused=statuses.count("reused"),
                steps_skipped=statuses.count("skipped"),
                steps_failed=statuses.count("failed"),
                steps_not_run=statuses.count("not_run"),
                tool_calls=len(run.trace),
                duplicate_tool_calls=len(run.trace)
                - len({(c.tool_name, json.dumps(c.input, sort_keys=True, default=str)) for c in run.trace}),
            ),
            timings=run.timings,
            llm_provider=self.runtime.llm.provider,
            llm_model=self.runtime.llm.model,
            errors=run.errors,
            security_events=run.events,
            tool_results=run.results,
        )
        log_event(
            run.investigation_id,
            "investigation_final",
            status=investigation.status,
            tool_calls=len(run.trace),
            findings=len(findings),
        )
        run.progress("finished", "Investigation finished", status=investigation.status)
        return investigation

    def _synthesize(self, run: _Run, plan: AnalysisPlan) -> tuple[list, list, Any]:
        clock = time.perf_counter()
        run.progress("validating", "Validating findings")
        successful = [c.call_id for c in run.trace if c.success]
        if not successful:
            run.status = "failed" if run.budget_stop is None else "budget_exhausted"
            run.message = (
                STOP_MESSAGE
                if run.budget_stop
                else "The analysis steps could not be completed, so there are no findings."
            )
            run.timings.validation_ms = round((time.perf_counter() - clock) * 1000, 1)
            return [], [], None
        request = run.request
        claims_request = ValidatedRequest(
            intent=CLAIM_INTENTS[plan.template],
            metric=plan.outcome_metric,
            period=plan.period,
            comparison_period=plan.comparison_period,
            filters=dict(request.filters) if request else {},
            causal_question=request.causal_question if request else False,
            assumptions=list(plan.assumptions),
        )
        graph = run.graph
        build_claims(graph, claims_request)
        for claim_id in [c.claim_id for c in graph.claims.values() if c.claim_type == "recommendation"]:
            graph.claims.pop(claim_id)  # investigations derive their own recommendations, from findings
        failed = [
            f"{c.tool_name}: {safe_message(c.error.code if c.error else c.status)}" for c in run.trace if not c.success
        ]
        checked = validate_evidence(graph, successful_call_ids=successful, failed_tools=failed, require_primary=False)
        for claim_id in checked.unsupported_claim_ids:
            graph.claims.pop(claim_id, None)
            run.issues.append(
                ValidationIssue(item_id=claim_id, action="removed", reason="the claim is not supported by its evidence")
            )
        tampered = graph.verify_integrity()
        if tampered:
            run.events.append(
                security_event(
                    run.investigation_id,
                    "evidence_integrity_failed",
                    Severity.CRITICAL,
                    component="evidence",
                    action="verify",
                    decision="deny",
                    reason=f"{len(tampered)} evidence items no longer match their fingerprint",
                )
            )
        findings = build_findings(graph, plan, run.records)
        findings, issues = validate_findings(findings, graph, plan, successful_call_ids=successful)
        run.issues += issues
        outcome = outcome_finding(findings)
        complete = run.budget_stop is None
        analysis = analyze_drivers(findings, outcome, graph) if complete else DriverAnalysis()
        relationships, issues = validate_relationships(analysis.relationships, findings)
        run.issues += issues
        analysis.drivers, issues = validate_drivers(analysis.drivers, findings, relationships)
        run.issues += issues
        analysis.contradictions, issues = validate_drivers(analysis.contradictions, findings, relationships)
        run.issues += issues
        analysis.context, issues = validate_drivers(analysis.context, findings, relationships)
        run.issues += issues
        run.timings.validation_ms = round((time.perf_counter() - clock) * 1000, 1)

        clock = time.perf_counter()
        run.progress("synthesizing", "Preparing the decision brief")
        recommendations = recommend(findings, outcome, analysis, graph, plan) if complete else []
        if recommendations:
            checked = validate_evidence(
                graph, successful_call_ids=successful, failed_tools=failed, require_primary=False
            )
            for rec in recommendations:
                if rec.claim_id in checked.unsupported_claim_ids:
                    graph.claims.pop(rec.claim_id, None)
        recommendations, issues = validate_recommendations(recommendations, findings, graph)
        run.issues += issues
        brief, issues = compose_brief(
            plan,
            run.objective,
            findings=findings,
            outcome=outcome,
            analysis=analysis,
            recommendations=recommendations,
            records=run.records,
            graph=graph,
            causal_question=claims_request.causal_question,
            complete=complete,
            max_chars=self.config.max_investigation_output_chars,
        )
        run.issues += issues
        brief = _redacted(brief)
        run.timings.synthesis_ms = round((time.perf_counter() - clock) * 1000, 1)
        if not complete:
            run.status, run.message = "budget_exhausted", STOP_MESSAGE
        elif claims_request.causal_question:
            run.status = "insufficient_evidence"
            run.message = "The available data cannot establish the cause asked about; the observed findings are shown."
        elif outcome is None and plan.template != "management_brief":
            run.status = "insufficient_evidence"
            run.message = "The outcome of the investigation could not be measured from the available data."
        elif not findings:
            run.status, run.message = "insufficient_evidence", "No finding passed validation."
        else:
            run.status = "completed"
        return findings, relationships, brief


def _redacted(brief: Any) -> Any:
    """Remove secret-like values from every user-facing text of the brief (the last output boundary)."""
    return brief.model_copy(
        update={
            "objective": redact(brief.objective),
            "executive_summary": redact(brief.executive_summary),
            "uncertainty": [redact(u) for u in brief.uncertainty],
        }
    )
