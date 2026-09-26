"""Build the API response from one agent run. Selection and copying only: no number is computed here.

- ``outcome``/``refusal``: from the agent status and the kind of denial the run recorded (the
  denial's pattern names and reasons stay internal).
- ``kpis``: the headline evidence items (observed or calculated, not a dimension member).
- ``trace``: the tool calls with their plan purpose, timing and user-safe status.
- ``forecasts``/``anomalies``: the shared views (``app/tools/views.py``) of the typed results that
  the run's evidence was built from.
- ``visualizations``: chart specs (``app/api/visualizations.py``).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import get_args

from app.agent.records import AgentStatus
from app.agent.response import trace_entries
from app.agent.runner import AgentRunResult
from app.anomalies import AnomalyReport
from app.api.labels import metric_label
from app.api.schemas.responses import (
    AnomalySection,
    AskResponse,
    ForecastSection,
    KPIValue,
    Outcome,
    Refusal,
    RunSummary,
    StageTiming,
    TraceStep,
)
from app.api.visualizations import build_visualizations
from app.evidence.models import Claim, Evidence
from app.forecasting import ForecastResult
from app.tools.views import AnomalyReportView, ForecastView

MAX_KPIS = 8

# What each agent graph node does, in words a user can follow (``/ask/stream`` progress events).
STAGE_LABELS = {
    "question_received": "Screening the question",
    "understand_question": "Understanding the question",
    "validate_request": "Validating the request",
    "plan_investigation": "Planning the analysis",
    "execute_tools": "Running analysis tools",
    "collect_evidence": "Collecting evidence",
    "validate_evidence": "Validating the evidence",
    "generate_response": "Preparing the answer",
    "validate_response": "Checking the answer against the evidence",
    "done": "Answer ready",
    "unsupported_request": "Request declined",
    "insufficient_evidence": "Stopped: insufficient evidence",
    "tool_error": "Stopped: analysis tools failed",
    "validation_failure": "Stopped: explanation failed validation",
    "planning_failure": "Stopped: no valid analysis plan",
}
_STOPPING_STAGES = frozenset(get_args(AgentStatus)) - {"running", "completed"}


def stage_label(node: str) -> str:
    return STAGE_LABELS.get(node, "Finishing")


def stage_timings(stages: Sequence[tuple[str, float]]) -> list[StageTiming]:
    """Durations between consecutive stage ends (timing only; no business number)."""
    timings, previous = [], 0.0
    for node, elapsed in stages:
        timings.append(
            StageTiming(
                stage=node,
                label=stage_label(node),
                duration_ms=round(max(0.0, elapsed - previous), 2),
                ok=node not in _STOPPING_STAGES,
            )
        )
        previous = elapsed
    return timings


def classify(result: AgentRunResult) -> tuple[Outcome, Refusal | None]:
    """The outcome of a run, and what was refused (if anything) in user-facing terms."""
    status = result.status
    if status == "completed":
        return "answered", None
    if status == "unsupported_request":
        denied = {e.event_type for e in result.security_events if e.decision == "deny"}
        if "suspicious_prompt" in denied:
            return "refused", Refusal(kind="policy", message=result.response.answer)
        if "input_rejected" in denied:
            return "refused", Refusal(kind="invalid_input", message=result.response.answer)
        return "unsupported", Refusal(kind="out_of_scope", message=result.response.answer)
    if status == "insufficient_evidence":
        return "insufficient_evidence", None
    if status == "validation_failure":
        return "partial", None
    return "failed", None


def _citations(claims: list[Claim]) -> dict[str, list[Claim]]:
    citing: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        for evidence_id in claim.evidence_ids:
            citing[evidence_id].append(claim)
    return citing


def _is_headline(e: Evidence) -> bool:
    return (
        e.evidence_type in ("observed", "calculated")
        and e.status == "ok"
        and e.metric is not None
        and e.dimension_value is None
        and isinstance(e.value, int | float)
        and not isinstance(e.value, bool)
    )


def kpi_values(result: AgentRunResult) -> list[KPIValue]:
    """Headline evidence, ordered: cited by a primary claim, then cited, then the rest (evidence order)."""
    citing = _citations(result.claims)

    def rank(e: Evidence) -> int:
        claims = citing.get(e.evidence_id, [])
        return 0 if any(c.primary for c in claims) else 1 if claims else 2

    headline = sorted((e for e in result.evidence if _is_headline(e)), key=rank)[:MAX_KPIS]
    values = []
    for e in headline:
        claims = citing.get(e.evidence_id, [])
        change = e.attributes.get("percentage_change")
        assert e.metric is not None and isinstance(e.value, int | float)
        values.append(
            KPIValue(
                evidence_id=e.evidence_id,
                claim_ids=[c.claim_id for c in claims],
                claim_type=claims[0].claim_type if claims else None,
                primary=any(c.primary for c in claims),
                metric=e.metric,
                label=metric_label(e.metric) + (" change" if e.comparison_label else ""),
                value=e.value,
                display_value=e.display_value,
                unit=e.unit,
                evidence_type=e.evidence_type,
                period=e.period_label,
                comparison_period=e.comparison_label,
                filters=e.filters,
                percentage_change=change if isinstance(change, int | float) and not isinstance(change, bool) else None,
                statement=e.statement,
            )
        )
    return values


def trace_steps(result: AgentRunResult) -> list[TraceStep]:
    purposes = {s.step_id: s.purpose for plan in result.plans for s in plan.steps}
    safe = trace_entries(result.tool_trace)  # the user-facing summary and error category of each call
    return [
        TraceStep(
            step=index,
            call_id=call.call_id,
            step_id=call.step_id,
            tool_name=call.tool_name,
            purpose=purposes.get(call.step_id, ""),
            status=call.status,
            success=call.success,
            started_at=call.start_time,
            execution_time_ms=call.execution_time_ms,
            attempts=call.attempts,
            result_summary=entry.result_summary,
            query_ids=call.query_ids,
            evidence_ids=call.evidence_ids,
            error=entry.error,
        )
        for index, (call, entry) in enumerate(zip(result.tool_trace, safe, strict=True), start=1)
    ]


def _evidence_by_call(result: AgentRunResult) -> dict[str, list[str]]:
    by_call: dict[str, list[str]] = defaultdict(list)
    for e in result.evidence:
        by_call[e.tool_call_id].append(e.evidence_id)
    return by_call


def forecast_sections(result: AgentRunResult) -> list[tuple[ForecastSection, ForecastResult]]:
    """Forecast results that produced evidence in this run (a result without evidence is not shown)."""
    by_call = _evidence_by_call(result)
    sections = []
    for r in result.tool_results:
        if r.success and isinstance(r.result, ForecastResult) and by_call.get(r.call_id):
            section = ForecastSection(
                call_id=r.call_id,
                evidence_ids=by_call[r.call_id],
                forecast=ForecastView.from_result(r.result),
                limitations=list(r.result.limitations),
            )
            sections.append((section, r.result))
    return sections


def anomaly_sections(result: AgentRunResult) -> list[tuple[AnomalySection, AnomalyReport]]:
    by_call = _evidence_by_call(result)
    sections = []
    for r in result.tool_results:
        if r.success and isinstance(r.result, AnomalyReport) and by_call.get(r.call_id):
            section = AnomalySection(
                call_id=r.call_id,
                evidence_ids=by_call[r.call_id],
                report=AnomalyReportView.from_report(r.result),
                limitations=list(r.result.limitations),
            )
            sections.append((section, r.result))
    return sections


def build_response(
    result: AgentRunResult, *, session_id: str | None = None, stages: Sequence[tuple[str, float]] = ()
) -> AskResponse:
    outcome, refusal = classify(result)
    kpis = kpi_values(result)
    forecasts = forecast_sections(result)
    anomalies = anomaly_sections(result)
    request = result.request
    return AskResponse(
        request_id=result.run_id,
        session_id=session_id,
        status=result.status,
        outcome=outcome,
        question=result.question,
        answer=result.response.answer,
        response=result.response,
        scope=request,
        period=request.period if request else None,
        comparison_period=request.comparison_period if request else None,
        kpis=kpis,
        claims=result.claims,
        evidence=result.evidence,
        trace=trace_steps(result),
        forecasts=[s for s, _ in forecasts],
        anomalies=[s for s, _ in anomalies],
        visualizations=build_visualizations(
            result.evidence, result.claims, kpis, forecasts=forecasts, anomalies=anomalies
        ),
        refusal=refusal,
        run=RunSummary(
            run_id=result.run_id,
            llm_provider=result.llm_provider,
            llm_model=result.llm_model,
            tool_calls=result.total_tool_calls,
            retries=result.total_retries,
            agent_time_ms=result.execution_time_ms,
            pipeline=result.transitions,
            stages=stage_timings(stages),
        ),
    )
