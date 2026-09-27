"""Build the API response from one investigation. Selection and copying only: no number is computed here.

The plan, the analysis trace, the headline KPIs, the forecast and anomaly sections and the chart specs
come from the same presenter functions as ``/ask``, applied to the investigation's evidence, claims and
tool calls. The decision brief, findings and relationships are the investigation's own objects.
"""

from __future__ import annotations

from app.api.presenter import (
    anomaly_sections_of,
    forecast_sections_of,
    kpi_values_of,
    trace_steps_of,
)
from app.api.schemas.investigations import InvestigationResponse, InvestigationRunSummary, PlanStepView
from app.api.schemas.responses import Outcome, Refusal
from app.api.visualizations import build_visualizations
from app.investigation import Investigation, InvestigationStatus

OUTCOMES: dict[InvestigationStatus, Outcome] = {
    "completed": "answered",
    "budget_exhausted": "partial",
    "insufficient_evidence": "insufficient_evidence",
    "refused": "refused",
    "unsupported": "unsupported",
    "failed": "failed",
    "cancelled": "failed",
    "planned": "failed",
    "running": "failed",
}

# What each investigation stage does, in words a user can follow (``/investigations/stream``).
STAGE_LABELS = {
    "started": "Investigation started",
    "understanding": "Understanding the objective",
    "planning": "Planning the analysis",
    "plan": "Analysis plan ready",
    "step_started": "Running an analysis step",
    "step_finished": "Analysis step finished",
    "validating": "Validating findings",
    "synthesizing": "Preparing the decision brief",
    "finished": "Investigation finished",
}


def classify(investigation: Investigation) -> tuple[Outcome, Refusal | None]:
    outcome = OUTCOMES[investigation.status]
    message = investigation.message or ""
    if investigation.status == "refused":
        policy = any(
            e.event_type == "suspicious_prompt" and e.decision == "deny" for e in investigation.security_events
        )
        return outcome, Refusal(kind="policy" if policy else "invalid_input", message=message)
    if investigation.status == "unsupported":
        return outcome, Refusal(kind="out_of_scope", message=message)
    return outcome, None


def plan_view(investigation: Investigation) -> list[PlanStepView]:
    if investigation.plan is None:
        return []
    records = {r.step_id: r for r in investigation.steps}
    views = []
    for step in investigation.plan.steps:
        record = records[step.step_id]
        views.append(
            PlanStepView(
                step_id=step.step_id,
                title=step.title,
                area=step.area,
                tool_name=step.tool_name,
                depends_on=step.depends_on,
                condition=step.condition,
                status=record.status,
                reason=record.reason,
                duration_ms=round(record.execution_time_ms, 1),
                evidence_ids=record.evidence_ids,
                reused_from=record.reused_from,
            )
        )
    return views


def build_investigation_response(
    investigation: Investigation, *, session_id: str | None = None
) -> InvestigationResponse:
    outcome, refusal = classify(investigation)
    evidence, claims = investigation.evidence, investigation.claims
    kpis = kpi_values_of(evidence, claims)
    forecasts = forecast_sections_of(investigation.tool_results, evidence)
    anomalies = anomaly_sections_of(investigation.tool_results, evidence)
    plan = investigation.plan
    purposes = {s.step_id: s.title for s in plan.steps} if plan else {}
    issues = investigation.validation_issues
    return InvestigationResponse(
        request_id=investigation.investigation_id,
        session_id=session_id,
        investigation_id=investigation.investigation_id,
        status=investigation.status,
        outcome=outcome,
        objective=investigation.objective,
        message=investigation.message,
        title=plan.title if plan else None,
        template=investigation.template,
        scope=investigation.scope,
        period=plan.period if plan else None,
        comparison_period=plan.comparison_period if plan else None,
        plan=plan_view(investigation),
        brief=investigation.brief,
        findings=investigation.findings,
        relationships=investigation.relationships,
        kpis=kpis,
        claims=claims,
        evidence=evidence,
        trace=trace_steps_of(investigation.tool_trace, purposes),
        forecasts=[s for s, _ in forecasts],
        anomalies=[s for s, _ in anomalies],
        visualizations=build_visualizations(evidence, claims, kpis, forecasts=forecasts, anomalies=anomalies),
        refusal=refusal,
        run=InvestigationRunSummary(
            investigation_id=investigation.investigation_id,
            llm_provider=investigation.llm_provider,
            llm_model=investigation.llm_model,
            tool_calls=len(investigation.tool_trace),
            efficiency=investigation.efficiency,
            timings=investigation.timings,
            budget=investigation.budget,
            validation={
                "removed": sum(1 for i in issues if i.action == "removed"),
                "downgraded": sum(1 for i in issues if i.action == "downgraded"),
            },
        ),
    )
