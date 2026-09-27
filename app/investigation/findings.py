"""Validated claims -> investigation findings, with their full identity copied from the evidence.

Claims are built by the Phase 4 claim builders (``app/agent/findings.py``) from the evidence every step
returned, and checked by the Phase 5/7.1 evidence validator before they get here. A finding restates a
claim and adds what the brief needs: the business area (from the step that produced its evidence),
the identity of what it is about (metric, unit, period, comparison period, dimension, member and
filters, taken from the claim's subject evidence, never re-derived), and whether it is the
investigation's outcome.

Left out: recommendation claims (investigations derive their own, from findings) and claims about
individual customers (risk rows name customer IDs; the brief works with aggregates).
"""

from __future__ import annotations

from app.evidence.models import Claim, Evidence, EvidenceGraph
from app.investigation.models import FINDING_LABELS, AnalysisPlan, Area, Finding, StepRecord

# Metrics that different tools report under different names for the same outcome.
OUTCOME_ALIASES: dict[str, frozenset[str]] = {
    "support_ticket_volume": frozenset({"support_ticket_volume", "tickets"}),
    "revenue_growth": frozenset({"revenue"}),
}
_CUSTOMER_LEVEL = frozenset({"customer_id"})


def outcome_metrics(outcome: str | None) -> frozenset[str]:
    if outcome is None:
        return frozenset()
    return OUTCOME_ALIASES.get(outcome, frozenset({outcome}))


def _identity(claim: Claim, graph: EvidenceGraph) -> Evidence | None:
    """The evidence a claim is about: its subject, else its only (or first) evidence item."""
    if claim.subject is not None and claim.subject.evidence_id in graph.evidence:
        return graph.evidence[claim.subject.evidence_id]
    cited = [graph.evidence[e] for e in claim.evidence_ids if e in graph.evidence]
    return cited[0] if cited else None


def build_findings(graph: EvidenceGraph, plan: AnalysisPlan, records: list[StepRecord]) -> list[Finding]:
    step_of_call = {r.call_id: r for r in records if r.call_id}
    outcomes = outcome_metrics(plan.outcome_metric)
    findings: list[Finding] = []
    outcome_taken = False
    for claim in graph.claims.values():
        if claim.claim_type == "recommendation" or claim.support_status == "unsupported":
            continue
        about = _identity(claim, graph)
        if about is None or about.dimension in _CUSTOMER_LEVEL:
            continue
        cited = [graph.evidence[e] for e in claim.evidence_ids if e in graph.evidence]
        steps = list(
            dict.fromkeys(step_of_call[e.tool_call_id].step_id for e in cited if e.tool_call_id in step_of_call)
        )
        area: Area = step_of_call[about.tool_call_id].area if about.tool_call_id in step_of_call else "revenue"
        is_outcome = (
            not outcome_taken
            and claim.kind == "change"
            and about.metric in outcomes
            and about.dimension_value is None
            and about.comparison_label == plan.comparison_label
            and about.period_label == plan.period_label
        )
        outcome_taken = outcome_taken or is_outcome
        findings.append(
            Finding(
                finding_id=f"F{len(findings) + 1}",
                claim_id=claim.claim_id,
                text=claim.text,
                claim_type=claim.claim_type,
                label=FINDING_LABELS[claim.claim_type],
                kind=claim.kind,
                area=area,
                metric=about.metric,
                unit=about.unit,
                period=about.period_label,
                comparison_period=about.comparison_label,
                dimension=about.dimension,
                breakdown=about.dimension_value,
                filters=dict(about.filters),
                evidence_ids=list(claim.evidence_ids),
                step_ids=steps,
                direction=claim.direction,
                confidence=claim.confidence,
                primary=is_outcome,
                limitations=list(claim.limitations),
            )
        )
    return findings


def outcome_finding(findings: list[Finding]) -> Finding | None:
    return next((f for f in findings if f.primary), None)
