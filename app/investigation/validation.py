"""Cross-finding validation (Phase 10): run before a decision brief is written.

Claims have already passed the Phase 5/7.1 evidence validator (provenance, successful tool calls,
numbers, direction, observed vs calculated, causal wording, and the claim-evidence identity rules).
Investigations add checks across findings:

1. every finding cites evidence that exists, is unmodified (fingerprint) and comes from a successful call;
2. its identity (metric, unit, period, comparison period, dimension, member, filters) equals that of the
   evidence it is about: the Phase 7.1 identity rules, re-applied after the findings were assembled;
3. every number in its text appears in its evidence, it names no other KPI, and it uses no causal
   wording (the response validator, applied to the finding alone);
4. a comparison uses the investigation's periods, and a level uses one of them;
5. relationships join findings about the same period pair, with a non-causal type;
6. recommendations cite surviving findings and are worded as suggested next steps.

Nothing is repaired. An invalid finding, relationship or recommendation is removed; a partially
supported finding is downgraded to low confidence. Each action is recorded as a ``ValidationIssue``.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.evidence.models import EvidenceGraph
from app.evidence.validation import causal_sentences, recommendation_wording_problems, validate_response
from app.investigation.models import (
    AnalysisPlan,
    Driver,
    Finding,
    FindingRelationship,
    Recommendation,
    RelationshipType,
    ValidationIssue,
)
from app.llm.schemas import ResponseDraftOutput

_DATED_ELSEWHERE = frozenset({"anomaly", "anomaly_summary", "forecast", "forecast_quality"})
NON_CAUSAL: frozenset[RelationshipType] = frozenset(
    {"supports", "correlates_with", "contributes_to", "contradicts", "contextualizes"}
)
_PAIRED: frozenset[RelationshipType] = frozenset({"supports", "contributes_to", "contradicts"})


def text_problems(text: str, claim_ids: list[str], graph: EvidenceGraph, *, max_chars: int = 20000) -> list[str]:
    """The response validator applied to one text: numbers, KPI names, causal wording, labels, direction."""
    draft = ResponseDraftOutput(answer=text, answer_claim_ids=claim_ids)
    return validate_response(draft, graph, max_chars=max_chars).errors


def _identity_problems(finding: Finding, graph: EvidenceGraph) -> list[str]:
    claim = graph.claims.get(finding.claim_id)
    if claim is None:
        return ["its claim no longer exists"]
    about = graph.evidence.get(claim.subject.evidence_id) if claim.subject is not None else None
    if about is None:
        cited = [graph.evidence[e] for e in finding.evidence_ids if e in graph.evidence]
        about = cited[0] if cited else None
    if about is None:
        return ["it cites no evidence"]
    expected = {
        "metric": about.metric,
        "unit": about.unit,
        "period": about.period_label,
        "comparison_period": about.comparison_label,
        "dimension": about.dimension,
        "breakdown": about.dimension_value,
        "filters": about.filters,
    }
    return [
        f"its {name} {getattr(finding, name)!r} differs from its evidence ({value!r})"
        for name, value in expected.items()
        if getattr(finding, name) != value
    ]


def _period_problems(finding: Finding, plan: AnalysisPlan) -> list[str]:
    if finding.kind in _DATED_ELSEWHERE or finding.period is None:
        return []
    if finding.comparison_period is not None:
        if (finding.period, finding.comparison_period) != (plan.period_label, plan.comparison_label):
            return [
                f"it compares {finding.period} with {finding.comparison_period}, not the investigation's "
                f"{plan.period_label} with {plan.comparison_label}"
            ]
        return []
    if finding.period not in (plan.period_label, plan.comparison_label):
        return [f"it is about {finding.period}, outside the investigation's periods"]
    return []


def validate_findings(
    findings: list[Finding],
    graph: EvidenceGraph,
    plan: AnalysisPlan,
    *,
    successful_call_ids: Iterable[str],
) -> tuple[list[Finding], list[ValidationIssue]]:
    successful = set(successful_call_ids)
    tampered = set(graph.verify_integrity())
    kept: list[Finding] = []
    issues: list[ValidationIssue] = []
    for finding in findings:
        problems: list[str] = []
        if not finding.evidence_ids:
            problems.append("it cites no evidence")
        for evidence_id in finding.evidence_ids:
            evidence = graph.evidence.get(evidence_id)
            if evidence is None:
                problems.append(f"it cites unknown evidence {evidence_id}")
            elif evidence_id in tampered:
                problems.append(f"its evidence {evidence_id} was modified after it was recorded")
            elif evidence.tool_call_id not in successful:
                problems.append(f"its evidence {evidence_id} does not come from a successful step")
        problems += _identity_problems(finding, graph)
        problems += _period_problems(finding, plan)
        if not problems:
            problems += text_problems(finding.text, [finding.claim_id], graph)
        if problems:
            issues.append(ValidationIssue(item_id=finding.finding_id, action="removed", reason="; ".join(problems)))
            continue
        claim = graph.claims[finding.claim_id]
        if claim.support_status == "partially_supported" and finding.confidence != "low":
            finding = finding.model_copy(update={"confidence": "low"})
            issues.append(
                ValidationIssue(
                    item_id=finding.finding_id,
                    action="downgraded",
                    reason="only part of its evidence is usable, so its confidence is low",
                )
            )
        kept.append(finding)
    return kept, issues


def same_window(a: Finding, b: Finding) -> bool:
    """The same period, and the same comparison period when both findings have one (a movement within a
    period, such as the MRR bridge, has none)."""
    if a.period != b.period:
        return False
    return a.comparison_period is None or b.comparison_period is None or a.comparison_period == b.comparison_period


def validate_relationships(
    relationships: list[FindingRelationship], findings: list[Finding]
) -> tuple[list[FindingRelationship], list[ValidationIssue]]:
    by_id = {f.finding_id: f for f in findings}
    kept: list[FindingRelationship] = []
    issues: list[ValidationIssue] = []
    for rel in relationships:
        source, target = by_id.get(rel.source_finding_id), by_id.get(rel.target_finding_id)
        problems: list[str] = []
        if source is None or target is None:
            problems.append("it joins a finding that did not pass validation")
        elif rel.relationship not in NON_CAUSAL:
            problems.append(f"{rel.relationship} is not a permitted relationship")
        elif rel.relationship in _PAIRED and not same_window(source, target):
            problems.append("its findings are about different periods")
        if problems:
            item = f"{rel.source_finding_id}->{rel.target_finding_id}"
            issues.append(ValidationIssue(item_id=item, action="removed", reason="; ".join(problems)))
        else:
            kept.append(rel)
    return kept, issues


def validate_drivers(
    drivers: list[Driver], findings: list[Finding], relationships: list[FindingRelationship]
) -> tuple[list[Driver], list[ValidationIssue]]:
    """A driver needs validated findings and a validated relationship of its own type (from its first finding)."""
    known = {f.finding_id for f in findings}
    edges = {(r.source_finding_id, r.relationship) for r in relationships}
    kept: list[Driver] = []
    issues: list[ValidationIssue] = []
    for driver in drivers:
        problems: list[str] = []
        if not driver.finding_ids or not set(driver.finding_ids) <= known:
            problems.append("it rests on a finding that did not pass validation")
        elif (driver.finding_ids[0], driver.relationship) not in edges:
            problems.append("its relationship to the outcome did not pass validation")
        if not driver.evidence_ids:
            problems.append("it cites no evidence")
        if driver.relationship not in NON_CAUSAL:
            problems.append(f"{driver.relationship} is not a permitted relationship")
        if causal_sentences(driver.statement):
            problems.append("its statement uses causal wording the evidence does not establish")
        if problems:
            issues.append(ValidationIssue(item_id=driver.driver_id, action="removed", reason="; ".join(problems)))
        else:
            kept.append(driver)
    return kept, issues


def validate_recommendations(
    recommendations: list[Recommendation], findings: list[Finding], graph: EvidenceGraph
) -> tuple[list[Recommendation], list[ValidationIssue]]:
    known = {f.finding_id for f in findings}
    kept: list[Recommendation] = []
    issues: list[ValidationIssue] = []
    for rec in recommendations:
        problems: list[str] = []
        if not rec.supporting_finding_ids:
            problems.append("it cites no supporting finding")
        elif not set(rec.supporting_finding_ids) <= known:
            problems.append("it cites a finding that did not pass validation")
        problems += recommendation_wording_problems(rec.text)
        if causal_sentences(rec.text) or causal_sentences(rec.rationale):
            problems.append("it uses causal wording the evidence does not establish")
        claim = graph.claims.get(rec.claim_id)
        if claim is None or claim.support_status == "unsupported":
            problems.append("its claim is not supported by evidence")
        if problems:
            issues.append(ValidationIssue(item_id=rec.recommendation_id, action="removed", reason="; ".join(problems)))
        else:
            kept.append(rec)
    return kept, issues
