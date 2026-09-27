"""The decision brief: an evidence-backed summary of an investigation, assembled by rules.

- **Executive summary**: sentences taken from validated findings and drivers (the outcome, the largest
  contribution, what moved in line with it and what moved the other way), then the non-causal caveat.
  It is checked like any answer: every number must appear in the cited evidence, and causal wording is
  rejected. A summary that fails is replaced by the outcome finding alone.
- **Key findings**: the outcome, the findings behind each driver and contradiction, then the first
  finding of each other area.
- **Risks**: adverse movements (``ADVERSE_DIRECTION``), each citing its finding.
- **Uncertainty**: what the evidence cannot establish, from rules (causal limits, one period pair,
  contradictions, steps that did not run, claim limitations and forecast, anomaly and association notes).
- **Sections** (management brief): one per area that produced validated findings; no empty section.

The brief is bounded by ``max_investigation_output_chars``: lower-priority items are dropped first and a
note says so.
"""

from __future__ import annotations

import re

from app.agent.response import build_caveats
from app.evidence.models import EvidenceGraph
from app.investigation.drivers import DriverAnalysis, indicator_name, phrase
from app.investigation.models import (
    STOP_MESSAGE,
    AnalysisPlan,
    Area,
    BriefSection,
    DecisionBrief,
    Driver,
    Finding,
    Recommendation,
    RiskItem,
    StepRecord,
    ValidationIssue,
)
from app.investigation.recommendations import is_adverse
from app.investigation.validation import text_problems

MAX_KEY_FINDINGS = 10
MAX_RISKS = 6
SECTION_TITLES: dict[Area, str] = {
    "revenue": "Revenue",
    "customers": "Customer health",
    "sales": "Sales",
    "marketing": "Marketing",
    "product": "Product",
    "support": "Support",
    "anomalies": "Anomalies",
    "forecast": "Forecast",
}
NOT_CAUSAL = "These are contributions and co-movements in the data, not established causes."
TRIMMED = "Some lower-priority items were omitted to keep the brief within its size limit."


def _names(drivers: list[Driver]) -> str:
    names = list(dict.fromkeys(phrase(d.name) for d in drivers))
    if len(names) > 4:
        names = [*names[:3], f"{len(names) - 3} other indicators"]
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


_RISES = re.compile(r"\b(increas\w*|ris(?:e|es|ing|en)|rose|grow\w*|climb\w*|surg\w*|going up)\b", re.IGNORECASE)
_FALLS = re.compile(r"\b(declin\w*|decreas\w*|fall\w*|fell|drop\w*|slow\w*|shrink\w*|going down)\b", re.IGNORECASE)


def premise_note(objective: str, outcome: Finding | None) -> str | None:
    """When the objective presumes a direction the evidence contradicts, say so (the premise is not adopted)."""
    if outcome is None or outcome.direction not in ("increase", "decrease"):
        return None
    rises, falls = bool(_RISES.search(objective)), bool(_FALLS.search(objective))
    if rises == falls:
        return None  # no direction, or both ("growth is slowing")
    presumed = "increase" if rises else "decrease"
    if presumed == outcome.direction:
        return None
    actual = "a decline" if outcome.direction == "decrease" else "an increase"
    return (
        f"The objective describes {'an increase' if rises else 'a decline'}, but the evidence shows {actual} "
        f"from {outcome.comparison_period} to {outcome.period}."
    )


def _claims(finding_ids: list[str], by_id: dict[str, Finding]) -> list[str]:
    return [by_id[i].claim_id for i in finding_ids if i in by_id]


class _Composer:
    def __init__(
        self,
        plan: AnalysisPlan,
        objective: str,
        findings: list[Finding],
        outcome: Finding | None,
        analysis: DriverAnalysis,
        graph: EvidenceGraph,
    ):
        self.plan = plan
        self.objective = objective
        self.findings = findings
        self.by_id = {f.finding_id: f for f in findings}
        self.outcome = outcome
        self.analysis = analysis
        self.graph = graph
        self.issues: list[ValidationIssue] = []

    def summary(self) -> tuple[str, list[str]]:
        if self.plan.template == "management_brief":
            return self.brief_summary()
        outcome, analysis = self.outcome, self.analysis
        if outcome is None:
            return "The outcome of this investigation could not be measured from the available data.", []
        parts, cited = [outcome.text], [outcome.finding_id]
        premise = premise_note(self.objective, outcome)
        if premise is not None:
            parts.append(premise)
        contribution = next((d for d in analysis.drivers if d.relationship == "contributes_to"), None)
        if contribution is not None:
            parts.append(contribution.statement)
            cited += contribution.finding_ids
        supporting = [d for d in analysis.drivers if d.relationship in ("supports", "correlates_with")]
        if supporting:
            parts.append(f"In the same period, {_names(supporting)} moved in line with this change.")
            cited += [i for d in supporting for i in d.finding_ids]
        if analysis.contradictions:
            pronoun = "it" if len(analysis.contradictions) == 1 else "them"
            parts.append(
                f"{_sentence(_names(analysis.contradictions))} moved the other way, so the evidence does not point "
                f"to {pronoun}."
            )
            cited += [i for d in analysis.contradictions for i in d.finding_ids]
        if analysis.drivers:
            parts.append(NOT_CAUSAL)
        return " ".join(parts), list(dict.fromkeys(cited))

    def brief_summary(self) -> tuple[str, list[str]]:
        parts = [f"Management brief for {self.plan.period_label}, compared with {self.plan.comparison_label}."]
        cited: list[str] = []
        for area in ("revenue", "customers", "sales", "support"):
            finding = next(
                (f for f in self.findings if f.area == area and f.kind == "change" and f.dimension is None), None
            )
            if finding is not None:
                parts.append(finding.text)
                cited.append(finding.finding_id)
        return " ".join(parts), cited

    def checked_summary(self) -> tuple[str, list[str]]:
        text, cited = self.summary()
        claim_ids = _claims(cited, self.by_id)
        problems = text_problems(text, claim_ids, self.graph) if claim_ids else []
        if not problems:
            return text, cited
        self.issues.append(
            ValidationIssue(item_id="executive_summary", action="removed", reason="; ".join(problems[:3]))
        )
        if self.outcome is not None:
            return self.outcome.text, [self.outcome.finding_id]
        return "The findings below summarise the evidence.", []

    def key_findings(self) -> list[str]:
        ordered: list[str] = []
        if self.outcome is not None:
            ordered.append(self.outcome.finding_id)
        for driver in [*self.analysis.drivers, *self.analysis.contradictions]:
            ordered.append(driver.finding_ids[0])
        seen_areas = {self.by_id[i].area for i in ordered if i in self.by_id}
        for finding in self.findings:
            if finding.area not in seen_areas and finding.claim_type != "inference":
                ordered.append(finding.finding_id)
                seen_areas.add(finding.area)
        return list(dict.fromkeys(ordered))[:MAX_KEY_FINDINGS]

    def risks(self) -> list[RiskItem]:
        if self.plan.template == "management_brief":
            adverse = [f for f in self.findings if is_adverse(f) and f.dimension is None]
        else:
            candidates = [self.outcome] if self.outcome is not None else []
            candidates += [
                self.by_id[d.finding_ids[0]]
                for d in self.analysis.drivers
                if d.relationship == "supports" and d.finding_ids[0] in self.by_id
            ]
            adverse = [f for f in candidates if is_adverse(f)]
        risks = [RiskItem(text=f.text, finding_ids=[f.finding_id]) for f in adverse]
        risks += [
            RiskItem(text=d.statement, finding_ids=d.finding_ids)
            for d in self.analysis.context
            if d.category == "anomaly"
        ]
        return risks[:MAX_RISKS]

    def sections(self) -> list[BriefSection]:
        if self.plan.template != "management_brief":
            return []
        sections = []
        for area, title in SECTION_TITLES.items():
            ids = [f.finding_id for f in self.findings if f.area == area and f.claim_type != "recommendation"]
            if ids:
                sections.append(BriefSection(area=area, title=title, finding_ids=ids))
        return sections


def uncertainty_notes(
    plan: AnalysisPlan,
    *,
    objective: str,
    findings: list[Finding],
    outcome: Finding | None,
    analysis: DriverAnalysis,
    records: list[StepRecord],
    graph: EvidenceGraph,
    key_claim_ids: list[str],
    causal_question: bool,
    complete: bool,
) -> list[str]:
    notes: list[str] = []
    outcome_name = phrase(indicator_name(plan.outcome_metric or "revenue"))
    premise = premise_note(objective, outcome)
    if premise is not None:
        notes.append(premise)
    if outcome is not None and outcome.metric == "product_adoption" and outcome.filters:
        scope = ", ".join(outcome.filters.values())
        notes.append(
            f"Product adoption is measured per feature: the outcome shown is for {scope}, the first feature the "
            "adoption analysis lists; other features may have moved differently."
        )
    if not complete:
        notes.append(
            f"{STOP_MESSAGE} The findings cover only the steps that ran; no drivers or recommendations were derived."
        )
    if causal_question:
        notes.append(
            "The available data describes what was observed; it cannot establish whether the factor named in the "
            "objective produced the change."
        )
        notes.append(
            "Pricing changes, product releases and market conditions are not recorded in this dataset, so their "
            "effect cannot be assessed."
        )
    if plan.template == "management_brief":
        notes.append(
            f"This brief compares {plan.period_label} with {plan.comparison_label}; it describes the current state "
            "and does not explain causes."
        )
    if analysis.drivers:
        notes.append(
            "Drivers are accounting contributions and same-period co-movements: they show where the change occurred "
            f"and what moved with it, not what produced the {outcome_name} change."
        )
    if any(d.relationship == "supports" for d in analysis.drivers):
        notes.append(
            f"Co-movements compare one pair of periods ({plan.period_label} with {plan.comparison_label}); they are "
            "consistent with, but do not demonstrate, a relationship."
        )
    for driver in analysis.contradictions:
        notes.append(
            f"{driver.name} moved opposite to what would accompany the {outcome_name} change; the evidence does not "
            f"establish that {driver.name.lower()} explains it."
        )
    if complete and outcome is not None and outcome.direction in ("increase", "decrease") and not analysis.drivers:
        notes.append(f"No measured factor moved in line with the {outcome_name} change, so no driver is identified.")
    if outcome is None and plan.template != "management_brief":
        notes.append(f"The {outcome_name} change could not be measured, so no drivers were analysed.")
    for record in records:
        if record.status == "failed" or (record.status == "skipped" and "depends on" in (record.reason or "")):
            notes.append(f"Not analysed: {record.title.lower()} ({(record.reason or 'failed').rstrip('.')}).")
    if any(f.metric is None and f.area == "product" and f.kind == "fact" for f in findings) and not any(
        f.metric == "product_adoption" and f.kind == "change" for f in findings
    ):
        notes.append(
            f"Product adoption was measured as levels per feature in {plan.period_label}, not as a change, so it is "
            "shown as context rather than assessed as a driver."
        )
    notes += build_caveats(graph, key_claim_ids)
    return list(dict.fromkeys(notes))


def cited_claim_ids(key: list[str], analysis: DriverAnalysis, by_id: dict[str, Finding]) -> list[str]:
    ids = list(key) + [
        i for d in [*analysis.drivers, *analysis.contradictions, *analysis.context] for i in d.finding_ids
    ]
    return _claims(list(dict.fromkeys(ids)), by_id)


def compose_brief(
    plan: AnalysisPlan,
    objective: str,
    *,
    findings: list[Finding],
    outcome: Finding | None,
    analysis: DriverAnalysis,
    recommendations: list[Recommendation],
    records: list[StepRecord],
    graph: EvidenceGraph,
    causal_question: bool,
    complete: bool,
    max_chars: int,
) -> tuple[DecisionBrief, list[ValidationIssue]]:
    composer = _Composer(plan, objective, findings, outcome, analysis, graph)
    summary, cited = composer.checked_summary()
    key = composer.key_findings()
    by_id = composer.by_id
    notes = uncertainty_notes(
        plan,
        objective=objective,
        findings=findings,
        outcome=outcome,
        analysis=analysis,
        records=records,
        graph=graph,
        key_claim_ids=cited_claim_ids(key, analysis, by_id),
        causal_question=causal_question,
        complete=complete,
    )
    brief = DecisionBrief(
        title=plan.title,
        objective=objective,
        executive_summary=summary,
        summary_finding_ids=cited,
        key_finding_ids=key,
        drivers=analysis.drivers if complete else [],
        contradictions=analysis.contradictions if complete else [],
        context=analysis.context if complete else [],
        risks=composer.risks(),
        recommendations=recommendations if complete else [],
        uncertainty=notes,
        sections=composer.sections(),
        complete=complete,
    )
    return _bounded(brief, by_id, max_chars), composer.issues


def brief_chars(brief: DecisionBrief, by_id: dict[str, Finding]) -> int:
    texts = [brief.executive_summary, *brief.uncertainty]
    texts += [by_id[i].text for i in brief.key_finding_ids if i in by_id]
    texts += [d.statement for d in [*brief.drivers, *brief.contradictions, *brief.context]]
    texts += [r.text for r in brief.risks] + [r.text for r in brief.recommendations]
    return sum(len(t) for t in texts)


def _bounded(brief: DecisionBrief, by_id: dict[str, Finding], max_chars: int) -> DecisionBrief:
    """Drop lower-priority items until the brief fits: context, risks, non-outcome key findings, contradictions,
    all but the first driver and recommendation, the last uncertainty notes (the first is kept), then the
    remaining drivers and recommendations, and finally the summary is reduced to the lead finding's text."""
    if brief_chars(brief, by_id) <= max_chars:
        return brief
    trimmed = brief.model_copy(update={"uncertainty": [*brief.uncertainty, TRIMMED]})
    while brief_chars(trimmed, by_id) > max_chars:
        lead = by_id.get(trimmed.key_finding_ids[0]) if trimmed.key_finding_ids else None
        if trimmed.context:
            trimmed.context = trimmed.context[:-1]
        elif len(trimmed.risks) > 1:
            trimmed.risks = trimmed.risks[:-1]
        elif len(trimmed.key_finding_ids) > 1:
            trimmed.key_finding_ids = trimmed.key_finding_ids[:-1]
        elif trimmed.contradictions:
            trimmed.contradictions = trimmed.contradictions[:-1]
        elif len(trimmed.drivers) > 1:
            trimmed.drivers = trimmed.drivers[:-1]
        elif len(trimmed.recommendations) > 1:
            trimmed.recommendations = trimmed.recommendations[:-1]
        elif len(trimmed.uncertainty) > 2:
            trimmed.uncertainty = [*trimmed.uncertainty[:-2], TRIMMED]
        elif trimmed.drivers or trimmed.recommendations or trimmed.risks:
            trimmed.drivers, trimmed.recommendations, trimmed.risks = [], [], []
        elif lead is not None and trimmed.executive_summary != lead.text:
            trimmed.executive_summary, trimmed.summary_finding_ids = lead.text, [lead.finding_id]
        else:
            break
    return trimmed
