"""Driver analysis: which observable factors appear associated with the outcome in the available evidence.

This is not causal inference. A driver is a finding linked to the outcome by one of these rules, each
over validated findings about the same pair of periods:

- ``contributes_to`` (accounting): a member of the outcome's own decomposition (region, segment,
  country) whose change moved with the outcome. The decomposition sums to the change, so its share is
  a measured part of it. With the MRR bridge, a component contributes to an MRR outcome.
- ``supports`` (co-movement): another KPI changed in the direction that usually accompanies the outcome's
  change (``EXPECTED_CO_MOVEMENT``, a documented table: for a revenue decline, churn rising, retention,
  pipeline or win rate falling). It is consistent with the outcome, not a demonstration of a relationship.
- ``correlates_with`` (association): only the explicit association analysis (usage and support load of
  customers who churned versus those retained), linked to the churn finding.
- ``contradicts``: a KPI in the table moved the other way. It is reported as a contradicting signal,
  never as a driver.
- ``contextualizes``: anomalies, forecasts, rankings of levels and indicators without an expected
  direction. Shown as context.

Magnitude, share and confidence are copied from the evidence and the claims; no score is invented.
Wording is non-causal ("in line with", "moved the other way"), and the cross-finding validator checks it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.analytics.labels import metric_label
from app.evidence.models import Direction, EvidenceGraph
from app.investigation.findings import outcome_metrics
from app.investigation.models import Driver, DriverCategory, Finding, FindingRelationship, RelationshipType

_REVENUE_FAMILY = frozenset({"revenue", "revenue_growth", "mrr", "arr"})
_CHURN_FAMILY = frozenset({"logo_churn_rate", "revenue_churn_rate"})
# (expected sign relative to the outcome, category): +1 moves with the outcome, -1 moves against it.
_REVENUE_CO_MOVEMENT: dict[str, tuple[int, DriverCategory]] = {
    "logo_churn_rate": (-1, "customer_churn"),
    "revenue_churn_rate": (-1, "customer_churn"),
    "nrr": (1, "retention"),
    "retention_rate": (1, "retention"),
    "customer_count": (1, "customer_base"),
    "win_rate": (1, "sales_performance"),
    "average_order_value": (1, "sales_performance"),
    "sales_cycle": (-1, "sales_performance"),
    "pipeline_value": (1, "pipeline"),
    "conversion_rate": (1, "conversion"),
    "product_adoption": (1, "product_adoption"),
}
_CHURN_CO_MOVEMENT: dict[str, tuple[int, DriverCategory]] = {
    "nrr": (-1, "retention"),
    "retention_rate": (-1, "retention"),
    "customer_count": (-1, "customer_base"),
    "product_adoption": (-1, "product_adoption"),
    "support_ticket_volume": (1, "support_activity"),
}
EXPECTED_CO_MOVEMENT: dict[str, dict[str, tuple[int, DriverCategory]]] = {
    **{m: _REVENUE_CO_MOVEMENT for m in _REVENUE_FAMILY},
    **{m: _CHURN_CO_MOVEMENT for m in _CHURN_FAMILY},
    "nrr": {
        "logo_churn_rate": (-1, "customer_churn"),
        "revenue_churn_rate": (-1, "customer_churn"),
        "customer_count": (1, "customer_base"),
    },
    "customer_count": {
        "logo_churn_rate": (-1, "customer_churn"),
        "nrr": (1, "retention"),
    },
    "retention_rate": {
        "logo_churn_rate": (-1, "customer_churn"),
        "revenue_churn_rate": (-1, "customer_churn"),
        "nrr": (1, "retention"),
        "customer_count": (1, "customer_base"),
    },
    "win_rate": {
        "sales_cycle": (-1, "sales_performance"),
        "pipeline_value": (1, "pipeline"),
    },
    "pipeline_value": {"win_rate": (1, "sales_performance")},
    "support_ticket_volume": {
        "tickets_per_active_customer": (1, "support_activity"),
        "customer_count": (1, "customer_base"),
    },
    "average_resolution_time": {"support_ticket_volume": (1, "support_activity"), "tickets": (1, "support_activity")},
}
_BRIDGE_CATEGORIES: dict[str, DriverCategory] = {
    "churn": "customer_churn",
    "contraction": "contraction",
    "expansion": "expansion",
    "new": "new_business",
}
_SIGN = {"increase": 1, "decrease": -1}
# Tools that report the same indicator under different names (one driver per indicator, not per tool).
SAME_INDICATOR = {"tickets": "support_ticket_volume"}


@dataclass
class DriverAnalysis:
    relationships: list[FindingRelationship] = field(default_factory=list)
    drivers: list[Driver] = field(default_factory=list)
    contradictions: list[Driver] = field(default_factory=list)
    context: list[Driver] = field(default_factory=list)


# Short, sentence-case names for the indicators a driver can be (acronyms kept as they are).
DRIVER_NAMES: dict[str, str] = {
    "revenue": "Revenue",
    "mrr": "MRR",
    "logo_churn_rate": "Logo churn",
    "revenue_churn_rate": "Revenue churn",
    "nrr": "Net revenue retention",
    "retention_rate": "Retention",
    "customer_count": "Active customers",
    "win_rate": "Win rate",
    "average_order_value": "Average order value",
    "sales_cycle": "Sales cycle",
    "pipeline_value": "Sales pipeline",
    "conversion_rate": "Conversion rate",
    "product_adoption": "Product adoption",
    "support_ticket_volume": "Support ticket volume",
    "tickets": "Support tickets",
    "tickets_per_active_customer": "Tickets per active customer",
    "average_resolution_hours": "Average resolution time",
    "median_resolution_hours": "Median resolution time",
    "average_resolution_time": "Average resolution time",
    "cac": "Customer acquisition cost",
}
_BRIDGE_NAMES = {"churn": "MRR lost to churn", "contraction": "MRR lost to contraction"}


def indicator_name(metric: str | None) -> str:
    """The short, sentence-case name of an indicator (for driver names and brief text)."""
    return _metric_name(metric)


def _metric_name(metric: str | None) -> str:
    if not metric:
        return "The indicator"
    return DRIVER_NAMES.get(metric) or metric_label(metric)


def phrase(name: str) -> str:
    """A name inside a sentence: lower-case first letter, except for acronyms (MRR, NRR)."""
    return name if len(name) > 1 and name[1].isupper() else name[:1].lower() + name[1:]


def _word(direction: Direction | None) -> str:
    return {"increase": "increase", "decrease": "decline"}.get(direction or "", "change")


class _Analysis:
    def __init__(self, findings: list[Finding], outcome: Finding | None, graph: EvidenceGraph):
        self.findings = findings
        self.outcome = outcome
        self.graph = graph
        self.result = DriverAnalysis()

    def evidence_ids(self, *findings: Finding) -> list[str]:
        return list(dict.fromkeys(e for f in findings for e in f.evidence_ids))

    def share(self, finding: Finding) -> float | None:
        claim = self.graph.claims.get(finding.claim_id)
        if claim is None or claim.subject is None:
            return None
        evidence = self.graph.evidence.get(claim.subject.evidence_id)
        if evidence is None:
            return None
        for key in ("share_of_gross_decline", "share_of_gross_increase"):
            value = evidence.attributes.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        return None

    def link(self, source: Finding, target: Finding, relationship: RelationshipType, rule: str) -> None:
        self.result.relationships.append(
            FindingRelationship(
                source_finding_id=source.finding_id,
                target_finding_id=target.finding_id,
                relationship=relationship,
                rule=rule,
                evidence_ids=self.evidence_ids(source, target),
            )
        )

    def add(
        self,
        bucket: list[Driver],
        *,
        name: str,
        category: DriverCategory,
        relationship: RelationshipType,
        statement: str,
        findings: list[Finding],
        direction: Direction | None = None,
        share: float | None = None,
    ) -> None:
        primary = findings[0]
        magnitude = primary.text.split(": ", 1)[1] if ": " in primary.text else None
        bucket.append(
            Driver(
                driver_id="",
                name=name,
                category=category,
                relationship=relationship,
                statement=statement,
                direction=direction,
                magnitude=magnitude,
                share=share,
                confidence=min((f.confidence for f in findings), key=["low", "medium", "high"].index),
                finding_ids=[f.finding_id for f in findings],
                evidence_ids=self.evidence_ids(*findings),
            )
        )

    def same_periods(self, finding: Finding) -> bool:
        assert self.outcome is not None
        return (finding.period, finding.comparison_period) == (self.outcome.period, self.outcome.comparison_period)

    # ------------------------------------------------------------------ rules
    def contributions(self) -> None:
        outcome = self.outcome
        assert outcome is not None
        if outcome.metric not in _REVENUE_FAMILY | {"tickets"} and outcome.metric != "revenue":
            return
        by_claim = {f.claim_id: f for f in self.findings}
        inferences = [f for f in self.findings if f.kind == "concentration"]
        tops = {f.finding_id: f for f in self.findings if f.kind == "contribution" and self.same_periods(f)}
        for top in tops.values():
            if top.metric != outcome.metric or top.direction != outcome.direction:
                continue
            # A member of a drill-down (filters set) contributes to the member it was drilled from.
            parent = outcome
            if top.filters:
                parent = next(
                    (
                        f
                        for f in tops.values()
                        if f is not top
                        and not f.filters
                        and f.dimension in top.filters
                        and f.breakdown in top.filters.values()
                    ),
                    outcome,
                )
            self.link(top, parent, "contributes_to", "decomposition_member")
            support = [top]
            inference = next(
                (
                    i
                    for i in inferences
                    if i.dimension == top.dimension and i.breakdown == top.breakdown and i.filters == top.filters
                ),
                None,
            )
            if inference is not None:
                support.append(inference)
            within = "" if not top.filters else f" (within {', '.join(f'{k} {v}' for k, v in top.filters.items())})"
            name = f"{str(top.dimension).replace('_', ' ').capitalize()} {top.breakdown}{within}"
            category: DriverCategory = (
                "regional_performance" if top.dimension in ("region", "country") else "segment_performance"
            )
            self.add(
                self.result.drivers,
                name=name,
                category=category,
                relationship="contributes_to",
                statement=inference.text if inference is not None else top.text,
                findings=support,
                direction=top.direction,
                share=self.share(top),
            )
        del by_claim

    def bridge(self) -> None:
        outcome = self.outcome
        assert outcome is not None
        component = next((f for f in self.findings if f.kind == "bridge_component"), None)
        if component is None or outcome.metric not in _REVENUE_FAMILY or outcome.direction != "decrease":
            return
        if component.period != outcome.period:
            return
        member = (component.breakdown or "").lower()
        category = _BRIDGE_CATEGORIES.get(member, "contraction")
        relationship: RelationshipType = "contributes_to" if outcome.metric == "mrr" else "supports"
        self.link(component, outcome, relationship, "mrr_bridge_component")
        coincidence = next((f for f in self.findings if f.kind == "coincidence"), None)
        support = [component] + ([coincidence] if coincidence is not None else [])
        self.add(
            self.result.drivers,
            name=_BRIDGE_NAMES.get(member, "Negative MRR movement"),
            category=category,
            relationship=relationship,
            statement=(
                f"{component.text} This is in line with the {phrase(_metric_name(outcome.metric))} "
                f"{_word(outcome.direction)} in the same period."
            ),
            findings=support,
            direction="decrease",
        )

    def co_movements(self) -> None:
        outcome = self.outcome
        assert outcome is not None
        table = EXPECTED_CO_MOVEMENT.get(outcome.metric or "", {})
        outcome_sign = _SIGN.get(outcome.direction or "")
        seen: set[str] = set()
        for finding in self.findings:
            if finding is outcome or finding.kind != "change" or finding.dimension is not None:
                continue
            indicator = SAME_INDICATOR.get(str(finding.metric), str(finding.metric))
            if finding.metric in outcome_metrics(outcome.metric) or indicator in seen:
                continue
            if not self.same_periods(finding) or finding.filters != outcome.filters:
                continue
            seen.add(indicator)
            sign = _SIGN.get(finding.direction or "")
            name = _metric_name(finding.metric)
            if finding.metric not in table or sign is None or outcome_sign is None:
                self.link(finding, outcome, "contextualizes", "indicator_without_expected_direction")
                self.add(
                    self.result.context,
                    name=name,
                    category="support_activity" if "ticket" in str(finding.metric) else "customer_base",
                    relationship="contextualizes",
                    statement=finding.text,
                    findings=[finding],
                    direction=finding.direction,
                )
                continue
            expected, category = table[str(finding.metric)]
            outcome_word = f"{phrase(_metric_name(outcome.metric))} {_word(outcome.direction)}"
            if sign * outcome_sign == expected:
                self.link(finding, outcome, "supports", "same_period_co_movement")
                self.add(
                    self.result.drivers,
                    name=name,
                    category=category,
                    relationship="supports",
                    statement=f"{finding.text} This moved in line with the {outcome_word} in the same period.",
                    findings=[finding],
                    direction=finding.direction,
                )
            else:
                self.link(finding, outcome, "contradicts", "same_period_opposite_movement")
                self.add(
                    self.result.contradictions,
                    name=name,
                    category=category,
                    relationship="contradicts",
                    statement=(
                        f"{finding.text} This moved the other way from the {outcome_word}, so the available "
                        f"evidence does not point to {phrase(name)} as a contributor."
                    ),
                    findings=[finding],
                    direction=finding.direction,
                )

    def associations(self) -> None:
        outcome = self.outcome
        assert outcome is not None
        data = next((f for f in self.findings if f.kind == "association_data"), None)
        if data is None:
            return
        churn = outcome if outcome.metric in _CHURN_FAMILY else None
        if churn is None:
            churn = next(
                (
                    f
                    for f in self.findings
                    if f.kind == "change" and f.metric in _CHURN_FAMILY and f.dimension is None and self.same_periods(f)
                ),
                None,
            )
        if churn is None or churn.direction != "increase":
            return  # the association describes customers who churned: relevant when churn rose
        churn_supports = churn is outcome or any(
            d.relationship == "supports" and churn is not None and churn.finding_id in d.finding_ids
            for d in self.result.drivers
        )
        if churn is None or not churn_supports:
            return
        note = next((f for f in self.findings if f.kind == "association"), None)
        self.link(data, churn, "correlates_with", "association_analysis")
        self.add(
            self.result.drivers,
            name="Usage and support load before churn",
            category="product_adoption",
            relationship="correlates_with",
            statement=data.text + (f" {note.text}" if note is not None else ""),
            findings=[data] + ([note] if note is not None else []),
        )

    def context(self) -> None:
        outcome = self.outcome
        for finding in self.findings:
            if finding.kind == "anomaly" and outcome is not None and finding.metric in outcome_metrics(outcome.metric):
                claim = self.graph.claims.get(finding.claim_id)
                evidence = self.graph.evidence.get(claim.subject.evidence_id) if claim and claim.subject else None
                if evidence is None or not evidence.details.get("is_anomaly"):
                    continue
                self.link(finding, outcome, "contextualizes", "flagged_anomaly")
                self.add(
                    self.result.context,
                    name=f"Statistically unusual {phrase(_metric_name(finding.metric))}",
                    category="anomaly",
                    relationship="contextualizes",
                    statement=finding.text,
                    findings=[finding],
                )
            elif finding.kind == "ranking" and outcome is not None and finding.period == outcome.period:
                self.link(finding, outcome, "contextualizes", "level_ranking")
                self.add(
                    self.result.context,
                    name=f"{str(finding.dimension or 'member').replace('_', ' ').capitalize()} {finding.breakdown}",
                    category="segment_performance",
                    relationship="contextualizes",
                    statement=finding.text,
                    findings=[finding],
                )


def analyze_drivers(findings: list[Finding], outcome: Finding | None, graph: EvidenceGraph) -> DriverAnalysis:
    """Relationships and drivers for the outcome (none when the outcome could not be measured)."""
    analysis = _Analysis(findings, outcome, graph)
    if outcome is None or outcome.direction not in ("increase", "decrease"):
        return analysis.result
    analysis.contributions()
    analysis.bridge()
    analysis.co_movements()
    analysis.associations()
    analysis.context()
    result = analysis.result
    # Contributions first (largest share first), then co-movements and associations in plan order.
    order = {"contributes_to": 0, "supports": 1, "correlates_with": 2}
    result.drivers.sort(key=lambda d: (order.get(d.relationship, 3), -(d.share or 0.0)))
    for prefix, bucket in (("D", result.drivers), ("X", result.contradictions), ("K", result.context)):
        for index, driver in enumerate(bucket, start=1):
            driver.driver_id = f"{prefix}{index}"
    return result
