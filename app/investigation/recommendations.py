"""Grounded recommendations: suggested next analytical steps, each tied to the findings it rests on.

There is no free-form recommendation generator. Each rule below turns a validated driver or finding into
a next step worded as a suggestion ("Investigate", "Review", "Check"), never a directive or a promised
outcome. A recommendation is also a ``recommendation`` claim in the evidence graph, citing the evidence
of its findings, so it passes the same evidence validator as every other claim.

Rules, in order (at most ``MAX_RECOMMENDATIONS``):

1. A concentrated contribution (one member accounts for most of the change): review that member's
   accounts, the most specific member first (a drill-down before its region).
2. Churn or retention moved in line with the outcome: review the customers who churned, starting with
   the segment that had the highest churn when that ranking is available.
3. Pipeline or sales performance moved in line with the outcome: review the pipeline and win/loss outcomes.
4. The outcome month was statistically unusual: check it against operational context that the dataset
   does not record.
5. Management brief: each adverse indicator that no rule above covered.

A recommendation without a supporting finding cannot be built.
"""

from __future__ import annotations

import re

from app.analytics.labels import metric_label
from app.evidence.models import Claim, EvidenceGraph
from app.investigation.drivers import DriverAnalysis
from app.investigation.models import AnalysisPlan, Driver, Finding, Recommendation

MAX_RECOMMENDATIONS = 4
CONCENTRATED = 0.5
# The direction in which an indicator's change is adverse for the business.
ADVERSE_DIRECTION: dict[str, str] = {
    **dict.fromkeys(
        (
            "revenue",
            "mrr",
            "arr",
            "nrr",
            "retention_rate",
            "customer_count",
            "win_rate",
            "pipeline_value",
            "average_order_value",
            "conversion_rate",
            "product_adoption",
        ),
        "decrease",
    ),
    **dict.fromkeys(
        (
            "logo_churn_rate",
            "revenue_churn_rate",
            "support_ticket_volume",
            "tickets",
            "average_resolution_time",
            "cac",
            "sales_cycle",
        ),
        "increase",
    ),
}
_CLAIM_NUMBER = re.compile(r"^C(\d+)$")


def is_adverse(finding: Finding) -> bool:
    return (
        finding.kind == "change"
        and finding.direction is not None
        and ADVERSE_DIRECTION.get(finding.metric or "") == finding.direction
    )


def next_claim_id(graph: EvidenceGraph) -> str:
    """A claim ID after every existing one (IDs stay unique after claims were removed)."""
    numbers = [int(m.group(1)) for c in graph.claims if (m := _CLAIM_NUMBER.match(c))]
    return f"C{max(numbers, default=0) + 1}"


def _word(direction: str | None) -> str:
    return {"increase": "increase", "decrease": "decline"}.get(direction or "", "change")


class _Recommender:
    def __init__(self, findings: list[Finding], graph: EvidenceGraph, plan: AnalysisPlan):
        self.by_id = {f.finding_id: f for f in findings}
        self.graph = graph
        self.plan = plan
        self.items: list[Recommendation] = []
        self.used: set[str] = set()

    def add(self, text: str, findings: list[Finding], uncertainty: str) -> None:
        if len(self.items) >= MAX_RECOMMENDATIONS or not findings:
            return
        evidence_ids = list(dict.fromkeys(e for f in findings for e in f.evidence_ids))
        claim = Claim(
            claim_id=next_claim_id(self.graph),
            text=text,
            claim_type="recommendation",
            evidence_ids=evidence_ids,
            kind="next_step",
            confidence="medium",
        )
        self.graph.add_claim(claim)
        refs = ", ".join(f"{f.finding_id} ({f.label})" for f in findings)
        self.items.append(
            Recommendation(
                recommendation_id=f"R{len(self.items) + 1}",
                claim_id=claim.claim_id,
                text=text,
                rationale=f"Rests on {refs}.",
                supporting_finding_ids=[f.finding_id for f in findings],
                evidence_ids=evidence_ids,
                uncertainty=uncertainty,
            )
        )
        self.used.update(f.finding_id for f in findings)

    def findings_of(self, driver: Driver) -> list[Finding]:
        return [self.by_id[i] for i in driver.finding_ids if i in self.by_id]

    # ------------------------------------------------------------------ rules
    def concentration(self, analysis: DriverAnalysis, outcome: Finding) -> None:
        concentrated = [
            d
            for d in analysis.drivers
            if d.relationship == "contributes_to" and d.share is not None and d.share >= CONCENTRATED
        ]
        if not concentrated:
            return
        # The most specific member first: a drill-down names a member within a member.
        driver = max(concentrated, key=lambda d: (bool(self.findings_of(d)[0].filters), d.share or 0.0))
        member = self.findings_of(driver)[0]
        within = f" within {', '.join(f'{k} {v}' for k, v in member.filters.items())}" if member.filters else ""
        dimension = str(member.dimension or "member").replace("_", " ")
        self.add(
            f"Investigate the {dimension} {member.breakdown} accounts{within} behind the "
            f"{metric_label(outcome.metric or 'revenue').lower()} {_word(outcome.direction)}, for example their "
            f"churn and contraction in {member.period}.",
            self.findings_of(driver),
            "The contribution is an accounting share of the change; it does not show why those accounts changed.",
        )

    def churn(self, analysis: DriverAnalysis, outcome: Finding) -> None:
        churn = next(
            (
                d
                for d in analysis.drivers
                if d.relationship == "supports"
                and d.category in ("customer_churn", "retention")
                and self.findings_of(d)
                and self.findings_of(d)[0].kind == "change"
            ),
            None,
        )
        churn_finding = self.findings_of(churn)[0] if churn is not None else None
        if (
            churn_finding is None
            and outcome.metric in ("logo_churn_rate", "revenue_churn_rate")
            and is_adverse(outcome)
        ):
            churn_finding = outcome
        if churn_finding is None or not is_adverse(churn_finding):
            return
        ranking = next(
            (
                self.by_id[i]
                for d in analysis.context
                if d.category == "segment_performance"
                for i in d.finding_ids
                if i in self.by_id and self.by_id[i].metric in ("logo_churn_rate",) and self.by_id[i].breakdown
            ),
            None,
        )
        start = (
            f", starting with the {ranking.dimension} {ranking.breakdown}, which had the highest logo churn"
            if ranking is not None
            else ""
        )
        self.add(
            f"Review the customers who churned in {churn_finding.period}{start}, and check the risk signals of "
            "similar active accounts.",
            [churn_finding] + ([ranking] if ranking is not None else []),
            "Churn moved in line with the outcome over one pair of periods; that is not a demonstrated relationship.",
        )

    def sales(self, analysis: DriverAnalysis, outcome: Finding) -> None:
        driver = next(
            (
                d
                for d in analysis.drivers
                if d.relationship == "supports" and d.category in ("pipeline", "sales_performance", "conversion")
            ),
            None,
        )
        if driver is None:
            return
        finding = self.findings_of(driver)[0]
        if not is_adverse(finding):
            return
        self.add(
            f"Review the sales pipeline and recent win and loss outcomes for {finding.period} alongside the "
            f"{metric_label(outcome.metric or 'revenue').lower()} {_word(outcome.direction)}.",
            [finding],
            "Sales indicators moved in line with the outcome over one pair of periods; that is not a demonstrated "
            "relationship.",
        )

    def anomaly(self, analysis: DriverAnalysis) -> None:
        driver = next((d for d in analysis.context if d.category == "anomaly"), None)
        if driver is None:
            return
        finding = self.findings_of(driver)[0]
        self.add(
            f"Check the statistically unusual {metric_label(finding.metric or 'revenue').lower()} movement in "
            f"{finding.period} against operational context (pricing, releases, account events) that is not "
            "recorded in this dataset.",
            [finding],
            "An anomaly flag marks an unusual value against recent months; it does not explain it.",
        )

    def brief(self, findings: list[Finding]) -> None:
        for finding in findings:
            if not is_adverse(finding) or finding.finding_id in self.used or finding.dimension is not None:
                continue
            self.add(
                f"Investigate the {metric_label(finding.metric or 'revenue').lower()} "
                f"{_word(finding.direction)} in {finding.period} and which customer groups it affects.",
                [finding],
                "The brief reports what changed; it does not establish why.",
            )


def recommend(
    findings: list[Finding],
    outcome: Finding | None,
    analysis: DriverAnalysis,
    graph: EvidenceGraph,
    plan: AnalysisPlan,
) -> list[Recommendation]:
    recommender = _Recommender(findings, graph, plan)
    if outcome is not None:
        recommender.concentration(analysis, outcome)
        recommender.churn(analysis, outcome)
        recommender.sales(analysis, outcome)
    recommender.anomaly(analysis)
    if plan.template == "management_brief":
        recommender.brief(findings)
    return recommender.items
