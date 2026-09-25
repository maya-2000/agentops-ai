"""Deterministic finding builders: evidence -> typed claims.

Claims are built from the evidence that tools actually returned, so the agent's statements are
data-driven and never hard-coded. Each builder reads evidence of one kind and writes claims with
explicit types and support links:

- ``observed_fact`` / ``calculated_result``: restate one evidence item (its numbers are asserted
  and later checked against the evidence for contradictions).
- ``inference``: a reasoned reading across evidence ("was concentrated in", "coincided with",
  "is an association"). It is never presented as an observation.
- ``recommendation``: a next analytical step grounded in a supported inference or flagged anomaly.

Wording is non-causal by construction. ``CONCENTRATION_SHARE`` is the documented rule for calling a
change "concentrated": one member accounts for at least half of the gross decline (or increase).
"""

from __future__ import annotations

from collections.abc import Iterable

from app.agent.request import ValidatedRequest
from app.analytics.dimensions import DIMENSIONS
from app.evidence.formatting import format_percent, format_value
from app.evidence.models import Claim, ClaimType, Evidence, EvidenceGraph, NumericAssertion
from app.llm.schemas import Intent

CONCENTRATION_SHARE = 0.5
MAX_GENERIC_CLAIMS = 5
_SPECIFICITY = {"country": 0, "segment": 1, "plan": 2, "industry": 3, "region": 4}
_TOPICS = {
    "logo_churn_rate": "customer churn",
    "revenue_churn_rate": "revenue churn",
    "retention_rate": "retention",
    "support_ticket_volume": "the change in ticket volume",
    "revenue": "the change in revenue",
    "mrr": "the change in MRR",
}


def _within(filters: dict[str, str]) -> str:
    return f" within {', '.join(f'{k}={v}' for k, v in filters.items())}" if filters else ""


class _Builder:
    def __init__(self, graph: EvidenceGraph, request: ValidatedRequest):
        self.graph = graph
        self.request = request

    # ------------------------------------------------------------------ helpers
    def evidence(self, *, operation: str | None = None, tool: str | None = None) -> list[Evidence]:
        return [
            e
            for e in self.graph.evidence.values()
            if (operation is None or e.operation == operation) and (tool is None or e.tool_name == tool)
        ]

    def claim(
        self,
        text: str,
        claim_type: ClaimType,
        evidence: Iterable[Evidence],
        *,
        kind: str,
        primary: bool = False,
        assertions: Iterable[tuple[Evidence, str]] = (),
        direction_from: tuple[Evidence, str] | None = None,
        about: Evidence | None = None,
        limitations: Iterable[str] = (),
        confidence: str = "high",
    ) -> Claim:
        items = list({e.evidence_id: e for e in evidence}.values())
        numeric = []
        for item, field in assertions:
            value = item.value if field == "value" else item.attributes.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric.append(NumericAssertion(evidence_id=item.evidence_id, field=field, value=float(value)))
        direction = None
        direction_evidence = None
        if direction_from is not None:
            item, field = direction_from
            value = item.value if field == "value" else item.attributes.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                direction = "increase" if value > 0 else "decrease" if value < 0 else "none"
                direction_evidence = NumericAssertion(evidence_id=item.evidence_id, field=field, value=float(value))
        claim = Claim(
            claim_id=self.graph.next_claim_id(),
            text=text,
            claim_type=claim_type,
            evidence_ids=[e.evidence_id for e in items],
            kind=kind,
            primary=primary,
            numeric_assertions=numeric,
            direction=direction,  # type: ignore[arg-type]
            direction_evidence=direction_evidence,
            about_period_start=about.period_start if about else None,
            about_period_end=about.period_end if about else None,
            limitations=list(dict.fromkeys(limitations)),
            confidence=confidence,  # type: ignore[arg-type]
        )
        self.graph.add_claim(claim)
        return claim

    def fact(self, e: Evidence, *, kind: str, primary: bool = False) -> Claim:
        claim_type: ClaimType = "observed_fact" if e.evidence_type == "observed" else "calculated_result"
        return self.claim(e.statement, claim_type, [e], kind=kind, primary=primary, assertions=[(e, "value")], about=e)

    def in_request_period(self, e: Evidence) -> bool:
        period = self.request.period
        if period is None or e.period_start is None or e.period_end is None:
            return True
        return e.period_start <= period.end and e.period_end >= period.start

    # ------------------------------------------------------------------ builders
    def kpi_values(self) -> None:
        for e in self.evidence(tool="get_kpi"):
            if e.status != "ok" or e.value is None or "absolute_change" in e.attributes:
                continue
            if e.dimension_value is None:
                primary = e.metric == self.request.metric and self.in_request_period(e) and e.comparison_label is None
                self.fact(e, kind="kpi_value", primary=primary and self.request.intent != Intent.PERIOD_COMPARISON)
        self._kpi_rankings()

    def _kpi_rankings(self) -> None:
        rows = [e for e in self.evidence(tool="get_kpi") if e.dimension_value is not None and e.status == "ok"]
        if not rows:
            return
        ranked = sorted(rows, key=lambda e: int(e.attributes.get("rank") or 0))
        top, bottom = ranked[0], ranked[-1]
        name = self.request.metric_name or top.metric or "The metric"
        wants_lowest = self.request.analysis_type == "lowest"
        chosen = bottom if wants_lowest else top
        word = "lowest" if chosen is bottom else "highest"
        group = (
            DIMENSIONS[chosen.dimension].display_name.lower() if chosen.dimension in DIMENSIONS else chosen.dimension
        )
        self.claim(
            f"Among {group}s, {chosen.dimension_value} had the {word} {name} for {chosen.period_label}: "
            f"{chosen.display_value}.",
            "calculated_result",
            [chosen],
            kind="ranking",
            primary=True,
            assertions=[(chosen, "value")],
            about=chosen,
        )

    def changes(self) -> None:
        for e in self.graph.evidence.values():
            if e.status != "ok" or e.dimension_value is not None:
                continue
            if "absolute_change" not in e.attributes and "percentage_change" not in e.attributes:
                continue
            if e.operation == "analyze_revenue.decompose_revenue_change":
                continue
            field = "absolute_change" if "absolute_change" in e.attributes else "percentage_change"
            matches = e.metric == self.request.metric or (
                e.metric == "tickets" and self.request.metric == "support_ticket_volume"
            )
            primary = (self.request.intent == Intent.MIXED_INVESTIGATION and e.metric in ("revenue", "tickets")) or (
                self.request.intent in (Intent.PERIOD_COMPARISON, Intent.REVENUE_INVESTIGATION, Intent.SUPPORT_ANALYSIS)
                and matches
            )
            self.claim(
                e.statement,
                "calculated_result",
                [e],
                kind="change",
                primary=primary,
                assertions=[(e, field), (e, "percentage_change")],
                direction_from=(e, field),
                about=e,
            )
        for e in self.evidence(operation="analyze_revenue.revenue_change"):
            if e.evidence_type == "observed" and e.status == "ok":
                self.fact(e, kind="kpi_value")

    def contributions(self) -> list[Claim]:
        inferences: list[Claim] = []
        totals = [
            e for e in self.evidence(operation="analyze_revenue.decompose_revenue_change") if e.dimension_value is None
        ]
        totals.sort(key=lambda e: (not e.filters, _SPECIFICITY.get(e.dimension or "", 9)))
        most_specific = True
        for total in totals:
            members = [
                e
                for e in self.evidence(operation="analyze_revenue.decompose_revenue_change")
                if e.dimension_value is not None and e.dimension == total.dimension and e.filters == total.filters
            ]
            top = next((m for m in members if m.attributes.get("rank") == 1), None)
            if top is None or total.value in (None, 0):
                continue
            declining = float(total.value or 0) < 0
            share_key = "share_of_gross_decline" if declining else "share_of_gross_increase"
            share = top.attributes.get(share_key)
            word = "decline" if declining else "increase"
            within = _within(total.filters)
            primary = self.request.intent == Intent.DIMENSIONAL_COMPARISON and not total.filters
            self.claim(
                f"The largest contribution to the revenue {word}{within} by {top.dimension} came from "
                f"{top.dimension_value}: {top.statement.split(': ', 1)[1]}",
                "calculated_result",
                [top, total],
                kind="contribution",
                primary=primary,
                assertions=[(top, "absolute_change"), (top, share_key)]
                if share is not None
                else [(top, "absolute_change")],
                direction_from=(top, "absolute_change"),
                about=top,
            )
            if not isinstance(share, (int, float)):
                continue
            concentrated = share >= CONCENTRATION_SHARE
            if concentrated:
                text = (
                    f"The available evidence indicates the revenue {word}{within} was concentrated in {top.dimension} "
                    f"{top.dimension_value}, which accounted for {format_percent(share)} of the gross {word}."
                )
            else:
                text = (
                    f"The revenue {word}{within} was spread across several {top.dimension}s; the largest single "
                    f"contribution, {top.dimension_value}, accounted for {format_percent(share)} of the gross {word}."
                )
            primary_inference = most_specific and concentrated and self.request.intent == Intent.REVENUE_INVESTIGATION
            most_specific = most_specific and not concentrated
            inferences.append(
                self.claim(
                    text,
                    "inference",
                    [top, total],
                    kind="concentration",
                    primary=primary_inference,
                    assertions=[(top, share_key)],
                    about=top,
                    confidence="medium",
                )
            )
        return inferences

    def bridge(self) -> list[Claim]:
        rows = {e.dimension_value: e for e in self.evidence(operation="analyze_revenue.revenue_bridge")}
        inferences: list[Claim] = []
        net = rows.get("net_change")
        mrr_change_known = any(
            e.metric == "mrr" and "absolute_change" in e.attributes for e in self.evidence(tool="get_kpi")
        )
        if net is not None and net.status == "ok" and not mrr_change_known:
            primary = self.request.metric == "mrr" and self.request.intent == Intent.PERIOD_COMPARISON
            self.claim(
                net.statement,
                "calculated_result",
                [net],
                kind="change",
                primary=primary,
                assertions=[(net, "value")],
                direction_from=(net, "value"),
                about=net,
            )
        movements = [rows[k] for k in ("new", "expansion", "contraction", "churn") if k in rows]
        negatives = [m for m in movements if isinstance(m.value, (int, float)) and m.value < 0]
        if negatives:
            largest = min(negatives, key=lambda m: float(m.value or 0))
            self.claim(
                f"{str(largest.dimension_value).capitalize()} was the largest negative MRR movement in "
                f"{largest.period_label}: {largest.statement.split(': ', 1)[1]}",
                "calculated_result",
                [largest, *negatives],
                kind="bridge_component",
                assertions=[(largest, "value")],
                about=largest,
            )
            revenue_change = next(
                (
                    e
                    for e in self.graph.evidence.values()
                    if e.metric == "revenue"
                    and "absolute_change" in e.attributes
                    and e.dimension_value is None
                    and e.operation != "analyze_revenue.decompose_revenue_change"
                ),
                None,
            )
            if revenue_change is not None and float(revenue_change.value or 0) < 0:
                parts = " and ".join(
                    f"{m.dimension_value} of {format_value(abs(float(m.value or 0)), 'SGD')}" for m in negatives
                )
                inferences.append(
                    self.claim(
                        f"The revenue decline in {revenue_change.period_label} coincided with negative MRR movements: "
                        f"{parts}.",
                        "inference",
                        [revenue_change, *negatives],
                        kind="coincidence",
                        assertions=[(m, "value") for m in negatives],
                        about=revenue_change,
                        confidence="medium",
                    )
                )
        return inferences

    def churn(self) -> list[Claim]:
        claims: list[Claim] = []
        for e in self.evidence(operation="analyze_customers.churn_summary"):
            if e.metric == "logo_churn_rate" and e.status == "ok":
                claims.append(
                    self.fact(e, kind="kpi_value", primary=self.request.intent == Intent.CUSTOMER_INVESTIGATION)
                )
        rows = [e for e in self.evidence(operation="analyze_customers.churn_by_dimension") if e.status == "ok"]
        groups: dict[tuple[str | None, tuple[tuple[str, str], ...]], list[Evidence]] = {}
        for e in rows:
            groups.setdefault((e.dimension, tuple(sorted(e.filters.items()))), []).append(e)
        for members in groups.values():
            valid = [m for m in members if m.attributes.get("sufficient_sample") is True]
            if not valid:
                continue
            ranked = sorted(valid, key=lambda m: -float(m.value or 0))
            lowest = self.request.analysis_type == "lowest"
            top, word = (ranked[-1], "lowest") if lowest else (ranked[0], "highest")
            notes = []
            if len(ranked) > 1:
                other = ranked[-2] if lowest else ranked[1]
                low, high = top.attributes.get("ci_low"), top.attributes.get("ci_high")
                o_low, o_high = other.attributes.get("ci_low"), other.attributes.get("ci_high")
                if all(isinstance(v, (int, float)) for v in (low, high, o_low, o_high)) and not (
                    float(low) > float(o_high) or float(high) < float(o_low)  # type: ignore[arg-type]
                ):
                    notes.append(
                        f"The 95% intervals of {top.dimension_value} and {other.dimension_value} overlap, so the "
                        "ranking is not statistically distinguishable."
                    )
            primary = (
                self.request.intent in (Intent.DIMENSIONAL_COMPARISON, Intent.CUSTOMER_INVESTIGATION)
                and not top.filters
            )
            within = _within(top.filters)
            claims.append(
                self.claim(
                    f"{top.dimension_value} had the {word} logo churn among {top.dimension}s{within} in "
                    f"{top.period_label}: {top.display_value} ({top.statement.split('(', 1)[1].split(';')[0]}).",
                    "calculated_result",
                    [top],
                    kind="ranking",
                    primary=primary,
                    assertions=[(top, "value"), (top, "churned_customers"), (top, "opening_customers")],
                    about=top,
                    limitations=notes,
                )
            )
        return claims

    def associations(self) -> list[Claim]:
        rows = {e.dimension_value: e for e in self.evidence(operation="analyze_customers.usage_churn_relationship")}
        churned, retained = rows.get("churned in period"), rows.get("retained")
        if churned is None or retained is None:
            return []
        c_t, r_t = (
            churned.attributes.get("tickets_per_customer_recent_90d"),
            retained.attributes.get("tickets_per_customer_recent_90d"),
        )
        c_u, r_u = churned.attributes.get("median_usage_ratio"), retained.attributes.get("median_usage_ratio")
        self.claim(
            f"In {churned.period_label}, customers who churned averaged {float(c_t or 0):.2f} support tickets per "
            f"customer in the prior 90 days versus {float(r_t or 0):.2f} for retained customers; median usage "
            f"ratios were {float(c_u or 0):.2f} and {float(r_u or 0):.2f}.",
            "calculated_result",
            [churned, retained],
            kind="association_data",
            assertions=[
                (churned, "tickets_per_customer_recent_90d"),
                (retained, "tickets_per_customer_recent_90d"),
                (churned, "median_usage_ratio"),
                (retained, "median_usage_ratio"),
            ],
            about=churned,
        )
        return [
            self.claim(
                "These are associations observed before churn; the available analysis does not establish that "
                "support load or usage levels caused customers to churn.",
                "inference",
                [churned, retained],
                kind="association",
                confidence="medium",
            )
        ]

    def forecasts(self) -> None:
        points = [e for e in self.evidence(tool="forecast_metric")]
        for e in points:
            if e.evidence_type != "forecast":
                continue
            self.claim(
                f"{e.statement.replace('Forecast ', 'Forecast of ', 1)}",
                "calculated_result",
                [e],
                kind="forecast",
                primary=self.request.intent == Intent.FORECAST,
                assertions=[(e, "value"), (e, "lower_bound"), (e, "upper_bound")],
            )
        first = next((e for e in points if e.evidence_type == "forecast"), None)
        if first is not None:
            mae, base = first.details.get("backtest_mae"), first.details.get("baseline_mae")
            if isinstance(mae, (int, float)) and isinstance(base, (int, float)):
                self.claim(
                    f"In rolling-origin backtests the forecast model {first.details.get('model')} had a mean absolute "
                    f"error of {format_value(mae, first.unit)}, versus {format_value(base, first.unit)} for the naive "
                    "baseline (forecast quality, not a guarantee).",
                    "calculated_result",
                    [first],
                    kind="forecast_quality",
                )

    def anomalies(self) -> list[Claim]:
        flagged_claims: list[Claim] = []
        for e in self.evidence(tool="detect_anomalies"):
            if e.status != "ok":
                continue
            if e.evidence_type == "derived":
                self.claim(
                    e.statement,
                    "calculated_result",
                    [e],
                    kind="anomaly_summary",
                    primary=self.request.intent == Intent.ANOMALY_DETECTION,
                    assertions=[(e, "value")],
                )
            elif e.evidence_type == "anomaly":
                claim = self.claim(
                    e.statement,
                    "calculated_result",
                    [e],
                    kind="anomaly",
                    assertions=[(e, "value"), (e, "expected_value")],
                    about=e,
                )
                if e.details.get("is_anomaly"):
                    flagged_claims.append(claim)
        return flagged_claims

    def generic(self) -> None:
        handled = {
            "get_kpi",
            "analyze_revenue.revenue_change",
            "analyze_revenue.decompose_revenue_change",
            "analyze_revenue.revenue_bridge",
            "analyze_customers.churn_summary",
            "analyze_customers.churn_by_dimension",
            "analyze_customers.usage_churn_relationship",
            "forecast_metric",
            "detect_anomalies",
        }
        by_operation: dict[str, list[Evidence]] = {}
        for e in self.graph.evidence.values():
            if e.operation in handled or e.status != "ok":
                continue
            if "absolute_change" in e.attributes or "percentage_change" in e.attributes:
                continue  # already a change claim
            by_operation.setdefault(e.operation, []).append(e)
        domain = self.request.intent in (
            Intent.SALES_ANALYSIS,
            Intent.MARKETING_ANALYSIS,
            Intent.SUPPORT_ANALYSIS,
            Intent.PRODUCT_ANALYSIS,
            Intent.MIXED_INVESTIGATION,
            Intent.CUSTOMER_INVESTIGATION,
        )
        for items in by_operation.values():
            for index, e in enumerate(items[:MAX_GENERIC_CLAIMS]):
                claim_type: ClaimType = "observed_fact" if e.evidence_type == "observed" else "calculated_result"
                self.claim(
                    e.statement,
                    claim_type,
                    [e],
                    kind="fact",
                    primary=domain and index == 0,
                    assertions=[(e, "value")] if e.value is not None else [],
                    limitations=["Truncated result: the rows are not complete."] if e.truncated else [],
                )

    def causal_limit(self) -> None:
        if not self.request.causal_question:
            return
        support = [e for e in self.graph.evidence.values() if e.status == "ok"][:3]
        if not support:
            return
        topic = _TOPICS.get(self.request.metric or "", (self.request.metric_name or "this outcome").lower())
        self.claim(
            f"The available analysis describes what was observed; it does not establish what caused {topic}.",
            "inference",
            support,
            kind="causality_limit",
            primary=True,
            confidence="high",
        )

    def recommendations(self, inferences: list[Claim], flagged: list[Claim]) -> None:
        if self.request.intent in (Intent.KPI_LOOKUP, Intent.FORECAST):
            return
        made = 0
        for claim in inferences:
            if made >= 1:
                break
            evidence = self.graph.get_supporting_evidence(claim.claim_id)
            member = next((e for e in evidence if e.dimension_value is not None), None)
            if claim.kind == "concentration" and member is not None and claim.support_status == "supported":
                within = _within(member.filters)
                self.claim(
                    f"Review the {member.dimension} {member.dimension_value} accounts{within} behind this movement, "
                    "for example their churn and contraction in the same period, before drawing conclusions.",
                    "recommendation",
                    evidence,
                    kind="next_step",
                )
                made += 1
        for claim in flagged[:1]:
            evidence = self.graph.get_supporting_evidence(claim.claim_id)
            e = evidence[0]
            self.claim(
                f"Check the statistically unusual {e.metric} movement in {e.period_label} against operational context "
                "(pricing, releases, account events) that is not recorded in this dataset.",
                "recommendation",
                evidence,
                kind="next_step",
            )


def build_claims(graph: EvidenceGraph, request: ValidatedRequest) -> EvidenceGraph:
    """Rebuild all claims from the graph's evidence (idempotent: existing claims are replaced)."""
    graph.claims = {}
    builder = _Builder(graph, request)
    builder.kpi_values()
    builder.changes()
    inferences = builder.contributions()
    inferences += builder.bridge()
    builder.churn()
    inferences += builder.associations()
    builder.forecasts()
    flagged = builder.anomalies()
    builder.generic()
    builder.causal_limit()
    builder.recommendations(inferences, flagged)
    return graph
