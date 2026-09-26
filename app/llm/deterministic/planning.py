"""Deterministic investigation playbooks (the offline model's planner).

A playbook maps a validated intent to a short list of allow-listed tool calls. Arguments come only
from the validated request: metric, explicit period dates, filters and dimensions. Follow-up
planning is evidence-driven. For a revenue investigation, when the first round shows that one
member of a geographic dimension accounted for most of the change, the planner drills into that
member. The member's name is read from the evidence, never assumed.
"""

from __future__ import annotations

import json
from typing import Any

from app.llm.schemas import CHANGE_RANKINGS, Intent, PlanOutput, PlanStepOutput

CONCENTRATION_SHARE = 0.5  # same rule as the claim builders
DEFAULT_ANOMALY_METRICS = ("revenue", "mrr", "customer_count", "support_ticket_volume")


def plan(context: dict[str, Any]) -> dict[str, Any]:
    request: dict[str, Any] = context["request"]
    iteration = int(context.get("iteration", 1))
    if iteration > 1:
        steps = _follow_up(request, context.get("evidence", []), context.get("executed_steps", []))
        return PlanOutput(
            steps=steps,
            rationale="Drill into the largest contributor found so far."
            if steps
            else "Evidence collected so far is sufficient.",
            sufficient=not steps,
        ).model_dump()
    steps = _initial(request)
    # The full playbook is returned even if it exceeds the budget: the graph rejects over-budget plans
    # explicitly ("Investigation limit reached") instead of silently dropping steps.
    return PlanOutput(steps=steps, rationale=f"Playbook for intent {request['intent']}.").model_dump()


def _step(tool: str, purpose: str, **arguments: Any) -> PlanStepOutput:
    clean = {k: v for k, v in arguments.items() if v not in (None, {}, [])}
    return PlanStepOutput(tool_name=tool, arguments_json=json.dumps(clean, sort_keys=True), purpose=purpose)


def _period(request: dict[str, Any]) -> dict[str, Any]:
    period = request.get("period")
    return {"start_date": period["start"], "end_date": period["end"]} if period else {}


def _comparison(request: dict[str, Any]) -> dict[str, Any]:
    comparison = request.get("comparison_period")
    return (
        {"comparison_start_date": comparison["start"], "comparison_end_date": comparison["end"]} if comparison else {}
    )


def _anomaly_end(request: dict[str, Any]) -> dict[str, Any]:
    period = request.get("period")
    return {"end_date": period["end"]} if period else {}


def _initial(request: dict[str, Any]) -> list[PlanStepOutput]:
    intent = Intent(request["intent"])
    metric = request.get("metric")
    filters = request.get("filters") or {}
    dimensions = request.get("dimensions") or []
    period, comparison = _period(request), _comparison(request)
    analysis = request.get("analysis_type")

    if "sales_rep" in dimensions and intent in (
        Intent.KPI_LOOKUP,
        Intent.DIMENSIONAL_COMPARISON,
        Intent.SALES_ANALYSIS,
    ):
        # Individual reps are exposed only through rep performance (the Phase 5 data policy).
        return [
            _step(
                "analyze_sales",
                "Win rate per rep against the team median (reps with enough closed deals are ranked).",
                operation="rep_performance",
                filters=filters,
                **period,
            )
        ]
    if intent == Intent.KPI_LOOKUP:
        return [
            _step(
                "get_kpi",
                f"Calculate {metric}.",
                kpi=metric,
                filters=filters,
                dimension=dimensions[0] if dimensions else None,
                **period,
            )
        ]
    if intent == Intent.PERIOD_COMPARISON:
        return _comparison_steps(metric, filters, period, comparison)
    if intent == Intent.DIMENSIONAL_COMPARISON:
        dimension = dimensions[0] if dimensions else "segment"
        if (analysis == "contribution" or analysis in CHANGE_RANKINGS) and metric in (None, "revenue"):
            return [
                _step(
                    "analyze_revenue",
                    f"Decompose the revenue change by {dimension}.",
                    operation="decompose_revenue_change",
                    dimension=dimension,
                    filters=filters,
                    **period,
                    **comparison,
                )
            ]
        if metric in ("logo_churn_rate", "revenue_churn_rate", "retention_rate"):
            return [
                _step(
                    "analyze_customers",
                    f"Compare churn by {dimension}.",
                    operation="churn_by_dimension",
                    dimension=dimension,
                    filters=filters,
                    **period,
                )
            ]
        if metric == "support_ticket_volume":
            return [
                _step(
                    "analyze_support",
                    f"Compare tickets by {dimension}.",
                    operation="support_by_dimension",
                    dimension=dimension,
                    filters=filters,
                    **period,
                )
            ]
        return [
            _step(
                "get_kpi",
                f"Break {metric} down by {dimension}.",
                kpi=metric or "revenue",
                dimension=dimension,
                filters=filters,
                **period,
            )
        ]
    if intent == Intent.REVENUE_INVESTIGATION:
        return [
            _step(
                "analyze_revenue",
                "Measure the revenue change.",
                operation="revenue_change",
                filters=filters,
                **period,
                **comparison,
            ),
            _step(
                "analyze_revenue",
                "Which regions contributed.",
                operation="decompose_revenue_change",
                dimension="region",
                filters=filters,
                **period,
                **comparison,
            ),
            _step(
                "analyze_revenue",
                "Which segments contributed.",
                operation="decompose_revenue_change",
                dimension="segment",
                filters=filters,
                **period,
                **comparison,
            ),
            _step(
                "analyze_revenue",
                "Which countries contributed.",
                operation="decompose_revenue_change",
                dimension="country",
                filters=filters,
                **period,
                **comparison,
            ),
            _step(
                "analyze_revenue",
                "Recurring-revenue movements (new, expansion, contraction, churn).",
                operation="revenue_bridge",
                filters=filters,
                **period,
            ),
            _step(
                "detect_anomalies",
                "Was the movement statistically unusual?",
                metric="revenue",
                filters=filters,
                **_anomaly_end(request),
            ),
        ]
    if intent == Intent.CUSTOMER_INVESTIGATION:
        if analysis == "risk":
            return [
                _step(
                    "get_customer_risk",
                    "Customers with current risk signals.",
                    filters=filters,
                    min_band="high",
                    limit=20,
                )
            ]
        if analysis == "cohort":
            return [_step("get_cohort_analysis", "Retention by signup cohort.", filters=filters, max_months=12)]
        steps = [
            _step(
                "analyze_customers",
                "Churn and retention for the period.",
                operation="churn_summary",
                filters=filters,
                **period,
            ),
            _step(
                "analyze_customers",
                "Churn by segment.",
                operation="churn_by_dimension",
                dimension=dimensions[0] if dimensions else "segment",
                filters=filters,
                **period,
            ),
            _step(
                "analyze_customers", "Customer movements.", operation="customer_movements", filters=filters, **period
            ),
        ]
        if analysis in ("causal", "change"):
            steps.append(
                _step(
                    "analyze_customers",
                    "Usage and support before churn (associative).",
                    operation="usage_churn_relationship",
                    filters=filters,
                )
            )
        return steps
    if intent == Intent.SUPPORT_ANALYSIS:
        return [
            _step(
                "analyze_support",
                "Ticket volume change.",
                operation="support_volume_change",
                filters=filters,
                **period,
                **comparison,
            ),
            _step(
                "analyze_support",
                "Tickets by category.",
                operation="support_by_dimension",
                dimension="ticket_category",
                filters=filters,
                **period,
            ),
            _step(
                "detect_anomalies",
                "Was ticket volume statistically unusual?",
                metric="support_ticket_volume",
                filters=_series_filters(filters),
                **_anomaly_end(request),
            ),
        ]
    if intent == Intent.SALES_ANALYSIS:
        steps = [
            _step(
                "analyze_sales",
                "Closed-deal performance.",
                operation="sales_performance",
                dimension=dimensions[0] if dimensions else None,
                **period,
            )
        ]
        if metric in ("win_rate", "average_order_value", "sales_cycle", "pipeline_value"):
            steps.insert(0, _step("get_kpi", f"Calculate {metric}.", kpi=metric, **period))
        return steps
    if intent == Intent.MARKETING_ANALYSIS:
        steps = [_step("analyze_marketing", "Channel performance.", operation="channel_performance", **period)]
        if metric in ("cac", "conversion_rate"):
            steps.insert(0, _step("get_kpi", f"Calculate {metric}.", kpi=metric, **period))
        return steps
    if intent == Intent.PRODUCT_ANALYSIS:
        feature = filters.get("product_feature")
        if feature:
            return [
                _step("analyze_product", f"Adoption trend of {feature}.", operation="adoption_trend", feature=feature)
            ]
        return [_step("analyze_product", "Adoption by feature.", operation="feature_adoption", **period)]
    if intent == Intent.FORECAST:
        return [
            _step(
                "forecast_metric",
                f"Forecast {metric}.",
                metric=metric,
                horizon=request.get("horizon") or 1,
                filters=filters,
            )
        ]
    if intent == Intent.ANOMALY_DETECTION:
        if metric:
            return [
                _step(
                    "detect_anomalies",
                    "Rolling z-score check.",
                    metric=metric,
                    detector="rolling_zscore",
                    filters=filters,
                    **_anomaly_end(request),
                ),
                _step(
                    "detect_anomalies",
                    "Forecast-residual check (corroboration).",
                    metric=metric,
                    detector="forecast_residual",
                    filters=filters,
                    **_anomaly_end(request),
                ),
            ]
        return [
            _step("detect_anomalies", f"Check {m}.", metric=m, detector="rolling_zscore", **_anomaly_end(request))
            for m in DEFAULT_ANOMALY_METRICS
        ]
    if intent == Intent.MIXED_INVESTIGATION:
        steps = _comparison_steps("revenue", filters, period, comparison)[:1]
        steps.append(
            _step("analyze_support", "Ticket volume change.", operation="support_volume_change", **period, **comparison)
        )
        steps.append(_step("analyze_customers", "Churn and retention.", operation="churn_summary", **period))
        return steps
    return []


def _comparison_steps(
    metric: str | None, filters: dict[str, str], period: dict[str, Any], comparison: dict[str, Any]
) -> list[PlanStepOutput]:
    if metric == "revenue":
        return [
            _step(
                "analyze_revenue",
                "Compare revenue between the periods.",
                operation="revenue_change",
                filters=filters,
                **period,
                **comparison,
            )
        ]
    if metric == "mrr":
        return [
            _step(
                "get_kpi", "MRR at both period ends and the change.", kpi="mrr", filters=filters, **period, **comparison
            ),
            _step(
                "analyze_revenue",
                "MRR movements behind the change.",
                operation="revenue_bridge",
                filters=filters,
                **period,
            ),
        ]
    if metric == "support_ticket_volume":
        return [
            _step(
                "analyze_support",
                "Compare ticket volume.",
                operation="support_volume_change",
                filters=filters,
                **period,
                **comparison,
            )
        ]
    return [
        _step("get_kpi", f"Compare {metric} between the periods.", kpi=metric, filters=filters, **period, **comparison)
    ]


def _series_filters(filters: dict[str, str]) -> dict[str, str]:
    allowed = ("region", "country", "segment", "industry", "acquisition_channel", "ticket_category", "ticket_priority")
    return {k: v for k, v in filters.items() if k in allowed}


def _follow_up(
    request: dict[str, Any], evidence: list[dict[str, Any]], executed: list[dict[str, Any]]
) -> list[PlanStepOutput]:
    if Intent(request["intent"]) != Intent.REVENUE_INVESTIGATION:
        return []
    if any(step.get("arguments", {}).get("filters") for step in executed):
        return []  # already drilled down
    candidates = []
    for item in evidence:
        attributes = item.get("attributes") or {}
        if (
            item.get("operation") == "analyze_revenue.decompose_revenue_change"
            and item.get("dimension") in ("country", "region")
            and item.get("dimension_value")
            and not item.get("filters")
            and attributes.get("rank") == 1
        ):
            share = attributes.get("share_of_gross_decline") or attributes.get("share_of_gross_increase")
            if isinstance(share, (int, float)) and share >= CONCENTRATION_SHARE:
                candidates.append((item["dimension"] == "country", share, item))
    if not candidates:
        return []
    _, _, top = max(candidates, key=lambda c: (c[0], c[1]))
    drill = {top["dimension"]: top["dimension_value"]}
    period, comparison = _period(request), _comparison(request)
    return [
        _step(
            "analyze_revenue",
            f"Which segments drove the change within {top['dimension_value']}.",
            operation="decompose_revenue_change",
            dimension="segment",
            filters=drill,
            **period,
            **comparison,
        ),
        _step(
            "analyze_customers",
            f"Churn by segment within {top['dimension_value']}.",
            operation="churn_by_dimension",
            dimension="segment",
            filters=drill,
            min_customers=10,
            **period,
        ),
        _step(
            "detect_anomalies",
            f"Was MRR within {top['dimension_value']} statistically unusual?",
            metric="mrr",
            filters=drill,
            **_anomaly_end(request),
        ),
    ]
