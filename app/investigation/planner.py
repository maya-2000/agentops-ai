"""Turn a validated investigation objective into an analysis plan (Phase 10).

The objective is understood exactly like a question (the agent's understanding step and the central
request validation), so its metric, periods and filters are validated values. The planner then
chooses one of the closed set of templates (``app/investigation/templates.py``):

1. A management brief when the objective asks for one ("management brief", "state of the business",
   "what should management investigate next"): a fixed list of cues, never model output.
2. ``general`` for a mixed investigation (several business areas named together).
3. The template of the outcome metric (revenue, churn, sales, support/product metrics).
4. The template of the validated intent.

Periods: the validated period, or the latest complete month; the validated comparison period, or the
previous period of the same length. Both defaults are recorded as assumptions.
"""

from __future__ import annotations

import re
from datetime import date

from app.agent.request import ValidatedRequest
from app.analytics.periods import Period, previous_period, resolve_period
from app.investigation.models import AnalysisPlan, TemplateName
from app.investigation.templates import DEFAULT_OUTCOMES, TEMPLATE_METRICS, build_plan
from app.llm.schemas import Intent

BRIEF_CUES = re.compile(
    r"\b(management brief|executive (?:brief|summary|overview)|business (?:overview|brief|review|health|summary)|"
    r"state of the business|current state of|overall business|how (?:is|are) (?:the|our) business|"
    r"what should (?:management|we|leadership) (?:investigate|look at|focus on|examine|review)|"
    r"(?:recent )?business trends|key business risks)\b",
    re.IGNORECASE,
)
METRIC_TEMPLATES: dict[str, TemplateName] = {
    "revenue": "revenue",
    "mrr": "revenue",
    "arr": "revenue",
    "revenue_growth": "revenue",
    "arpu": "revenue",
    "logo_churn_rate": "customer",
    "revenue_churn_rate": "customer",
    "retention_rate": "customer",
    "nrr": "customer",
    "customer_count": "customer",
    "clv": "customer",
    "win_rate": "sales",
    "pipeline_value": "sales",
    "sales_cycle": "sales",
    "average_order_value": "sales",
    "conversion_rate": "sales",
    "cac": "sales",
    "support_ticket_volume": "product_support",
    "average_resolution_time": "product_support",
    "product_adoption": "product_support",
}
INTENT_TEMPLATES: dict[Intent, TemplateName] = {
    Intent.REVENUE_INVESTIGATION: "revenue",
    Intent.CUSTOMER_INVESTIGATION: "customer",
    Intent.SALES_ANALYSIS: "sales",
    Intent.MARKETING_ANALYSIS: "sales",
    Intent.SUPPORT_ANALYSIS: "product_support",
    Intent.PRODUCT_ANALYSIS: "product_support",
    Intent.MIXED_INVESTIGATION: "general",
}
DEFAULT_PERIOD = "last_month"


def asks_for_brief(objective: str) -> bool:
    return BRIEF_CUES.search(objective) is not None


def select_template(objective: str, request: ValidatedRequest | None) -> TemplateName | None:
    """The template for a validated objective, or ``None`` when the objective is outside every template."""
    if asks_for_brief(objective):
        return "management_brief"
    if request is None:
        return None
    if request.intent == Intent.MIXED_INVESTIGATION:
        return "general"
    if request.metric in METRIC_TEMPLATES:
        return METRIC_TEMPLATES[request.metric]
    return INTENT_TEMPLATES.get(request.intent, "general" if request.intent != Intent.UNSUPPORTED else None)


def plan_investigation(
    template: TemplateName,
    request: ValidatedRequest | None,
    *,
    as_of: date,
) -> AnalysisPlan:
    assumptions = list(request.assumptions) if request else []
    period: Period | None = request.period if request else None
    if period is None:
        period = resolve_period(DEFAULT_PERIOD, as_of=as_of)
        assumptions.append(f"No period was given; the latest complete month ({period.label}) was used.")
    comparison: Period | None = request.comparison_period if request else None
    if comparison is None:
        comparison = previous_period(period)
        assumptions.append(f"Compared with the previous period of the same length ({comparison.label}).")
    metric = request.metric if request else None
    outcome = metric if metric in TEMPLATE_METRICS[template] else DEFAULT_OUTCOMES[template]
    filters = dict(request.filters) if request else {}
    return build_plan(
        template,
        period=period,
        comparison=comparison,
        filters=filters,
        outcome_metric=outcome,
        assumptions=list(dict.fromkeys(assumptions)),
    )
