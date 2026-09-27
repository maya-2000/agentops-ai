"""Investigation templates: the closed set of analytical plans (Phase 10).

A template turns a validated request (outcome metric, period, comparison period, filters) into an
``AnalysisPlan``. Templates are code: the model never writes an investigation step, and the tools a
template may use are fixed here. Every step is still authorised call by call by the Phase 5 policy,
under the validated intent named in ``authorized_as`` (the existing intent permissions, unchanged),
and with the SQL privilege withdrawn: investigations never run ad-hoc SQL.

Templates (steps in execution order):

- ``revenue``: revenue change, decomposition by region and segment, MRR movements, churn, net
  revenue retention, win rate, pipeline, usage before churn, revenue anomalies, and a drill-down into a
  region that concentrates the change (conditional).
- ``customer``: churn change, net revenue retention, customer movements, churn by segment and region,
  cohorts, risk signals, usage before churn, product adoption, customer-count anomalies.
- ``sales``: win rate, pipeline, sales cycle, order value, performance by segment, acquisition cost,
  marketing channels, revenue context.
- ``product_support``: ticket volume, tickets per customer and resolution time, tickets by category and
  segment, adoption levels, adoption change of the two most adopted features (evidence-bound),
  ticket anomalies, usage before churn.
- ``general``: revenue with the cross-functional indicators (churn, retention, sales, support, product).
- ``management_brief``: a current-state overview (revenue, customers, sales, product, support,
  anomalies, forecast); no single outcome is explained.
"""

from __future__ import annotations

from typing import Any

from app.analytics.periods import Period
from app.investigation.models import AnalysisPlan, AnalysisStep, Area, StepBinding, StepCondition, TemplateName
from app.llm.schemas import Intent

TEMPLATE_TITLES: dict[TemplateName, str] = {
    "revenue": "Revenue investigation",
    "customer": "Customer and churn investigation",
    "sales": "Sales investigation",
    "product_support": "Product and support investigation",
    "general": "Business investigation",
    "management_brief": "Management brief",
}
# The outcome each template explains when the request does not name one of its metrics.
DEFAULT_OUTCOMES: dict[TemplateName, str | None] = {
    "revenue": "revenue",
    "customer": "logo_churn_rate",
    "sales": "win_rate",
    "product_support": "support_ticket_volume",
    "general": "revenue",
    "management_brief": "revenue",
}
TEMPLATE_METRICS: dict[TemplateName, frozenset[str]] = {
    "revenue": frozenset({"revenue"}),
    "customer": frozenset({"logo_churn_rate", "revenue_churn_rate", "retention_rate", "nrr", "customer_count"}),
    "sales": frozenset({"win_rate", "pipeline_value", "sales_cycle", "average_order_value", "cac"}),
    # Product adoption is measured per feature (the bound feature steps); the first feature is the outcome.
    "product_support": frozenset({"support_ticket_volume", "average_resolution_time", "product_adoption"}),
    "general": frozenset({"revenue"}),
    "management_brief": frozenset({"revenue"}),
}
_DECOMPOSITION_DIMENSIONS = ("region", "segment")


class _Builder:
    def __init__(self, period: Period, comparison: Period, filters: dict[str, str], outcome: str | None):
        self.period: dict[str, Any] = {"start_date": period.start.isoformat(), "end_date": period.end.isoformat()}
        self.comparison: dict[str, Any] = {
            "comparison_start_date": comparison.start.isoformat(),
            "comparison_end_date": comparison.end.isoformat(),
        }
        self.filters = dict(filters)
        self.outcome = outcome  # the metric the investigation explains (each template measures it)
        self.steps: list[AnalysisStep] = []

    def add(
        self,
        title: str,
        area: Area,
        tool: str,
        authorized_as: Intent,
        *,
        depends_on: tuple[str, ...] = (),
        condition: StepCondition | None = None,
        binding: StepBinding | None = None,
        **arguments: Any,
    ) -> str:
        step_id = f"S{len(self.steps) + 1}"
        clean = {k: v for k, v in arguments.items() if v not in (None, {}, [])}
        self.steps.append(
            AnalysisStep(
                step_id=step_id,
                title=title,
                area=area,
                tool_name=tool,
                arguments=clean,
                authorized_as=authorized_as,
                depends_on=list(depends_on),
                condition=condition,
                binding=binding,
            )
        )
        return step_id

    # ------------------------------------------------------------------ reusable steps
    def kpi_change(self, kpi: str, title: str, area: Area, intent: Intent, *, filters: bool = True) -> str:
        return self.add(
            title,
            area,
            "get_kpi",
            intent,
            kpi=kpi,
            filters=self.filters if filters else None,
            **self.period,
            **self.comparison,
        )

    def revenue_change(self) -> str:
        return self.add(
            "Measure the revenue change",
            "revenue",
            "analyze_revenue",
            Intent.REVENUE_INVESTIGATION,
            operation="revenue_change",
            filters=self.filters,
            **self.period,
            **self.comparison,
        )

    def decompose(self, dimension: str, after: str, title: str) -> str:
        return self.add(
            title,
            "revenue",
            "analyze_revenue",
            Intent.REVENUE_INVESTIGATION,
            depends_on=(after,),
            condition="outcome_changed",
            operation="decompose_revenue_change",
            dimension=dimension,
            filters=self.filters,
            **self.period,
            **self.comparison,
        )

    def bridge(self) -> str:
        return self.add(
            "Measure recurring-revenue movements (new, expansion, contraction, churn)",
            "revenue",
            "analyze_revenue",
            Intent.REVENUE_INVESTIGATION,
            operation="revenue_bridge",
            filters=self.filters,
            **self.period,
        )

    def usage_before_churn(self) -> str:
        return self.add(
            "Check usage and support before churn (associative)",
            "product",
            "analyze_customers",
            Intent.CUSTOMER_INVESTIGATION,
            operation="usage_churn_relationship",
            filters=self.filters,
            **self.period,
        )

    def feature_adoption(self) -> str:
        return self.add(
            "Measure product adoption by feature",
            "product",
            "analyze_product",
            Intent.PRODUCT_ANALYSIS,
            operation="feature_adoption",
            **self.period,
        )

    def anomalies(self, metric: str, area_intent: Intent, title: str) -> str:
        return self.add(
            title,
            "anomalies",
            "detect_anomalies",
            area_intent,
            metric=metric,
            filters=_series_filters(self.filters),
            end_date=self.period["end_date"],
        )

    def usable_dimensions(self) -> list[str]:
        return [d for d in _DECOMPOSITION_DIMENSIONS if d not in self.filters]


def _series_filters(filters: dict[str, str]) -> dict[str, str]:
    allowed = ("region", "country", "segment", "industry", "acquisition_channel")
    return {k: v for k, v in filters.items() if k in allowed}


# ------------------------------------------------------------------------------------------------ templates


def _revenue(b: _Builder) -> None:
    change = b.revenue_change()
    dims = b.usable_dimensions()
    first = None
    for dimension in dims:
        step = b.decompose(dimension, change, f"Compare {dimension} performance")
        first = first or step
    b.bridge()
    b.kpi_change("logo_churn_rate", "Analyze customer churn", "customers", Intent.CUSTOMER_INVESTIGATION)
    b.kpi_change("nrr", "Analyze net revenue retention", "customers", Intent.CUSTOMER_INVESTIGATION)
    b.kpi_change("win_rate", "Check sales win rate", "sales", Intent.SALES_ANALYSIS, filters=False)
    b.kpi_change("pipeline_value", "Check the sales pipeline", "sales", Intent.SALES_ANALYSIS, filters=False)
    b.usage_before_churn()
    b.anomalies("revenue", Intent.REVENUE_INVESTIGATION, "Check for statistically unusual revenue")
    if first is not None and "region" in dims and "country" not in b.filters:
        b.add(
            "Drill into the region that concentrates the change",
            "revenue",
            "analyze_revenue",
            Intent.REVENUE_INVESTIGATION,
            depends_on=(first,),
            condition="concentrated",
            binding=StepBinding(kind="concentrated_member", source_step=first),
            operation="decompose_revenue_change",
            dimension="country",
            filters=b.filters,
            **b.period,
            **b.comparison,
        )


_OUTCOME_TITLES = {
    "revenue_churn_rate": "Measure the change in revenue churn",
    "retention_rate": "Measure the change in customer retention",
    "customer_count": "Measure the change in active customers",
    "average_resolution_time": "Measure the change in ticket resolution time",
}


def _customer(b: _Builder) -> None:
    b.kpi_change("logo_churn_rate", "Measure the change in customer churn", "customers", Intent.CUSTOMER_INVESTIGATION)
    b.kpi_change("nrr", "Measure net revenue retention", "customers", Intent.CUSTOMER_INVESTIGATION)
    if b.outcome in _OUTCOME_TITLES:
        b.kpi_change(b.outcome, _OUTCOME_TITLES[b.outcome], "customers", Intent.CUSTOMER_INVESTIGATION)
    b.add(
        "Measure customer movements",
        "customers",
        "analyze_customers",
        Intent.CUSTOMER_INVESTIGATION,
        operation="customer_movements",
        filters=b.filters,
        **b.period,
    )
    for dimension in ("segment", "region"):
        if dimension not in b.filters:
            b.add(
                f"Compare churn by {dimension}",
                "customers",
                "analyze_customers",
                Intent.CUSTOMER_INVESTIGATION,
                operation="churn_by_dimension",
                dimension=dimension,
                filters=b.filters,
                **b.period,
            )
    b.add(
        "Check retention by signup cohort",
        "customers",
        "get_cohort_analysis",
        Intent.CUSTOMER_INVESTIGATION,
        max_months=6,
        filters=b.filters,
    )
    b.add(
        "Check current customer risk signals",
        "customers",
        "get_customer_risk",
        Intent.CUSTOMER_INVESTIGATION,
        min_band="high",
        limit=10,
        filters=b.filters,
    )
    b.usage_before_churn()
    b.feature_adoption()
    b.anomalies("customer_count", Intent.CUSTOMER_INVESTIGATION, "Check for statistically unusual customer counts")


def _sales(b: _Builder) -> None:
    b.kpi_change("win_rate", "Measure the change in win rate", "sales", Intent.SALES_ANALYSIS)
    b.kpi_change("pipeline_value", "Measure the sales pipeline", "sales", Intent.SALES_ANALYSIS)
    b.kpi_change("sales_cycle", "Measure the sales cycle", "sales", Intent.SALES_ANALYSIS)
    b.kpi_change("average_order_value", "Measure average order value", "sales", Intent.SALES_ANALYSIS)
    b.add(
        "Compare sales performance by segment",
        "sales",
        "analyze_sales",
        Intent.SALES_ANALYSIS,
        operation="sales_performance",
        dimension="segment",
        **b.period,
    )
    b.kpi_change("cac", "Check customer acquisition cost", "marketing", Intent.MARKETING_ANALYSIS, filters=False)
    b.add(
        "Compare marketing channels",
        "marketing",
        "analyze_marketing",
        Intent.MARKETING_ANALYSIS,
        operation="channel_performance",
        **b.period,
    )
    b.revenue_change()


def _product_support(b: _Builder) -> None:
    b.kpi_change("support_ticket_volume", "Measure the change in ticket volume", "support", Intent.SUPPORT_ANALYSIS)
    if b.outcome == "average_resolution_time":
        b.kpi_change(b.outcome, _OUTCOME_TITLES[b.outcome], "support", Intent.SUPPORT_ANALYSIS)
    b.add(
        "Measure tickets per customer and resolution time",
        "support",
        "analyze_support",
        Intent.SUPPORT_ANALYSIS,
        operation="support_volume_change",
        filters=b.filters,
        **b.period,
        **b.comparison,
    )
    for dimension in ("ticket_category", "segment"):
        if dimension not in b.filters:
            b.add(
                f"Compare tickets by {dimension.replace('_', ' ')}",
                "support",
                "analyze_support",
                Intent.SUPPORT_ANALYSIS,
                operation="support_by_dimension",
                dimension=dimension,
                filters=b.filters,
                **b.period,
            )
    adoption = b.feature_adoption()
    for index in (1, 2):
        b.add(
            f"Measure the adoption change of feature {index} (by adoption level)",
            "product",
            "get_kpi",
            Intent.PRODUCT_ANALYSIS,
            depends_on=(adoption,),
            binding=StepBinding(kind="feature", source_step=adoption, index=index),
            kpi="product_adoption",
            **b.period,
            **b.comparison,
        )
    b.anomalies("support_ticket_volume", Intent.SUPPORT_ANALYSIS, "Check for statistically unusual ticket volume")
    b.usage_before_churn()


def _general(b: _Builder) -> None:
    change = b.revenue_change()
    if "segment" not in b.filters:
        b.decompose("segment", change, "Compare segment performance")
    b.bridge()
    b.kpi_change("logo_churn_rate", "Analyze customer churn", "customers", Intent.CUSTOMER_INVESTIGATION)
    b.kpi_change("nrr", "Analyze net revenue retention", "customers", Intent.CUSTOMER_INVESTIGATION)
    b.kpi_change("win_rate", "Check sales win rate", "sales", Intent.SALES_ANALYSIS, filters=False)
    b.kpi_change("pipeline_value", "Check the sales pipeline", "sales", Intent.SALES_ANALYSIS, filters=False)
    b.kpi_change("support_ticket_volume", "Check support ticket volume", "support", Intent.SUPPORT_ANALYSIS)
    b.feature_adoption()
    b.usage_before_churn()
    b.anomalies("revenue", Intent.REVENUE_INVESTIGATION, "Check for statistically unusual revenue")


def _management_brief(b: _Builder) -> None:
    b.revenue_change()
    b.bridge()
    b.kpi_change("customer_count", "Measure the customer base", "customers", Intent.CUSTOMER_INVESTIGATION)
    b.kpi_change("logo_churn_rate", "Measure customer churn", "customers", Intent.CUSTOMER_INVESTIGATION)
    b.kpi_change("nrr", "Measure net revenue retention", "customers", Intent.CUSTOMER_INVESTIGATION)
    b.kpi_change("win_rate", "Measure sales win rate", "sales", Intent.SALES_ANALYSIS, filters=False)
    b.kpi_change("pipeline_value", "Measure the sales pipeline", "sales", Intent.SALES_ANALYSIS, filters=False)
    b.feature_adoption()
    b.kpi_change("support_ticket_volume", "Measure support ticket volume", "support", Intent.SUPPORT_ANALYSIS)
    b.anomalies("revenue", Intent.REVENUE_INVESTIGATION, "Check for statistically unusual revenue")
    b.anomalies("support_ticket_volume", Intent.SUPPORT_ANALYSIS, "Check for statistically unusual ticket volume")
    b.add(
        "Forecast revenue for the next three months",
        "forecast",
        "forecast_metric",
        Intent.FORECAST,
        metric="revenue",
        horizon=3,
        filters=_series_filters(b.filters),
    )


_TEMPLATES = {
    "revenue": _revenue,
    "customer": _customer,
    "sales": _sales,
    "product_support": _product_support,
    "general": _general,
    "management_brief": _management_brief,
}


def build_plan(
    template: TemplateName,
    *,
    period: Period,
    comparison: Period,
    filters: dict[str, str] | None = None,
    outcome_metric: str | None = None,
    assumptions: list[str] | None = None,
) -> AnalysisPlan:
    outcome = outcome_metric or DEFAULT_OUTCOMES[template]
    builder = _Builder(period, comparison, filters or {}, outcome)
    _TEMPLATES[template](builder)
    return AnalysisPlan(
        template=template,
        title=TEMPLATE_TITLES[template],
        outcome_metric=outcome,
        period=period,
        comparison_period=comparison,
        period_label=period.label,
        comparison_label=comparison.label,
        steps=builder.steps,
        assumptions=list(assumptions or []),
    )
