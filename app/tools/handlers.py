"""Tool handlers: validate-free thin wrappers (inputs are already validated) around Phase 1-3 services.

No handler contains a business formula. Each calls one existing function and passes its typed
result through with provenance. The one exception is the KPI period comparison: the change
between two KPI values is computed with Phase 2's own ``pct_change`` helper, and it is labelled as
such in the result's calculation.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from app.analytics import cohorts, customers, marketing, product, revenue, risk, sales, support
from app.analytics.common import pct_change
from app.analytics.errors import InvalidRequestError
from app.analytics.executor import QueryRunner
from app.analytics.kpis import KPIResult, get_kpi_definition
from app.analytics.models import AnalyticsResult, Scalar
from app.anomalies import AnomalyReport, AnomalyService
from app.forecasting import ForecastResult, ForecastService
from app.tools.base import ToolContext, ToolOutput, ToolStatus
from app.tools.inputs import (
    AnalyzeCustomersInput,
    AnalyzeMarketingInput,
    AnalyzeProductInput,
    AnalyzeRevenueInput,
    AnalyzeSalesInput,
    AnalyzeSupportInput,
    AnomalyInput,
    CohortAnalysisInput,
    CustomerRiskInput,
    ForecastInput,
    GetKPIInput,
    SafeSQLInput,
)
from app.tools.results import KPIComparison, SQLResult
from app.tools.sql_safety import UnsafeSQLError, validate_sql

# ------------------------------------------------------------------------------------------------ helpers


def _from_kpi(result: KPIResult) -> ToolOutput:
    return ToolOutput(
        result=result,
        status=result.status,
        source_tables=result.source_tables,
        query_ids=result.query_ids,
        calculation=result.calculation,
        limitations=list(result.limitations),
        message=result.message,
    )


def _from_analytics(result: AnalyticsResult[Any]) -> ToolOutput:
    return ToolOutput(
        result=result,
        status=result.status,
        source_tables=result.source_tables,
        query_ids=result.provenance.query_ids,
        calculation=result.provenance.calculation,
        limitations=list(result.limitations),
        message=result.message,
    )


_SEVERITY: tuple[ToolStatus, ...] = ("error", "no_data", "insufficient_history", "insufficient_data")


def _worst(*statuses: ToolStatus) -> ToolStatus:
    for status in _SEVERITY:
        if status in statuses:
            return status
    return "ok"


# ------------------------------------------------------------------------------------------------ KPI


def get_kpi(ctx: ToolContext, inp: GetKPIInput) -> ToolOutput:
    definition = get_kpi_definition(inp.kpi)
    base: dict[str, Any] = {"filters": dict(inp.filters)}
    if inp.start_date is not None:
        base |= {"start_date": inp.start_date, "end_date": inp.end_date}
    elif inp.period is not None:
        base["period"] = inp.period
    comparison: dict[str, Any] = {}
    if inp.comparison_start_date is not None:
        comparison = {"start_date": inp.comparison_start_date, "end_date": inp.comparison_end_date}
    elif inp.comparison_period is not None:
        comparison = {"period": inp.comparison_period}

    if not inp.has_comparison or definition.requires_comparison:
        params = dict(base)
        if inp.dimension is not None:
            params["dimension"] = inp.dimension
        params |= {f"comparison_{name}": value for name, value in comparison.items()}
        return _from_kpi(ctx.kpi_service.calculate_kpi(definition.key, params))

    if inp.dimension is not None:
        raise InvalidRequestError("A breakdown dimension cannot be combined with a comparison period in get_kpi")
    current = ctx.kpi_service.calculate_kpi(definition.key, base)
    previous = ctx.kpi_service.calculate_kpi(definition.key, {"filters": dict(inp.filters), **comparison})
    change = pct_change(current.value, previous.value)
    absolute = current.value - previous.value if current.value is not None and previous.value is not None else None
    direction: Literal["increase", "decline", "unchanged"] | None = None
    if absolute is not None:
        direction = "increase" if absolute > 0 else "decline" if absolute < 0 else "unchanged"
    result = KPIComparison(
        key=definition.key,
        name=definition.name,
        unit=definition.unit,
        current=current,
        comparison=previous,
        absolute_change=absolute,
        percentage_change=change,
        direction=direction,
    )
    return ToolOutput(
        result=result,
        status=_worst(current.status, previous.status),
        source_tables=sorted({*current.source_tables, *previous.source_tables}),
        query_ids=[*current.query_ids, *previous.query_ids],
        calculation=(
            f"{definition.name} in {current.period.label} and {previous.period.label}; change = current - comparison, "
            "percentage change = current / comparison - 1 (Phase 2 pct_change)."
        ),
        limitations=sorted({*current.limitations, *previous.limitations}),
        message=current.message or previous.message,
    )


# ------------------------------------------------------------------------------------------------ Phase 2 analytics

_Dispatch = dict[str, Callable[[ToolContext, Any], AnalyticsResult[Any]]]

_REVENUE: _Dispatch = {
    "revenue_by_period": lambda c, i: revenue.revenue_by_period(
        c.db, i.period_spec() or "trailing_12_months", grain=i.grain, filters=i.filters, as_of=c.as_of
    ),
    "revenue_change": lambda c, i: revenue.revenue_change(
        c.db, i.period_spec(), i.comparison_spec(), filters=i.filters, as_of=c.as_of
    ),
    "decompose_revenue_change": lambda c, i: revenue.decompose_revenue_change(
        c.db, i.dimension, i.period_spec(), i.comparison_spec(), filters=i.filters, as_of=c.as_of
    ),
    "revenue_bridge": lambda c, i: revenue.revenue_bridge(c.db, i.period_spec(), filters=i.filters, as_of=c.as_of),
    "mrr_series": lambda c, i: revenue.mrr_series(
        c.db, i.period_spec() or "trailing_12_months", filters=i.filters, as_of=c.as_of
    ),
    "revenue_concentration": lambda c, i: revenue.revenue_concentration(
        c.db, i.period_spec(), top_n=i.top_n, filters=i.filters, as_of=c.as_of
    ),
}

_CUSTOMERS: _Dispatch = {
    "churn_summary": lambda c, i: customers.churn_summary(c.db, i.period_spec(), filters=i.filters, as_of=c.as_of),
    "churn_by_dimension": lambda c, i: customers.churn_by_dimension(
        c.db, i.dimension, i.period_spec(), filters=i.filters, min_customers=i.min_customers, as_of=c.as_of
    ),
    "customer_movements": lambda c, i: customers.customer_movements(
        c.db, i.period_spec(), dimension=i.dimension, filters=i.filters, as_of=c.as_of
    ),
    "monthly_churn_series": lambda c, i: customers.monthly_churn_series(
        c.db, i.period_spec() or "trailing_12_months", filters=i.filters, as_of=c.as_of
    ),
    "usage_churn_relationship": lambda c, i: customers.usage_churn_relationship(
        c.db, i.period_spec() or "last_quarter", filters=i.filters, as_of=c.as_of
    ),
}

_SALES: _Dispatch = {
    "pipeline_summary": lambda c, i: sales.pipeline_summary(c.db, filters=i.filters, as_of=c.as_of),
    "sales_performance": lambda c, i: sales.sales_performance(
        c.db, i.period_spec(), dimension=i.dimension, filters=i.filters, as_of=c.as_of
    ),
    "rep_performance": lambda c, i: sales.rep_performance(
        c.db, i.period_spec() or "trailing_12_months", filters=i.filters, min_closed=i.min_closed, as_of=c.as_of
    ),
    "opportunity_conversion": lambda c, i: sales.opportunity_conversion(
        c.db, i.period_spec() or "trailing_12_months", dimension=i.dimension, filters=i.filters, as_of=c.as_of
    ),
    "funnel_stage_distribution": lambda c, i: sales.funnel_stage_distribution(
        c.db, i.period_spec() or "trailing_12_months", filters=i.filters, as_of=c.as_of
    ),
    "segment_performance": lambda c, i: sales.segment_performance(c.db, i.period_spec(), as_of=c.as_of),
}

_MARKETING: _Dispatch = {
    "channel_performance": lambda c, i: marketing.channel_performance(c.db, i.period_spec(), as_of=c.as_of),
    "campaign_performance": lambda c, i: marketing.campaign_performance(
        c.db,
        i.period_spec() or "trailing_12_months",
        channel=i.channel,
        min_conversions=i.min_conversions,
        as_of=c.as_of,
    ),
    "marketing_period_change": lambda c, i: marketing.marketing_period_change(
        c.db, i.period_spec(), i.comparison_spec(), as_of=c.as_of
    ),
    "channel_roas": lambda c, i: marketing.channel_roas(
        c.db, i.period_spec() or "last_quarter", window_days=i.window_days, as_of=c.as_of
    ),
}

_SUPPORT: _Dispatch = {
    "support_summary": lambda c, i: support.support_summary(c.db, i.period_spec(), filters=i.filters, as_of=c.as_of),
    "support_by_dimension": lambda c, i: support.support_by_dimension(
        c.db, i.dimension, i.period_spec(), filters=i.filters, as_of=c.as_of
    ),
    "support_volume_change": lambda c, i: support.support_volume_change(
        c.db, i.period_spec(), i.comparison_spec(), filters=i.filters, as_of=c.as_of
    ),
    "resolution_time_trend": lambda c, i: support.resolution_time_trend(
        c.db, i.period_spec() or "trailing_12_months", filters=i.filters, as_of=c.as_of
    ),
}

_PRODUCT: _Dispatch = {
    "feature_adoption": lambda c, i: product.feature_adoption(c.db, i.period_spec(), as_of=c.as_of),
    "adoption_trend": lambda c, i: product.adoption_trend(
        c.db, i.feature, i.period_spec() or "trailing_12_months", grain=i.grain, as_of=c.as_of
    ),
    "adoption_change": lambda c, i: product.adoption_change(
        c.db, i.feature, i.period_spec(), i.comparison_spec(), as_of=c.as_of
    ),
    "feature_launch_summary": lambda c, i: product.feature_launch_summary(
        c.db, i.feature, window_days=i.window_days, as_of=c.as_of
    ),
    "adoption_breadth_by": lambda c, i: product.adoption_breadth_by(
        c.db, i.dimension, i.period_spec(), filters=i.filters, as_of=c.as_of
    ),
    "feature_usage_distribution": lambda c, i: product.feature_usage_distribution(
        c.db, i.period_spec(), filters=i.filters, as_of=c.as_of
    ),
}


def _dispatcher(table: _Dispatch) -> Callable[[ToolContext, Any], ToolOutput]:
    def handler(ctx: ToolContext, inp: Any) -> ToolOutput:
        return _from_analytics(table[inp.operation](ctx, inp))

    return handler


analyze_revenue: Callable[[ToolContext, AnalyzeRevenueInput], ToolOutput] = _dispatcher(_REVENUE)
analyze_customers: Callable[[ToolContext, AnalyzeCustomersInput], ToolOutput] = _dispatcher(_CUSTOMERS)
analyze_sales: Callable[[ToolContext, AnalyzeSalesInput], ToolOutput] = _dispatcher(_SALES)
analyze_marketing: Callable[[ToolContext, AnalyzeMarketingInput], ToolOutput] = _dispatcher(_MARKETING)
analyze_support: Callable[[ToolContext, AnalyzeSupportInput], ToolOutput] = _dispatcher(_SUPPORT)
analyze_product: Callable[[ToolContext, AnalyzeProductInput], ToolOutput] = _dispatcher(_PRODUCT)


def get_cohort_analysis(ctx: ToolContext, inp: CohortAnalysisInput) -> ToolOutput:
    return _from_analytics(
        cohorts.cohort_retention(
            ctx.db,
            first_cohort=inp.first_cohort,
            last_cohort=inp.last_cohort,
            max_months=inp.max_months,
            filters=inp.filters,
            as_of=ctx.as_of,
        )
    )


def get_customer_risk(ctx: ToolContext, inp: CustomerRiskInput) -> ToolOutput:
    return _from_analytics(
        risk.score_customer_risk(ctx.db, as_of=ctx.as_of, filters=inp.filters, min_band=inp.min_band, limit=inp.limit)
    )


# ------------------------------------------------------------------------------------------------ Phase 3


def forecast_metric(ctx: ToolContext, inp: ForecastInput) -> ToolOutput:
    result: ForecastResult = ForecastService(ctx.db, as_of=ctx.as_of).forecast(
        metric=inp.metric,
        horizon=inp.horizon,
        cutoff_date=inp.cutoff_date,
        filters=inp.filters,
        model=inp.model,
        confidence_level=inp.confidence_level,
    )
    return ToolOutput(
        result=result,
        status=result.status,
        source_tables=result.source_tables,
        query_ids=result.query_ids,
        calculation=result.calculation,
        limitations=list(result.limitations),
        message=result.message,
    )


def detect_anomalies(ctx: ToolContext, inp: AnomalyInput) -> ToolOutput:
    report: AnomalyReport = AnomalyService(ctx.db, as_of=ctx.as_of).detect(
        inp.metric,
        inp.start_date,
        inp.end_date,
        inp.detector,
        inp.filters,
        inp.window,
        transform=inp.transform,
    )
    return ToolOutput(
        result=report,
        status=report.status,
        source_tables=report.source_tables,
        query_ids=report.provenance.query_ids,
        calculation=report.calculation,
        limitations=list(report.limitations),
        message=report.message,
    )


# ------------------------------------------------------------------------------------------------ SQL


def run_safe_sql(ctx: ToolContext, inp: SafeSQLInput) -> ToolOutput:
    max_rows = min(inp.max_rows or ctx.sql_row_limit, ctx.sql_row_limit)
    validated = validate_sql(inp.sql, inp.parameters, max_rows)
    description = inp.description.strip() or "Ad-hoc read-only query"
    runner = QueryRunner(ctx.db, "run_safe_sql")
    try:
        result = runner.run(validated.sql, dict(validated.parameters), calculation=description)
    except InvalidRequestError as exc:
        raise UnsafeSQLError(str(exc)) from None
    rows = [[_scalar(v) for v in row] for row in result.rows]
    truncated = len(rows) > max_rows
    rows = rows[:max_rows]
    sql_result = SQLResult(
        columns=result.columns,
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
        max_rows=max_rows,
        sql=validated.sql,
        parameters={k: _scalar(v) for k, v in validated.parameters.items()},
        query_id=result.query_id,
        source_tables=validated.source_tables,
        execution_timestamp=result.lineage.execution_timestamp,
        description=description,
    )
    limitations = ["Result of an ad-hoc query, not a registered KPI definition."]
    if truncated:
        limitations.append(
            f"Truncated: more than {max_rows} rows matched; only the first {max_rows} are returned, so the rows are "
            "not complete and must not be summed or presented as exhaustive."
        )
    return ToolOutput(
        result=sql_result,
        status="ok" if rows else "no_data",
        source_tables=validated.source_tables,
        query_ids=[result.query_id],
        calculation=description,
        limitations=limitations,
        message=None if rows else "The query returned no rows.",
    )


def _scalar(value: Any) -> Scalar:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)
