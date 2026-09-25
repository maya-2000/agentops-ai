"""The central, allow-listed tool catalogue and its execution boundary.

The planner can only name tools registered here. ``ToolRegistry.execute`` validates arguments
against the tool's typed input model, runs the handler, times it and converts every failure into a
typed ``ToolError``. A failed call is recorded, never replaced by a fabricated result.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from app.analytics.errors import AnalyticsDatabaseError, AnalyticsError, InvalidRequestError
from app.tools import handlers
from app.tools.base import ToolContext, ToolDefinition, ToolError, ToolOutput, ToolRequest, ToolResult
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
from app.tools.sql_safety import UnsafeSQLError

_PERIODS = (
    "Periods are specs relative to the business as-of date (last_month, previous_month, last_quarter, "
    "trailing_N_months, ytd, YYYY-MM, YYYY-Qn, YYYY) or explicit start_date/end_date."
)

TOOL_DEFINITIONS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        name="get_kpi",
        description=(
            "Calculate one registered business KPI (revenue, mrr, arr, revenue_growth, logo_churn_rate, "
            "revenue_churn_rate, retention_rate, nrr, cac, clv, arpu, average_order_value, conversion_rate, "
            "pipeline_value, win_rate, sales_cycle, support_ticket_volume, average_resolution_time, "
            "product_adoption, customer_count) for a period, optionally filtered or broken down by one "
            f"validated dimension. With comparison_period it returns both periods and their change. {_PERIODS}"
        ),
        when_to_use="The user asks for the value of a defined KPI, or how a KPI changed between two periods.",
        not_for="Explaining why a KPI moved, forecasting, anomaly detection, or metrics that are not registered KPIs.",
        output_description=(
            "KPIResult (value, unit, components, breakdown rows) or KPIComparison (both periods and the change)."
        ),
        limitations="One KPI per call; a breakdown cannot be combined with a comparison period.",
        source_layer="phase2_kpi",
        input_model=GetKPIInput,
        handler=handlers.get_kpi,
    ),
    ToolDefinition(
        name="analyze_revenue",
        description=(
            "Revenue analytics from the Phase 2 revenue module. operation: revenue_change (current vs comparison "
            "period revenue and % change), decompose_revenue_change (each member of a dimension's contribution to "
            "the revenue change; needs dimension), revenue_bridge (MRR bridge: opening, new, expansion, "
            "contraction, churn, closing), revenue_by_period (monthly/quarterly revenue), mrr_series (month-end "
            f"MRR), revenue_concentration (top-N customer share). {_PERIODS}"
        ),
        when_to_use="Revenue or MRR changes, which dimension members drove a revenue change, MRR movements.",
        not_for="Customer counts or churn rates (use analyze_customers), forecasts or anomaly scores.",
        output_description="AnalyticsResult with typed rows, summary figures and provenance.",
        limitations="Comparison periods must lie fully within the data coverage.",
        source_layer="phase2_analytics",
        input_model=AnalyzeRevenueInput,
        handler=handlers.analyze_revenue,
    ),
    ToolDefinition(
        name="analyze_customers",
        description=(
            "Customer and churn analytics. operation: churn_summary (logo/revenue churn, retention, NRR), "
            "churn_by_dimension (churn rate per segment/region/country/plan with Wilson intervals; needs dimension), "
            "customer_movements (opening, new, churned, closing customers), monthly_churn_series, "
            f"usage_churn_relationship (usage and tickets before churn vs retained; associative only). {_PERIODS}"
        ),
        when_to_use="Churn, retention, customer counts and movements, which group churns most.",
        not_for="Establishing why customers churned (results are associative), revenue decompositions.",
        output_description="AnalyticsResult with churn rows, confidence intervals and sample-size flags.",
        limitations="Churn rates are for the requested period and not annualised; small groups are flagged.",
        source_layer="phase2_analytics",
        input_model=AnalyzeCustomersInput,
        handler=handlers.analyze_customers,
    ),
    ToolDefinition(
        name="analyze_sales",
        description=(
            "Sales analytics. operation: sales_performance (win rate, AOV, cycle for closed deals; optional "
            "dimension), pipeline_summary (open pipeline by stage at the as-of date), rep_performance (win rate "
            "vs team median with intervals), opportunity_conversion, funnel_stage_distribution, "
            f"segment_performance. {_PERIODS}"
        ),
        when_to_use="Pipeline, win rate, deal size, sales cycle, rep or segment sales performance.",
        not_for="Recognised revenue (use analyze_revenue) or marketing spend.",
        output_description="AnalyticsResult with sales rows and summary.",
        limitations="Rep comparisons need a minimum number of closed deals (min_closed).",
        source_layer="phase2_analytics",
        input_model=AnalyzeSalesInput,
        handler=handlers.analyze_sales,
    ),
    ToolDefinition(
        name="analyze_marketing",
        description=(
            "Marketing analytics. operation: channel_performance (spend, leads, conversions, CAC by channel), "
            "campaign_performance (per campaign; optional channel), marketing_period_change (channel change "
            f"between periods), channel_roas (first-N-day revenue per channel). {_PERIODS}"
        ),
        when_to_use="Marketing spend, CAC, conversion by channel or campaign, campaign efficiency.",
        not_for="Sales pipeline or revenue decompositions.",
        output_description="AnalyticsResult with channel or campaign rows.",
        limitations="ROAS is channel-level; customers carry no campaign id.",
        source_layer="phase2_analytics",
        input_model=AnalyzeMarketingInput,
        handler=handlers.analyze_marketing,
    ),
    ToolDefinition(
        name="analyze_support",
        description=(
            "Support analytics. operation: support_summary (tickets, resolution time), support_by_dimension "
            "(tickets by ticket_category, ticket_priority, segment, region, ...; needs dimension), "
            "support_volume_change (tickets, tickets per customer and resolution time vs a comparison period), "
            f"resolution_time_trend. {_PERIODS}"
        ),
        when_to_use="Ticket volume, its change, category mix, resolution times.",
        not_for="Establishing why tickets changed; anomaly scoring (use detect_anomalies).",
        output_description="AnalyticsResult with support rows and changes.",
        limitations="Recent tickets are more often unresolved, which biases their resolution times.",
        source_layer="phase2_analytics",
        input_model=AnalyzeSupportInput,
        handler=handlers.analyze_support,
    ),
    ToolDefinition(
        name="analyze_product",
        description=(
            "Product analytics. operation: feature_adoption (average daily adoption per feature), adoption_trend "
            "(one feature over time; needs feature), adoption_change (needs feature), feature_launch_summary "
            "(needs feature), adoption_breadth_by (features used per customer by dimension; needs dimension), "
            f"feature_usage_distribution. {_PERIODS}"
        ),
        when_to_use="Feature adoption levels, trends and launches.",
        not_for="Revenue or churn questions.",
        output_description="AnalyticsResult with feature rows.",
        limitations="Feature adoption is platform-level; breadth by customer attributes uses weekly usage.",
        source_layer="phase2_analytics",
        input_model=AnalyzeProductInput,
        handler=handlers.analyze_product,
    ),
    ToolDefinition(
        name="get_cohort_analysis",
        description="Monthly signup cohorts x months since signup: logo and revenue retention (Phase 2 cohorts).",
        when_to_use="Retention by signup cohort, how cohorts age.",
        not_for="Single-period churn rates (use analyze_customers).",
        output_description="AnalyticsResult of cohort cells.",
        limitations="Young cohorts have few observed months.",
        source_layer="phase2_analytics",
        input_model=CohortAnalysisInput,
        handler=handlers.get_cohort_analysis,
    ),
    ToolDefinition(
        name="get_customer_risk",
        description=(
            "Transparent rule-based churn-risk score per active customer from observable signals (seat "
            "utilisation, usage decline, feature breadth, ticket sentiment and trend, contraction)."
        ),
        when_to_use="Which customers show risk signals now.",
        not_for="Predicting churn with certainty or explaining past churn.",
        output_description="AnalyticsResult of customers with score, band and the signals that fired.",
        limitations="A rule score from observable signals, backtested in Phase 2; not a probability.",
        source_layer="phase2_analytics",
        input_model=CustomerRiskInput,
        handler=handlers.get_customer_risk,
    ),
    ToolDefinition(
        name="forecast_metric",
        description=(
            "Forecast revenue, mrr, customer_count, support_ticket_volume or product_adoption (needs "
            "product_feature filter) 1-6 months after the cutoff (default: business as-of date) with prediction "
            "intervals, rolling-origin backtests and a naive-baseline comparison (Phase 3)."
        ),
        when_to_use="Expected future values of a supported metric.",
        not_for="Explaining history, horizons beyond 6 months, or metrics without a monthly series.",
        output_description=(
            "ForecastResult: forecast points with bounds, the selected model, backtest and baseline metrics."
        ),
        limitations="24 monthly observations; intervals ignore parameter uncertainty (backtest coverage reported).",
        source_layer="phase3_forecasting",
        input_model=ForecastInput,
        handler=handlers.forecast_metric,
    ),
    ToolDefinition(
        name="detect_anomalies",
        description=(
            "Score months of a supported metric (revenue, mrr, customer_count, support_ticket_volume, "
            "product_adoption) against the preceding months with a rolling z-score, IQR or forecast-residual "
            "detector (Phase 3). Default range: the 12 months to the as-of date."
        ),
        when_to_use="Whether a month was statistically unusual, and when an unusual movement began.",
        not_for="Causes of a movement; metrics without a monthly series.",
        output_description="AnomalyReport: every scored month and the ranked flagged anomalies with explanations.",
        limitations="Direction is statistical only; short windows make scores noisy.",
        source_layer="phase3_anomalies",
        input_model=AnomalyInput,
        handler=handlers.detect_anomalies,
    ),
    ToolDefinition(
        name="run_safe_sql",
        description=(
            "Run ONE read-only SELECT over the allow-listed business tables and views, with values bound as $name "
            "parameters. Rows are capped at the configured limit and truncation is reported."
        ),
        when_to_use="Only for questions no other tool covers (e.g. a specific count or list from the tables).",
        not_for="Registered KPIs or analytics (use the dedicated tools), data changes, files, or system tables.",
        output_description="SQLResult: columns, rows, row_count, truncated flag, query id and source tables.",
        limitations="Ad-hoc results are not KPI definitions; PII-tagged columns are not queryable.",
        source_layer="phase1_database",
        input_model=SafeSQLInput,
        handler=handlers.run_safe_sql,
    ),
)


class ToolRegistry:
    def __init__(self, definitions: tuple[ToolDefinition, ...] = TOOL_DEFINITIONS):
        self._tools = {d.name: d for d in definitions}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def catalog(self) -> list[dict[str, Any]]:
        return [d.catalog_entry() for d in self._tools.values()]

    def validate_arguments(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate arguments for a tool; returns the canonical argument dict or raises ``InvalidRequestError``."""
        definition = self._tools.get(name)
        if definition is None:
            raise InvalidRequestError(f"Unknown tool {name!r}. Registered tools: {', '.join(self._tools)}")
        try:
            parsed = definition.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise InvalidRequestError(f"Invalid arguments for {name}: {_summarise(exc)}") from None
        return parsed.model_dump(mode="json", exclude_unset=True)

    def execute(self, request: ToolRequest, context: ToolContext) -> ToolResult:
        started = datetime.now(UTC)
        clock = time.perf_counter()
        definition = self._tools.get(request.tool_name)

        def failure(code: str, message: str, retryable: bool) -> ToolResult:
            return ToolResult(
                call_id=request.call_id,
                tool_name=request.tool_name,
                arguments=request.arguments,
                success=False,
                status="error",
                error=ToolError(code=code, message=message, retryable=retryable),
                message=message,
                started_at=started,
                finished_at=datetime.now(UTC),
                execution_time_ms=round((time.perf_counter() - clock) * 1000, 3),
            )

        if definition is None:
            return failure("unknown_tool", f"Unknown tool {request.tool_name!r}", False)
        try:
            parsed = definition.input_model.model_validate(request.arguments)
        except ValidationError as exc:
            return failure("invalid_arguments", _summarise(exc), False)
        except InvalidRequestError as exc:
            return failure("invalid_arguments", str(exc), False)
        try:
            output: ToolOutput = definition.handler(context, parsed)
        except UnsafeSQLError as exc:
            return failure("unsafe_sql", str(exc), False)
        except AnalyticsDatabaseError as exc:
            return failure("database_error", str(exc), True)
        except InvalidRequestError as exc:
            return failure(getattr(exc, "code", "invalid_request"), str(exc), False)
        except AnalyticsError as exc:
            return failure(getattr(exc, "code", "analytics_error"), str(exc), False)
        except Exception as exc:  # an unexpected failure is reported, never hidden or replaced
            return failure("internal_error", f"{type(exc).__name__}: {exc}", False)
        return ToolResult(
            call_id=request.call_id,
            tool_name=request.tool_name,
            arguments=parsed.model_dump(mode="json", exclude_unset=True),
            success=True,
            status=output.status,
            result=output.result,
            result_type=type(output.result).__name__,
            message=output.message,
            started_at=started,
            finished_at=datetime.now(UTC),
            execution_time_ms=round((time.perf_counter() - clock) * 1000, 3),
            source_tables=output.source_tables,
            query_ids=output.query_ids,
            calculation=output.calculation,
            limitations=output.limitations,
        )


def _summarise(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors()[:5]:
        location = ".".join(str(p) for p in error["loc"]) or "arguments"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)
