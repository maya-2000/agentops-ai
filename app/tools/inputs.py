"""Typed input models for every tool. Arguments are validated before any service is called.

Analytics tools take an ``operation`` naming one existing Phase 2 function. Each operation declares
which arguments it requires and which it accepts, so a planner cannot pass an argument the
underlying function would silently ignore.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import Field, model_validator

from app.analytics.errors import InvalidRequestError
from app.analytics.periods import Period, explicit_period
from app.anomalies.config import DetectorName
from app.timeseries.metrics import Transform
from app.tools.base import ToolInput

Grain = Literal["month", "quarter"]


class PeriodArgs(ToolInput):
    """A period as a spec (``last_month``, ``2026-08``, ``2026-Q2``, ``trailing_3_months``, ...) or explicit dates."""

    period: str | None = None
    start_date: date | None = None
    end_date: date | None = None

    @model_validator(mode="after")
    def _period_consistent(self) -> PeriodArgs:
        if (self.start_date is None) != (self.end_date is None):
            raise InvalidRequestError("start_date and end_date must be given together")
        if self.start_date is not None and self.period is not None:
            raise InvalidRequestError("Give either period or start_date/end_date, not both")
        return self

    def period_spec(self) -> str | Period | None:
        if self.start_date is not None and self.end_date is not None:
            return explicit_period(self.start_date, self.end_date)
        return self.period


class ComparisonArgs(PeriodArgs):
    comparison_period: str | None = None
    comparison_start_date: date | None = None
    comparison_end_date: date | None = None

    @model_validator(mode="after")
    def _comparison_consistent(self) -> ComparisonArgs:
        if (self.comparison_start_date is None) != (self.comparison_end_date is None):
            raise InvalidRequestError("comparison_start_date and comparison_end_date must be given together")
        if self.comparison_start_date is not None and self.comparison_period is not None:
            raise InvalidRequestError("Give either comparison_period or comparison dates, not both")
        return self

    def comparison_spec(self) -> str | Period | None:
        if self.comparison_start_date is not None and self.comparison_end_date is not None:
            return explicit_period(self.comparison_start_date, self.comparison_end_date)
        return self.comparison_period

    @property
    def has_comparison(self) -> bool:
        return self.comparison_period is not None or self.comparison_start_date is not None


def check_operation_arguments(model: ToolInput, operation: str, rules: dict[str, tuple[set[str], set[str]]]) -> None:
    """Reject missing required arguments and arguments the operation does not use."""
    required, allowed = rules[operation]
    given = {name for name in model.model_fields_set if name != "operation"}
    missing = required - given
    if missing:
        raise InvalidRequestError(f"Operation {operation!r} requires: {', '.join(sorted(missing))}")
    extra = given - required - allowed
    if extra:
        raise InvalidRequestError(f"Operation {operation!r} does not accept: {', '.join(sorted(extra))}")


_PERIOD = {"period", "start_date", "end_date"}
_COMPARISON = _PERIOD | {"comparison_period", "comparison_start_date", "comparison_end_date"}

# ------------------------------------------------------------------------------------------------ KPI


class GetKPIInput(ComparisonArgs):
    kpi: str = Field(description="Registered KPI key, e.g. revenue, mrr, logo_churn_rate, win_rate")
    dimension: str | None = Field(default=None, description="Optional breakdown dimension, e.g. segment, region")
    filters: dict[str, str] = Field(default_factory=dict)


# ------------------------------------------------------------------------------------------------ revenue

RevenueOperation = Literal[
    "revenue_by_period",
    "revenue_change",
    "decompose_revenue_change",
    "revenue_bridge",
    "mrr_series",
    "revenue_concentration",
]
REVENUE_RULES: dict[str, tuple[set[str], set[str]]] = {
    "revenue_by_period": (set(), _PERIOD | {"grain", "filters"}),
    "revenue_change": (set(), _COMPARISON | {"filters"}),
    "decompose_revenue_change": ({"dimension"}, _COMPARISON | {"filters"}),
    "revenue_bridge": (set(), _PERIOD | {"filters"}),
    "mrr_series": (set(), _PERIOD | {"filters"}),
    "revenue_concentration": (set(), _PERIOD | {"top_n", "filters"}),
}


class AnalyzeRevenueInput(ComparisonArgs):
    operation: RevenueOperation
    dimension: str | None = None
    filters: dict[str, str] = Field(default_factory=dict)
    grain: Grain = "month"
    top_n: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def _operation_arguments(self) -> AnalyzeRevenueInput:
        check_operation_arguments(self, self.operation, REVENUE_RULES)
        return self


# ------------------------------------------------------------------------------------------------ customers

CustomerOperation = Literal[
    "churn_summary", "churn_by_dimension", "customer_movements", "monthly_churn_series", "usage_churn_relationship"
]
CUSTOMER_RULES: dict[str, tuple[set[str], set[str]]] = {
    "churn_summary": (set(), _PERIOD | {"filters"}),
    "churn_by_dimension": ({"dimension"}, _PERIOD | {"filters", "min_customers"}),
    "customer_movements": (set(), _PERIOD | {"dimension", "filters"}),
    "monthly_churn_series": (set(), _PERIOD | {"filters"}),
    "usage_churn_relationship": (set(), _PERIOD | {"filters"}),
}


class AnalyzeCustomersInput(PeriodArgs):
    operation: CustomerOperation
    dimension: str | None = None
    filters: dict[str, str] = Field(default_factory=dict)
    min_customers: int = Field(default=30, ge=1, le=1000)

    @model_validator(mode="after")
    def _operation_arguments(self) -> AnalyzeCustomersInput:
        check_operation_arguments(self, self.operation, CUSTOMER_RULES)
        return self


# ------------------------------------------------------------------------------------------------ sales

SalesOperation = Literal[
    "pipeline_summary",
    "sales_performance",
    "rep_performance",
    "opportunity_conversion",
    "funnel_stage_distribution",
    "segment_performance",
]
SALES_RULES: dict[str, tuple[set[str], set[str]]] = {
    "pipeline_summary": (set(), {"filters"}),
    "sales_performance": (set(), _PERIOD | {"dimension", "filters"}),
    "rep_performance": (set(), _PERIOD | {"filters", "min_closed"}),
    "opportunity_conversion": (set(), _PERIOD | {"dimension", "filters"}),
    "funnel_stage_distribution": (set(), _PERIOD | {"filters"}),
    "segment_performance": (set(), _PERIOD),
}


class AnalyzeSalesInput(PeriodArgs):
    operation: SalesOperation
    dimension: str | None = None
    filters: dict[str, str] = Field(default_factory=dict)
    min_closed: int = Field(default=30, ge=1, le=1000)

    @model_validator(mode="after")
    def _operation_arguments(self) -> AnalyzeSalesInput:
        check_operation_arguments(self, self.operation, SALES_RULES)
        return self


# ------------------------------------------------------------------------------------------------ marketing

MarketingOperation = Literal["channel_performance", "campaign_performance", "marketing_period_change", "channel_roas"]
MARKETING_RULES: dict[str, tuple[set[str], set[str]]] = {
    "channel_performance": (set(), _PERIOD),
    "campaign_performance": (set(), _PERIOD | {"channel", "min_conversions"}),
    "marketing_period_change": (set(), _COMPARISON),
    "channel_roas": (set(), _PERIOD | {"window_days"}),
}


class AnalyzeMarketingInput(ComparisonArgs):
    operation: MarketingOperation
    channel: str | None = None
    min_conversions: int = Field(default=10, ge=1, le=10000)
    window_days: int = Field(default=90, ge=7, le=365)

    @model_validator(mode="after")
    def _operation_arguments(self) -> AnalyzeMarketingInput:
        check_operation_arguments(self, self.operation, MARKETING_RULES)
        return self


# ------------------------------------------------------------------------------------------------ support

SupportOperation = Literal["support_summary", "support_by_dimension", "support_volume_change", "resolution_time_trend"]
SUPPORT_RULES: dict[str, tuple[set[str], set[str]]] = {
    "support_summary": (set(), _PERIOD | {"filters"}),
    "support_by_dimension": ({"dimension"}, _PERIOD | {"filters"}),
    "support_volume_change": (set(), _COMPARISON | {"filters"}),
    "resolution_time_trend": (set(), _PERIOD | {"filters"}),
}


class AnalyzeSupportInput(ComparisonArgs):
    operation: SupportOperation
    dimension: str | None = None
    filters: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _operation_arguments(self) -> AnalyzeSupportInput:
        check_operation_arguments(self, self.operation, SUPPORT_RULES)
        return self


# ------------------------------------------------------------------------------------------------ product

ProductOperation = Literal[
    "feature_adoption",
    "adoption_trend",
    "adoption_change",
    "feature_launch_summary",
    "adoption_breadth_by",
    "feature_usage_distribution",
]
PRODUCT_RULES: dict[str, tuple[set[str], set[str]]] = {
    "feature_adoption": (set(), _PERIOD),
    "adoption_trend": ({"feature"}, _PERIOD | {"grain"}),
    "adoption_change": ({"feature"}, _COMPARISON),
    "feature_launch_summary": ({"feature"}, {"window_days"}),
    "adoption_breadth_by": ({"dimension"}, _PERIOD | {"filters"}),
    "feature_usage_distribution": (set(), _PERIOD | {"filters"}),
}


class AnalyzeProductInput(ComparisonArgs):
    operation: ProductOperation
    feature: str | None = None
    dimension: str | None = None
    filters: dict[str, str] = Field(default_factory=dict)
    grain: Grain = "month"
    window_days: int = Field(default=28, ge=7, le=180)

    @model_validator(mode="after")
    def _operation_arguments(self) -> AnalyzeProductInput:
        check_operation_arguments(self, self.operation, PRODUCT_RULES)
        return self


# ------------------------------------------------------------------------------------------------ cohorts / risk


class CohortAnalysisInput(ToolInput):
    first_cohort: str | None = Field(default=None, description="First signup month, YYYY-MM")
    last_cohort: str | None = Field(default=None, description="Last signup month, YYYY-MM")
    max_months: int | None = Field(default=None, ge=1, le=24)
    filters: dict[str, str] = Field(default_factory=dict)


class CustomerRiskInput(ToolInput):
    filters: dict[str, str] = Field(default_factory=dict)
    min_band: Literal["low", "medium", "high"] = "medium"
    limit: int = Field(default=20, ge=1, le=200)


# ------------------------------------------------------------------------------------------------ Phase 3


class ForecastInput(ToolInput):
    metric: str = Field(description="revenue, mrr, customer_count, support_ticket_volume or product_adoption")
    horizon: int = Field(default=3, description="Months ahead, 1-6")
    cutoff_date: date | None = None
    filters: dict[str, str] = Field(default_factory=dict)
    model: str = "auto"
    confidence_level: float | None = Field(default=None, gt=0.5, lt=1.0)


class AnomalyInput(ToolInput):
    metric: str = Field(description="revenue, mrr, customer_count, support_ticket_volume or product_adoption")
    start_date: date | None = None
    end_date: date | None = None
    detector: DetectorName = "rolling_zscore"
    filters: dict[str, str] = Field(default_factory=dict)
    window: int = Field(default=12, ge=3, le=36)
    transform: Transform | None = None


# ------------------------------------------------------------------------------------------------ SQL


class SafeSQLInput(ToolInput):
    sql: str = Field(description="One read-only SELECT over allow-listed tables, with $name parameters for values")
    parameters: dict[str, Any] = Field(default_factory=dict)
    max_rows: int | None = Field(default=None, ge=1)
    description: str = Field(default="", description="What the query computes, in one sentence")
