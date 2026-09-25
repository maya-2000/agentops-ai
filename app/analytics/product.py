"""Product analytics: feature adoption, adoption over time and change, launch view, usage breadth.

Feature adoption comes from the registered ``product_adoption`` KPI: the mean of the daily
``product_features.adoption_rate``, where adoption_rate = feature daily active users / platform
daily active users. ``product_features`` has no customer attributes, so feature-specific
adoption cannot be split by segment or region. For those comparisons this module reports
*adoption breadth* from ``usage_events.feature_usage``: the mean number of distinct features an
account used per week.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel

from app.analytics.common import CUSTOMER_FILTER_COLUMNS, PeriodSpec, filter_clause, to_filters, to_period
from app.analytics.dimensions import Filters
from app.analytics.errors import InvalidRequestError
from app.analytics.executor import QueryRunner
from app.analytics.kpis.service import KPIService
from app.analytics.models import AnalyticsResult, to_number
from app.analytics.periods import explicit_period, previous_period
from app.database.base import Database

BREADTH_DIMENSIONS = ("segment", "region", "country", "industry", "acquisition_channel")


class FeatureAdoptionRow(BaseModel):
    feature: str
    average_daily_adoption_rate: float | None
    average_daily_feature_users: float | None
    observed_days: int
    first_observed_date: str | None


class AdoptionTrendRow(BaseModel):
    period: str
    average_daily_adoption_rate: float | None
    average_daily_feature_users: float | None
    observed_days: int


class AdoptionBreadthRow(BaseModel):
    dimension_value: str
    account_weeks: int
    accounts: int
    average_features_used: float
    average_weekly_active_users: float


class UsageDistributionRow(BaseModel):
    features_used: int
    account_weeks: int
    share: float


def feature_adoption(
    db: Database, period: PeriodSpec = None, *, as_of: date | None = None
) -> AnalyticsResult[FeatureAdoptionRow]:
    """Average daily adoption rate and feature users for every feature observed in the period, highest first."""
    current = to_period(period, as_of)
    kpi = KPIService(db, as_of=as_of).calculate_kpi(
        "product_adoption", start_date=current.start, end_date=current.end, dimension="product_feature"
    )
    rows = sorted(
        (_feature_row(r.dimension_value, r.components) for r in kpi.breakdown),
        key=lambda r: -(r.average_daily_adoption_rate or 0.0),
    )
    return AnalyticsResult[FeatureAdoptionRow](
        operation="feature_adoption",
        status="ok" if rows else "no_data",
        period=current,
        dimensions=["product_feature"],
        data=rows,
        summary={"features": len(rows), "highest_adoption": rows[0].feature if rows else None},
        message=None if rows else "No feature observations in the period.",
        limitations=list(kpi.limitations),
        provenance=kpi.provenance,
    )


def adoption_trend(
    db: Database,
    feature: str,
    period: PeriodSpec = "trailing_12_months",
    *,
    grain: Literal["month", "quarter"] = "month",
    as_of: date | None = None,
) -> AnalyticsResult[AdoptionTrendRow]:
    """Adoption of one feature per month or quarter (months before the feature existed are absent)."""
    current = to_period(period, as_of)
    kpi = KPIService(db, as_of=as_of).calculate_kpi(
        "product_adoption", start_date=current.start, end_date=current.end, product_feature=feature, dimension=grain
    )
    rows = [
        AdoptionTrendRow(
            period=r.dimension_value,
            average_daily_adoption_rate=_float(r.components["average_daily_adoption_rate"]),
            average_daily_feature_users=_float(r.components["average_daily_feature_users"]),
            observed_days=int(r.components["observed_feature_days"] or 0),
        )
        for r in kpi.breakdown
    ]
    first, last = (rows[0], rows[-1]) if rows else (None, None)
    return AnalyticsResult[AdoptionTrendRow](
        operation="adoption_trend",
        status=kpi.status,
        period=current,
        filters=kpi.filters,
        dimensions=[grain],
        data=rows,
        summary={
            "feature": kpi.filters.get("product_feature"),
            "first_period": first.period if first else None,
            "first_period_adoption": first.average_daily_adoption_rate if first else None,
            "last_period": last.period if last else None,
            "last_period_adoption": last.average_daily_adoption_rate if last else None,
            "change_in_percentage_points": (
                (last.average_daily_adoption_rate or 0) - (first.average_daily_adoption_rate or 0)
            )
            * 100
            if first and last
            else None,
        },
        message=kpi.message,
        limitations=list(kpi.limitations),
        provenance=kpi.provenance,
    )


def adoption_change(
    db: Database, feature: str, period: PeriodSpec = None, comparison: PeriodSpec = None, *, as_of: date | None = None
) -> AnalyticsResult[AdoptionTrendRow]:
    """Adoption of a feature in a period vs a comparison period (default: the preceding period)."""
    current = to_period(period, as_of)
    cmp_period = to_period(comparison, as_of) if comparison is not None else previous_period(current)
    service = KPIService(db, as_of=as_of)
    now = service.calculate_kpi(
        "product_adoption", start_date=current.start, end_date=current.end, product_feature=feature
    )
    before = service.calculate_kpi(
        "product_adoption", start_date=cmp_period.start, end_date=cmp_period.end, product_feature=feature
    )
    runner = QueryRunner(db, "adoption_change")
    runner.absorb(now.provenance)
    runner.absorb(before.provenance)
    rows = [
        AdoptionTrendRow(
            period=p.label,
            average_daily_adoption_rate=k.value,
            average_daily_feature_users=_float(k.components.get("average_daily_feature_users")),
            observed_days=int(k.components.get("observed_feature_days") or 0),
        )
        for p, k in ((cmp_period, before), (current, now))
    ]
    change_pp = (now.value - before.value) * 100 if now.value is not None and before.value is not None else None
    message = None
    if before.status != "ok":
        message = f"No adoption observed for {feature} in {cmp_period.label} (e.g. before launch)."
    return AnalyticsResult[AdoptionTrendRow](
        operation="adoption_change",
        status="ok" if now.status == "ok" and before.status == "ok" else "insufficient_data",
        period=current,
        comparison_period=cmp_period,
        filters=now.filters,
        data=rows,
        summary={
            "feature": now.filters.get("product_feature"),
            "current_adoption": now.value,
            "comparison_adoption": before.value,
            "change_in_percentage_points": change_pp,
        },
        message=message,
        limitations=list(now.limitations),
        provenance=runner.provenance("product_adoption in both periods; change in percentage points"),
    )


def feature_launch_summary(
    db: Database, feature: str, *, window_days: int = 28, as_of: date | None = None
) -> AnalyticsResult[AdoptionTrendRow]:
    """Adoption in the first ``window_days`` after a feature's first observation vs the latest ``window_days``."""
    service = KPIService(db, as_of=as_of)
    runner = QueryRunner(db, "feature_launch_summary")
    first_seen = runner.records(
        "SELECT MIN(date) AS first_date, MAX(date) AS last_date FROM product_features WHERE feature_name = $feature",
        {"feature": feature},
        calculation="first and last observed date for the feature",
    )[0]
    if first_seen["first_date"] is None:
        raise InvalidRequestError(f"No product feature {feature!r} exists in the data")
    first_date, last_date = first_seen["first_date"], min(first_seen["last_date"], service.as_of)
    launch = explicit_period(
        first_date, min(first_date + timedelta(days=window_days - 1), last_date), f"first {window_days} days"
    )
    latest = explicit_period(
        max(first_date, last_date - timedelta(days=window_days - 1)), last_date, f"latest {window_days} days"
    )
    change = adoption_change(db, feature, latest, launch, as_of=as_of)
    runner.absorb(change.provenance)
    return change.model_copy(
        update={
            "operation": "feature_launch_summary",
            "summary": {
                **change.summary,
                "first_observed_date": first_date.isoformat(),
                "latest_date": last_date.isoformat(),
                "days_since_first_observation": (last_date - first_date).days,
            },
            "provenance": runner.provenance(
                "first observed date of the feature; product_adoption in its first and latest windows"
            ),
        }
    )


def adoption_breadth_by(
    db: Database,
    dimension: str,
    period: PeriodSpec = None,
    *,
    filters: Filters | dict[str, str] | None = None,
    as_of: date | None = None,
) -> AnalyticsResult[AdoptionBreadthRow]:
    """Mean distinct features used per account-week (and mean weekly active users) by a customer attribute."""
    if dimension not in BREADTH_DIMENSIONS:
        raise InvalidRequestError(f"adoption_breadth_by supports {', '.join(BREADTH_DIMENSIONS)}")
    current = to_period(period, as_of)
    active = to_filters(filters)
    clause, bind = filter_clause(active, CUSTOMER_FILTER_COLUMNS, "adoption_breadth_by")
    runner = QueryRunner(db, "adoption_breadth_by")
    records = runner.records(
        f"""
SELECT CAST({CUSTOMER_FILTER_COLUMNS[dimension]} AS VARCHAR) AS dimension_value,
       COUNT(*) AS account_weeks, COUNT(DISTINCT u.customer_id) AS accounts,
       AVG(u.feature_usage) AS average_features_used, AVG(u.active_users) AS average_weekly_active_users
FROM usage_events AS u
JOIN customers AS c ON c.customer_id = u.customer_id
WHERE u.event_date BETWEEN $start_date AND $end_date{clause}
GROUP BY 1
ORDER BY 1
""",
        {"start_date": current.start, "end_date": current.end, **bind},
        calculation="mean distinct features used per account-week, by customer attribute",
    )
    rows = [
        AdoptionBreadthRow(
            dimension_value=str(r["dimension_value"]),
            account_weeks=int(r["account_weeks"]),
            accounts=int(r["accounts"]),
            average_features_used=float(r["average_features_used"]),
            average_weekly_active_users=float(r["average_weekly_active_users"]),
        )
        for r in records
    ]
    return AnalyticsResult[AdoptionBreadthRow](
        operation="adoption_breadth_by",
        status="ok" if rows else "no_data",
        period=current,
        filters=active,
        dimensions=[dimension],
        data=rows,
        message=None if rows else "No usage observations in the period.",
        limitations=[
            "Breadth counts distinct features used per account-week; it does not identify which features.",
            "Usage weeks belong to the period containing their Monday.",
        ],
        provenance=runner.provenance("AVG(usage_events.feature_usage) per group"),
    )


def feature_usage_distribution(
    db: Database,
    period: PeriodSpec = None,
    *,
    filters: Filters | dict[str, str] | None = None,
    as_of: date | None = None,
) -> AnalyticsResult[UsageDistributionRow]:
    """Distribution of distinct features used per account-week."""
    current = to_period(period, as_of)
    active = to_filters(filters)
    clause, bind = filter_clause(active, CUSTOMER_FILTER_COLUMNS, "feature_usage_distribution")
    runner = QueryRunner(db, "feature_usage_distribution")
    records = runner.records(
        f"""
SELECT u.feature_usage AS features_used, COUNT(*) AS account_weeks
FROM usage_events AS u
JOIN customers AS c ON c.customer_id = u.customer_id
WHERE u.event_date BETWEEN $start_date AND $end_date{clause}
GROUP BY 1
ORDER BY 1
""",
        {"start_date": current.start, "end_date": current.end, **bind},
        calculation="account-weeks by number of distinct features used",
    )
    total = sum(int(r["account_weeks"]) for r in records)
    rows = [
        UsageDistributionRow(
            features_used=int(r["features_used"]),
            account_weeks=int(r["account_weeks"]),
            share=int(r["account_weeks"]) / total,
        )
        for r in records
    ]
    return AnalyticsResult[UsageDistributionRow](
        operation="feature_usage_distribution",
        status="ok" if rows else "no_data",
        period=current,
        filters=active,
        dimensions=["features_used"],
        data=rows,
        summary={"account_weeks": total},
        provenance=runner.provenance("share = account-weeks with N features used / all account-weeks"),
    )


def _float(value: object) -> float | None:
    number = to_number(value)
    return None if number is None else float(number)


def _feature_row(feature: str, c: dict[str, float | int | str | None]) -> FeatureAdoptionRow:
    first = c.get("first_observed_date")
    return FeatureAdoptionRow(
        feature=feature,
        average_daily_adoption_rate=_float(c.get("average_daily_adoption_rate")),
        average_daily_feature_users=_float(c.get("average_daily_feature_users")),
        observed_days=int(c.get("observed_feature_days") or 0),
        first_observed_date=str(first) if first is not None else None,
    )
