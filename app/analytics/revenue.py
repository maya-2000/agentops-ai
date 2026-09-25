"""Revenue analytics: revenue over time, growth, change decomposition, MRR bridge and concentration.

Revenue figures come from the registered KPIs (``revenue``, ``revenue_growth``, ``mrr``), so the
numbers here are identical to what ``calculate_kpi`` reports. Operations that are not KPIs
(the MRR bridge, month-end MRR series, concentration) use their own documented SQL, built on
the same point-in-time rule.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel

from app.analytics.common import CUSTOMER_FILTER_COLUMNS, PeriodSpec, filter_clause, pct_change, to_filters
from app.analytics.dimensions import Filters
from app.analytics.errors import InvalidRequestError, ResultStatus
from app.analytics.executor import QueryRunner
from app.analytics.kpis.models import KPIResult
from app.analytics.kpis.service import KPIService
from app.analytics.kpis.sql import in_force
from app.analytics.models import AnalyticsResult, safe_ratio, to_number
from app.analytics.periods import Period, previous_period, resolve_period
from app.database.base import Database

DECOMPOSITION_DIMENSIONS = ("region", "country", "segment", "plan", "industry", "acquisition_channel", "revenue_type")
RECONCILIATION_TOLERANCE_SGD = 0.01  # float conversion of exact DECIMAL sums


# ---------------------------------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------------------------------


class RevenuePeriodRow(BaseModel):
    period: str
    revenue: float
    subscription_revenue: float
    usage_revenue: float


class DecompositionRow(BaseModel):
    """One member's share of a revenue change. Negative ``absolute_change`` = decline."""

    dimension: str
    dimension_value: str
    current_value: float
    previous_value: float
    absolute_change: float
    percentage_change: float | None  # None when the member had no revenue in the comparison period
    contribution_to_total_change: float | None  # absolute_change / previous total (sums to total growth rate)
    share_of_total_change: float | None  # absolute_change / total change (sums to 1)
    share_of_gross_decline: float | None  # this member's decline / sum of all members' declines
    share_of_gross_increase: float | None  # this member's increase / sum of all members' increases


class BridgeRow(BaseModel):
    component: Literal["opening_mrr", "new", "expansion", "contraction", "churn", "reactivation", "closing_mrr"]
    mrr: float | None
    customers: int | None
    note: str | None = None


class MRRSeriesRow(BaseModel):
    month: str
    month_end: date
    mrr: float
    arr: float
    active_customers: int


class ConcentrationRow(BaseModel):
    rank: int
    customer_id: str
    company_name: str
    revenue: float
    share_of_revenue: float
    cumulative_share: float


# ---------------------------------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------------------------------


def revenue_by_period(
    db: Database,
    period: PeriodSpec = "trailing_12_months",
    *,
    grain: Literal["month", "quarter"] = "month",
    filters: Filters | dict[str, str] | None = None,
    as_of: date | None = None,
) -> AnalyticsResult[RevenuePeriodRow]:
    """Recognised revenue per month or quarter (subscription and usage split)."""
    params: dict[str, object] = {"period": _spec(period), "dimension": grain, "filters": to_filters(filters)}
    kpi = KPIService(db, as_of=as_of).calculate_kpi("revenue", {**params, **_dates(period)})
    rows = [
        RevenuePeriodRow(
            period=r.dimension_value,
            revenue=float(r.components["revenue"] or 0),
            subscription_revenue=float(r.components["subscription_revenue"] or 0),
            usage_revenue=float(r.components["usage_revenue"] or 0),
        )
        for r in kpi.breakdown
    ]
    return AnalyticsResult[RevenuePeriodRow](
        operation="revenue_by_period",
        status=kpi.status,
        period=kpi.period,
        filters=kpi.filters,
        dimensions=[grain],
        data=rows,
        summary={"total_revenue": kpi.value, "periods": len(rows)},
        message=kpi.message,
        limitations=kpi.limitations,
        provenance=kpi.provenance,
    )


def revenue_growth(
    db: Database,
    period: PeriodSpec = None,
    comparison: PeriodSpec = None,
    *,
    filters: Filters | dict[str, str] | None = None,
    as_of: date | None = None,
) -> KPIResult:
    """Revenue growth KPI (comparison defaults to the preceding period of the same shape)."""
    service = KPIService(db, as_of=as_of)
    current = _resolve(period, service)
    params = {"start_date": current.start, "end_date": current.end, "filters": to_filters(filters)}
    if comparison is not None:
        cmp_period = _resolve(comparison, service)
        params |= {"comparison_start_date": cmp_period.start, "comparison_end_date": cmp_period.end}
    return service.calculate_kpi("revenue_growth", params)


def revenue_change(
    db: Database,
    period: PeriodSpec = None,
    comparison: PeriodSpec = None,
    *,
    filters: Filters | dict[str, str] | None = None,
    as_of: date | None = None,
) -> AnalyticsResult[DecompositionRow]:
    """Total revenue change between two periods (no breakdown)."""
    growth = revenue_growth(db, period, comparison, filters=filters, as_of=as_of)
    current = growth.components.get("current_revenue")
    previous = growth.components.get("comparison_revenue")
    summary = {
        "current_revenue": _num(current),
        "comparison_revenue": _num(previous),
        "absolute_change": _num(current) - _num(previous) if current is not None and previous is not None else None,
        "percentage_change": growth.value,
        "direction": _direction(growth.value),
    }
    return AnalyticsResult[DecompositionRow](
        operation="revenue_change",
        status=growth.status,
        period=growth.period,
        comparison_period=growth.comparison_period,
        filters=growth.filters,
        summary=summary,
        message=growth.message,
        limitations=growth.limitations,
        provenance=growth.provenance,
    )


def decompose_revenue_change(
    db: Database,
    dimension: str,
    period: PeriodSpec = None,
    comparison: PeriodSpec = None,
    *,
    filters: Filters | dict[str, str] | None = None,
    as_of: date | None = None,
) -> AnalyticsResult[DecompositionRow]:
    """Split the revenue change between two periods across the members of ``dimension``.

    The member changes sum exactly to the total change (checked and reported as
    ``reconciliation_difference``). Rows are ordered from the largest decline to the largest increase.
    """
    if dimension not in DECOMPOSITION_DIMENSIONS:
        raise InvalidRequestError(f"Revenue can be decomposed by {', '.join(DECOMPOSITION_DIMENSIONS)}")
    service = KPIService(db, as_of=as_of)
    current = _resolve(period, service)
    cmp_period = _resolve(comparison, service) if comparison is not None else previous_period(current)
    kpi = service.calculate_kpi(
        "revenue_growth",
        start_date=current.start,
        end_date=current.end,
        comparison_start_date=cmp_period.start,
        comparison_end_date=cmp_period.end,
        dimension=dimension,
        filters=to_filters(filters),
    )
    if kpi.status != "ok":
        return AnalyticsResult[DecompositionRow](
            operation="decompose_revenue_change",
            status=kpi.status,
            period=kpi.period,
            comparison_period=kpi.comparison_period,
            filters=kpi.filters,
            dimensions=[dimension],
            message=kpi.message,
            limitations=kpi.limitations,
            provenance=kpi.provenance,
        )

    total_current = _num(kpi.components["current_revenue"])
    total_previous = _num(kpi.components["comparison_revenue"])
    total_change = total_current - total_previous
    changes = {
        r.dimension_value: (_num(r.components["current_revenue"]), _num(r.components["comparison_revenue"]))
        for r in kpi.breakdown
    }
    gross_decline = sum(cur - prev for cur, prev in changes.values() if cur < prev)
    gross_increase = sum(cur - prev for cur, prev in changes.values() if cur > prev)
    rows = []
    for value, (cur, prev) in changes.items():
        change = cur - prev
        rows.append(
            DecompositionRow(
                dimension=dimension,
                dimension_value=value,
                current_value=cur,
                previous_value=prev,
                absolute_change=change,
                percentage_change=pct_change(cur, prev),
                contribution_to_total_change=safe_ratio(change, total_previous),
                share_of_total_change=safe_ratio(change, total_change),
                share_of_gross_decline=safe_ratio(change, gross_decline) if change < 0 else None,
                share_of_gross_increase=safe_ratio(change, gross_increase) if change > 0 else None,
            )
        )
    rows.sort(key=lambda r: r.absolute_change)
    component_sum = sum(r.absolute_change for r in rows)
    difference = component_sum - total_change
    summary = {
        "current_revenue": total_current,
        "comparison_revenue": total_previous,
        "total_change": total_change,
        "total_percentage_change": kpi.value,
        "direction": _direction(kpi.value),
        "sum_of_member_changes": component_sum,
        "reconciliation_difference": difference,
        "reconciled": abs(difference) <= RECONCILIATION_TOLERANCE_SGD,
        "gross_decline": gross_decline,
        "gross_increase": gross_increase,
        "largest_decline": rows[0].dimension_value if rows and rows[0].absolute_change < 0 else None,
        "largest_increase": rows[-1].dimension_value if rows and rows[-1].absolute_change > 0 else None,
    }
    return AnalyticsResult[DecompositionRow](
        operation="decompose_revenue_change",
        status="ok",
        period=kpi.period,
        comparison_period=kpi.comparison_period,
        filters=kpi.filters,
        dimensions=[dimension],
        data=rows,
        summary=summary,
        limitations=[
            *kpi.limitations,
            "share_of_total_change can exceed 100% or be negative when members move in opposite directions; "
            "share_of_gross_decline / share_of_gross_increase show each member's part of all declines / increases.",
            "Decomposition attributes change to members; it does not establish why revenue changed.",
        ],
        provenance=kpi.provenance,
    )


def revenue_bridge(
    db: Database,
    period: PeriodSpec = None,
    *,
    filters: Filters | dict[str, str] | None = None,
    as_of: date | None = None,
) -> AnalyticsResult[BridgeRow]:
    """MRR bridge from subscription events: opening + new + expansion - contraction - churn = closing.

    Opening and closing MRR come from the ``mrr`` KPI (at the close of the day before the period
    and of the period's last day). The identity holds exactly under the point-in-time rule, and
    the residual is reported. The data model has no reactivations (a churned customer never
    returns), so that component is reported as not observed rather than as zero.
    """
    service = KPIService(db, as_of=as_of)
    current = _resolve(period, service)
    active = to_filters(filters)
    runner = QueryRunner(db, "revenue_bridge")
    opening = service.calculate_kpi(
        "mrr", start_date=current.opening_date, end_date=current.opening_date, filters=active
    )
    closing = service.calculate_kpi("mrr", start_date=current.start, end_date=current.end, filters=active)
    runner.absorb(opening.provenance)
    runner.absorb(closing.provenance)
    if opening.status != "ok" or closing.status != "ok":
        return AnalyticsResult[BridgeRow](
            operation="revenue_bridge",
            status="insufficient_data",
            period=current,
            filters=active,
            message=opening.message or closing.message,
            limitations=[*opening.limitations],
            provenance=runner.provenance("MRR bridge: opening or closing MRR not observable"),
        )

    clause, bind = filter_clause(active, CUSTOMER_FILTER_COLUMNS, "revenue_bridge")
    sql = f"""
SELECT
    COALESCE(SUM(s.monthly_recurring_revenue) FILTER (
        WHERE s.change_type = 'new' AND s.start_date BETWEEN $start_date AND $end_date), 0) AS new_mrr,
    COUNT(*) FILTER (WHERE s.change_type = 'new' AND s.start_date BETWEEN $start_date AND $end_date) AS new_customers,
    COALESCE(SUM(s.monthly_recurring_revenue - s.previous_mrr) FILTER (
        WHERE s.change_type = 'expansion' AND s.start_date BETWEEN $start_date AND $end_date), 0) AS expansion_mrr,
    COUNT(DISTINCT s.customer_id) FILTER (
        WHERE s.change_type = 'expansion' AND s.start_date BETWEEN $start_date AND $end_date) AS expanding_customers,
    COALESCE(SUM(s.previous_mrr - s.monthly_recurring_revenue) FILTER (
        WHERE s.change_type = 'contraction' AND s.start_date BETWEEN $start_date AND $end_date), 0) AS contraction_mrr,
    COUNT(DISTINCT s.customer_id) FILTER (
        WHERE s.change_type = 'contraction' AND s.start_date BETWEEN $start_date AND $end_date)
        AS contracting_customers,
    COALESCE(SUM(s.monthly_recurring_revenue) FILTER (
        WHERE s.status = 'churned' AND s.end_date BETWEEN $start_date AND $end_date), 0) AS churned_mrr,
    COUNT(*) FILTER (WHERE s.status = 'churned' AND s.end_date BETWEEN $start_date AND $end_date) AS churned_customers
FROM subscriptions AS s
JOIN customers AS c ON c.customer_id = s.customer_id
WHERE TRUE{clause}
"""
    flows = runner.records(
        sql,
        {"start_date": current.start, "end_date": current.end, **bind},
        calculation="MRR movements from subscription records in the period",
    )[0]
    opening_mrr, closing_mrr = _num(opening.value), _num(closing.value)
    new, expansion = _num(flows["new_mrr"]), _num(flows["expansion_mrr"])
    contraction, churn = _num(flows["contraction_mrr"]), _num(flows["churned_mrr"])
    expected_closing = opening_mrr + new + expansion - contraction - churn
    rows = [
        BridgeRow(component="opening_mrr", mrr=opening_mrr, customers=int(opening.components["active_customers"] or 0)),
        BridgeRow(component="new", mrr=new, customers=int(flows["new_customers"])),
        BridgeRow(component="expansion", mrr=expansion, customers=int(flows["expanding_customers"])),
        BridgeRow(component="contraction", mrr=-contraction, customers=int(flows["contracting_customers"])),
        BridgeRow(component="churn", mrr=-churn, customers=int(flows["churned_customers"])),
        BridgeRow(
            component="reactivation",
            mrr=None,
            customers=None,
            note="Not observed: the data model has no reactivations (churned customers do not return).",
        ),
        BridgeRow(component="closing_mrr", mrr=closing_mrr, customers=int(closing.components["active_customers"] or 0)),
    ]
    difference = closing_mrr - expected_closing
    return AnalyticsResult[BridgeRow](
        operation="revenue_bridge",
        status="ok",
        period=current,
        filters=active,
        data=rows,
        summary={
            "opening_mrr": opening_mrr,
            "closing_mrr": closing_mrr,
            "net_change": closing_mrr - opening_mrr,
            "net_new_mrr": new + expansion - contraction - churn,
            "reconciliation_difference": difference,
            "reconciled": abs(difference) <= RECONCILIATION_TOLERANCE_SGD,
        },
        limitations=[
            "MRR bridge (recurring revenue only); usage revenue is not part of MRR.",
            "Signs: contraction and churn are shown as negative MRR movements.",
        ],
        provenance=runner.provenance(
            "closing MRR = opening MRR + new + expansion - contraction - churn (subscription records with "
            "start_date / end_date in the period; opening and closing from the mrr KPI)"
        ),
    )


def mrr_series(
    db: Database,
    period: PeriodSpec = "trailing_12_months",
    *,
    filters: Filters | dict[str, str] | None = None,
    as_of: date | None = None,
) -> AnalyticsResult[MRRSeriesRow]:
    """Month-end MRR, ARR and active customers for each month ending in the period."""
    current = _resolve(period, KPIService(db, as_of=as_of))
    active = to_filters(filters)
    clause, bind = filter_clause(active, {**CUSTOMER_FILTER_COLUMNS, "plan": "s.plan"}, "mrr_series")
    runner = QueryRunner(db, "mrr_series")
    sql = f"""
WITH month_ends AS (
    SELECT CAST(m + INTERVAL 1 MONTH - INTERVAL 1 DAY AS DATE) AS month_end
    FROM range(CAST(date_trunc('month', CAST($start_date AS DATE)) AS DATE), CAST($end_date AS DATE) + INTERVAL 1 DAY,
               INTERVAL 1 MONTH) AS t(m)
),
coverage AS (SELECT MIN(date) - 1 AS first_state, MAX(date) AS last_state FROM daily_revenue)
SELECT
    strftime(e.month_end, '%Y-%m') AS month,
    e.month_end,
    COALESCE(SUM(s.monthly_recurring_revenue), 0) AS mrr,
    COUNT(DISTINCT s.customer_id) AS active_customers
FROM month_ends AS e
CROSS JOIN coverage
JOIN subscriptions AS s ON {in_force("s", "e.month_end")}
JOIN customers AS c ON c.customer_id = s.customer_id
WHERE e.month_end BETWEEN $start_date AND $end_date
  AND e.month_end BETWEEN coverage.first_state AND coverage.last_state{clause}
GROUP BY 1, 2
ORDER BY 2
"""
    records = runner.records(
        sql,
        {"start_date": current.start, "end_date": current.end, **bind},
        calculation="MRR at each month end (records in force at the close of the month end)",
    )
    rows = [
        MRRSeriesRow(
            month=r["month"],
            month_end=r["month_end"],
            mrr=_num(r["mrr"]),
            arr=_num(r["mrr"]) * 12,
            active_customers=int(r["active_customers"]),
        )
        for r in records
    ]
    status: ResultStatus = "ok" if rows else "no_data"
    return AnalyticsResult[MRRSeriesRow](
        operation="mrr_series",
        status=status,
        period=current,
        filters=active,
        dimensions=["month"],
        data=rows,
        summary={"months": len(rows)},
        message=None if rows else "No month end within the period is covered by the data.",
        limitations=["Only month ends inside both the period and the data coverage are reported."],
        provenance=runner.provenance("MRR per month end = SUM(monthly_recurring_revenue) of records in force"),
    )


def revenue_concentration(
    db: Database,
    period: PeriodSpec = None,
    *,
    top_n: int = 10,
    filters: Filters | dict[str, str] | None = None,
    as_of: date | None = None,
) -> AnalyticsResult[ConcentrationRow]:
    """Share of recognised revenue from the top-N customers in the period."""
    if not 1 <= top_n <= 100:
        raise InvalidRequestError("top_n must be between 1 and 100")
    current = _resolve(period, KPIService(db, as_of=as_of))
    active = to_filters(filters)
    clause, bind = filter_clause(active, {**CUSTOMER_FILTER_COLUMNS, "plan": "r.plan"}, "revenue_concentration")
    runner = QueryRunner(db, "revenue_concentration")
    sql = f"""
WITH by_customer AS (
    SELECT r.customer_id, c.company_name, SUM(r.revenue) AS revenue
    FROM daily_revenue AS r
    JOIN customers AS c ON c.customer_id = r.customer_id
    WHERE r.date BETWEEN $start_date AND $end_date{clause}
    GROUP BY 1, 2
)
SELECT customer_id, company_name, revenue, SUM(revenue) OVER () AS total_revenue, COUNT(*) OVER () AS customers
FROM by_customer
ORDER BY revenue DESC, customer_id
LIMIT $top_n
"""
    records = runner.records(
        sql,
        {"start_date": current.start, "end_date": current.end, "top_n": top_n, **bind},
        calculation="customer revenue / total revenue, ranked",
    )
    if not records:
        return AnalyticsResult[ConcentrationRow](
            operation="revenue_concentration",
            status="no_data",
            period=current,
            filters=active,
            message="No revenue observations for the requested period and filters.",
            provenance=runner.provenance("no revenue rows"),
        )
    total = _num(records[0]["total_revenue"])
    cumulative = 0.0
    rows = []
    for rank, r in enumerate(records, start=1):
        share = _num(r["revenue"]) / total
        cumulative += share
        rows.append(
            ConcentrationRow(
                rank=rank,
                customer_id=r["customer_id"],
                company_name=r["company_name"],
                revenue=_num(r["revenue"]),
                share_of_revenue=share,
                cumulative_share=cumulative,
            )
        )
    return AnalyticsResult[ConcentrationRow](
        operation="revenue_concentration",
        status="ok",
        period=current,
        filters=active,
        data=rows,
        summary={
            "total_revenue": total,
            "paying_customers": int(records[0]["customers"]),
            f"top_{top_n}_share": cumulative,
        },
        provenance=runner.provenance("share of revenue = customer revenue / total revenue in the period"),
    )


# ---------------------------------------------------------------------------------------------------


def _num(value: object) -> float:
    number = to_number(value)
    return float(number) if number is not None else 0.0


def _direction(change: float | None) -> str | None:
    if change is None:
        return None
    return "increase" if change > 0 else "decline" if change < 0 else "unchanged"


def _spec(period: PeriodSpec) -> str | None:
    return None if isinstance(period, Period) else period


def _dates(period: PeriodSpec) -> dict[str, date]:
    return {"start_date": period.start, "end_date": period.end} if isinstance(period, Period) else {}


def _resolve(period: PeriodSpec, service: KPIService) -> Period:
    return period if isinstance(period, Period) else resolve_period(period, as_of=service.as_of)
