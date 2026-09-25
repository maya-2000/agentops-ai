"""Support analytics: volume, tickets per active customer, resolution times and period change.

Counts and mean resolution times come from the registered ``support_ticket_volume`` and
``average_resolution_time`` KPIs, which share one template. Medians are computed here with the
same allow-listed columns. Unresolved tickets are always reported explicitly and never averaged
as if resolved.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel

from app.analytics.common import PeriodSpec, filter_clause, pct_change, to_filters, to_period
from app.analytics.dimensions import Filters
from app.analytics.errors import InvalidRequestError
from app.analytics.executor import QueryRunner
from app.analytics.kpis.service import KPIService
from app.analytics.kpis.sql import CUSTOMER_LIFETIMES_CTE, SUPPORT_TICKETS
from app.analytics.models import AnalyticsResult, safe_ratio, to_number
from app.analytics.periods import Period, previous_period
from app.database.base import Database

SUPPORT_DIMENSIONS = tuple(SUPPORT_TICKETS.dimension_columns)


class SupportGroupRow(BaseModel):
    dimension: str
    dimension_value: str
    tickets: int
    resolved_tickets: int
    unresolved_tickets: int
    average_resolution_hours: float | None
    median_resolution_hours: float | None
    share_of_tickets: float | None


class SupportChangeRow(BaseModel):
    metric: str
    current: float | None
    comparison: float | None
    change: float | None


def support_summary(
    db: Database,
    period: PeriodSpec = None,
    *,
    filters: Filters | dict[str, str] | None = None,
    as_of: date | None = None,
) -> AnalyticsResult[SupportGroupRow]:
    """Ticket volume, resolved vs unresolved, mean and median resolution, and tickets per active customer."""
    current = to_period(period, as_of)
    active = to_filters(filters)
    kpi = KPIService(db, as_of=as_of).calculate_kpi(
        "support_ticket_volume", start_date=current.start, end_date=current.end, filters=active
    )
    runner = QueryRunner(db, "support_summary")
    runner.absorb(kpi.provenance)
    median = _medians(runner, current, active, None).get("All")
    customers = _active_customers(runner, current, active)
    c = kpi.components
    tickets, resolved = int(c.get("tickets") or 0), int(c.get("resolved_tickets") or 0)
    summary: dict[str, Any] = {
        "tickets": tickets,
        "resolved_tickets": resolved,
        "unresolved_tickets": int(c.get("unresolved_tickets") or 0),
        "average_resolution_hours": safe_ratio(to_number(c.get("total_resolution_hours")), resolved),
        "median_resolution_hours": median,
        "active_customers_in_period": customers,
        "tickets_per_active_customer": safe_ratio(tickets, customers),
    }
    return AnalyticsResult[SupportGroupRow](
        operation="support_summary",
        status=kpi.status,
        period=current,
        filters=active,
        summary=summary,
        message=kpi.message,
        limitations=[
            *kpi.limitations,
            "Tickets per active customer = tickets / customers with a subscription in force on any day of the period.",
            "Resolution times cover resolved tickets only; unresolved tickets are counted separately.",
        ],
        provenance=runner.provenance(
            "support_ticket_volume KPI; median resolution over resolved tickets; tickets / active customers"
        ),
    )


def support_by_dimension(
    db: Database,
    dimension: str,
    period: PeriodSpec = None,
    *,
    filters: Filters | dict[str, str] | None = None,
    as_of: date | None = None,
) -> AnalyticsResult[SupportGroupRow]:
    """Ticket volume and resolution time (mean and median) per category, priority, segment, region or month."""
    if dimension not in SUPPORT_DIMENSIONS:
        raise InvalidRequestError(f"support_by_dimension supports {', '.join(SUPPORT_DIMENSIONS)}")
    current = to_period(period, as_of)
    active = to_filters(filters)
    kpi = KPIService(db, as_of=as_of).calculate_kpi(
        "support_ticket_volume", start_date=current.start, end_date=current.end, dimension=dimension, filters=active
    )
    runner = QueryRunner(db, "support_by_dimension")
    runner.absorb(kpi.provenance)
    medians = _medians(runner, current, active, dimension)
    total = int(kpi.components.get("tickets") or 0)
    rows = [
        _group_row(dimension, r.dimension_value, r.components, medians.get(r.dimension_value), total)
        for r in kpi.breakdown
    ]
    if dimension not in ("month", "quarter"):
        rows.sort(key=lambda r: -r.tickets)
    return AnalyticsResult[SupportGroupRow](
        operation="support_by_dimension",
        status=kpi.status,
        period=current,
        filters=active,
        dimensions=[dimension],
        data=rows,
        summary={"tickets": total},
        message=kpi.message,
        limitations=kpi.limitations,
        provenance=runner.provenance("ticket counts, mean and median resolution hours per group"),
    )


def resolution_time_trend(
    db: Database,
    period: PeriodSpec = "trailing_12_months",
    *,
    filters: Filters | dict[str, str] | None = None,
    as_of: date | None = None,
) -> AnalyticsResult[SupportGroupRow]:
    """Monthly ticket volume and resolution time."""
    result = support_by_dimension(db, "month", period, filters=filters, as_of=as_of)
    return result.model_copy(update={"operation": "resolution_time_trend"})


def support_volume_change(
    db: Database,
    period: PeriodSpec = None,
    comparison: PeriodSpec = None,
    *,
    filters: Filters | dict[str, str] | None = None,
    as_of: date | None = None,
) -> AnalyticsResult[SupportChangeRow]:
    """How much ticket volume, tickets per active customer and resolution time changed between periods."""
    current = to_period(period, as_of)
    cmp_period = to_period(comparison, as_of) if comparison is not None else previous_period(current)
    now = support_summary(db, current, filters=filters, as_of=as_of)
    before = support_summary(db, cmp_period, filters=filters, as_of=as_of)
    runner = QueryRunner(db, "support_volume_change")
    runner.absorb(now.provenance)
    runner.absorb(before.provenance)
    rows = []
    for metric in (
        "tickets",
        "tickets_per_active_customer",
        "average_resolution_hours",
        "median_resolution_hours",
        "unresolved_tickets",
    ):
        a, b = _float(now.summary.get(metric)), _float(before.summary.get(metric))
        rows.append(SupportChangeRow(metric=metric, current=a, comparison=b, change=pct_change(a, b)))
    return AnalyticsResult[SupportChangeRow](
        operation="support_volume_change",
        status="ok" if now.status == "ok" and before.status == "ok" else "insufficient_data",
        period=current,
        comparison_period=cmp_period,
        filters=to_filters(filters),
        data=rows,
        summary={"ticket_volume_change": rows[0].change, "tickets_per_customer_change": rows[1].change},
        limitations=[
            "Change = current / comparison - 1. Normalise by active customers before attributing changes to demand.",
            "Recent periods have more unresolved tickets, which biases their resolution times downwards.",
        ],
        provenance=runner.provenance("support_summary for both periods; change = current / comparison - 1"),
    )


# ---------------------------------------------------------------------------------------------------


def _medians(runner: QueryRunner, period: Period, filters: dict[str, str], dimension: str | None) -> dict[str, float]:
    columns = dict(SUPPORT_TICKETS.filter_columns)
    clause, bind = filter_clause(filters, columns, "support medians")
    group = f"CAST({SUPPORT_TICKETS.dimension_columns[dimension]} AS VARCHAR)" if dimension else "'All'"
    records = runner.records(
        f"""
SELECT {group} AS dimension_value, MEDIAN(t.resolution_time) AS median_hours
FROM support_tickets AS t
JOIN customers AS c ON c.customer_id = t.customer_id
WHERE t.status = 'Resolved' AND t.created_at >= $start_date AND t.created_at < $end_date + INTERVAL 1 DAY{clause}
GROUP BY 1
""",
        {"start_date": period.start, "end_date": period.end, **bind},
        calculation="median resolution hours of resolved tickets created in the period",
    )
    return {str(r["dimension_value"]): float(r["median_hours"]) for r in records if r["median_hours"] is not None}


def _active_customers(runner: QueryRunner, period: Period, filters: dict[str, str]) -> int:
    columns = {k: v for k, v in SUPPORT_TICKETS.filter_columns.items() if v.startswith("c.")}
    clause, bind = filter_clause({k: v for k, v in filters.items() if k in columns}, columns, "active customers")
    row = runner.records(
        f"""
WITH {CUSTOMER_LIFETIMES_CTE}
SELECT COUNT(*) AS n FROM customer_lifetimes AS c
WHERE c.signup_date <= $end_date AND (c.churn_date IS NULL OR c.churn_date >= $start_date){clause}
""",
        {"start_date": period.start, "end_date": period.end, **bind},
        calculation="customers with a subscription in force on any day of the period",
    )[0]
    return int(row["n"])


def _group_row(dimension: str, value: str, c: dict[str, Any], median: float | None, total: int) -> SupportGroupRow:
    tickets, resolved = int(c.get("tickets") or 0), int(c.get("resolved_tickets") or 0)
    return SupportGroupRow(
        dimension=dimension,
        dimension_value=value,
        tickets=tickets,
        resolved_tickets=resolved,
        unresolved_tickets=int(c.get("unresolved_tickets") or 0),
        average_resolution_hours=safe_ratio(to_number(c.get("total_resolution_hours")), resolved),
        median_resolution_hours=median,
        share_of_tickets=safe_ratio(tickets, total),
    )


def _float(value: object) -> float | None:
    number = to_number(value)
    return None if number is None else float(number)


__all__ = [
    "SupportChangeRow",
    "SupportGroupRow",
    "resolution_time_trend",
    "support_by_dimension",
    "support_summary",
    "support_volume_change",
]
