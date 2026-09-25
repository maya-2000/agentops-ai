"""Customer analytics: movements, churn, retention, NRR and the usage/support-churn relationship.

Rates come from the registered churn KPIs (``logo_churn_rate``, ``revenue_churn_rate``,
``retention_rate``, ``nrr``). All wording is associative ("associated with", "was higher among"),
never causal.

Cohort retention lives in ``app.analytics.cohorts`` and risk scoring in ``app.analytics.risk``;
both are re-exported here for convenience.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from app.analytics.cohorts import CohortCell, cohort_retention
from app.analytics.common import (
    CUSTOMER_FILTER_COLUMNS,
    PeriodSpec,
    filter_clause,
    median,
    to_filters,
    to_period,
    wilson_interval,
)
from app.analytics.dimensions import Filters
from app.analytics.errors import InvalidRequestError, ResultStatus
from app.analytics.executor import QueryRunner
from app.analytics.kpis.service import KPIService
from app.analytics.kpis.sql import CUSTOMER_LIFETIMES_CTE
from app.analytics.models import AnalyticsResult, safe_ratio, to_number
from app.analytics.periods import months_in
from app.analytics.risk import CustomerRisk, RiskSignal, score_customer_risk
from app.database.base import Database

__all__ = [
    "ChurnKPIRow",
    "ChurnSegmentRow",
    "CohortCell",
    "CustomerMovementRow",
    "CustomerRisk",
    "MonthlyChurnRow",
    "RiskSignal",
    "UsageChurnRow",
    "churn_by_dimension",
    "churn_summary",
    "cohort_retention",
    "customer_movements",
    "monthly_churn_series",
    "score_customer_risk",
    "usage_churn_relationship",
]

CHURN_DIMENSIONS = ("segment", "region", "country", "plan", "industry", "acquisition_channel")
MIN_CUSTOMERS_FOR_COMPARISON = 30  # below this, a group's churn rate is reported but not compared


class CustomerMovementRow(BaseModel):
    dimension_value: str
    opening_customers: int
    new_customers: int
    churned_customers: int
    closing_customers: int


class ChurnKPIRow(BaseModel):
    key: str
    name: str
    status: ResultStatus
    value: float | None
    unit: str


class ChurnSegmentRow(BaseModel):
    dimension: str
    dimension_value: str
    opening_customers: int
    churned_customers: int
    logo_churn_rate: float | None
    logo_churn_ci_low: float | None
    logo_churn_ci_high: float | None
    opening_mrr: float
    churned_mrr: float
    revenue_churn_rate: float | None
    sufficient_sample: bool


class MonthlyChurnRow(BaseModel):
    month: str
    opening_customers: int
    churned_customers: int
    logo_churn_rate: float | None
    revenue_churn_rate: float | None
    net_revenue_retention: float | None


class UsageChurnRow(BaseModel):
    segment: str
    outcome: str  # "churned in period" or "retained"
    customers: int
    median_usage_ratio: float | None  # recent 4-week WAU / WAU 8-16 weeks earlier
    share_with_usage_decline: float | None  # share with ratio <= 0.8
    tickets_per_customer_recent_90d: float
    tickets_per_customer_prior_90d: float


def customer_movements(
    db: Database,
    period: PeriodSpec = None,
    *,
    dimension: str | None = None,
    filters: Filters | dict[str, str] | None = None,
    as_of: date | None = None,
) -> AnalyticsResult[CustomerMovementRow]:
    """Opening, new, churned and closing customers; closing = opening + new - churned (always)."""
    current = to_period(period, as_of)
    active = to_filters(filters)
    columns = {k: v for k, v in CUSTOMER_FILTER_COLUMNS.items() if k != "customer_id"}
    if dimension is not None and dimension not in columns:
        raise InvalidRequestError(f"customer_movements supports dimensions {', '.join(columns)}")
    clause, bind = filter_clause(active, CUSTOMER_FILTER_COLUMNS, "customer_movements")
    group = f"CAST({columns[dimension]} AS VARCHAR)" if dimension else "'All'"
    runner = QueryRunner(db, "customer_movements")
    sql = f"""
WITH {CUSTOMER_LIFETIMES_CTE}
SELECT
    {group} AS dimension_value,
    COUNT(*) FILTER (WHERE c.signup_date <= $opening_date
                       AND (c.churn_date IS NULL OR c.churn_date > $opening_date)) AS opening_customers,
    COUNT(*) FILTER (WHERE c.signup_date BETWEEN $start_date AND $end_date) AS new_customers,
    COUNT(*) FILTER (WHERE c.churn_date BETWEEN $start_date AND $end_date) AS churned_customers,
    COUNT(*) FILTER (WHERE c.signup_date <= $end_date
                       AND (c.churn_date IS NULL OR c.churn_date > $end_date)) AS closing_customers
FROM customer_lifetimes AS c
WHERE TRUE{clause}
GROUP BY 1
ORDER BY 1
"""
    records = runner.records(
        sql,
        {"opening_date": current.opening_date, "start_date": current.start, "end_date": current.end, **bind},
        calculation="customer counts from signup_date and churn_date (last day of service)",
    )
    rows = [
        CustomerMovementRow(
            dimension_value=str(r["dimension_value"]),
            opening_customers=int(r["opening_customers"]),
            new_customers=int(r["new_customers"]),
            churned_customers=int(r["churned_customers"]),
            closing_customers=int(r["closing_customers"]),
        )
        for r in records
    ]
    return AnalyticsResult[CustomerMovementRow](
        operation="customer_movements",
        status="ok" if rows else "no_data",
        period=current,
        filters=active,
        dimensions=[dimension] if dimension else [],
        data=rows,
        summary={
            "opening_customers": sum(r.opening_customers for r in rows),
            "new_customers": sum(r.new_customers for r in rows),
            "churned_customers": sum(r.churned_customers for r in rows),
            "closing_customers": sum(r.closing_customers for r in rows),
        },
        limitations=["A customer who joins and churns within the period counts as new and churned."],
        provenance=runner.provenance(
            "opening = signed up by the opening date and not churned by it; new = signup in period; churned = last "
            "day of service in period; closing = opening + new - churned"
        ),
    )


def churn_summary(
    db: Database,
    period: PeriodSpec = None,
    *,
    filters: Filters | dict[str, str] | None = None,
    as_of: date | None = None,
) -> AnalyticsResult[ChurnKPIRow]:
    """Logo churn, revenue churn, retention and NRR for the same opening customer base."""
    service = KPIService(db, as_of=as_of)
    current = to_period(period, as_of)
    runner = QueryRunner(db, "churn_summary")
    rows, components = [], {}
    for key in ("logo_churn_rate", "revenue_churn_rate", "retention_rate", "nrr"):
        kpi = service.calculate_kpi(key, start_date=current.start, end_date=current.end, filters=to_filters(filters))
        runner.absorb(kpi.provenance)
        rows.append(ChurnKPIRow(key=kpi.key, name=kpi.name, status=kpi.status, value=kpi.value, unit=kpi.unit))
        components = kpi.components
    status: ResultStatus = "ok" if all(r.status == "ok" for r in rows) else rows[0].status
    return AnalyticsResult[ChurnKPIRow](
        operation="churn_summary",
        status=status,
        period=current,
        filters=to_filters(filters),
        data=rows,
        summary={**{r.key: r.value for r in rows}, **components},
        limitations=["Rates are over the requested period and are not annualised."],
        provenance=runner.provenance("churn KPIs over one opening cohort (see each KPI's formula)"),
    )


def churn_by_dimension(
    db: Database,
    dimension: str,
    period: PeriodSpec = None,
    *,
    filters: Filters | dict[str, str] | None = None,
    min_customers: int = MIN_CUSTOMERS_FOR_COMPARISON,
    as_of: date | None = None,
) -> AnalyticsResult[ChurnSegmentRow]:
    """Logo and revenue churn per member of a dimension, with 95% Wilson intervals, highest first."""
    if dimension not in CHURN_DIMENSIONS:
        raise InvalidRequestError(f"churn_by_dimension supports {', '.join(CHURN_DIMENSIONS)}")
    current = to_period(period, as_of)
    kpi = KPIService(db, as_of=as_of).calculate_kpi(
        "logo_churn_rate",
        start_date=current.start,
        end_date=current.end,
        dimension=dimension,
        filters=to_filters(filters),
    )
    rows = []
    for r in kpi.breakdown:
        opening, churned = int(r.components["opening_customers"] or 0), int(r.components["churned_customers"] or 0)
        interval = wilson_interval(churned, opening)
        opening_mrr, churned_mrr = _num(r.components["opening_mrr"]), _num(r.components["churned_mrr"])
        rows.append(
            ChurnSegmentRow(
                dimension=dimension,
                dimension_value=r.dimension_value,
                opening_customers=opening,
                churned_customers=churned,
                logo_churn_rate=r.value,
                logo_churn_ci_low=interval[0] if interval else None,
                logo_churn_ci_high=interval[1] if interval else None,
                opening_mrr=opening_mrr,
                churned_mrr=churned_mrr,
                revenue_churn_rate=safe_ratio(churned_mrr, opening_mrr),
                sufficient_sample=opening >= min_customers,
            )
        )
    rows.sort(key=lambda r: (r.logo_churn_rate is None, -(r.logo_churn_rate or 0.0)))
    comparable = [r for r in rows if r.sufficient_sample and r.logo_churn_rate is not None]
    return AnalyticsResult[ChurnSegmentRow](
        operation="churn_by_dimension",
        status=kpi.status,
        period=current,
        filters=kpi.filters,
        dimensions=[dimension],
        data=rows,
        summary={
            "overall_logo_churn_rate": kpi.value,
            "highest_logo_churn": comparable[0].dimension_value if comparable else None,
            "lowest_logo_churn": comparable[-1].dimension_value if comparable else None,
            "min_customers_for_comparison": min_customers,
        },
        message=kpi.message,
        limitations=[
            *kpi.limitations,
            f"Members with fewer than {min_customers} opening customers are reported but not ranked.",
        ],
        provenance=kpi.provenance,
    )


def monthly_churn_series(
    db: Database,
    period: PeriodSpec = "trailing_12_months",
    *,
    filters: Filters | dict[str, str] | None = None,
    as_of: date | None = None,
) -> AnalyticsResult[MonthlyChurnRow]:
    """Logo churn, revenue churn and NRR for each calendar month in the period."""
    service = KPIService(db, as_of=as_of)
    current = to_period(period, as_of)
    runner = QueryRunner(db, "monthly_churn_series")
    rows = []
    for month in months_in(current):
        kpi = service.calculate_kpi(
            "logo_churn_rate", start_date=month.start, end_date=month.end, filters=to_filters(filters)
        )
        runner.absorb(kpi.provenance)
        c = kpi.components
        if kpi.status == "insufficient_data" and not c:
            continue  # month outside the observable window
        opening_mrr = _num(c.get("opening_mrr"))
        retained = (
            opening_mrr - _num(c.get("churned_mrr")) - _num(c.get("contraction_mrr")) + _num(c.get("expansion_mrr"))
        )
        rows.append(
            MonthlyChurnRow(
                month=month.label,
                opening_customers=int(c.get("opening_customers") or 0),
                churned_customers=int(c.get("churned_customers") or 0),
                logo_churn_rate=kpi.value,
                revenue_churn_rate=safe_ratio(c.get("churned_mrr"), c.get("opening_mrr")),  # type: ignore[arg-type]
                net_revenue_retention=safe_ratio(retained, opening_mrr),
            )
        )
    return AnalyticsResult[MonthlyChurnRow](
        operation="monthly_churn_series",
        status="ok" if rows else "no_data",
        period=current,
        filters=to_filters(filters),
        dimensions=["month"],
        data=rows,
        summary={"months": len(rows)},
        limitations=["Monthly rates use each month's own opening base (logo_churn_rate KPI per month)."],
        provenance=runner.provenance("logo_churn_rate / revenue churn / NRR evaluated per calendar month"),
    )


def usage_churn_relationship(
    db: Database,
    period: PeriodSpec = "last_quarter",
    *,
    filters: Filters | dict[str, str] | None = None,
    as_of: date | None = None,
) -> AnalyticsResult[UsageChurnRow]:
    """Compare pre-churn usage and ticket activity of churned vs retained customers, within segment.

    Population: customers active at the period opening with at least 16 weeks of tenure.
    Reference date: the churn date (last day of service) for churned customers; the period end
    for retained customers. Usage ratio = mean weekly active users in the 4 weeks before the
    reference date / mean in weeks 8-16 before it. Tickets are counted in the 90 days before the
    reference date and in the 90 days before that. Results are split by segment because pooling
    segments with different sizes and churn rates can reverse comparisons (Simpson's paradox).
    """
    current = to_period(period, as_of)
    active = to_filters(filters)
    clause, bind = filter_clause(
        active, {k: v for k, v in CUSTOMER_FILTER_COLUMNS.items() if k != "customer_id"}, "usage_churn_relationship"
    )
    runner = QueryRunner(db, "usage_churn_relationship")
    sql = f"""
WITH {CUSTOMER_LIFETIMES_CTE},
population AS (
    SELECT
        c.customer_id,
        c.segment,
        CASE WHEN c.churn_date BETWEEN $start_date AND $end_date THEN 'churned in period' ELSE 'retained' END
            AS outcome,
        CASE WHEN c.churn_date BETWEEN $start_date AND $end_date THEN c.churn_date ELSE $end_date END AS reference_date
    FROM customer_lifetimes AS c
    WHERE c.signup_date <= $opening_date - INTERVAL 112 DAY
      AND (c.churn_date IS NULL OR c.churn_date >= $start_date){clause}
),
usage AS (
    SELECT
        p.customer_id,
        AVG(u.active_users) FILTER (WHERE u.event_date > p.reference_date - INTERVAL 28 DAY
                                      AND u.event_date <= p.reference_date) AS recent_wau,
        AVG(u.active_users) FILTER (WHERE u.event_date > p.reference_date - INTERVAL 112 DAY
                                      AND u.event_date <= p.reference_date - INTERVAL 56 DAY) AS baseline_wau
    FROM population AS p
    JOIN usage_events AS u ON u.customer_id = p.customer_id
     AND u.event_date > p.reference_date - INTERVAL 112 DAY AND u.event_date <= p.reference_date
    GROUP BY 1
),
tickets AS (
    SELECT
        p.customer_id,
        COUNT(*) FILTER (WHERE t.created_at > p.reference_date - INTERVAL 90 DAY) AS recent_tickets,
        COUNT(*) FILTER (WHERE t.created_at <= p.reference_date - INTERVAL 90 DAY) AS prior_tickets
    FROM population AS p
    JOIN support_tickets AS t ON t.customer_id = p.customer_id
     AND t.created_at > p.reference_date - INTERVAL 180 DAY AND t.created_at < p.reference_date + INTERVAL 1 DAY
    GROUP BY 1
)
SELECT p.segment, p.outcome, p.customer_id, u.recent_wau, u.baseline_wau,
       COALESCE(t.recent_tickets, 0) AS recent_tickets, COALESCE(t.prior_tickets, 0) AS prior_tickets
FROM population AS p
LEFT JOIN usage AS u ON u.customer_id = p.customer_id
LEFT JOIN tickets AS t ON t.customer_id = p.customer_id
"""
    records = runner.records(
        sql,
        {"opening_date": current.opening_date, "start_date": current.start, "end_date": current.end, **bind},
        calculation="per-customer usage ratio and ticket counts relative to a reference date",
    )
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for r in records:
        groups.setdefault((str(r["segment"]), str(r["outcome"])), []).append(r)
        groups.setdefault(("All", str(r["outcome"])), []).append(r)
    rows = []
    for (segment, outcome), members in sorted(groups.items()):
        ratios = [
            float(r["recent_wau"]) / float(r["baseline_wau"])  # type: ignore[arg-type]
            for r in members
            if r["recent_wau"] is not None and r["baseline_wau"] not in (None, 0)
        ]
        rows.append(
            UsageChurnRow(
                segment=segment,
                outcome=outcome,
                customers=len(members),
                median_usage_ratio=median(ratios),
                share_with_usage_decline=(sum(x <= 0.8 for x in ratios) / len(ratios)) if ratios else None,
                tickets_per_customer_recent_90d=sum(int(r["recent_tickets"]) for r in members) / len(members),  # type: ignore[call-overload]
                tickets_per_customer_prior_90d=sum(int(r["prior_tickets"]) for r in members) / len(members),  # type: ignore[call-overload]
            )
        )
    summary: dict[str, float | int | str | None] = {}
    for outcome in ("churned in period", "retained"):
        row = next((r for r in rows if r.segment == "All" and r.outcome == outcome), None)
        key = "churned" if outcome.startswith("churned") else "retained"
        summary[f"{key}_customers"] = row.customers if row else 0
        summary[f"{key}_median_usage_ratio"] = row.median_usage_ratio if row else None
        summary[f"{key}_tickets_recent_90d"] = row.tickets_per_customer_recent_90d if row else None
    return AnalyticsResult[UsageChurnRow](
        operation="usage_churn_relationship",
        status="ok" if records else "no_data",
        period=current,
        filters=active,
        dimensions=["segment", "outcome"],
        data=rows,
        summary=summary,
        limitations=[
            "Associative comparison only: it shows what was observed before churn, not why customers churned.",
            "Compare within a segment; the 'All' rows pool segments with different sizes and churn rates.",
            "Customers with less than 16 weeks of tenure at the period opening are excluded.",
        ],
        provenance=runner.provenance(
            "usage ratio = mean WAU (4 weeks before reference) / mean WAU (weeks 8-16 before); tickets in the 90 days "
            "before the reference date vs the 90 days before that"
        ),
    )


def _num(value: object) -> float:
    number = to_number(value)
    return float(number) if number is not None else 0.0
