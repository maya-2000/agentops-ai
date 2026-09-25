"""Monthly signup-cohort retention (logo and revenue).

A cohort is the calendar month of ``customers.signup_date``. Only cohorts whose signup month
lies inside the data window are reported. Customers acquired before the window keep their
true (earlier) signup month, and their early history is not observed. So the start of the
window never creates an artificial cohort.

For cohort month C and k months since signup, the cell is measured at the close of the last
day of month C + k (the point-in-time rule used across the analytics layer):

- ``active_customers``: cohort members with a subscription in force; never exceeds the cohort size
- ``logo_retention`` = active_customers / cohort_size
- ``revenue_retention`` = cohort MRR at that month end / cohort MRR at the end of the signup month (k = 0)
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from app.analytics.common import CUSTOMER_FILTER_COLUMNS, business_as_of, filter_clause, to_filters
from app.analytics.dimensions import Filters
from app.analytics.executor import QueryRunner
from app.analytics.kpis.sql import in_force
from app.analytics.models import AnalyticsResult, safe_ratio, to_number
from app.database.base import Database


class CohortCell(BaseModel):
    cohort_month: str
    months_since_signup: int
    cohort_size: int
    active_customers: int
    logo_retention: float
    cohort_initial_mrr: float
    mrr: float
    revenue_retention: float | None


def cohort_retention(
    db: Database,
    *,
    first_cohort: str | None = None,
    last_cohort: str | None = None,
    max_months: int | None = None,
    filters: Filters | dict[str, str] | None = None,
    as_of: date | None = None,
) -> AnalyticsResult[CohortCell]:
    """Cohort x months-since-signup matrix in long format (one row per cell)."""
    active = to_filters(filters)
    columns = {k: v for k, v in CUSTOMER_FILTER_COLUMNS.items() if k != "customer_id"}
    clause, bind = filter_clause(active, columns, "cohort_retention")
    as_of_date = business_as_of(as_of)
    runner = QueryRunner(db, "cohort_retention")
    sql = f"""
WITH window_bounds AS (
    SELECT MIN(date) AS first_day, LEAST(MAX(date), CAST($as_of AS DATE)) AS last_day FROM daily_revenue
),
members AS (
    SELECT c.customer_id, strftime(c.signup_date, '%Y-%m') AS cohort_month,
           CAST(date_trunc('month', c.signup_date) AS DATE) AS cohort_start
    FROM customers AS c, window_bounds AS w
    WHERE c.signup_date >= w.first_day AND c.signup_date <= w.last_day{clause}
),
sizes AS (
    SELECT cohort_month, cohort_start, COUNT(*) AS cohort_size FROM members GROUP BY 1, 2
),
month_ends AS (
    SELECT CAST(m AS DATE) AS month_start, CAST(m + INTERVAL 1 MONTH - INTERVAL 1 DAY AS DATE) AS month_end
    FROM window_bounds AS w,
         range(CAST(date_trunc('month', w.first_day) AS DATE), w.last_day + INTERVAL 1 DAY, INTERVAL 1 MONTH) AS t(m)
),
cells AS (
    SELECT
        m.cohort_month,
        datediff('month', m.cohort_start, e.month_start) AS months_since_signup,
        COUNT(DISTINCT s.customer_id) AS active_customers,
        COALESCE(SUM(s.monthly_recurring_revenue), 0) AS mrr
    FROM members AS m
    JOIN month_ends AS e ON e.month_start >= m.cohort_start
    CROSS JOIN window_bounds AS w
    LEFT JOIN subscriptions AS s ON s.customer_id = m.customer_id AND {in_force("s", "e.month_end")}
    WHERE e.month_end <= w.last_day
    GROUP BY 1, 2
)
SELECT cells.cohort_month, cells.months_since_signup, sizes.cohort_size, cells.active_customers, cells.mrr
FROM cells JOIN sizes USING (cohort_month)
ORDER BY 1, 2
"""
    records = runner.records(
        sql, {"as_of": as_of_date, **bind}, calculation="active cohort members and their MRR at each month end"
    )
    initial = {r["cohort_month"]: _num(r["mrr"]) for r in records if r["months_since_signup"] == 0}
    rows = []
    for r in records:
        month, k = str(r["cohort_month"]), int(r["months_since_signup"])
        if (first_cohort and month < first_cohort) or (last_cohort and month > last_cohort):
            continue
        if max_months is not None and k > max_months:
            continue
        size = int(r["cohort_size"])
        rows.append(
            CohortCell(
                cohort_month=month,
                months_since_signup=k,
                cohort_size=size,
                active_customers=int(r["active_customers"]),
                logo_retention=int(r["active_customers"]) / size,
                cohort_initial_mrr=initial.get(month, 0.0),
                mrr=_num(r["mrr"]),
                revenue_retention=safe_ratio(_num(r["mrr"]), initial.get(month)),
            )
        )
    excluded = runner.records(
        "SELECT COUNT(*) AS n FROM customers WHERE signup_date < (SELECT MIN(date) FROM daily_revenue)",
        calculation="customers acquired before the data window (no cohort)",
    )[0]["n"]
    return AnalyticsResult[CohortCell](
        operation="cohort_retention",
        status="ok" if rows else "no_data",
        filters=active,
        dimensions=["cohort_month", "months_since_signup"],
        data=rows,
        summary={
            "cohorts": len({r.cohort_month for r in rows}),
            "cells": len(rows),
            "customers_in_cohorts": sum(r.cohort_size for r in rows if r.months_since_signup == 0),
            "pre_window_customers_excluded": int(excluded),
            "as_of": as_of_date.isoformat(),
        },
        limitations=[
            "Only cohorts whose signup month lies inside the data window; earlier customers are excluded because "
            "their early history is not observed.",
            "Cells are measured at month ends up to the as-of date; recent cohorts have fewer cells.",
            "Revenue retention is relative to the cohort's MRR at the end of its signup month and can exceed 1.",
        ],
        provenance=runner.provenance(
            "logo_retention = active members at month end / cohort size; revenue_retention = cohort MRR at month end "
            "/ cohort MRR at the end of the signup month"
        ),
    )


def cohort_matrix(result: AnalyticsResult[CohortCell], metric: str = "logo_retention") -> dict[str, dict[int, float]]:
    """Pivot cells into {cohort_month: {months_since_signup: value}} for tables and heatmaps."""
    matrix: dict[str, dict[int, float]] = {}
    for cell in result.data:
        value = getattr(cell, metric)
        if value is not None:
            matrix.setdefault(cell.cohort_month, {})[cell.months_since_signup] = float(value)
    return matrix


def _num(value: object) -> float:
    number = to_number(value)
    return float(number) if number is not None else 0.0
