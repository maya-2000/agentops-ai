"""Registered, parameterised SQL templates for the KPI registry.

Rules every template follows:

- Values are **bound** named parameters (``$start_date``, ``$end_date``, ``$opening_date``,
  ``$comparison_start_date``, ``$comparison_end_date``, ``$period_months`` and one
  ``$f_<dimension>`` per filter). Nothing supplied by a caller is formatted into the SQL text.
- The only substitutions are two slots filled from this module's own allow-listed column maps:
  ``{dimension}`` (a breakdown column expression, or the literal ``'All'``) and ``{filters}``
  (``AND <column> = $f_<dimension>`` clauses).
- Every template returns ``dimension_value`` plus additive *components* (sums and counts); the
  KPI value is derived from components by the KPI's documented value rule, so totals and
  breakdowns always use identical logic.

Point-in-time convention (used by every "as of a date" calculation):
a subscription record is **in force at the close of day X** when
``start_date <= X <= end_date`` (open-ended when ``end_date`` is NULL), excluding a *churned*
record whose last day of service is X (the customer has churned by the close of that day).
A superseded record whose last day is X still counts: its successor starts on X + 1. With this
rule, opening + new + expansion - contraction - churn = closing reconciles exactly for any period.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from app.analytics.errors import UnsupportedDimensionError


def in_force(alias: str, date_param: str) -> str:
    """SQL predicate: record ``alias`` is in force at the close of ``date_param``."""
    return (
        f"{alias}.start_date <= {date_param} AND ({alias}.end_date IS NULL OR {alias}.end_date > {date_param} "
        f"OR ({alias}.end_date = {date_param} AND {alias}.status = 'superseded'))"
    )


CUSTOMER_LIFETIMES_CTE = """customer_lifetimes AS (
    SELECT cu.*, ch.churn_date
    FROM customers AS cu
    LEFT JOIN (
        SELECT customer_id, end_date AS churn_date FROM subscriptions WHERE status = 'churned'
    ) AS ch ON ch.customer_id = cu.customer_id
)"""

_CUSTOMER_ATTRIBUTES = ("region", "country", "segment", "industry", "acquisition_channel", "customer_id")


def _customer_columns(alias: str = "c") -> dict[str, str]:
    return {key: f"{alias}.{key}" for key in _CUSTOMER_ATTRIBUTES}


def _time_columns(date_expression: str) -> dict[str, str]:
    return {
        "month": f"strftime(CAST({date_expression} AS DATE), '%Y-%m')",
        "quarter": f"CAST(year({date_expression}) AS VARCHAR) || '-Q' || CAST(quarter({date_expression}) AS VARCHAR)",
    }


@dataclass(frozen=True)
class SQLTemplate:
    """A reviewable SQL template with its allow-listed filter and breakdown columns."""

    name: str
    description: str
    sql: str
    filter_columns: Mapping[str, str]
    dimension_columns: Mapping[str, str]
    temporal: str  # "flow", "point_in_time" or "cohort_flow" (see KPI service coverage rules)
    source_tables: tuple[str, ...]
    parameters: tuple[str, ...] = field(default=())

    def render(self, dimension: str | None, filters: Mapping[str, str]) -> str:
        """Fill the two slots. ``dimension``/``filters`` keys must be allow-listed for this template."""
        if dimension is None:
            dimension_sql = "'All'"
        else:
            if dimension not in self.dimension_columns:
                raise UnsupportedDimensionError(f"Template {self.name!r} cannot break down by {dimension!r}")
            dimension_sql = f"CAST({self.dimension_columns[dimension]} AS VARCHAR)"
        clauses = []
        for key in filters:
            if key not in self.filter_columns:
                raise UnsupportedDimensionError(f"Template {self.name!r} cannot filter by {key!r}")
            clauses.append(f"\n  AND {self.filter_columns[key]} = $f_{key}")
        return self.sql.replace("{dimension}", dimension_sql).replace("{filters}", "".join(clauses))


def _template(
    name: str,
    description: str,
    sql: str,
    *,
    filters: dict[str, str],
    dimensions: dict[str, str],
    temporal: str,
    tables: tuple[str, ...],
    parameters: tuple[str, ...],
) -> SQLTemplate:
    return SQLTemplate(
        name=name,
        description=description,
        sql=sql.strip("\n"),
        filter_columns=MappingProxyType(filters),
        dimension_columns=MappingProxyType(dimensions),
        temporal=temporal,
        source_tables=tables,
        parameters=parameters,
    )


# ---------------------------------------------------------------------------------------------------
# Revenue (flows over daily_revenue)
# ---------------------------------------------------------------------------------------------------

_REVENUE_FILTERS = {**_customer_columns(), "plan": "r.plan", "revenue_type": "r.revenue_type"}
_REVENUE_DIMENSIONS = {k: v for k, v in _REVENUE_FILTERS.items() if k != "customer_id"} | _time_columns("r.date")

REVENUE = _template(
    "revenue",
    "Recognised revenue (subscription + usage) for days in the period.",
    """
SELECT
    {dimension} AS dimension_value,
    SUM(r.revenue) AS revenue,
    COALESCE(SUM(r.revenue) FILTER (WHERE r.revenue_type = 'subscription'), 0) AS subscription_revenue,
    COALESCE(SUM(r.revenue) FILTER (WHERE r.revenue_type = 'usage'), 0) AS usage_revenue,
    COUNT(*) AS observations
FROM daily_revenue AS r
JOIN customers AS c ON c.customer_id = r.customer_id
WHERE r.date BETWEEN $start_date AND $end_date{filters}
GROUP BY 1
""",
    filters=_REVENUE_FILTERS,
    dimensions=_REVENUE_DIMENSIONS,
    temporal="flow",
    tables=("daily_revenue", "customers"),
    parameters=("start_date", "end_date"),
)

REVENUE_COMPARISON = _template(
    "revenue_comparison",
    "Recognised revenue in the current and the comparison period, side by side.",
    """
SELECT
    {dimension} AS dimension_value,
    COALESCE(SUM(r.revenue) FILTER (WHERE r.date BETWEEN $start_date AND $end_date), 0) AS current_revenue,
    COALESCE(SUM(r.revenue) FILTER (
        WHERE r.date BETWEEN $comparison_start_date AND $comparison_end_date), 0) AS comparison_revenue,
    COUNT(*) FILTER (WHERE r.date BETWEEN $start_date AND $end_date) AS current_observations,
    COUNT(*) FILTER (WHERE r.date BETWEEN $comparison_start_date AND $comparison_end_date) AS comparison_observations
FROM daily_revenue AS r
JOIN customers AS c ON c.customer_id = r.customer_id
WHERE (r.date BETWEEN $start_date AND $end_date
       OR r.date BETWEEN $comparison_start_date AND $comparison_end_date){filters}
GROUP BY 1
""",
    filters=_REVENUE_FILTERS,
    dimensions={k: v for k, v in _REVENUE_DIMENSIONS.items() if k not in ("month", "quarter")},
    temporal="flow",
    tables=("daily_revenue", "customers"),
    parameters=("start_date", "end_date", "comparison_start_date", "comparison_end_date"),
)

# ---------------------------------------------------------------------------------------------------
# Recurring state at a point in time (MRR, ARR, ARPU, customer count)
# ---------------------------------------------------------------------------------------------------

_STATE_FILTERS = {**_customer_columns(), "plan": "s.plan"}

RECURRING_STATE = _template(
    "recurring_state",
    "Subscriptions in force at the close of the period's end date.",
    f"""
SELECT
    {{dimension}} AS dimension_value,
    SUM(s.monthly_recurring_revenue) AS mrr,
    COUNT(DISTINCT s.customer_id) AS active_customers
FROM subscriptions AS s
JOIN customers AS c ON c.customer_id = s.customer_id
WHERE {in_force("s", "$end_date")}{{filters}}
GROUP BY 1
""",
    filters=_STATE_FILTERS,
    dimensions={k: v for k, v in _STATE_FILTERS.items() if k != "customer_id"},
    temporal="point_in_time",
    tables=("subscriptions", "customers"),
    parameters=("end_date",),
)

# ---------------------------------------------------------------------------------------------------
# Opening-cohort movements (logo/revenue churn, retention, NRR)
# ---------------------------------------------------------------------------------------------------

_COHORT_FILTERS = {**_customer_columns(), "plan": "o.plan"}

OPENING_COHORT = _template(
    "opening_cohort",
    "Customers active at the period opening and what happened to them during the period.",
    f"""
WITH {CUSTOMER_LIFETIMES_CTE},
opening AS (
    SELECT s.customer_id, s.plan, s.monthly_recurring_revenue AS opening_mrr
    FROM subscriptions AS s
    WHERE {in_force("s", "$opening_date")}
),
cohort AS (
    SELECT {{dimension}} AS dimension_value, o.customer_id, o.opening_mrr
    FROM opening AS o
    JOIN customer_lifetimes AS c ON c.customer_id = o.customer_id
    WHERE TRUE{{filters}}
),
movements AS (
    SELECT
        customer_id,
        SUM(monthly_recurring_revenue - previous_mrr) FILTER (WHERE change_type = 'expansion') AS expansion_mrr,
        SUM(previous_mrr - monthly_recurring_revenue) FILTER (WHERE change_type = 'contraction') AS contraction_mrr
    FROM subscriptions
    WHERE change_type IN ('expansion', 'contraction') AND start_date BETWEEN $start_date AND $end_date
    GROUP BY customer_id
),
churned AS (
    SELECT customer_id, monthly_recurring_revenue AS churned_mrr
    FROM subscriptions
    WHERE status = 'churned' AND end_date BETWEEN $start_date AND $end_date
)
SELECT
    cohort.dimension_value,
    COUNT(*) AS opening_customers,
    COUNT(churned.customer_id) AS churned_customers,
    SUM(cohort.opening_mrr) AS opening_mrr,
    COALESCE(SUM(churned.churned_mrr), 0) AS churned_mrr,
    COALESCE(SUM(movements.expansion_mrr), 0) AS expansion_mrr,
    COALESCE(SUM(movements.contraction_mrr), 0) AS contraction_mrr
FROM cohort
LEFT JOIN churned ON churned.customer_id = cohort.customer_id
LEFT JOIN movements ON movements.customer_id = cohort.customer_id
GROUP BY 1
""",
    filters=_COHORT_FILTERS,
    dimensions={k: v for k, v in _COHORT_FILTERS.items() if k != "customer_id"},
    temporal="cohort_flow",
    tables=("subscriptions", "customers"),
    parameters=("opening_date", "start_date", "end_date"),
)

CUSTOMER_LIFETIME_VALUE = _template(
    "customer_lifetime_value",
    "Closing recurring revenue per customer and the period's logo churn among opening customers.",
    f"""
WITH {CUSTOMER_LIFETIMES_CTE},
flow AS (
    SELECT
        {{dimension}} AS dimension_value,
        COUNT(*) AS opening_customers,
        COUNT(*) FILTER (WHERE c.churn_date BETWEEN $start_date AND $end_date) AS churned_customers
    FROM subscriptions AS s
    JOIN customer_lifetimes AS c ON c.customer_id = s.customer_id
    WHERE {in_force("s", "$opening_date")}{{filters}}
    GROUP BY 1
),
closing AS (
    SELECT
        {{dimension}} AS dimension_value,
        SUM(s.monthly_recurring_revenue) AS closing_mrr,
        COUNT(DISTINCT s.customer_id) AS closing_customers
    FROM subscriptions AS s
    JOIN customer_lifetimes AS c ON c.customer_id = s.customer_id
    WHERE {in_force("s", "$end_date")}{{filters}}
    GROUP BY 1
)
SELECT
    flow.dimension_value,
    flow.opening_customers,
    flow.churned_customers,
    closing.closing_mrr,
    closing.closing_customers,
    CAST($period_months AS DOUBLE) AS period_months
FROM flow
LEFT JOIN closing ON closing.dimension_value = flow.dimension_value
""",
    filters=_customer_columns(),
    dimensions={k: v for k, v in _customer_columns().items() if k != "customer_id"},
    temporal="cohort_flow",
    tables=("subscriptions", "customers"),
    parameters=("opening_date", "start_date", "end_date", "period_months"),
)

# ---------------------------------------------------------------------------------------------------
# Marketing (weekly campaign rows; a week belongs to the period of its Monday)
# ---------------------------------------------------------------------------------------------------

MARKETING_FUNNEL = _template(
    "marketing_funnel",
    "Campaign spend and funnel volumes for campaign-weeks starting in the period.",
    """
SELECT
    {dimension} AS dimension_value,
    SUM(m.spend) AS spend,
    SUM(m.impressions) AS impressions,
    SUM(m.clicks) AS clicks,
    SUM(m.leads) AS leads,
    SUM(m.conversions) AS conversions,
    COUNT(*) AS observations
FROM marketing_campaigns AS m
WHERE m.date BETWEEN $start_date AND $end_date{filters}
GROUP BY 1
""",
    filters={"acquisition_channel": "m.channel", "campaign": "m.campaign_id"},
    dimensions={"acquisition_channel": "m.channel", "campaign": "m.campaign_id"} | _time_columns("m.date"),
    temporal="flow",
    tables=("marketing_campaigns",),
    parameters=("start_date", "end_date"),
)

# ---------------------------------------------------------------------------------------------------
# Sales
# ---------------------------------------------------------------------------------------------------

_SALES_FILTERS = {
    "segment": "o.segment",
    "region": "o.region",
    "sales_rep": "o.sales_rep",
    "opportunity_type": "o.opportunity_type",
    "customer_id": "o.customer_id",
}
_SALES_DIMENSIONS = {k: v for k, v in _SALES_FILTERS.items() if k != "customer_id"}

CLOSED_OPPORTUNITIES = _template(
    "closed_opportunities",
    "Opportunities closed (won or lost) with a close date in the period.",
    """
SELECT
    {dimension} AS dimension_value,
    COUNT(*) AS closed_opportunities,
    COUNT(*) FILTER (WHERE o.stage = 'Won') AS won_opportunities,
    COUNT(*) FILTER (WHERE o.stage = 'Lost') AS lost_opportunities,
    COALESCE(SUM(o.deal_value) FILTER (WHERE o.stage = 'Won'), 0) AS won_value,
    COALESCE(SUM(o.deal_value) FILTER (WHERE o.stage = 'Lost'), 0) AS lost_value,
    SUM(datediff('day', o.created_date, o.close_date)) AS total_cycle_days
FROM sales_opportunities AS o
WHERE o.stage IN ('Won', 'Lost') AND o.close_date BETWEEN $start_date AND $end_date{filters}
GROUP BY 1
""",
    filters=_SALES_FILTERS,
    dimensions=_SALES_DIMENSIONS | _time_columns("o.close_date"),
    temporal="flow",
    tables=("sales_opportunities",),
    parameters=("start_date", "end_date"),
)

OPEN_PIPELINE = _template(
    "open_pipeline",
    "Opportunities open at the close of the period's end date (created, not yet closed).",
    """
SELECT
    {dimension} AS dimension_value,
    COUNT(*) AS open_opportunities,
    COALESCE(SUM(o.deal_value), 0) AS pipeline_value
FROM sales_opportunities AS o
WHERE o.created_date <= $end_date AND (o.close_date IS NULL OR o.close_date > $end_date){filters}
GROUP BY 1
""",
    filters=_SALES_FILTERS,
    dimensions=_SALES_DIMENSIONS,
    temporal="point_in_time",
    tables=("sales_opportunities",),
    parameters=("end_date",),
)

# ---------------------------------------------------------------------------------------------------
# Support
# ---------------------------------------------------------------------------------------------------

_SUPPORT_FILTERS = {**_customer_columns(), "ticket_category": "t.category", "ticket_priority": "t.priority"}

SUPPORT_TICKETS = _template(
    "support_tickets",
    "Tickets created in the period, with their resolution status as of the dataset end.",
    """
SELECT
    {dimension} AS dimension_value,
    COUNT(*) AS tickets,
    COUNT(*) FILTER (WHERE t.status = 'Resolved') AS resolved_tickets,
    COUNT(*) FILTER (WHERE t.status = 'Open') AS unresolved_tickets,
    SUM(t.resolution_time) FILTER (WHERE t.status = 'Resolved') AS total_resolution_hours
FROM support_tickets AS t
JOIN customers AS c ON c.customer_id = t.customer_id
WHERE t.created_at >= $start_date AND t.created_at < $end_date + INTERVAL 1 DAY{filters}
GROUP BY 1
""",
    filters=_SUPPORT_FILTERS,
    dimensions={k: v for k, v in _SUPPORT_FILTERS.items() if k != "customer_id"} | _time_columns("t.created_at"),
    temporal="flow",
    tables=("support_tickets", "customers"),
    parameters=("start_date", "end_date"),
)

# ---------------------------------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------------------------------

PRODUCT_FEATURES = _template(
    "product_features",
    "Daily feature adoption rows (feature DAU / platform DAU) for days in the period.",
    """
SELECT
    {dimension} AS dimension_value,
    AVG(p.adoption_rate) AS average_daily_adoption_rate,
    AVG(p.active_users) AS average_daily_feature_users,
    COUNT(*) AS observed_feature_days,
    strftime(MIN(p.date), '%Y-%m-%d') AS first_observed_date
FROM product_features AS p
WHERE p.date BETWEEN $start_date AND $end_date{filters}
GROUP BY 1
""",
    filters={"product_feature": "p.feature_name"},
    dimensions={"product_feature": "p.feature_name"} | _time_columns("p.date"),
    temporal="flow",
    tables=("product_features",),
    parameters=("start_date", "end_date"),
)

TEMPLATES: dict[str, SQLTemplate] = {
    t.name: t
    for t in (
        REVENUE,
        REVENUE_COMPARISON,
        RECURRING_STATE,
        OPENING_COHORT,
        CUSTOMER_LIFETIME_VALUE,
        MARKETING_FUNNEL,
        CLOSED_OPPORTUNITIES,
        OPEN_PIPELINE,
        SUPPORT_TICKETS,
        PRODUCT_FEATURES,
    )
}
