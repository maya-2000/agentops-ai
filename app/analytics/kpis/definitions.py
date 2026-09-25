"""The KPI registry: 20 KPI definitions (strongly typed; single source of truth).

The Phase 0 plan headed this list "KPIs (19)" while enumerating 20 KPI concepts (churn is
split into logo and revenue churn). The enumerated list is authoritative, so all 20 are
implemented here. See docs/analytics.md.

The LLM (Phase 4) retrieves these definitions and calls the KPI service. It never writes a KPI
formula itself.
"""

from __future__ import annotations

from app.analytics.kpis.models import KPIDefinition, ValueRule
from app.analytics.kpis.sql import TEMPLATES

TEMPLATE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue", "subscription_revenue", "usage_revenue", "observations"),
    "revenue_comparison": ("current_revenue", "comparison_revenue", "current_observations", "comparison_observations"),
    "recurring_state": ("mrr", "active_customers"),
    "opening_cohort": (
        "opening_customers",
        "churned_customers",
        "opening_mrr",
        "churned_mrr",
        "expansion_mrr",
        "contraction_mrr",
    ),
    "customer_lifetime_value": (
        "opening_customers",
        "churned_customers",
        "closing_mrr",
        "closing_customers",
        "period_months",
    ),
    "marketing_funnel": ("spend", "impressions", "clicks", "leads", "conversions", "observations"),
    "closed_opportunities": (
        "closed_opportunities",
        "won_opportunities",
        "lost_opportunities",
        "won_value",
        "lost_value",
        "total_cycle_days",
    ),
    "open_pipeline": ("open_opportunities", "pipeline_value"),
    "support_tickets": ("tickets", "resolved_tickets", "unresolved_tickets", "total_resolution_hours"),
    "product_features": (
        "average_daily_adoption_rate",
        "average_daily_feature_users",
        "observed_feature_days",
        "first_observed_date",
    ),
}

_CUSTOMER_GRAINS = ("region", "country", "segment", "industry", "plan", "customer_id")
_MARKETING_GRAIN_REASON = (
    "Marketing spend is recorded per campaign and channel only. It cannot be attributed to customer "
    "segments, regions, countries, industries, plans or individual customers without inventing an "
    "allocation rule, so CAC is not reported at this grain."
)
_PRODUCT_GRAIN_REASON = (
    "product_features records platform-wide daily adoption per feature. It has no customer, segment or "
    "region attributes, so feature adoption cannot be broken down by them. Use product analytics' "
    "adoption breadth (distinct features used per account) for segment/region comparisons."
)
_NO_SPEND_REASON = "No marketing spend is recorded for {value}: the channel has no campaigns, so CAC is undefined."


def _kpi(
    key: str,
    name: str,
    *,
    template: str,
    rule: ValueRule,
    definition: str,
    formula: str,
    unit: str,
    time_grain: str,
    interpretation: str,
    limitations: tuple[str, ...],
    dependencies: tuple[str, ...],
    **options: object,
) -> KPIDefinition:
    tpl = TEMPLATES[template]
    components = TEMPLATE_COMPONENTS[template]
    for name_used in (rule.component, rule.numerator, rule.denominator):
        if name_used is not None and name_used not in components:
            raise ValueError(f"{key}: rule references unknown component {name_used!r}")
    return KPIDefinition(
        key=key,
        name=name,
        definition=definition,
        formula=formula,
        sql=tpl.sql,
        unit=unit,
        time_grain=time_grain,
        interpretation=interpretation,
        limitations=limitations,
        dependencies=dependencies,
        template=template,
        value_rule=rule,
        components=components,
        supported_filters=tuple(tpl.filter_columns),
        supported_dimensions=tuple(tpl.dimension_columns),
        **options,  # type: ignore[arg-type]
    )


_DEFINITIONS: tuple[KPIDefinition, ...] = (
    # ------------------------------------------------------------------ revenue & recurring revenue
    _kpi(
        "revenue",
        "Revenue",
        template="revenue",
        rule=ValueRule(kind="component", component="revenue"),
        definition="Recognised revenue (subscription plus usage) for all days in the period.",
        formula="SUM(daily_revenue.revenue) over dates in [start, end]",
        unit="SGD",
        time_grain="Flow over any period (day, month, quarter, year, custom range).",
        interpretation="Total revenue recognised in the period; compare with the prior period via revenue_growth.",
        limitations=(
            "Includes subscription and usage (API overage) revenue; filter revenue_type to isolate one stream.",
            "Subscription revenue is recognised as MRR / days-in-month per day in force, so a full calendar "
            "month recognises exactly the MRR; partial months are prorated.",
            "If the period extends beyond the data coverage, only observed days are included.",
        ),
        dependencies=("daily_revenue", "customers"),
        observation_component="observations",
    ),
    _kpi(
        "mrr",
        "Monthly Recurring Revenue",
        template="recurring_state",
        rule=ValueRule(kind="component", component="mrr"),
        definition="Contracted monthly recurring revenue of all subscriptions in force at the close of the "
        "period's end date.",
        formula="SUM(subscriptions.monthly_recurring_revenue) for records in force at the close of end_date",
        unit="SGD per month",
        time_grain="Point in time: the state at the period end (e.g. MRR for 2026-Q2 = MRR at 30 June 2026).",
        interpretation="Recurring run-rate at the end of the period; excludes one-off and usage revenue.",
        limitations=(
            "Point-in-time metric: movements inside the period are not visible (see the MRR bridge).",
            "Excludes usage (overage) revenue.",
            "Customers acquired before the data window carry a single opening record, so MRR history before "
            "2024-09-01 is not available.",
        ),
        dependencies=("subscriptions", "customers"),
        observation_component="active_customers",
    ),
    _kpi(
        "arr",
        "Annual Recurring Revenue",
        template="recurring_state",
        rule=ValueRule(kind="scaled", component="mrr", factor=12.0),
        definition="Annualised recurring run-rate at the period end.",
        formula="MRR x 12",
        unit="SGD per year",
        time_grain="Point in time (period end).",
        interpretation="Annualised equivalent of MRR; a run-rate, not contracted annual value.",
        limitations=("Inherits MRR limitations; assumes the current MRR persists for 12 months.",),
        dependencies=("mrr",),
        observation_component="active_customers",
    ),
    _kpi(
        "revenue_growth",
        "Revenue Growth",
        template="revenue_comparison",
        rule=ValueRule(kind="growth", numerator="current_revenue", denominator="comparison_revenue"),
        definition="Relative change in recognised revenue between the period and a comparison period.",
        formula="(current period revenue - comparison period revenue) / comparison period revenue",
        unit="ratio",
        time_grain="Two flows; the comparison defaults to the immediately preceding period of the same shape.",
        interpretation="Positive = growth, negative = decline (e.g. -0.01 = a 1% decline).",
        limitations=(
            "Both periods must lie fully within the data coverage.",
            "Usage revenue and partial periods scale with day counts; full-month subscription revenue does not.",
        ),
        dependencies=("revenue",),
        requires_comparison=True,
        observation_component="current_observations",
    ),
    # ------------------------------------------------------------------ churn, retention, NRR
    _kpi(
        "logo_churn_rate",
        "Churn Rate - Logo",
        template="opening_cohort",
        rule=ValueRule(kind="ratio", numerator="churned_customers", denominator="opening_customers"),
        definition="Share of customers active at the period opening whose last day of service falls in the period.",
        formula="churned opening customers / customers active at the close of (start - 1 day)",
        unit="ratio",
        time_grain="Over the requested period (not annualised: a quarterly rate is roughly 3x a monthly rate).",
        interpretation="Higher = more accounts lost. Compare like-for-like periods.",
        limitations=(
            "Customers acquired during the period are excluded from numerator and denominator.",
            "Churn date = last day of service (subscriptions.end_date of the churned record).",
            "A plan filter uses the plan in force at the period opening.",
        ),
        dependencies=("subscriptions", "customers"),
        zero_when_empty=True,
    ),
    _kpi(
        "revenue_churn_rate",
        "Churn Rate - Revenue",
        template="opening_cohort",
        rule=ValueRule(kind="ratio", numerator="churned_mrr", denominator="opening_mrr"),
        definition="Share of opening MRR lost because opening customers churned during the period.",
        formula="MRR of churned opening customers (at churn) / MRR in force at the close of (start - 1 day)",
        unit="ratio",
        time_grain="Over the requested period (not annualised).",
        interpretation="Revenue-weighted churn; large accounts churning raise it more than logo churn.",
        limitations=(
            "Excludes contraction (downgrades); NRR captures contraction.",
            "Churned MRR is the MRR at the time of churn, which may differ from opening MRR if the customer "
            "expanded or contracted before churning.",
        ),
        dependencies=("subscriptions", "customers"),
        zero_when_empty=True,
    ),
    _kpi(
        "retention_rate",
        "Retention Rate",
        template="opening_cohort",
        rule=ValueRule(kind="complement_ratio", numerator="churned_customers", denominator="opening_customers"),
        definition="Share of customers active at the period opening who had not churned by the period end.",
        formula="1 - logo churn rate (same opening base)",
        unit="ratio",
        time_grain="Over the requested period.",
        interpretation="Logo retention; complements logo churn exactly.",
        limitations=("Inherits logo churn limitations.",),
        dependencies=("logo_churn_rate",),
        zero_when_empty=True,
    ),
    _kpi(
        "nrr",
        "Net Revenue Retention",
        template="opening_cohort",
        rule=ValueRule(kind="net_retention"),
        definition="Recurring revenue kept from the opening customer base, including their expansion, "
        "contraction and churn, relative to their opening MRR.",
        formula="(opening MRR - churned MRR - contraction MRR + expansion MRR) / opening MRR",
        unit="ratio",
        time_grain="Over the requested period; annual NRR uses a 12-month period (e.g. trailing_12_months).",
        interpretation="Above 1.0 = the existing base grew net of losses; below 1.0 = net shrinkage.",
        limitations=(
            "New-business MRR and all movements of customers acquired during the period are excluded by design.",
            "Usage (overage) revenue is excluded.",
        ),
        dependencies=("subscriptions", "customers"),
        zero_when_empty=True,
    ),
    # ------------------------------------------------------------------ unit economics
    _kpi(
        "cac",
        "Customer Acquisition Cost",
        template="marketing_funnel",
        rule=ValueRule(kind="ratio", numerator="spend", denominator="conversions"),
        definition="Marketing programme spend per customer acquired by marketing campaigns.",
        formula="SUM(marketing_campaigns.spend) / SUM(marketing_campaigns.conversions) "
        "for campaign-weeks in the period",
        unit="SGD per customer",
        time_grain="Flow; weekly campaign rows belong to the period containing their Monday.",
        interpretation="Cost to acquire one marketing-sourced customer; lower is more efficient.",
        limitations=(
            "Only marketing programme spend is recorded; sales salaries, commissions and tooling are not in the "
            "dataset, so this understates fully loaded CAC.",
            "Covers marketing-sourced customers only; Outbound Sales and Referral acquisitions have no recorded spend.",
            "Conversions are counted in the week they occur, not linked to the week the lead was generated.",
        ),
        dependencies=("marketing_campaigns",),
        observation_component="observations",
        insufficient_grains={grain: _MARKETING_GRAIN_REASON for grain in _CUSTOMER_GRAINS},
        insufficient_filter_values={
            "acquisition_channel": {
                channel: _NO_SPEND_REASON.format(value=channel) for channel in ("Outbound Sales", "Referral")
            }
        },
    ),
    _kpi(
        "clv",
        "Customer Lifetime Value",
        template="customer_lifetime_value",
        rule=ValueRule(kind="lifetime_value"),
        definition="Revenue-based lifetime value: recurring revenue per customer multiplied by the expected "
        "customer lifetime implied by the period's logo churn.",
        formula="(closing MRR / closing customers) / (churned customers / opening customers / months in period)",
        unit="SGD per customer",
        time_grain="Uses the requested period for churn and its end for ARPA; use >= 12 months for stability.",
        interpretation="Approximate recurring revenue a typical customer generates over their lifetime.",
        limitations=(
            "Revenue-based: the dataset contains no gross-margin assumption, so CLV is not margin-adjusted and "
            "overstates profit-based CLV.",
            "Assumes a constant monthly churn rate (geometric lifetime = 1 / monthly churn).",
            "Ignores expansion, contraction and usage revenue; short periods make it volatile.",
        ),
        dependencies=("arpu", "logo_churn_rate"),
        zero_when_empty=True,
    ),
    _kpi(
        "arpu",
        "Average Revenue Per User (Account)",
        template="recurring_state",
        rule=ValueRule(kind="ratio", numerator="mrr", denominator="active_customers"),
        definition="Recurring revenue per active customer account at the period end (ARPA).",
        formula="MRR / active customers (both at the close of end_date)",
        unit="SGD per customer per month",
        time_grain="Point in time (period end).",
        interpretation="Average monthly recurring revenue per account; the 'user' is the customer account.",
        limitations=("Excludes usage revenue.", "Accounts, not individual seats or end users."),
        dependencies=("mrr", "customer_count"),
        observation_component="active_customers",
    ),
    # ------------------------------------------------------------------ sales
    _kpi(
        "average_order_value",
        "Average Order Value",
        template="closed_opportunities",
        rule=ValueRule(kind="ratio", numerator="won_value", denominator="won_opportunities"),
        definition="Average annual contract value of opportunities won in the period.",
        formula="SUM(deal_value of Won opportunities) / COUNT(Won opportunities), close_date in the period",
        unit="SGD (annual contract value)",
        time_grain="Flow over close dates.",
        interpretation="Typical deal size; segment mix strongly influences it.",
        limitations=(
            "Includes New Business and Expansion deals unless filtered by opportunity_type.",
            "deal_value is annual contract value (12 x MRR added), not the first invoice.",
        ),
        dependencies=("sales_opportunities",),
        zero_when_empty=True,
    ),
    _kpi(
        "conversion_rate",
        "Conversion Rate",
        template="marketing_funnel",
        rule=ValueRule(kind="ratio", numerator="conversions", denominator="leads"),
        definition="Marketing lead-to-customer conversion: customers converted per marketing-qualified lead.",
        formula="SUM(marketing_campaigns.conversions) / SUM(marketing_campaigns.leads) "
        "for campaign-weeks in the period",
        unit="ratio",
        time_grain="Flow; same-period ratio of weekly campaign rows.",
        interpretation="Share of marketing leads that became paying customers.",
        limitations=(
            "Numerator and denominator are counted in the same weeks; it is not a lead-cohort conversion.",
            "Covers the marketing funnel only; sales-opportunity conversion is reported by win_rate and sales "
            "analytics.",
        ),
        dependencies=("marketing_campaigns",),
        observation_component="observations",
        insufficient_grains={grain: _MARKETING_GRAIN_REASON for grain in _CUSTOMER_GRAINS},
    ),
    _kpi(
        "pipeline_value",
        "Pipeline Value",
        template="open_pipeline",
        rule=ValueRule(kind="component", component="pipeline_value"),
        definition="Total annual contract value of opportunities open at the close of the period end date.",
        formula="SUM(deal_value) where created_date <= end_date and (close_date IS NULL or close_date > end_date)",
        unit="SGD (annual contract value)",
        time_grain="Point in time (period end).",
        interpretation="Unweighted value of deals still in progress.",
        limitations=(
            "Unweighted: stage probabilities are only known for the current (as-of) pipeline; see sales analytics.",
            "Historical pipeline is reconstructed from created/close dates; the stage at that time is not recorded.",
        ),
        dependencies=("sales_opportunities",),
        zero_when_empty=True,
    ),
    _kpi(
        "win_rate",
        "Win Rate",
        template="closed_opportunities",
        rule=ValueRule(kind="ratio", numerator="won_opportunities", denominator="closed_opportunities"),
        definition="Share of opportunities closed in the period that were won.",
        formula="COUNT(Won) / COUNT(Won or Lost), close_date in the period",
        unit="ratio",
        time_grain="Flow over close dates.",
        interpretation="Higher = a larger share of closed deals won. Compare segments separately.",
        limitations=(
            "Every CRM record is an opportunity, including deals lost at the Lead stage.",
            "Open opportunities are excluded until they close.",
        ),
        dependencies=("sales_opportunities",),
        zero_when_empty=True,
    ),
    _kpi(
        "sales_cycle",
        "Sales Cycle",
        template="closed_opportunities",
        rule=ValueRule(kind="ratio", numerator="total_cycle_days", denominator="closed_opportunities"),
        definition="Average days from opportunity creation to close for opportunities closed in the period.",
        formula="SUM(close_date - created_date) / COUNT(closed), Won and Lost, close_date in the period",
        unit="days",
        time_grain="Flow over close dates.",
        interpretation="Average elapsed time to a decision; Enterprise deals take longer.",
        limitations=(
            "Includes won and lost deals; open opportunities are excluded.",
            "The mean is sensitive to long deals; sales analytics also reports the median for won deals.",
        ),
        dependencies=("sales_opportunities",),
        zero_when_empty=True,
    ),
    # ------------------------------------------------------------------ support
    _kpi(
        "support_ticket_volume",
        "Support Ticket Volume",
        template="support_tickets",
        rule=ValueRule(kind="component", component="tickets"),
        definition="Number of support tickets created in the period.",
        formula="COUNT(support_tickets) with created_at in the period",
        unit="tickets",
        time_grain="Flow over ticket creation dates.",
        interpretation="Support demand; normalise by active customers when comparing periods.",
        limitations=("Volume is not normalised for customer growth; see tickets per active customer.",),
        dependencies=("support_tickets", "customers"),
        zero_when_empty=True,
    ),
    _kpi(
        "average_resolution_time",
        "Average Resolution Time",
        template="support_tickets",
        rule=ValueRule(kind="ratio", numerator="total_resolution_hours", denominator="resolved_tickets"),
        definition="Mean hours from creation to resolution for tickets created in the period and resolved.",
        formula="SUM(resolution_time of Resolved tickets) / COUNT(Resolved tickets), created_at in the period",
        unit="hours",
        time_grain="Flow over ticket creation dates.",
        interpretation="Higher = slower support.",
        limitations=(
            "Unresolved tickets are excluded from the denominator (reported as unresolved_tickets); recent periods "
            "are biased towards faster tickets because slow ones may still be open.",
            "The mean is skewed by long tickets; support analytics also reports the median.",
        ),
        dependencies=("support_tickets", "customers"),
        zero_when_empty=True,
    ),
    # ------------------------------------------------------------------ product & customers
    _kpi(
        "product_adoption",
        "Product Adoption",
        template="product_features",
        rule=ValueRule(kind="component", component="average_daily_adoption_rate"),
        definition="Average daily share of platform daily active users who used the feature.",
        formula="AVG(product_features.adoption_rate) over days in the period "
        "(adoption_rate = feature daily active users / platform daily active users)",
        unit="ratio",
        time_grain="Flow over days; one value per feature.",
        interpretation="Feature penetration among active users.",
        limitations=(
            "The denominator is all platform daily active users; plan eligibility is not recorded, so no "
            "eligible-user denominator is available.",
            "Platform-level only: no segment, region or customer breakdown.",
            "A feature has no rows before its launch date.",
        ),
        dependencies=("product_features",),
        observation_component="observed_feature_days",
        required_any_of=("product_feature",),
        insufficient_grains={grain: _PRODUCT_GRAIN_REASON for grain in _CUSTOMER_GRAINS},
    ),
    _kpi(
        "customer_count",
        "Customer Count",
        template="recurring_state",
        rule=ValueRule(kind="component", component="active_customers"),
        definition="Distinct customers with a subscription in force at the close of the period end date.",
        formula="COUNT(DISTINCT customer_id) of subscription records in force at the close of end_date",
        unit="customers",
        time_grain="Point in time (period end).",
        interpretation="Active paying accounts; see customer analytics for opening, new and churned counts.",
        limitations=("Point in time: customers who joined and left within the period are not counted.",),
        dependencies=("subscriptions", "customers"),
        zero_when_empty=True,
    ),
)

KPI_REGISTRY: dict[str, KPIDefinition] = {definition.key: definition for definition in _DEFINITIONS}

KPI_KEYS: tuple[str, ...] = tuple(KPI_REGISTRY)
