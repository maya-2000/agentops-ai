"""Single source of truth for the agent-facing business schema.

The DDL (``schema.py``), the data dictionary (``docs/data-dictionary.md``) and, in later
phases, the agent's schema context are all generated from these definitions, so they
cannot drift apart. Only business-observable data is described here: hidden generator
variables and injected-event ground truth are deliberately absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SCHEMA_VERSION = "1.0.0"

# --------------------------------------------------------------------------------------
# Controlled vocabularies (business reference data shared by the generator and the schema)
# --------------------------------------------------------------------------------------

REGIONS: tuple[str, ...] = ("APAC", "EMEA", "North America", "LATAM")

COUNTRY_REGION: dict[str, str] = {
    "Singapore": "APAC",
    "Australia": "APAC",
    "Japan": "APAC",
    "India": "APAC",
    "Indonesia": "APAC",
    "United Kingdom": "EMEA",
    "Germany": "EMEA",
    "France": "EMEA",
    "Netherlands": "EMEA",
    "United States": "North America",
    "Canada": "North America",
    "Brazil": "LATAM",
    "Mexico": "LATAM",
}
COUNTRIES: tuple[str, ...] = tuple(COUNTRY_REGION)

SEGMENTS: tuple[str, ...] = ("SMB", "Mid-Market", "Enterprise")
COMPANY_SIZES: tuple[str, ...] = ("1-50", "51-200", "201-1000", "1001-5000", "5000+")
INDUSTRIES: tuple[str, ...] = (
    "Fintech",
    "Retail & E-commerce",
    "Healthcare",
    "Logistics",
    "Manufacturing",
    "Media & Entertainment",
    "Education",
    "Professional Services",
)
PLANS: tuple[str, ...] = ("Starter", "Growth", "Professional", "Enterprise")

MARKETING_CHANNELS: tuple[str, ...] = (
    "Paid Search",
    "Paid Social",
    "Email",
    "Organic",
    "Partner",
    "Events",
)
ACQUISITION_CHANNELS: tuple[str, ...] = (*MARKETING_CHANNELS, "Outbound Sales", "Referral")

CUSTOMER_STATUSES: tuple[str, ...] = ("active", "churned")
SUBSCRIPTION_STATUSES: tuple[str, ...] = ("active", "superseded", "churned")
SUBSCRIPTION_CHANGE_TYPES: tuple[str, ...] = ("new", "expansion", "contraction")

OPPORTUNITY_TYPES: tuple[str, ...] = ("New Business", "Expansion")
OPPORTUNITY_STAGES: tuple[str, ...] = ("Lead", "Qualified", "Proposal", "Negotiation", "Won", "Lost")
OPEN_OPPORTUNITY_STAGES: tuple[str, ...] = ("Lead", "Qualified", "Proposal", "Negotiation")
FUNNEL_STAGES: tuple[str, ...] = ("Lead", "Qualified", "Proposal", "Negotiation", "Won")

TICKET_PRIORITIES: tuple[str, ...] = ("Low", "Medium", "High", "Urgent")
TICKET_CATEGORIES: tuple[str, ...] = (
    "How-To",
    "Bug",
    "Integration",
    "Performance",
    "Billing",
    "Account Access",
    "Feature Request",
)
TICKET_STATUSES: tuple[str, ...] = ("Open", "Resolved")
SENTIMENTS: tuple[str, ...] = ("Positive", "Neutral", "Negative")

REVENUE_TYPES: tuple[str, ...] = ("subscription", "usage")


# --------------------------------------------------------------------------------------
# Specification types
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ForeignKey:
    table: str
    column: str


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    sql_type: str
    description: str
    nullable: bool = False
    allowed_values: tuple[str, ...] | None = None
    expected_values: str | None = None  # free-text range/format when not an enumeration
    foreign_key: ForeignKey | None = None
    business_meaning: str = ""
    pii: bool = False


@dataclass(frozen=True)
class TableSpec:
    name: str
    business_purpose: str
    grain: str
    columns: tuple[ColumnSpec, ...]
    primary_key: tuple[str, ...]
    indexes: tuple[tuple[str, ...], ...] = field(default_factory=tuple)

    def column(self, name: str) -> ColumnSpec:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(f"{self.name} has no column {name!r}")

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(col.name for col in self.columns)


@dataclass(frozen=True)
class ViewSpec:
    name: str
    description: str
    sql: str
    source_tables: tuple[str, ...]


MONEY = "DECIMAL(14,2)"
_CUSTOMER_FK = ForeignKey("customers", "customer_id")

# --------------------------------------------------------------------------------------
# Tables (listed in load order: parents before children)
# --------------------------------------------------------------------------------------

CUSTOMERS = TableSpec(
    name="customers",
    business_purpose=(
        "Master record for every customer account (company) that has ever held a "
        "subscription, including accounts acquired before the reporting window."
    ),
    grain="One row per customer account.",
    primary_key=("customer_id",),
    columns=(
        ColumnSpec(
            "customer_id",
            "VARCHAR",
            "Stable unique customer identifier.",
            expected_values="Format CUST-000001; assigned in signup order.",
            business_meaning="Join key used across all customer-level tables.",
        ),
        ColumnSpec(
            "company_name",
            "VARCHAR",
            "Legal name of the customer company (synthetic).",
            expected_values="Unique, synthetic company names with a country-specific legal suffix.",
            business_meaning="Human-readable account name for reporting.",
        ),
        ColumnSpec(
            "country",
            "VARCHAR",
            "Country of the customer's billing entity.",
            allowed_values=COUNTRIES,
            business_meaning="Geographic market; Singapore is the headquarters market.",
        ),
        ColumnSpec(
            "region",
            "VARCHAR",
            "Sales region derived from country.",
            allowed_values=REGIONS,
            business_meaning="Territory used for sales coverage and regional P&L.",
        ),
        ColumnSpec(
            "industry",
            "VARCHAR",
            "Customer's primary industry.",
            allowed_values=INDUSTRIES,
            business_meaning="Vertical used for segmentation and marketing targeting.",
        ),
        ColumnSpec(
            "company_size",
            "VARCHAR",
            "Employee-count band of the customer company.",
            allowed_values=COMPANY_SIZES,
            business_meaning="Determines segment: 1-200 SMB, 201-1000 Mid-Market, 1001+ Enterprise.",
        ),
        ColumnSpec(
            "segment",
            "VARCHAR",
            "Commercial segment derived from company size.",
            allowed_values=SEGMENTS,
            business_meaning="Primary segmentation for pricing, sales motion and reporting.",
        ),
        ColumnSpec(
            "acquisition_channel",
            "VARCHAR",
            "Channel that sourced the customer.",
            allowed_values=ACQUISITION_CHANNELS,
            business_meaning=(
                "Marketing channels match marketing_campaigns.channel; 'Outbound Sales' and "
                "'Referral' have no campaign spend."
            ),
        ),
        ColumnSpec(
            "signup_date",
            "DATE",
            "Date the first subscription started.",
            expected_values="2021-01-01 to the dataset end date.",
            business_meaning="Start of the customer relationship; basis for cohorts and tenure.",
        ),
        ColumnSpec(
            "status",
            "VARCHAR",
            "Account status as of the dataset end date.",
            allowed_values=CUSTOMER_STATUSES,
            business_meaning="'churned' means no active subscription at the dataset end date.",
        ),
    ),
    indexes=(("segment",), ("country",)),
)

SUBSCRIPTIONS = TableSpec(
    name="subscriptions",
    business_purpose=(
        "Versioned subscription history. A new record starts whenever a customer's contracted "
        "MRR changes (expansion or contraction); the prior record is closed as 'superseded'. "
        "Churn closes the final record with status 'churned'."
    ),
    grain="One row per customer subscription period with constant plan, seats and MRR.",
    primary_key=("subscription_id",),
    columns=(
        ColumnSpec(
            "subscription_id",
            "VARCHAR",
            "Unique subscription record identifier.",
            expected_values="Format SUB-0000001.",
        ),
        ColumnSpec(
            "customer_id",
            "VARCHAR",
            "Customer holding the subscription.",
            foreign_key=_CUSTOMER_FK,
        ),
        ColumnSpec(
            "plan",
            "VARCHAR",
            "Subscription plan (price tier).",
            allowed_values=PLANS,
            business_meaning="Starter < Growth < Professional < Enterprise in price and features.",
        ),
        ColumnSpec(
            "seats",
            "INTEGER",
            "Number of licensed user seats.",
            expected_values="Positive integer.",
            business_meaning="Main pricing driver: MRR is approximately seats x plan price per seat.",
        ),
        ColumnSpec(
            "monthly_recurring_revenue",
            MONEY,
            "Contracted MRR in SGD for this subscription period.",
            expected_values=">= 0",
            business_meaning="Recurring revenue while this record is in force.",
        ),
        ColumnSpec(
            "start_date",
            "DATE",
            "First day this subscription record is in force.",
        ),
        ColumnSpec(
            "end_date",
            "DATE",
            "Last day this subscription record is in force; NULL while active.",
            nullable=True,
            expected_values=">= start_date; NULL only when status = 'active'.",
            business_meaning="For churned records this is the last day of service.",
        ),
        ColumnSpec(
            "status",
            "VARCHAR",
            "Record status as of the dataset end date.",
            allowed_values=SUBSCRIPTION_STATUSES,
            business_meaning=(
                "'active' = in force; 'superseded' = replaced by a later record for the same "
                "customer (expansion/contraction); 'churned' = customer cancelled."
            ),
        ),
        ColumnSpec(
            "change_type",
            "VARCHAR",
            "How this record began.",
            allowed_values=SUBSCRIPTION_CHANGE_TYPES,
            business_meaning=(
                "'new' = first subscription; 'expansion' / 'contraction' = MRR increased / "
                "decreased versus previous_mrr."
            ),
        ),
        ColumnSpec(
            "previous_mrr",
            MONEY,
            "Customer MRR immediately before this record started (0 for new customers).",
            expected_values=">= 0",
            business_meaning="MRR movement at start = monthly_recurring_revenue - previous_mrr.",
        ),
        ColumnSpec(
            "current_mrr",
            MONEY,
            "MRR contributed by this record as of the dataset end date.",
            expected_values="= monthly_recurring_revenue when status = 'active', otherwise 0.",
            business_meaning="SUM(current_mrr) gives total MRR at the dataset end date.",
        ),
    ),
    indexes=(("customer_id",), ("start_date",)),
)

USAGE_EVENTS = TableSpec(
    name="usage_events",
    business_purpose="Weekly product-usage telemetry per customer account.",
    grain=(
        "One row per customer per week (weeks start on Monday). A week is recorded when the "
        "customer was subscribed for at least 4 of its 7 days."
    ),
    primary_key=("event_id",),
    columns=(
        ColumnSpec("event_id", "BIGINT", "Unique usage record identifier."),
        ColumnSpec("customer_id", "VARCHAR", "Customer account.", foreign_key=_CUSTOMER_FK),
        ColumnSpec(
            "event_date",
            "DATE",
            "Monday that starts the usage week.",
            business_meaning="Weekly time key for usage trend analysis.",
        ),
        ColumnSpec(
            "active_users",
            "INTEGER",
            "Distinct users active during the week (weekly active users).",
            expected_values="0 <= active_users <= licensed seats.",
            business_meaning="Core engagement measure; declining values signal disengagement.",
        ),
        ColumnSpec(
            "sessions",
            "INTEGER",
            "Number of user sessions during the week.",
            expected_values=">= 0",
        ),
        ColumnSpec(
            "api_calls",
            "BIGINT",
            "API calls made during the week.",
            expected_values=">= 0; always 0 on the Starter plan (no API access).",
            business_meaning="Drives usage (overage) revenue above the plan's included quota.",
        ),
        ColumnSpec(
            "feature_usage",
            "INTEGER",
            "Number of distinct product features used during the week.",
            expected_values="0 to 12 (bounded by the features available on the plan).",
            business_meaning="Breadth of adoption; broader usage indicates deeper product fit.",
        ),
    ),
    indexes=(("customer_id",), ("event_date",)),
)

SALES_OPPORTUNITIES = TableSpec(
    name="sales_opportunities",
    business_purpose=(
        "CRM opportunities for sales-assisted deals: new business (prospects and won "
        "customers) and expansion of existing accounts."
    ),
    grain="One row per opportunity.",
    primary_key=("opportunity_id",),
    columns=(
        ColumnSpec("opportunity_id", "VARCHAR", "Unique opportunity identifier.", expected_values="Format OPP-000001."),
        ColumnSpec(
            "customer_id",
            "VARCHAR",
            "Customer account; NULL for new-business prospects that have not become customers.",
            nullable=True,
            foreign_key=_CUSTOMER_FK,
            business_meaning="Populated for all won and expansion opportunities.",
        ),
        ColumnSpec(
            "sales_rep",
            "VARCHAR",
            "Sales representative who owns the opportunity (synthetic name).",
            business_meaning="Owner used for rep-level performance analysis.",
            pii=True,
        ),
        ColumnSpec(
            "opportunity_type",
            "VARCHAR",
            "Deal type.",
            allowed_values=OPPORTUNITY_TYPES,
        ),
        ColumnSpec("segment", "VARCHAR", "Segment of the prospect or customer.", allowed_values=SEGMENTS),
        ColumnSpec("region", "VARCHAR", "Sales region (rep territory).", allowed_values=REGIONS),
        ColumnSpec("created_date", "DATE", "Date the opportunity was created."),
        ColumnSpec(
            "close_date",
            "DATE",
            "Date the opportunity was won or lost; NULL while open.",
            nullable=True,
            expected_values=">= created_date; within the reporting window.",
        ),
        ColumnSpec(
            "stage",
            "VARCHAR",
            "Current stage (final stage for closed opportunities).",
            allowed_values=OPPORTUNITY_STAGES,
            business_meaning="Lead -> Qualified -> Proposal -> Negotiation -> Won; Lost can occur at any stage.",
        ),
        ColumnSpec(
            "furthest_stage",
            "VARCHAR",
            "Furthest funnel stage reached (for lost deals: the stage at which the deal was lost).",
            allowed_values=FUNNEL_STAGES,
            business_meaning="Enables stage-to-stage conversion and drop-off analysis.",
        ),
        ColumnSpec(
            "deal_value",
            MONEY,
            "Annual contract value (SGD) of the opportunity.",
            expected_values="> 0. For won deals = 12 x MRR added.",
        ),
        ColumnSpec(
            "probability",
            "DECIMAL(4,2)",
            "CRM win probability for the current stage.",
            expected_values=("Lead 0.10, Qualified 0.25, Proposal 0.50, Negotiation 0.75, Won 1.00, Lost 0.00."),
            business_meaning="Used to weight open pipeline.",
        ),
    ),
    indexes=(("customer_id",), ("sales_rep",), ("created_date",)),
)

SUPPORT_TICKETS = TableSpec(
    name="support_tickets",
    business_purpose="Customer support tickets with priority, category, resolution and sentiment.",
    grain="One row per support ticket.",
    primary_key=("ticket_id",),
    columns=(
        ColumnSpec("ticket_id", "VARCHAR", "Unique ticket identifier.", expected_values="Format TCK-0000001."),
        ColumnSpec("customer_id", "VARCHAR", "Customer that raised the ticket.", foreign_key=_CUSTOMER_FK),
        ColumnSpec("created_at", "TIMESTAMP", "When the ticket was opened."),
        ColumnSpec(
            "resolved_at",
            "TIMESTAMP",
            "When the ticket was resolved; NULL if still open at the dataset end.",
            nullable=True,
            expected_values=">= created_at",
        ),
        ColumnSpec("priority", "VARCHAR", "Ticket priority.", allowed_values=TICKET_PRIORITIES),
        ColumnSpec("category", "VARCHAR", "Ticket category.", allowed_values=TICKET_CATEGORIES),
        ColumnSpec("status", "VARCHAR", "Ticket status at the dataset end date.", allowed_values=TICKET_STATUSES),
        ColumnSpec(
            "resolution_time",
            "DOUBLE",
            "Hours from creation to resolution; NULL while open.",
            nullable=True,
            expected_values="> 0 hours",
            business_meaning="Service-level measure; long resolution times hurt customer experience.",
        ),
        ColumnSpec(
            "sentiment",
            "VARCHAR",
            "Customer sentiment classified from the ticket conversation.",
            allowed_values=SENTIMENTS,
        ),
    ),
    indexes=(("customer_id",), ("created_at",)),
)

MARKETING_CAMPAIGNS = TableSpec(
    name="marketing_campaigns",
    business_purpose="Weekly marketing campaign performance: spend and funnel metrics.",
    grain="One row per campaign per week (weeks start on Monday).",
    primary_key=("campaign_id", "date"),
    columns=(
        ColumnSpec("campaign_id", "VARCHAR", "Campaign identifier.", expected_values="Format CMP-001."),
        ColumnSpec("campaign_name", "VARCHAR", "Descriptive campaign name (channel | theme | quarter)."),
        ColumnSpec("channel", "VARCHAR", "Marketing channel.", allowed_values=MARKETING_CHANNELS),
        ColumnSpec("date", "DATE", "Monday that starts the reporting week."),
        ColumnSpec("spend", MONEY, "Media / programme spend in SGD for the week.", expected_values=">= 0"),
        ColumnSpec("impressions", "BIGINT", "Ad impressions, sends or content views.", expected_values=">= clicks"),
        ColumnSpec("clicks", "BIGINT", "Clicks or engaged visits.", expected_values=">= leads"),
        ColumnSpec("leads", "INTEGER", "Marketing-qualified leads generated.", expected_values=">= conversions"),
        ColumnSpec(
            "conversions",
            "INTEGER",
            "Leads that became paying customers during the week.",
            expected_values=">= 0",
            business_meaning=(
                "Each conversion corresponds to one customer with acquisition_channel = channel "
                "and a signup_date in the same week."
            ),
        ),
    ),
    indexes=(("channel",),),
)

DAILY_REVENUE = TableSpec(
    name="daily_revenue",
    business_purpose=(
        "Daily recognised revenue per customer. Subscription revenue is recognised daily as "
        "MRR x 12 / 365; usage revenue is API overage above the plan quota, recognised evenly "
        "across the customer's active days in the month."
    ),
    grain="One row per customer per day per revenue type.",
    primary_key=("date", "customer_id", "revenue_type"),
    columns=(
        ColumnSpec("date", "DATE", "Revenue recognition date."),
        ColumnSpec("customer_id", "VARCHAR", "Customer account.", foreign_key=_CUSTOMER_FK),
        ColumnSpec("region", "VARCHAR", "Customer region (denormalised for reporting).", allowed_values=REGIONS),
        ColumnSpec("segment", "VARCHAR", "Customer segment (denormalised).", allowed_values=SEGMENTS),
        ColumnSpec("plan", "VARCHAR", "Plan in force on this date.", allowed_values=PLANS),
        ColumnSpec("revenue_type", "VARCHAR", "Revenue stream.", allowed_values=REVENUE_TYPES),
        ColumnSpec(
            "revenue",
            MONEY,
            "Revenue recognised on this date in SGD.",
            expected_values=">= 0",
            business_meaning="SUM over a period gives recognised revenue for that period.",
        ),
    ),
    indexes=(("customer_id",), ("date",)),
)

PRODUCT_FEATURES = TableSpec(
    name="product_features",
    business_purpose="Daily feature-level adoption across the whole customer base.",
    grain="One row per feature per day, from the feature's launch date onwards.",
    primary_key=("feature_id", "date"),
    columns=(
        ColumnSpec("feature_id", "VARCHAR", "Feature identifier.", expected_values="Format F01-F12."),
        ColumnSpec("feature_name", "VARCHAR", "Feature name."),
        ColumnSpec("date", "DATE", "Calendar date."),
        ColumnSpec("active_users", "INTEGER", "Users who used the feature on this date.", expected_values=">= 0"),
        ColumnSpec(
            "adoption_rate",
            "DOUBLE",
            "Share of the platform's daily active users who used the feature.",
            expected_values="0 to 1",
            business_meaning="Feature penetration; rising values indicate successful adoption.",
        ),
    ),
    indexes=(("date",),),
)

TABLES: tuple[TableSpec, ...] = (
    CUSTOMERS,
    SUBSCRIPTIONS,
    USAGE_EVENTS,
    SALES_OPPORTUNITIES,
    SUPPORT_TICKETS,
    MARKETING_CAMPAIGNS,
    DAILY_REVENUE,
    PRODUCT_FEATURES,
)
TABLES_BY_NAME: dict[str, TableSpec] = {t.name: t for t in TABLES}


@dataclass(frozen=True)
class Relationship:
    parent: str
    child: str
    label: str
    optional_parent: bool = False  # child rows may have no parent (nullable FK)
    logical: bool = False  # business relationship without a declared FK constraint
    join: str = ""


RELATIONSHIPS: tuple[Relationship, ...] = (
    Relationship("customers", "subscriptions", "has", join="customer_id"),
    Relationship("customers", "usage_events", "generates", join="customer_id"),
    Relationship("customers", "sales_opportunities", "is sold through", optional_parent=True, join="customer_id"),
    Relationship("customers", "support_tickets", "submits", join="customer_id"),
    Relationship("customers", "daily_revenue", "generates", join="customer_id"),
    Relationship(
        "marketing_campaigns",
        "customers",
        "acquires",
        optional_parent=True,
        logical=True,
        join="marketing_campaigns.channel = customers.acquisition_channel (same week as signup_date)",
    ),
)

# --------------------------------------------------------------------------------------
# Analytical views (descriptive only; KPI formulas are defined in the Phase 2 KPI framework)
# --------------------------------------------------------------------------------------

VIEWS: tuple[ViewSpec, ...] = (
    ViewSpec(
        name="v_monthly_revenue",
        description="Recognised revenue by calendar month, region, segment, plan and revenue type.",
        source_tables=("daily_revenue",),
        sql="""
SELECT
    CAST(date_trunc('month', date) AS DATE) AS month,
    region,
    segment,
    plan,
    revenue_type,
    SUM(revenue) AS revenue,
    COUNT(DISTINCT customer_id) AS paying_customers
FROM daily_revenue
GROUP BY ALL
""",
    ),
    ViewSpec(
        name="v_monthly_mrr",
        description=(
            "Month-end MRR snapshot per customer: subscriptions in force on the last day of "
            "each month in the reporting window."
        ),
        source_tables=("subscriptions", "customers", "daily_revenue"),
        sql="""
WITH bounds AS (
    SELECT CAST(date_trunc('month', MIN(date)) AS DATE) AS first_month, MAX(date) AS last_date
    FROM daily_revenue
),
months AS (
    SELECT
        CAST(m AS DATE) AS month,
        CAST(m + INTERVAL 1 MONTH - INTERVAL 1 DAY AS DATE) AS month_end
    FROM bounds, range(bounds.first_month, bounds.last_date + INTERVAL 1 DAY, INTERVAL 1 MONTH) AS t(m)
)
SELECT
    months.month,
    months.month_end,
    s.customer_id,
    c.region,
    c.country,
    c.segment,
    s.plan,
    s.seats,
    s.monthly_recurring_revenue AS mrr
FROM months
JOIN subscriptions AS s
    ON s.start_date <= months.month_end
   AND (s.end_date IS NULL OR s.end_date >= months.month_end)
JOIN customers AS c ON c.customer_id = s.customer_id
""",
    ),
    ViewSpec(
        name="v_subscription_events",
        description=(
            "Subscription movements derived from the versioned subscriptions table: new, "
            "expansion, contraction (at record start) and churn (at the last day of service)."
        ),
        source_tables=("subscriptions",),
        sql="""
SELECT
    subscription_id,
    customer_id,
    start_date AS event_date,
    change_type AS event_type,
    plan,
    previous_mrr,
    monthly_recurring_revenue AS new_mrr,
    monthly_recurring_revenue - previous_mrr AS mrr_change
FROM subscriptions
UNION ALL
SELECT
    subscription_id,
    customer_id,
    end_date AS event_date,
    'churn' AS event_type,
    plan,
    monthly_recurring_revenue AS previous_mrr,
    CAST(0 AS DECIMAL(14,2)) AS new_mrr,
    -monthly_recurring_revenue AS mrr_change
FROM subscriptions
WHERE status = 'churned'
""",
    ),
    ViewSpec(
        name="v_campaign_summary",
        description="Campaign-level totals of spend and funnel volumes across all active weeks.",
        source_tables=("marketing_campaigns",),
        sql="""
SELECT
    campaign_id,
    campaign_name,
    channel,
    MIN(date) AS first_week,
    MAX(date) AS last_week,
    COUNT(*) AS active_weeks,
    SUM(spend) AS spend,
    SUM(impressions) AS impressions,
    SUM(clicks) AS clicks,
    SUM(leads) AS leads,
    SUM(conversions) AS conversions
FROM marketing_campaigns
GROUP BY campaign_id, campaign_name, channel
""",
    ),
)
VIEWS_BY_NAME: dict[str, ViewSpec] = {v.name: v for v in VIEWS}


def table_names() -> tuple[str, ...]:
    return tuple(t.name for t in TABLES)


def view_names() -> tuple[str, ...]:
    return tuple(v.name for v in VIEWS)
