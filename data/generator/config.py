"""Generator configuration.

``GeneratorConfig`` holds run-level settings (seed, scale, dates, paths). The module-level
parameter objects hold the business assumptions of the synthetic company. Changing any of
them changes the dataset; the same seed + configuration always yields the same dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

DATASET_NAME = "Northwind Cloud"
DATASET_VERSION = "1.0.0"
CURRENCY = "SGD"
COMPANY_FOUNDED = date(2021, 1, 1)


class GeneratorConfig(BaseModel):
    """Run-level configuration (CLI flags map 1:1 onto these fields)."""

    seed: int = 42
    customer_count: int = Field(default=5000, ge=50)
    start_date: date = date(2024, 9, 1)
    end_date: date = date(2026, 8, 31)
    monthly_new_customer_rate: float = Field(
        default=0.05,
        gt=0,
        description=(
            "New customers per month as a share of the opening customer base; sets the split "
            "between accounts acquired before and during the window."
        ),
    )
    db_path: Path = Path("database/northwind_cloud.duckdb")
    parquet_dir: Path | None = Path("data/seeds/parquet")
    metadata_dir: Path = Path("data/metadata")
    ground_truth_path: Path = Path("data/seeds/injected_events.json")
    strict_event_validation: bool = True

    @model_validator(mode="after")
    def _check_dates(self) -> GeneratorConfig:
        if self.start_date.day != 1:
            raise ValueError("start_date must be the first day of a month")
        next_day = date.fromordinal(self.end_date.toordinal() + 1)
        if next_day.day != 1:
            raise ValueError("end_date must be the last day of a month")
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        if self.start_date <= COMPANY_FOUNDED:
            raise ValueError(f"start_date must be after the company founding date {COMPANY_FOUNDED}")
        return self

    @property
    def n_months(self) -> int:
        return (self.end_date.year - self.start_date.year) * 12 + self.end_date.month - self.start_date.month + 1

    @property
    def opening_customer_count(self) -> int:
        share = 1.0 / (1.0 + self.monthly_new_customer_rate * self.n_months)
        return round(self.customer_count * share)

    @property
    def new_customer_count(self) -> int:
        return self.customer_count - self.opening_customer_count


# --------------------------------------------------------------------------------------
# Business assumptions
# --------------------------------------------------------------------------------------

COUNTRY_WEIGHTS: dict[str, float] = {
    "Singapore": 0.15,
    "Australia": 0.09,
    "Japan": 0.06,
    "India": 0.06,
    "Indonesia": 0.04,
    "United Kingdom": 0.09,
    "Germany": 0.07,
    "France": 0.05,
    "Netherlands": 0.04,
    "United States": 0.20,
    "Canada": 0.05,
    "Brazil": 0.06,
    "Mexico": 0.04,
}

LEGAL_SUFFIX: dict[str, str] = {
    "Singapore": "Pte Ltd",
    "Australia": "Pty Ltd",
    "Japan": "K.K.",
    "India": "Pvt Ltd",
    "Indonesia": "PT",
    "United Kingdom": "Ltd",
    "Germany": "GmbH",
    "France": "SAS",
    "Netherlands": "B.V.",
    "United States": "Inc.",
    "Canada": "Inc.",
    "Brazil": "Ltda",
    "Mexico": "S.A. de C.V.",
}

INDUSTRY_WEIGHTS: dict[str, float] = {
    "Fintech": 0.16,
    "Retail & E-commerce": 0.17,
    "Healthcare": 0.11,
    "Logistics": 0.12,
    "Manufacturing": 0.12,
    "Media & Entertainment": 0.08,
    "Education": 0.08,
    "Professional Services": 0.16,
}

# Segment mix of newly acquired customers by acquisition channel.
SEGMENT_MIX_BY_CHANNEL: dict[str, dict[str, float]] = {
    "Paid Search": {"SMB": 0.75, "Mid-Market": 0.22, "Enterprise": 0.03},
    "Paid Social": {"SMB": 0.78, "Mid-Market": 0.19, "Enterprise": 0.03},
    "Email": {"SMB": 0.65, "Mid-Market": 0.28, "Enterprise": 0.07},
    "Organic": {"SMB": 0.72, "Mid-Market": 0.23, "Enterprise": 0.05},
    "Partner": {"SMB": 0.35, "Mid-Market": 0.45, "Enterprise": 0.20},
    "Events": {"SMB": 0.25, "Mid-Market": 0.45, "Enterprise": 0.30},
    "Outbound Sales": {"SMB": 0.15, "Mid-Market": 0.50, "Enterprise": 0.35},
    "Referral": {"SMB": 0.50, "Mid-Market": 0.35, "Enterprise": 0.15},
}
# Opening (pre-window) base: survivors skew towards larger accounts.
OPENING_SEGMENT_MIX: dict[str, float] = {"SMB": 0.45, "Mid-Market": 0.37, "Enterprise": 0.18}
OPENING_CHANNEL_MIX: dict[str, float] = {
    "Paid Search": 0.20,
    "Paid Social": 0.10,
    "Email": 0.07,
    "Organic": 0.16,
    "Partner": 0.12,
    "Events": 0.08,
    "Outbound Sales": 0.17,
    "Referral": 0.10,
}
NON_MARKETING_CHANNEL_SPLIT: dict[str, float] = {"Outbound Sales": 0.6, "Referral": 0.4}

COMPANY_SIZE_BY_SEGMENT: dict[str, dict[str, float]] = {
    "SMB": {"1-50": 0.6, "51-200": 0.4},
    "Mid-Market": {"201-1000": 1.0},
    "Enterprise": {"1001-5000": 0.7, "5000+": 0.3},
}

PLAN_MIX_BY_SEGMENT: dict[str, dict[str, float]] = {
    "SMB": {"Starter": 0.50, "Growth": 0.38, "Professional": 0.12},
    "Mid-Market": {"Starter": 0.03, "Growth": 0.32, "Professional": 0.50, "Enterprise": 0.15},
    "Enterprise": {"Professional": 0.30, "Enterprise": 0.70},
}


@dataclass(frozen=True)
class PlanParams:
    price_per_seat: float  # SGD per seat per month (list price)
    min_seats: int
    api_quota_per_seat: int  # included API calls per seat per month
    api_calls_per_user_week: float  # typical API calls per weekly active user
    sessions_per_user_week: float
    features_available: int  # of the 11 core features (AI Insights handled separately)


PLAN_PARAMS: dict[str, PlanParams] = {
    "Starter": PlanParams(19.0, 3, 0, 0.0, 4.0, 5),
    "Growth": PlanParams(32.0, 5, 400, 60.0, 6.0, 8),
    "Professional": PlanParams(49.0, 10, 1500, 250.0, 8.0, 10),
    "Enterprise": PlanParams(69.0, 25, 3000, 450.0, 10.0, 11),
}
PLAN_ORDER: tuple[str, ...] = ("Starter", "Growth", "Professional", "Enterprise")
API_OVERAGE_PRICE_PER_1000 = 6.0  # SGD per 1,000 calls above quota


@dataclass(frozen=True)
class SegmentParams:
    seats_median: float
    seats_sigma: float
    seats_min: int
    seats_max: int
    discount_low: float
    discount_high: float
    churn_intercept: float  # logit of monthly churn hazard at neutral health
    expansion_intercept: float
    contraction_intercept: float
    engagement_mean: float
    ticket_rate: float  # tickets per sqrt(seat) per month at neutral engagement
    sales_cycle_median_days: float


SEGMENT_PARAMS: dict[str, SegmentParams] = {
    "SMB": SegmentParams(9, 0.55, 3, 60, 0.00, 0.05, -3.75, -4.4, -4.8, 0.00, 0.15, 25),
    "Mid-Market": SegmentParams(30, 0.45, 10, 250, 0.05, 0.12, -4.75, -3.7, -4.8, 0.15, 0.14, 55),
    "Enterprise": SegmentParams(80, 0.50, 25, 800, 0.10, 0.25, -5.35, -3.4, -4.9, 0.25, 0.13, 110),
}
NEW_CUSTOMER_SEAT_FACTOR = 0.8  # new logos land smaller, then expand


@dataclass(frozen=True)
class HealthParams:
    """Weights of the hidden customer-health mechanism (never written to the database)."""

    engagement_persistence: float = 0.80
    engagement_noise: float = 0.22
    support_effect_on_engagement: float = 0.30
    adoption_engagement_lift: float = 0.10
    utilisation_slope: float = 1.3
    utilisation_intercept: float = 0.6
    ticket_engagement_elasticity: float = 0.45
    health_engagement_weight: float = 0.80
    health_support_weight: float = 0.35
    health_plan_fit_penalty: float = 0.25
    health_adoption_weight: float = 0.10
    health_noise: float = 0.20
    churn_health_slope: float = 1.2
    expansion_health_slope: float = 1.0
    contraction_health_slope: float = 1.0


HEALTH = HealthParams()

# Calendar-month multipliers. Acquisition seasonality also scales expansion (bookings follow budgets).
ACQUISITION_SEASONALITY: dict[int, float] = {
    1: 0.85,
    2: 0.90,
    3: 1.05,
    4: 1.00,
    5: 1.00,
    6: 1.08,
    7: 0.92,
    8: 0.78,
    9: 1.00,
    10: 1.05,
    11: 1.10,
    12: 1.15,
}
USAGE_SEASONALITY: dict[str, dict[int, float]] = {
    "APAC": {2: 0.92, 12: 0.90},
    "EMEA": {7: 0.93, 8: 0.84, 12: 0.86},
    "North America": {7: 0.96, 8: 0.95, 11: 0.96, 12: 0.87},
    "LATAM": {1: 0.90, 2: 0.92, 12: 0.88},
}
ANNUAL_SPEND_GROWTH = 0.20  # marketing budget growth per year
SPEND_REFERENCE_DATE = date(2024, 9, 1)


@dataclass(frozen=True)
class ChannelParams:
    conversion_share: float  # share of marketing-sourced customers
    target_cac: float  # SGD, sets the spend level
    cpm: float  # SGD per 1,000 impressions
    ctr: float
    lead_to_customer: float
    fatigue: float = 0.25  # CTR elasticity to spend (diminishing returns)


CHANNEL_PARAMS: dict[str, ChannelParams] = {
    "Paid Search": ChannelParams(0.30, 2200, 38.0, 0.032, 0.090),
    "Paid Social": ChannelParams(0.18, 2800, 11.0, 0.0085, 0.060),
    "Email": ChannelParams(0.12, 700, 4.0, 0.024, 0.080),
    "Organic": ChannelParams(0.20, 900, 3.0, 0.020, 0.100),
    "Partner": ChannelParams(0.12, 2400, 22.0, 0.015, 0.120),
    "Events": ChannelParams(0.08, 3600, 85.0, 0.050, 0.070),
}
MARKETING_SHARE_OF_NEW = 0.70
EVENT_CAMPAIGN_WEEKS = 3

CAMPAIGN_THEMES: dict[str, tuple[str, ...]] = {
    "Paid Search": (
        "Brand + Category Keywords",
        "Competitor Conquest",
        "Analytics Software Intent",
        "BI Dashboard Keywords",
    ),
    "Paid Social": (
        "LinkedIn Lead Gen",
        "Decision-Maker Retargeting",
        "Data Leaders Carousel",
        "Thought Leadership Video",
    ),
    "Email": ("Trial Nurture", "Product Newsletter", "Win-Back Sequence", "Webinar Invitations"),
    "Organic": ("SEO Content Hub", "Benchmark Report", "Customer Stories", "Templates Library"),
    "Partner": ("Cloud Marketplace Co-Sell", "Reseller Programme", "SI Alliance", "Consulting Partner Referrals"),
    "Events": ("Data Summit", "Industry Roadshow", "Executive Roundtable", "Partner Conference"),
}


@dataclass(frozen=True)
class FunnelParams:
    """Stage-to-stage conversion probabilities (Lead->Qualified->Proposal->Negotiation->Won)."""

    new_business: tuple[float, float, float, float] = (0.55, 0.62, 0.68, 0.62)
    expansion: tuple[float, float, float, float] = (0.80, 0.80, 0.85, 0.80)
    rep_skill_sd: float = 0.03
    rep_skill_bounds: tuple[float, float] = (0.94, 1.06)
    stage_probability: dict[str, float] = field(
        default_factory=lambda: {
            "Lead": 0.10,
            "Qualified": 0.25,
            "Proposal": 0.50,
            "Negotiation": 0.75,
            "Won": 1.00,
            "Lost": 0.00,
        }
    )
    sales_assisted_channels: tuple[str, ...] = ("Outbound Sales", "Partner", "Events")
    expansion_opportunity_share: float = 0.7  # MM/Ent expansions that run through a CRM opportunity


FUNNEL = FunnelParams()

SALES_TEAM: dict[str, int] = {"APAC": 7, "EMEA": 5, "North America": 6, "LATAM": 2}


@dataclass(frozen=True)
class SupportParams:
    category_shares: dict[str, float] = field(
        default_factory=lambda: {
            "How-To": 0.26,
            "Bug": 0.12,
            "Integration": 0.12,
            "Performance": 0.08,
            "Billing": 0.10,
            "Account Access": 0.20,
            "Feature Request": 0.12,
        }
    )
    priority_shares: dict[str, tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            # Low, Medium, High, Urgent
            "SMB": (0.36, 0.44, 0.16, 0.04),
            "Mid-Market": (0.30, 0.45, 0.20, 0.05),
            "Enterprise": (0.24, 0.44, 0.24, 0.08),
        }
    )
    resolution_median_hours: dict[str, float] = field(
        default_factory=lambda: {"Low": 55.0, "Medium": 30.0, "High": 14.0, "Urgent": 5.0}
    )
    resolution_sigma: float = 0.7
    enterprise_resolution_factor: float = 0.8


SUPPORT = SupportParams()

# Product features: (feature_id, name, baseline adoption, adoption drift per year, min plan).
CORE_FEATURES: tuple[tuple[str, str, float, float], ...] = (
    ("F01", "Dashboards", 0.82, 0.01),
    ("F02", "Reports", 0.64, 0.00),
    ("F03", "Data Connectors", 0.45, 0.03),
    ("F04", "Alerts", 0.38, 0.02),
    ("F05", "Collaboration", 0.33, 0.04),
    ("F06", "Scheduled Exports", 0.27, -0.01),
    ("F07", "Custom Metrics", 0.22, 0.02),
    ("F08", "API Access", 0.18, 0.02),
    ("F09", "Mobile App", 0.15, 0.03),
    ("F10", "SSO & Permissions", 0.12, 0.01),
    ("F11", "Embedded Analytics", 0.09, 0.02),
)
AI_INSIGHTS_FEATURE_ID = "F12"
AI_INSIGHTS_FEATURE_NAME = "AI Insights"
DAU_TO_WAU_RATIO = 0.62
WEEKDAY_ACTIVITY = (1.0, 1.02, 1.03, 1.0, 0.92, 0.30, 0.24)  # Monday..Sunday
