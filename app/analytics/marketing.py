"""Marketing analytics: spend, funnel, CAC, CPL, conversion and ROAS by channel and campaign.

Spend and funnel figures come from the registered ``cac`` / ``conversion_rate`` KPIs, which share
one template. Campaign comparisons are descriptive cross-sectional ratios (CAC relative to the
channel and overall medians). They are not statistical anomaly detection, which is out of
Phase 2's scope.

ROAS attribution: customers carry an ``acquisition_channel`` but no campaign identifier, so
revenue can be attributed to a *channel* but not to an individual campaign. Channel ROAS =
revenue recognised in each customer's first ``window_days`` days / the channel's spend in the
same period, for customers who signed up in the period via that channel.
"""

from __future__ import annotations

from datetime import date, timedelta

from pydantic import BaseModel

from app.analytics.common import PeriodSpec, median, pct_change, to_period
from app.analytics.errors import InvalidRequestError
from app.analytics.executor import QueryRunner
from app.analytics.kpis.models import KPIBreakdownRow, KPIResult
from app.analytics.kpis.service import KPIService
from app.analytics.models import AnalyticsResult, safe_ratio, to_number
from app.analytics.periods import previous_period
from app.database.base import Database
from app.database.metadata import MARKETING_CHANNELS

MIN_CONVERSIONS_FOR_COMPARISON = 10
HIGH_CAC_MULTIPLE = 2.0  # descriptive flag: CAC at least 2x the channel median
# Code constant from the schema vocabulary (not user input).
_MARKETING_CHANNEL_LIST = ", ".join(f"'{channel}'" for channel in MARKETING_CHANNELS)


class ChannelRow(BaseModel):
    channel: str
    spend: float
    impressions: int
    clicks: int
    leads: int
    conversions: int
    click_through_rate: float | None
    cost_per_lead: float | None
    conversion_rate: float | None
    cac: float | None


class CampaignRow(ChannelRow):
    campaign_id: str
    campaign_name: str
    first_week: date
    last_week: date
    channel_median_cac: float | None
    cac_vs_channel_median: float | None
    overall_median_cac: float | None
    cac_vs_overall_median: float | None
    sufficient_sample: bool
    high_cac_relative_to_channel: bool
    observation: str


class ChannelChangeRow(BaseModel):
    channel: str
    current_spend: float
    comparison_spend: float
    spend_change: float | None
    current_conversions: int
    comparison_conversions: int
    current_cac: float | None
    comparison_cac: float | None
    cac_change: float | None


class ROASRow(BaseModel):
    channel: str
    spend: float
    customers_acquired: int
    attributed_revenue: float
    roas: float | None


def channel_performance(
    db: Database, period: PeriodSpec = None, *, as_of: date | None = None
) -> AnalyticsResult[ChannelRow]:
    """Spend, funnel volumes, CPL, conversion rate and CAC per marketing channel."""
    kpi = _funnel(db, period, "acquisition_channel", as_of)
    rows = sorted((_channel_row(r.dimension_value, r) for r in kpi.breakdown), key=lambda r: -r.spend)
    total = _channel_row(
        "All", KPIBreakdownRow(dimension_value="All", status=kpi.status, value=kpi.value, components=kpi.components)
    )
    return AnalyticsResult[ChannelRow](
        operation="channel_performance",
        status=kpi.status,
        period=kpi.period,
        dimensions=["acquisition_channel"],
        data=rows,
        summary=total.model_dump(exclude={"channel"}) if kpi.components else {},
        message=kpi.message,
        limitations=kpi.limitations,
        provenance=kpi.provenance,
    )


def campaign_performance(
    db: Database,
    period: PeriodSpec = "trailing_12_months",
    *,
    channel: str | None = None,
    min_conversions: int = MIN_CONVERSIONS_FOR_COMPARISON,
    as_of: date | None = None,
) -> AnalyticsResult[CampaignRow]:
    """Per-campaign metrics with descriptive CAC comparisons against channel and overall medians."""
    current = to_period(period, as_of)
    service = KPIService(db, as_of=as_of)
    params: dict[str, object] = {"start_date": current.start, "end_date": current.end, "dimension": "campaign"}
    if channel is not None:
        params["acquisition_channel"] = channel
    kpi = service.calculate_kpi("cac", params)
    runner = QueryRunner(db, "campaign_performance")
    runner.absorb(kpi.provenance)
    meta = {
        r["campaign_id"]: r
        for r in runner.records(
            """
SELECT campaign_id, MIN(campaign_name) AS campaign_name, MIN(channel) AS channel,
       MIN(date) AS first_week, MAX(date) AS last_week
FROM marketing_campaigns
WHERE date BETWEEN $start_date AND $end_date
GROUP BY 1
""",
            {"start_date": current.start, "end_date": current.end},
            calculation="campaign names, channels and active weeks in the period",
        )
    }
    base = [(r, _channel_row(str(meta[r.dimension_value]["channel"]), r)) for r in kpi.breakdown]
    comparable = [(r, row) for r, row in base if row.conversions >= min_conversions and row.cac is not None]
    overall_median = median([row.cac for _, row in comparable if row.cac is not None])
    channel_medians = {
        ch: median([row.cac for _, row in comparable if row.channel == ch and row.cac is not None])
        for ch in {row.channel for _, row in base}
    }
    rows = []
    for r, row in base:
        m = meta[r.dimension_value]
        sufficient = row.conversions >= min_conversions and row.cac is not None
        ch_median = channel_medians.get(row.channel)
        vs_channel = safe_ratio(row.cac, ch_median) if sufficient else None
        vs_overall = safe_ratio(row.cac, overall_median) if sufficient else None
        high = bool(vs_channel is not None and vs_channel >= HIGH_CAC_MULTIPLE)
        if not sufficient:
            observation = f"Fewer than {min_conversions} conversions; CAC not compared."
        elif high:
            observation = f"CAC is {vs_channel:.1f}x the {row.channel} campaign median (descriptive comparison)."
        else:
            observation = f"CAC is {vs_channel:.1f}x the {row.channel} campaign median." if vs_channel else "Compared."
        rows.append(
            CampaignRow(
                **row.model_dump(),
                campaign_id=r.dimension_value,
                campaign_name=str(m["campaign_name"]),
                first_week=m["first_week"],
                last_week=m["last_week"],
                channel_median_cac=ch_median,
                cac_vs_channel_median=vs_channel,
                overall_median_cac=overall_median,
                cac_vs_overall_median=vs_overall,
                sufficient_sample=sufficient,
                high_cac_relative_to_channel=high,
                observation=observation,
            )
        )
    rows.sort(key=lambda r: (not r.sufficient_sample, -(r.cac or 0.0)))
    flagged = [r.campaign_id for r in rows if r.high_cac_relative_to_channel]
    return AnalyticsResult[CampaignRow](
        operation="campaign_performance",
        status=kpi.status,
        period=current,
        filters=kpi.filters,
        dimensions=["campaign"],
        data=rows,
        summary={
            "campaigns": len(rows),
            "campaigns_compared": len(comparable),
            "overall_median_cac": overall_median,
            "highest_cac_campaign": rows[0].campaign_id if rows and rows[0].sufficient_sample else None,
            "campaigns_with_cac_at_least_2x_channel_median": ", ".join(flagged) or None,
        },
        message=kpi.message,
        limitations=[
            *kpi.limitations,
            "Descriptive comparison only (ratio to medians); not a statistical anomaly test.",
            f"Campaigns with fewer than {min_conversions} conversions are listed but not compared.",
            "Campaigns spanning the period boundary contribute only their weeks inside the period.",
        ],
        provenance=runner.provenance(
            "CAC per campaign = spend / conversions; compared with channel and overall medians"
        ),
    )


def marketing_period_change(
    db: Database, period: PeriodSpec = None, comparison: PeriodSpec = None, *, as_of: date | None = None
) -> AnalyticsResult[ChannelChangeRow]:
    """Channel spend, conversions and CAC in the period vs a comparison period (default: previous)."""
    current = to_period(period, as_of)
    cmp_period = to_period(comparison, as_of) if comparison is not None else previous_period(current)
    now, before = (
        _funnel(db, current, "acquisition_channel", as_of),
        _funnel(db, cmp_period, "acquisition_channel", as_of),
    )
    runner = QueryRunner(db, "marketing_period_change")
    runner.absorb(now.provenance)
    runner.absorb(before.provenance)
    cur = {r.dimension_value: _channel_row(r.dimension_value, r) for r in now.breakdown}
    prev = {r.dimension_value: _channel_row(r.dimension_value, r) for r in before.breakdown}
    rows = []
    for ch in sorted(set(cur) | set(prev)):
        a, b = cur.get(ch), prev.get(ch)
        rows.append(
            ChannelChangeRow(
                channel=ch,
                current_spend=a.spend if a else 0.0,
                comparison_spend=b.spend if b else 0.0,
                spend_change=pct_change(a.spend if a else 0.0, b.spend if b else None),
                current_conversions=a.conversions if a else 0,
                comparison_conversions=b.conversions if b else 0,
                current_cac=a.cac if a else None,
                comparison_cac=b.cac if b else None,
                cac_change=pct_change(a.cac if a else None, b.cac if b else None),
            )
        )
    return AnalyticsResult[ChannelChangeRow](
        operation="marketing_period_change",
        status="ok" if rows else "no_data",
        period=current,
        comparison_period=cmp_period,
        dimensions=["acquisition_channel"],
        data=rows,
        limitations=[*now.limitations],
        provenance=runner.provenance("channel metrics in both periods; change = current / comparison - 1"),
    )


def channel_roas(
    db: Database, period: PeriodSpec = "last_quarter", *, window_days: int = 90, as_of: date | None = None
) -> AnalyticsResult[ROASRow]:
    """Channel ROAS: first-``window_days`` revenue of customers acquired in the period / channel spend.

    The whole revenue window must be observed. If any signup in the period is fewer than
    ``window_days`` days before the end of the data, the result is ``insufficient_data`` rather
    than a ROAS understated by truncated windows.
    """
    if not 7 <= window_days <= 365:
        raise InvalidRequestError("window_days must be between 7 and 365")
    current = to_period(period, as_of)
    service = KPIService(db, as_of=as_of)
    _, last_day = service.coverage()
    spend = service.calculate_kpi(
        "cac", start_date=current.start, end_date=current.end, dimension="acquisition_channel"
    )
    runner = QueryRunner(db, "channel_roas")
    runner.absorb(spend.provenance)
    latest_complete = last_day - timedelta(days=window_days - 1)
    if current.end > latest_complete:
        return AnalyticsResult[ROASRow](
            operation="channel_roas",
            status="insufficient_data",
            period=current,
            message=(
                f"A {window_days}-day revenue window is not yet observable for customers acquired after "
                f"{latest_complete.isoformat()}; choose a period ending on or before that date."
            ),
            provenance=runner.provenance("ROAS not computed: incomplete attribution window"),
        )
    records = runner.records(
        f"""
WITH acquired AS (
    SELECT customer_id, acquisition_channel, signup_date
    FROM customers
    WHERE signup_date BETWEEN $start_date AND $end_date AND acquisition_channel IN ({_MARKETING_CHANNEL_LIST})
)
SELECT a.acquisition_channel AS channel, COUNT(DISTINCT a.customer_id) AS customers,
       COALESCE(SUM(r.revenue), 0) AS revenue
FROM acquired AS a
LEFT JOIN daily_revenue AS r
  ON r.customer_id = a.customer_id
 AND r.date BETWEEN a.signup_date AND a.signup_date + CAST($window_days - 1 AS INTEGER)
GROUP BY 1
""",
        {"start_date": current.start, "end_date": current.end, "window_days": window_days},
        calculation=f"revenue in each acquired customer's first {window_days} days, by acquisition channel",
    )
    revenue = {r["channel"]: r for r in records}
    rows = []
    for r in spend.breakdown:
        ch_spend = float(to_number(r.components["spend"]) or 0.0)
        rev = revenue.get(r.dimension_value, {})
        attributed = float(to_number(rev.get("revenue")) or 0.0)
        rows.append(
            ROASRow(
                channel=r.dimension_value,
                spend=ch_spend,
                customers_acquired=int(rev.get("customers", 0)),
                attributed_revenue=attributed,
                roas=safe_ratio(attributed, ch_spend),
            )
        )
    rows.sort(key=lambda row: -(row.roas or 0.0))
    return AnalyticsResult[ROASRow](
        operation="channel_roas",
        status="ok" if rows else "no_data",
        period=current,
        dimensions=["acquisition_channel"],
        data=rows,
        summary={"window_days": window_days, "channels": ", ".join(MARKETING_CHANNELS)},
        limitations=[
            "Channel-level attribution only: customers record their acquisition channel, not the campaign.",
            f"Attributed revenue = recognised revenue (subscription + usage) in the first {window_days} days after "
            "signup; it is not lifetime revenue and is not margin-adjusted.",
            "Spend is attributed by campaign week and customers by signup date, so period edges may not align exactly.",
        ],
        provenance=runner.provenance(
            f"ROAS = first-{window_days}-day attributed revenue / channel spend in the period"
        ),
    )


# ---------------------------------------------------------------------------------------------------


def _funnel(db: Database, period: PeriodSpec, dimension: str, as_of: date | None) -> KPIResult:
    current = to_period(period, as_of)
    return KPIService(db, as_of=as_of).calculate_kpi(
        "cac", start_date=current.start, end_date=current.end, dimension=dimension
    )


def _channel_row(channel: str, r: KPIBreakdownRow) -> ChannelRow:
    c = r.components
    spend = float(to_number(c.get("spend")) or 0.0)
    impressions, clicks = int(c.get("impressions") or 0), int(c.get("clicks") or 0)
    leads, conversions = int(c.get("leads") or 0), int(c.get("conversions") or 0)
    return ChannelRow(
        channel=channel,
        spend=spend,
        impressions=impressions,
        clicks=clicks,
        leads=leads,
        conversions=conversions,
        click_through_rate=safe_ratio(clicks, impressions),
        cost_per_lead=safe_ratio(spend, leads),
        conversion_rate=safe_ratio(conversions, leads),
        cac=safe_ratio(spend, conversions),
    )
