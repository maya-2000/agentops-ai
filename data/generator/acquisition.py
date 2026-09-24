"""Marketing campaigns and customer acquisition.

Funnel per campaign-week: spend -> impressions -> clicks -> leads -> conversions. Click-through
rate falls as spend rises above the channel's usual weekly level (ad fatigue), so leads and
conversions grow sub-linearly with spend (diminishing returns). Each conversion becomes one
customer with that campaign's channel and a signup date inside the same week, so
marketing conversions reconcile exactly with customer acquisition.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.generator.config import (
    ACQUISITION_SEASONALITY,
    CAMPAIGN_THEMES,
    CHANNEL_PARAMS,
    COMPANY_FOUNDED,
    EVENT_CAMPAIGN_WEEKS,
    MARKETING_SHARE_OF_NEW,
    NON_MARKETING_CHANNEL_SPLIT,
    GeneratorConfig,
)
from data.generator.entities import choice
from data.generator.events import EVENTS, GroundTruthLog
from data.generator.timeline import Timeline, spend_growth_factor


@dataclass
class CampaignPlan:
    channel: str
    name: str
    week_indices: list[int]
    spend_weight: float = 1.0  # share of the channel's weekly budget
    spend_multiplier: float = 1.0  # event distortion (E3)
    lead_quality: float = 1.0  # event distortion (E3)
    is_e3: bool = False
    campaign_id: str = ""


def _quarter_key(d: date) -> tuple[int, int]:
    return d.year, (d.month - 1) // 3 + 1


def plan_campaigns(timeline: Timeline) -> list[CampaignPlan]:
    weeks = timeline.week_starts
    quarters: dict[tuple[int, int], list[int]] = {}
    for idx, week in enumerate(weeks):
        quarters.setdefault(_quarter_key(week), []).append(idx)

    e3 = EVENTS.e3_paid_social
    plans: list[CampaignPlan] = []
    for q_order, (quarter, week_idx) in enumerate(sorted(quarters.items())):
        label = f"{quarter[0]}-Q{quarter[1]}"
        for channel, themes in CAMPAIGN_THEMES.items():
            theme = themes[q_order % len(themes)]
            if channel == "Events":
                offset = min(5, max(0, len(week_idx) - EVENT_CAMPAIGN_WEEKS))
                flight = week_idx[offset : offset + EVENT_CAMPAIGN_WEEKS]
                plans.append(CampaignPlan(channel, f"{channel} | {theme} | {label}", flight))
                continue
            plan = CampaignPlan(channel, f"{channel} | {theme} | {label}", list(week_idx))
            if channel == e3.channel and quarter == _quarter_key(e3.quarter_start):
                plan.name = f"{channel} | {e3.theme} | {label}"
                plan.spend_multiplier = e3.spend_multiplier
                plan.lead_quality = e3.lead_quality
                plan.is_e3 = True
            plans.append(plan)

    # Launch campaign for AI Insights (8 weeks from launch), when the launch is in the window.
    launch = EVENTS.e5_ai_insights.launch_date
    launch_weeks = [i for i, w in enumerate(weeks) if launch <= w < launch + timedelta(weeks=8)]
    if launch_weeks:
        plans.append(
            CampaignPlan(
                "Email",
                f"Email | AI Insights Launch | {launch.year}-Q{(launch.month - 1) // 3 + 1}",
                launch_weeks,
                spend_weight=0.5,
            )
        )

    plans.sort(key=lambda p: (weeks[p.week_indices[0]], list(CAMPAIGN_THEMES).index(p.channel), p.name))
    for i, plan in enumerate(plans, start=1):
        plan.campaign_id = f"CMP-{i:03d}"
    return plans


def simulate_marketing(
    config: GeneratorConfig, rng: np.random.Generator, truth: GroundTruthLog
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (marketing_campaigns table, marketing-sourced signups)."""
    timeline = Timeline(config.start_date, config.end_date)
    weeks = timeline.week_starts
    n_weeks = len(weeks)
    plans = plan_campaigns(timeline)
    target_conversions = MARKETING_SHARE_OF_NEW * config.new_customer_count

    week_growth = np.array([spend_growth_factor(w) for w in weeks])
    week_growth /= week_growth.mean()
    week_season = np.array([ACQUISITION_SEASONALITY[w.month] for w in weeks])

    rows = []
    for plan in plans:
        params = CHANNEL_PARAMS[plan.channel]
        channel_target = params.conversion_share * target_conversions
        base_weekly_spend = channel_target * params.target_cac / n_weeks
        if plan.channel == "Events":  # a quarter's budget concentrated into the event flight
            base_weekly_spend *= 13 / EVENT_CAMPAIGN_WEEKS
        for w in plan.week_indices:
            noise = rng.lognormal(0.0, 0.12)
            baseline_spend = base_weekly_spend * plan.spend_weight * week_growth[w] * week_season[w] * noise
            rows.append(
                {
                    "campaign_id": plan.campaign_id,
                    "campaign_name": plan.name,
                    "channel": plan.channel,
                    "week_index": w,
                    "date": weeks[w],
                    "baseline_spend": baseline_spend,
                    "reference_spend": base_weekly_spend,
                    "spend_multiplier": plan.spend_multiplier,
                    "lead_quality": plan.lead_quality,
                    "season": week_season[w],
                }
            )
    df = pd.DataFrame(rows)
    df["spend"] = np.round(df["baseline_spend"] * df["spend_multiplier"], 2)

    ctr = df["channel"].map(lambda c: CHANNEL_PARAMS[c].ctr).to_numpy()
    cpm = df["channel"].map(lambda c: CHANNEL_PARAMS[c].cpm).to_numpy()
    fatigue = df["channel"].map(lambda c: CHANNEL_PARAMS[c].fatigue).to_numpy()
    l2c = df["channel"].map(lambda c: CHANNEL_PARAMS[c].lead_to_customer).to_numpy()
    rel_spend = df["spend"].to_numpy() / df["reference_spend"].to_numpy()
    ctr_eff = np.clip(ctr * rel_spend ** (-fatigue), 1e-5, 0.5)
    df["impressions"] = rng.poisson(df["spend"].to_numpy() / cpm * 1000.0)
    df["clicks"] = rng.binomial(df["impressions"].to_numpy(), ctr_eff)

    # Calibrate click->lead per channel so expected *baseline* conversions hit the target.
    base_rel = df["baseline_spend"].to_numpy() / df["reference_spend"].to_numpy()
    base_clicks = df["baseline_spend"].to_numpy() / cpm * 1000.0 * np.clip(ctr * base_rel ** (-fatigue), 1e-5, 0.5)
    df["_expected_base"] = base_clicks * l2c * df["season"].to_numpy()
    click_to_lead: dict[str, float] = {}
    for channel, params in CHANNEL_PARAMS.items():
        expected = df.loc[df["channel"] == channel, "_expected_base"].sum()
        target = params.conversion_share * target_conversions
        click_to_lead[channel] = min(0.9, target / expected) if expected > 0 else 0.0
    c2l = df["channel"].map(click_to_lead).to_numpy()
    df["leads"] = rng.binomial(df["clicks"].to_numpy(), c2l)
    conv_p = np.clip(l2c * df["season"].to_numpy() * df["lead_quality"].to_numpy(), 0, 1)
    df["conversions"] = rng.binomial(df["leads"].to_numpy(), conv_p)

    # Marketing-sourced signups: one customer per conversion, dated within the week.
    signup_rows = []
    for rec in df.loc[df["conversions"] > 0, ["campaign_id", "channel", "date", "conversions"]].itertuples(index=False):
        max_offset = min(6, (config.end_date - rec.date).days)
        offsets = rng.integers(0, max_offset + 1, rec.conversions)
        for off in offsets:
            signup_rows.append((rec.date + timedelta(days=int(off)), rec.channel, rec.campaign_id))
    signups = pd.DataFrame(signup_rows, columns=["signup_date", "acquisition_channel", "acquisition_campaign_id"])

    e3_plans = [p for p in plans if p.is_e3]
    if e3_plans:
        e3 = e3_plans[0]
        agg = df.loc[df["campaign_id"] == e3.campaign_id, ["spend", "conversions"]].sum()
        truth.record(
            "E3",
            injected=True,
            period=(weeks[e3.week_indices[0]], weeks[e3.week_indices[-1]] + timedelta(days=6)),
            campaign_id=e3.campaign_id,
            campaign_name=e3.name,
            spend_multiplier=e3.spend_multiplier,
            lead_quality=e3.lead_quality,
            total_spend=round(float(agg["spend"]), 2),
            total_conversions=int(agg["conversions"]),
        )
    else:
        truth.record("E3", injected=False, period=None, reason="Event quarter outside the generated window.")

    table = df[
        ["campaign_id", "campaign_name", "channel", "date", "spend", "impressions", "clicks", "leads", "conversions"]
    ].copy()
    table = table.sort_values(["campaign_id", "date"]).reset_index(drop=True)
    return table, signups


def non_marketing_signups(config: GeneratorConfig, rng: np.random.Generator, count: int) -> pd.DataFrame:
    """Outbound-sales and referral signups spread over the window with seasonality and growth."""
    if count < 0:
        raise ValueError(
            "Marketing produced more conversions than the configured number of new customers; "
            "lower MARKETING_SHARE_OF_NEW or raise customer_count."
        )
    days = [config.start_date + timedelta(days=i) for i in range((config.end_date - config.start_date).days + 1)]
    weights = np.array(
        [ACQUISITION_SEASONALITY[d.month] * spend_growth_factor(d) * (0.35 if d.weekday() >= 5 else 1.0) for d in days]
    )
    picks = rng.choice(len(days), size=count, p=weights / weights.sum())
    return pd.DataFrame(
        {
            "signup_date": [days[i] for i in picks],
            "acquisition_channel": choice(rng, NON_MARKETING_CHANNEL_SPLIT, count),
            "acquisition_campaign_id": None,
        }
    )


def opening_signups(config: GeneratorConfig, rng: np.random.Generator, count: int) -> np.ndarray:
    """Signup dates of customers acquired before the window (density rising over time)."""
    span = (config.start_date - COMPANY_FOUNDED).days - 1
    offsets = np.floor(span * np.sqrt(rng.random(count))).astype(int)
    return np.array([COMPANY_FOUNDED + timedelta(days=int(o)) for o in offsets], dtype=object)
