"""Materialise relational tables from the simulation state.

Every table is derived from the same underlying customer histories, which is what makes the
relationships between them consistent (e.g. revenue exists exactly while a subscription is in
force; usage stops when a customer churns; won deals match new MRR).
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from data.generator.config import (
    AI_INSIGHTS_FEATURE_ID,
    AI_INSIGHTS_FEATURE_NAME,
    API_OVERAGE_PRICE_PER_1000,
    CORE_FEATURES,
    DAU_TO_WAU_RATIO,
    PLAN_PARAMS,
    WEEKDAY_ACTIVITY,
    GeneratorConfig,
)
from data.generator.events import EVENTS
from data.generator.lifecycle import SimulationResult
from data.generator.timeline import Timeline, holiday_factor, month_index_array, to_day

_ONE_DAY = np.timedelta64(1, "D")


def build_customers_table(customers: pd.DataFrame, sim: SimulationResult) -> pd.DataFrame:
    table = customers[
        [
            "customer_id",
            "company_name",
            "country",
            "region",
            "industry",
            "company_size",
            "segment",
            "acquisition_channel",
            "signup_date",
        ]
    ].copy()
    table["status"] = np.where(np.isnat(sim.churn_date), "active", "churned")
    return table


def build_subscriptions(customers: pd.DataFrame, sim: SimulationResult) -> pd.DataFrame:
    """Versioned subscription records: one per contiguous period of constant plan/seats/MRR."""
    ch = sim.changes.sort_values(["customer_idx", "change_date"], kind="stable").reset_index(drop=True)
    cust = ch["customer_idx"].to_numpy()
    start = ch["change_date"].to_numpy(dtype="datetime64[D]")
    same_next = np.append(cust[1:] == cust[:-1], False)
    next_start = np.append(start[1:], np.datetime64("NaT"))
    churn = sim.churn_date[cust]

    end = np.where(same_next, next_start - _ONE_DAY, churn)
    status = np.where(same_next, "superseded", np.where(np.isnat(churn), "active", "churned"))
    prev_mrr = np.where(np.append(False, cust[1:] == cust[:-1]), np.append(0.0, ch["mrr"].to_numpy()[:-1]), 0.0)
    mrr = ch["mrr"].to_numpy()

    subs = pd.DataFrame(
        {
            "customer_id": customers["customer_id"].to_numpy()[cust],
            "plan": ch["plan"].to_numpy(),
            "seats": ch["seats"].to_numpy(dtype=int),
            "monthly_recurring_revenue": mrr,
            "start_date": start,
            "end_date": end,
            "status": status,
            "change_type": ch["change_type"].to_numpy(),
            "previous_mrr": np.round(prev_mrr, 2),
            "current_mrr": np.where(status == "active", mrr, 0.0),
        }
    )
    subs = subs.sort_values(["start_date", "customer_id"], kind="stable").reset_index(drop=True)
    subs.insert(0, "subscription_id", [f"SUB-{i + 1:07d}" for i in range(len(subs))])
    return subs


def _expand_days(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (row index, date) for every day in each [start, end] interval."""
    lengths = (ends - starts).astype(int) + 1
    lengths = np.maximum(lengths, 0)
    row = np.repeat(np.arange(starts.size), lengths)
    first = np.repeat(np.cumsum(lengths) - lengths, lengths)
    offset = np.arange(row.size) - first
    return row, starts[row] + offset.astype("timedelta64[D]")


def build_usage_events(
    config: GeneratorConfig,
    customers: pd.DataFrame,
    subscriptions: pd.DataFrame,
    sim: SimulationResult,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Weekly usage per customer for weeks with >= 4 subscribed days."""
    timeline = Timeline(config.start_date, config.end_date)
    weeks = np.array([to_day(w) for w in timeline.week_starts])
    signup = np.array(customers["signup_date"].tolist(), dtype="datetime64[D]")
    last_day = np.where(np.isnat(sim.churn_date), to_day(config.end_date), sim.churn_date)

    week_end = weeks + 6
    first_active = np.maximum(signup[:, None], weeks[None, :])
    last_active = np.minimum(last_day[:, None], week_end[None, :])
    active_days = (last_active - first_active).astype(int) + 1
    ci, wi = np.nonzero(active_days >= 4)

    week_dates = weeks[wi]
    mid_week = week_dates + 3
    month_idx = np.clip(month_index_array(mid_week, config.start_date), 0, timeline.n_months - 1)
    util = sim.utilisation[ci, month_idx]
    prev_idx = np.clip(month_idx - 1, 0, None)
    util = np.where(np.isnan(util), sim.utilisation[ci, prev_idx], util)
    util = np.nan_to_num(util, nan=0.5)

    # Plan and seats in force at mid-week.
    lookup = pd.DataFrame(
        {"customer_id": customers["customer_id"].to_numpy()[ci], "mid": mid_week, "row": np.arange(ci.size)}
    )
    subs = subscriptions[["customer_id", "start_date", "plan", "seats"]].copy()
    subs["start_date"] = subs["start_date"].astype("datetime64[ns]")
    lookup["mid"] = lookup["mid"].astype("datetime64[ns]")
    merged = pd.merge_asof(
        lookup.sort_values("mid"),
        subs.sort_values("start_date"),
        left_on="mid",
        right_on="start_date",
        by="customer_id",
        direction="backward",
    ).sort_values("row")
    plan = merged["plan"].to_numpy(dtype=object)
    seats = merged["seats"].to_numpy(dtype=int)

    region = customers["region"].to_numpy(dtype=object)[ci]
    week_py = [w.astype(object) for w in weeks]
    holiday = np.array([[holiday_factor(w, r) for w in week_py] for r in ("APAC", "EMEA", "North America", "LATAM")])
    region_code = pd.Series(region).map({"APAC": 0, "EMEA": 1, "North America": 2, "LATAM": 3}).to_numpy()
    hol = holiday[region_code, wi]

    wau_p = np.clip(util * 0.95 * hol * rng.lognormal(0.0, 0.07, ci.size), 0.0, 1.0)
    active_users = rng.binomial(seats, wau_p)
    spu = np.array([PLAN_PARAMS[p].sessions_per_user_week for p in plan])
    sessions = rng.poisson(active_users * spu * rng.lognormal(0.0, 0.15, ci.size))
    api_per_user = np.array([PLAN_PARAMS[p].api_calls_per_user_week for p in plan])
    api_calls = np.round(active_users * api_per_user * sim.api_intensity[ci] * rng.lognormal(0.0, 0.25, ci.size))

    available = np.array([PLAN_PARAMS[p].features_available for p in plan])
    ai_eligible = np.array([p in EVENTS.e5_ai_insights.eligible_plans for p in plan])
    adopted = ~np.isnat(sim.ai_adoption_date[ci]) & (sim.ai_adoption_date[ci] <= week_end[wi]) & ai_eligible
    available = available + adopted
    feature_usage = rng.binomial(available, np.clip(0.2 + 0.7 * util, 0.0, 1.0))
    feature_usage = np.where(active_users > 0, feature_usage, 0)

    usage = pd.DataFrame(
        {
            "customer_id": customers["customer_id"].to_numpy()[ci],
            "event_date": week_dates,
            "active_users": active_users.astype(int),
            "sessions": sessions.astype(int),
            "api_calls": api_calls.astype(np.int64),
            "feature_usage": feature_usage.astype(int),
            "_plan": plan,
            "_seats": seats,
            "_customer_idx": ci,
        }
    )
    usage = usage.sort_values(["event_date", "customer_id"], kind="stable").reset_index(drop=True)
    usage.insert(0, "event_id", np.arange(1, len(usage) + 1, dtype=np.int64))
    return usage


def _days_in_month(days: np.ndarray) -> np.ndarray:
    month = days.astype("datetime64[M]")
    return ((month + 1).astype("datetime64[D]") - month.astype("datetime64[D]")).astype(int)


def build_daily_revenue(
    config: GeneratorConfig,
    customers: pd.DataFrame,
    subscriptions: pd.DataFrame,
    usage: pd.DataFrame,
) -> pd.DataFrame:
    """Subscription revenue (MRR / days in month, per day in force) plus API overage revenue.

    A subscription in force for a whole calendar month therefore recognises exactly its MRR
    (up to cent rounding) in that month, and partial months are prorated by day.
    """
    window_start, window_end = to_day(config.start_date), to_day(config.end_date)
    start = subscriptions["start_date"].to_numpy(dtype="datetime64[D]")
    end = subscriptions["end_date"].to_numpy(dtype="datetime64[D]")
    end = np.where(np.isnat(end), window_end, end)
    clipped_start = np.maximum(start, window_start)
    clipped_end = np.minimum(end, window_end)
    row, day = _expand_days(clipped_start, clipped_end)

    seg = customers.set_index("customer_id")
    cust_ids = subscriptions["customer_id"].to_numpy(dtype=object)[row]
    mrr = subscriptions["monthly_recurring_revenue"].to_numpy(dtype=float)[row]
    daily_sub = pd.DataFrame(
        {
            "date": day,
            "customer_id": cust_ids,
            "region": seg["region"].reindex(cust_ids).to_numpy(),
            "segment": seg["segment"].reindex(cust_ids).to_numpy(),
            "plan": subscriptions["plan"].to_numpy(dtype=object)[row],
            "revenue_type": "subscription",
            "revenue": np.round(mrr / _days_in_month(day), 2),
        }
    )

    # Usage (overage) revenue: calendar-month API calls above the monthly quota, spread over the
    # customer's active days in that month. Weekly calls are split across months by day.
    week_start = usage["event_date"].to_numpy(dtype="datetime64[D]")
    days_left = (week_start.astype("datetime64[M]") + 1).astype("datetime64[D]") - week_start
    in_first = np.minimum(days_left.astype(int), 7)
    quota_rate = usage["_seats"].to_numpy() * np.array([PLAN_PARAMS[p].api_quota_per_seat for p in usage["_plan"]])
    parts = []
    for n_days, month_start in (
        (in_first, week_start.astype("datetime64[M]")),
        (7 - in_first, week_start.astype("datetime64[M]") + 1),
    ):
        dim = _days_in_month(month_start.astype("datetime64[D]"))
        parts.append(
            pd.DataFrame(
                {
                    "customer_id": usage["customer_id"].to_numpy(),
                    "month": month_start,
                    "api_calls": usage["api_calls"].to_numpy() * n_days / 7,
                    "quota": quota_rate * n_days / dim,
                }
            )
        )
    monthly = pd.concat(parts).groupby(["customer_id", "month"], as_index=False)[["api_calls", "quota"]].sum()
    monthly["overage"] = np.maximum(monthly["api_calls"] - monthly["quota"], 0) / 1000 * API_OVERAGE_PRICE_PER_1000
    monthly = monthly[monthly["overage"] >= 1.0]

    daily_sub["month"] = daily_sub["date"].to_numpy(dtype="datetime64[D]").astype("datetime64[M]")
    # Overage is recognised only on days when a plan with API access is in force.
    api_plans = [p for p, params in PLAN_PARAMS.items() if params.api_quota_per_seat > 0]
    api_days = daily_sub[daily_sub["plan"].isin(api_plans)]
    days_per = api_days.groupby(["customer_id", "month"], as_index=False).size().rename(columns={"size": "n_days"})
    monthly = monthly.merge(days_per, on=["customer_id", "month"], how="inner")
    overage_days = api_days.merge(monthly[["customer_id", "month", "overage", "n_days"]], on=["customer_id", "month"])
    daily_usage = overage_days.assign(
        revenue_type="usage",
        revenue=np.round(overage_days["overage"] / overage_days["n_days"], 2),
    )[["date", "customer_id", "region", "segment", "plan", "revenue_type", "revenue"]]
    daily_usage = daily_usage[daily_usage["revenue"] > 0]

    revenue = pd.concat([daily_sub.drop(columns="month"), daily_usage], ignore_index=True)
    return revenue.sort_values(["date", "customer_id", "revenue_type"], kind="stable").reset_index(drop=True)


def build_support_tickets(customers: pd.DataFrame, sim: SimulationResult, config: GeneratorConfig) -> pd.DataFrame:
    t = sim.tickets.sort_values(["created_at", "customer_idx"], kind="stable").reset_index(drop=True)
    window_close = np.datetime64(config.end_date + timedelta(days=1), "s")
    resolved_at = t["created_at"].to_numpy(dtype="datetime64[s]") + np.round(
        t["resolution_hours"].to_numpy() * 3600
    ).astype("int64").astype("timedelta64[s]")
    open_ = resolved_at >= window_close
    score = t["sentiment_score"].to_numpy()
    sentiment = np.select([score < -0.9, score > 0.75], ["Negative", "Positive"], "Neutral")
    return pd.DataFrame(
        {
            "ticket_id": [f"TCK-{i + 1:07d}" for i in range(len(t))],
            "customer_id": customers["customer_id"].to_numpy()[t["customer_idx"].to_numpy()],
            "created_at": t["created_at"].to_numpy(dtype="datetime64[s]"),
            "resolved_at": np.where(open_, np.datetime64("NaT"), resolved_at),
            "priority": t["priority"].to_numpy(),
            "category": t["category"].to_numpy(),
            "status": np.where(open_, "Open", "Resolved"),
            "resolution_time": np.where(open_, np.nan, t["resolution_hours"].to_numpy()),
            "sentiment": sentiment,
        }
    )


def build_product_features(
    config: GeneratorConfig, usage: pd.DataFrame, sim: SimulationResult, rng: np.random.Generator
) -> pd.DataFrame:
    """Daily feature adoption: feature DAU / platform DAU."""
    timeline = Timeline(config.start_date, config.end_date)
    days = timeline.days
    weeks = np.array([to_day(w) for w in timeline.week_starts])
    week_of_day = np.clip(np.searchsorted(weeks, days, side="right") - 1, 0, len(weeks) - 1)
    dow = (days.view("int64") - 4) % 7
    weekday = np.asarray(WEEKDAY_ACTIVITY)[dow] / np.mean(WEEKDAY_ACTIVITY)

    wau = usage.groupby("event_date")["active_users"].sum().reindex(weeks.astype("datetime64[ns]")).to_numpy()
    wau = np.nan_to_num(wau)
    platform_dau = np.maximum(
        np.round(wau[week_of_day] * DAU_TO_WAU_RATIO * weekday * rng.lognormal(0, 0.02, days.size)), 1
    )
    years = (days - days[0]).astype(int) / 365.25

    frames = []
    for feature_id, name, base, drift in CORE_FEATURES:
        rate = np.clip(base + drift * years + rng.normal(0.0, 0.008, days.size), 0.01, 0.98)
        users = np.round(platform_dau * rate)
        frames.append(
            pd.DataFrame(
                {
                    "feature_id": feature_id,
                    "feature_name": name,
                    "date": days,
                    "active_users": users.astype(int),
                    "adoption_rate": users / platform_dau,
                }
            )
        )

    launch = to_day(EVENTS.e5_ai_insights.launch_date)
    if launch <= days[-1]:
        adopt = sim.ai_adoption_date[usage["_customer_idx"].to_numpy()]
        week_dates = usage["event_date"].to_numpy(dtype="datetime64[D]")
        months_since = np.where(np.isnat(adopt), 0.0, (week_dates + 6 - adopt).astype(float) / 30.44)
        share = np.where(
            ~np.isnat(adopt) & (adopt <= week_dates + 6),
            0.25 + 0.35 * (1 - np.exp(-np.maximum(months_since, 0) / 2)),
            0.0,
        )
        ai_wau = pd.Series(usage["active_users"].to_numpy() * share).groupby(week_dates).sum()
        ai_wau = ai_wau.reindex(weeks).fillna(0.0).to_numpy()
        mask = days >= launch
        ai_users = np.round(ai_wau[week_of_day] * DAU_TO_WAU_RATIO * weekday * rng.lognormal(0, 0.04, days.size))
        ai_users = np.minimum(ai_users, platform_dau)[mask]
        frames.append(
            pd.DataFrame(
                {
                    "feature_id": AI_INSIGHTS_FEATURE_ID,
                    "feature_name": AI_INSIGHTS_FEATURE_NAME,
                    "date": days[mask],
                    "active_users": ai_users.astype(int),
                    "adoption_rate": ai_users / platform_dau[mask],
                }
            )
        )

    features = pd.concat(frames, ignore_index=True)
    features["adoption_rate"] = features["adoption_rate"].round(4)
    return features.sort_values(["feature_id", "date"]).reset_index(drop=True)
