"""Monthly customer-lifecycle simulation driven by a hidden customer-health mechanism.

For every month of the window, for every live customer:

1. **Engagement** (latent, AR(1)) is updated. It reverts towards a segment mean, is pushed
   down by last month's poor support experience, up by AI Insights adoption, and hit by
   injected events (E2 faulty-release shock, E7 disengagement drift).
2. **Utilisation** (share of seats active) follows engagement and regional seasonality.
3. **Support tickets** are drawn: larger and less engaged customers raise more tickets;
   injected events add ticket bursts (E1 billing complaints, E2 bug spike).
4. **Support experience** is scored from ticket volume, slow resolutions and negative
   sentiment.
5. **Health** (hidden) = engagement + support experience + plan fit + tenure + adoption.
6. Health drives **churn**, **expansion** and **contraction** probabilities.

Engagement, utilisation and health never reach the database. Only their observable
consequences do: usage telemetry, tickets, subscription changes and revenue.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from data.generator.config import (
    ACQUISITION_SEASONALITY,
    HEALTH,
    PLAN_ORDER,
    PLAN_PARAMS,
    SEGMENT_PARAMS,
    SUPPORT,
    USAGE_SEASONALITY,
    WEEKDAY_ACTIVITY,
    GeneratorConfig,
)
from data.generator.entities import mrr_for
from data.generator.events import EVENTS, GroundTruthLog
from data.generator.timeline import Timeline, to_day

_PLAN_FIT = {"Starter": (1, 20), "Growth": (5, 80), "Professional": (10, 300), "Enterprise": (30, 10_000)}
_MIN_PLAN_BY_SEGMENT = {"SMB": 0, "Mid-Market": 0, "Enterprise": 2}
_MAX_PLAN_BY_SEGMENT = {"SMB": 2, "Mid-Market": 3, "Enterprise": 3}
_CATEGORIES = tuple(SUPPORT.category_shares)
_PRIORITIES = ("Low", "Medium", "High", "Urgent")


@dataclass
class SimulationResult:
    churn_date: np.ndarray  # datetime64[D], NaT if active at the end of the window
    changes: pd.DataFrame  # customer_idx, change_date, plan, seats, mrr, change_type
    utilisation: np.ndarray  # [n_customers, n_months], NaN when not live (hidden)
    engagement: np.ndarray  # [n_customers, n_months] (hidden, diagnostics only)
    health: np.ndarray  # [n_customers, n_months] (hidden, diagnostics only)
    ai_adoption_date: np.ndarray  # datetime64[D], NaT if never adopted (hidden)
    api_intensity: np.ndarray  # per-customer API usage multiplier (hidden)
    tickets: pd.DataFrame  # customer_idx, created_at, priority, category, resolution_hours, sentiment_score


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sample_days(rng: np.random.Generator, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Draw one day per ticket uniformly in [lo, hi], thinned by weekday activity."""
    span = (hi - lo).astype(int) + 1
    out = lo + np.floor(rng.random(lo.size) * span).astype("timedelta64[D]")
    weights = np.asarray(WEEKDAY_ACTIVITY) / max(WEEKDAY_ACTIVITY)
    for _ in range(8):
        dow = (out.astype("datetime64[D]").view("int64") - 4) % 7  # 1970-01-01 was a Thursday
        reject = rng.random(out.size) > weights[dow]
        if not reject.any():
            break
        out[reject] = lo[reject] + np.floor(rng.random(reject.sum()) * span[reject]).astype("timedelta64[D]")
    return out


def _timestamps(rng: np.random.Generator, days: np.ndarray) -> np.ndarray:
    hours = np.clip(rng.normal(12.5, 3.0, days.size), 0.0, 23.99)
    seconds = np.round(hours * 3600).astype("int64").astype("timedelta64[s]")
    return days.astype("datetime64[s]") + seconds


def simulate_lifecycle(
    config: GeneratorConfig,
    customers: pd.DataFrame,
    rng: np.random.Generator,
    truth: GroundTruthLog,
) -> SimulationResult:
    timeline = Timeline(config.start_date, config.end_date)
    month_starts = [to_day(d) for d in timeline.month_starts]
    month_ends = [to_day(d) for d in timeline.month_ends]
    n, n_months = len(customers), timeline.n_months

    segment = customers["segment"].to_numpy(dtype=object)
    region = customers["region"].to_numpy(dtype=object)
    country = customers["country"].to_numpy(dtype=object)
    signup = np.array(customers["signup_date"].tolist(), dtype="datetime64[D]")
    plan = customers["plan"].to_numpy(dtype=object).copy()
    seats = customers["seats"].to_numpy(dtype=int).copy()
    discount = customers["discount"].to_numpy(dtype=float)
    mrr = customers["mrr"].to_numpy(dtype=float).copy()

    seg_mean = np.array([SEGMENT_PARAMS[s].engagement_mean for s in segment])
    churn_int = np.array([SEGMENT_PARAMS[s].churn_intercept for s in segment])
    exp_int = np.array([SEGMENT_PARAMS[s].expansion_intercept for s in segment])
    con_int = np.array([SEGMENT_PARAMS[s].contraction_intercept for s in segment])
    ticket_rate = np.array([SEGMENT_PARAMS[s].ticket_rate for s in segment])
    is_sg_ent = (country == "Singapore") & (segment == "Enterprise")
    is_sg_mm = (country == "Singapore") & (segment == "Mid-Market")

    churn_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[D]")
    adoption_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[D]")
    engagement = np.where(signup < month_starts[0], rng.normal(seg_mean, 0.55), np.nan)
    support_prev = np.zeros(n)
    e2_hit_prev = np.zeros(n, dtype=bool)
    e7_cohort = np.zeros(n, dtype=bool)
    api_intensity = rng.lognormal(0.0, 0.9, n)

    eng_hist = np.full((n, n_months), np.nan)
    util_hist = np.full((n, n_months), np.nan)
    health_hist = np.full((n, n_months), np.nan)
    # (customer index, change date, plan, seats, MRR, change type)
    change_rows: list[tuple[Any, ...]] = [
        (i, signup[i], plan[i], int(seats[i]), float(mrr[i]), "new") for i in range(n)
    ]
    ticket_frames: list[pd.DataFrame] = []

    e1, e2, e5, e7 = (
        EVENTS.e1_sg_enterprise_churn,
        EVENTS.e2_support_spike,
        EVENTS.e5_ai_insights,
        EVENTS.e7_pre_churn_decline,
    )
    e2_start, e2_end = to_day(e2.start), to_day(e2.end)
    launch = to_day(e5.launch_date)
    e1_details: dict[str, Any] = {"churned_customer_ids": [], "contracted_customer_ids": []}
    e2_counts = {"extra_tickets": 0, "customers_hit": 0}
    ids = customers["customer_id"].to_numpy(dtype=object)

    for m in range(n_months):
        ms, me = month_starts[m], month_ends[m]
        month_date: date = timeline.month_starts[m]
        days_in_month = int((me - ms).astype(int)) + 1

        not_churned = np.isnat(churn_date)
        active_start = (signup < ms) & not_churned
        new_now = (signup >= ms) & (signup <= me)
        live = active_start | new_now

        # ---- 1. engagement ------------------------------------------------------------
        engagement[new_now] = rng.normal(seg_mean[new_now] - 0.15, 0.5)
        target = seg_mean + HEALTH.adoption_engagement_lift * ~np.isnat(adoption_date)
        updated = (
            HEALTH.engagement_persistence * engagement
            + (1 - HEALTH.engagement_persistence) * target
            + HEALTH.support_effect_on_engagement * support_prev
            + rng.normal(0.0, HEALTH.engagement_noise, n)
        )
        updated -= e2.engagement_shock * e2_hit_prev
        if month_date == e7.drift_start and not e7_cohort.any():
            tenure_at_selection = (to_day(e7.selection_date) - signup).astype(int) / 30.44
            pool = np.flatnonzero(active_start & ~is_sg_ent & (tenure_at_selection >= e7.min_tenure_months))
            k = round(e7.cohort_share * active_start.sum())
            if pool.size and k:
                e7_cohort[rng.choice(pool, size=min(k, pool.size), replace=False)] = True
        if month_date >= e7.drift_start:
            updated -= e7.monthly_engagement_drift * e7_cohort
        engagement = np.where(active_start, updated, engagement)

        # ---- 2. utilisation -----------------------------------------------------------
        season = np.array([USAGE_SEASONALITY[r].get(month_date.month, 1.0) for r in region])
        util = np.clip(
            _sigmoid(HEALTH.utilisation_slope * engagement + HEALTH.utilisation_intercept) * season, 0.03, 0.97
        )

        # ---- AI Insights adoption (E5) ----------------------------------------------------
        if me >= launch:
            elig = np.array([e5.eligible_plans.get(p, 0.0) for p in plan])
            ramp = min(1.0, (m - timeline.month_index(e5.launch_date) + 1) / 3)
            hazard = e5.monthly_adoption_hazard * elig * (0.5 + util) * ramp
            adopt = live & np.isnat(adoption_date) & (rng.random(n) < hazard)
            lo = np.maximum(np.maximum(ms, launch), signup)
            span = np.maximum((me - lo).astype(int), 0) + 1
            adoption_date[adopt] = (lo + np.floor(rng.random(n) * span).astype("timedelta64[D]"))[adopt]

        # ---- 3. support tickets ------------------------------------------------------------
        active_from = np.maximum(ms, signup)
        active_frac = np.where(live, ((me - active_from).astype(int) + 1) / days_in_month, 0.0)
        lam = ticket_rate * np.sqrt(seats) * np.exp(-HEALTH.ticket_engagement_elasticity * np.nan_to_num(engagement))
        lam = np.where(live, lam * active_frac, 0.0)

        t_cust, t_cat, t_lo, t_hi, t_e1, t_e2 = [], [], [], [], [], []
        for cat in _CATEGORIES:
            counts = rng.poisson(lam * SUPPORT.category_shares[cat])
            idx = np.repeat(np.arange(n), counts)
            t_cust.append(idx)
            t_cat.append(np.full(idx.size, cat, dtype=object))
            t_lo.append(active_from[idx])
            t_hi.append(np.full(idx.size, me))
            t_e1.append(np.zeros(idx.size, dtype=bool))
            t_e2.append(np.zeros(idx.size, dtype=bool))
            multiplier = e2.category_multipliers.get(cat)
            win_lo, win_hi = np.maximum(active_from, e2_start), np.full(n, min(me, e2_end))
            if multiplier and (win_lo <= win_hi).any():
                window_days = np.where(win_lo <= win_hi, (win_hi - win_lo).astype(int) + 1, 0)
                extra = rng.poisson(
                    lam
                    / np.maximum(active_frac, 1e-9)
                    / days_in_month
                    * window_days
                    * SUPPORT.category_shares[cat]
                    * (multiplier - 1)
                    * live
                )
                idx = np.repeat(np.arange(n), extra)
                e2_counts["extra_tickets"] += int(idx.size)
                t_cust.append(idx)
                t_cat.append(np.full(idx.size, cat, dtype=object))
                t_lo.append(win_lo[idx])
                t_hi.append(win_hi[idx])
                t_e1.append(np.zeros(idx.size, dtype=bool))
                t_e2.append(np.ones(idx.size, dtype=bool))
        if month_date.year == e1.month.year and month_date.month in e1.billing_ticket_rate:
            extra = rng.poisson(e1.billing_ticket_rate[month_date.month] * (live & is_sg_ent))
            idx = np.repeat(np.arange(n), extra)
            t_cust.append(idx)
            t_cat.append(np.full(idx.size, "Billing", dtype=object))
            t_lo.append(active_from[idx])
            t_hi.append(np.full(idx.size, me))
            t_e1.append(np.ones(idx.size, dtype=bool))
            t_e2.append(np.zeros(idx.size, dtype=bool))

        tc = np.concatenate(t_cust)
        tcat = np.concatenate(t_cat)
        tday = _sample_days(rng, np.concatenate(t_lo), np.concatenate(t_hi))
        te1 = np.concatenate(t_e1)
        te2 = np.concatenate(t_e2)
        prio_shares = np.array([SUPPORT.priority_shares[s] for s in segment[tc]]).reshape(-1, 4)
        cum = prio_shares.cumsum(axis=1)
        prio_idx = (rng.random(tc.size)[:, None] > cum).sum(axis=1).clip(0, 3)
        priority = np.array(_PRIORITIES, dtype=object)[prio_idx]
        median = np.array([SUPPORT.resolution_median_hours[p] for p in priority])
        hours = rng.lognormal(np.log(median), SUPPORT.resolution_sigma)
        hours *= np.where(segment[tc] == "Enterprise", SUPPORT.enterprise_resolution_factor, 1.0)
        in_e2 = (tday >= e2_start) & (tday <= e2_end)
        hours *= np.where(in_e2, e2.resolution_time_factor, 1.0)
        technical = np.isin(tcat, ("Bug", "Integration", "Performance"))
        sentiment_score = (
            -0.9 * np.log(hours / median)
            - 0.5 * technical
            - 0.9 * te1
            + 0.35 * np.nan_to_num(engagement[tc])
            + rng.normal(0.0, 0.8, tc.size)
        )

        # ---- 4. support experience ----------------------------------------------------------
        n_t = np.bincount(tc, minlength=n)
        n_slow = np.bincount(tc, weights=(hours > 2 * median).astype(float), minlength=n)
        n_neg = np.bincount(tc, weights=(sentiment_score < -0.9).astype(float), minlength=n)
        expected = ticket_rate * np.sqrt(seats) * active_frac
        support = (
            0.05
            - 0.35 * (n_t - expected) / np.sqrt(expected + 0.5)
            - 0.8 * n_slow / np.maximum(n_t, 1)
            - 0.8 * n_neg / np.maximum(n_t, 1)
        )
        support = np.clip(np.where(live, support, 0.0), -2.5, 0.3)
        # Customers who hit the faulty release (raised an incident-driven Bug/Integration ticket).
        e2_hit = np.bincount(tc, weights=(te2 & np.isin(tcat, ("Bug", "Integration"))).astype(float), minlength=n) > 0
        e2_counts["customers_hit"] += int(e2_hit.sum())

        # ---- 5. hidden health -------------------------------------------------------------------
        fit_lo = np.array([_PLAN_FIT[p][0] for p in plan])
        fit_hi = np.array([_PLAN_FIT[p][1] for p in plan])
        misfit = (seats < fit_lo) | (seats > fit_hi)
        tenure = (ms - signup).astype(int) / 30.44
        tenure_term = np.select([tenure < 3, tenure < 6, tenure > 24], [-0.35, -0.15, 0.10], 0.0)
        adopted_now = ~np.isnat(adoption_date) & (adoption_date <= me)
        health = (
            HEALTH.health_engagement_weight * np.nan_to_num(engagement)
            + HEALTH.health_support_weight * support
            - HEALTH.health_plan_fit_penalty * misfit
            + tenure_term
            + HEALTH.health_adoption_weight * adopted_now
            + rng.normal(0.0, HEALTH.health_noise, n)
        )

        # ---- 6. churn -------------------------------------------------------------------------
        logit = churn_int - HEALTH.churn_health_slope * health
        if month_date in e7.late_churn_months:
            logit = logit + e7.late_churn_logit_boost * e7_cohort
        is_e1_month = month_date == e1.month
        if is_e1_month:
            logit = logit + np.log(e1.mid_market_churn_multiplier) * is_sg_mm
        churn = active_start & (rng.random(n) < _sigmoid(logit))
        if is_e1_month:
            churn &= ~is_sg_ent  # SG Enterprise churn this month is governed by the event
        churn_offsets = np.floor(rng.random(n) * (days_in_month - 1)).astype("timedelta64[D]")
        churn_date[churn] = (ms + churn_offsets)[churn]

        e1_contract = np.zeros(n, dtype=bool)
        if is_e1_month:
            pool = np.flatnonzero(active_start & is_sg_ent)
            if pool.size:
                weights = _sigmoid(-health[pool])
                k_churn = round(e1.churn_share * pool.size)
                chosen = rng.choice(pool, size=k_churn, replace=False, p=weights / weights.sum())
                w_lo, w_hi = (to_day(d) for d in e1.last_service_day_window)
                wave_days = w_lo + np.floor(rng.random(chosen.size) * ((w_hi - w_lo).astype(int) + 1)).astype(
                    "timedelta64[D]"
                )
                churn_date[chosen] = wave_days
                churn[chosen] = True
                remaining = np.setdiff1d(pool, chosen)
                k_con = round(e1.contraction_share * pool.size)
                if remaining.size and k_con:
                    e1_contract[rng.choice(remaining, size=min(k_con, remaining.size), replace=False)] = True
                e1_details["churned_customer_ids"] = sorted(ids[chosen].tolist())
                e1_details["contracted_customer_ids"] = sorted(ids[e1_contract].tolist())
                e1_details["sg_enterprise_active_at_month_start"] = int(pool.size)

        # Tickets cannot be raised after the last day of service: re-date them before churn.
        late = churn[tc] & (tday > churn_date[tc])
        if late.any():
            lo = np.minimum(np.maximum(ms, signup[tc[late]]), churn_date[tc[late]])
            tday[late] = _sample_days(rng, lo, churn_date[tc[late]])
        adoption_after_churn = churn & ~np.isnat(adoption_date) & (adoption_date > churn_date)
        adoption_date[adoption_after_churn] = np.datetime64("NaT")

        created_at = _timestamps(rng, tday)
        ticket_frames.append(
            pd.DataFrame(
                {
                    "customer_idx": tc,
                    "created_at": created_at,
                    "priority": priority,
                    "category": tcat,
                    "resolution_hours": np.round(hours, 2),
                    "sentiment_score": sentiment_score,
                }
            )
        )

        # ---- 7. expansion / contraction ----------------------------------------------------------
        eligible = active_start & ~churn & (tenure >= 2)
        p_exp = _sigmoid(exp_int + HEALTH.expansion_health_slope * health) * ACQUISITION_SEASONALITY[month_date.month]
        p_con = _sigmoid(con_int - HEALTH.contraction_health_slope * health)
        u = rng.random(n)
        expand = eligible & (u < p_exp) & ~e1_contract
        contract = (eligible & ~expand & (u < p_exp + p_con)) | e1_contract
        change_day = ms + np.floor(rng.random(n) * days_in_month).astype("timedelta64[D]")
        if is_e1_month and e1_contract.any():
            w_lo, w_hi = (to_day(d) for d in e1.last_service_day_window)
            e1_days = w_lo + np.floor(rng.random(n) * ((w_hi - w_lo).astype(int) + 1)).astype("timedelta64[D]")
            change_day = np.where(e1_contract, e1_days, change_day)

        for i in np.flatnonzero(expand | contract):
            old_plan, old_seats, old_mrr = plan[i], int(seats[i]), float(mrr[i])
            plan_idx = PLAN_ORDER.index(old_plan)
            if expand[i]:
                hi = 1.3 if segment[i] == "Enterprise" else 1.5
                new_seats = int(np.ceil(old_seats * rng.uniform(1.08, hi)))
                if rng.random() < 0.2 and plan_idx < _MAX_PLAN_BY_SEGMENT[segment[i]]:
                    plan_idx += 1
            else:
                lo_f, hi_f = e1.contraction_factor if e1_contract[i] else (0.6, 0.9)
                new_seats = int(np.floor(old_seats * rng.uniform(lo_f, hi_f)))
                if not e1_contract[i] and rng.random() < 0.15 and plan_idx > _MIN_PLAN_BY_SEGMENT[segment[i]]:
                    plan_idx -= 1
            new_plan = PLAN_ORDER[plan_idx]
            new_seats = max(new_seats, PLAN_PARAMS[new_plan].min_seats)
            new_mrr = float(mrr_for(np.array([new_plan]), np.array([new_seats]), np.array([discount[i]]))[0])
            if (expand[i] and new_mrr <= old_mrr) or (contract[i] and new_mrr >= old_mrr):
                continue
            plan[i], seats[i], mrr[i] = new_plan, new_seats, new_mrr
            change_rows.append(
                (i, change_day[i], new_plan, new_seats, new_mrr, "expansion" if expand[i] else "contraction")
            )

        eng_hist[live, m] = engagement[live]
        util_hist[live, m] = util[live]
        health_hist[live, m] = health[live]
        support_prev = support
        e2_hit_prev = e2_hit

    # ---- ground truth ---------------------------------------------------------------------------
    window = (config.start_date, config.end_date)
    e1_in = window[0] <= e1.month <= window[1]
    truth.record(
        "E1",
        injected=e1_in and bool(e1_details["churned_customer_ids"]),
        period=e1.last_service_day_window if e1_in else None,
        **e1_details,
    )
    e2_in = e2.start <= window[1] and e2.end >= window[0]
    truth.record(
        "E2",
        injected=e2_in,
        period=(e2.start, e2.end) if e2_in else None,
        extra_tickets=e2_counts["extra_tickets"],
        customer_months_hit=e2_counts["customers_hit"],
        category_multipliers=e2.category_multipliers,
        resolution_time_factor=e2.resolution_time_factor,
    )
    e5_in = window[0] <= e5.launch_date <= window[1]
    truth.record(
        "E5",
        injected=e5_in,
        period=(e5.launch_date, window[1]) if e5_in else None,
        feature_id=e5.feature_id,
        feature_name=e5.feature_name,
        launch_date=e5.launch_date,
        adopting_customers=int((~np.isnat(adoption_date)).sum()),
    )
    truth.record(
        "E6",
        injected=True,
        period=window,
        segment=EVENTS.e6_smb_churn.segment,
        churn_intercepts={s: p.churn_intercept for s, p in SEGMENT_PARAMS.items()},
    )
    cohort_ids = sorted(ids[e7_cohort].tolist())
    churned_cohort = sorted(ids[e7_cohort & ~np.isnat(churn_date)].tolist())
    truth.record(
        "E7",
        injected=bool(cohort_ids),
        period=(e7.drift_start, window[1]) if cohort_ids else None,
        cohort_customer_ids=cohort_ids,
        churned_customer_ids=churned_cohort,
        monthly_engagement_drift=e7.monthly_engagement_drift,
    )

    changes = pd.DataFrame(change_rows, columns=["customer_idx", "change_date", "plan", "seats", "mrr", "change_type"])
    tickets = pd.concat(ticket_frames, ignore_index=True)
    return SimulationResult(
        churn_date=churn_date,
        changes=changes,
        utilisation=util_hist,
        engagement=eng_hist,
        health=health_hist,
        ai_adoption_date=adoption_date,
        api_intensity=api_intensity,
        tickets=tickets,
    )
