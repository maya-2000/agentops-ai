"""Sales pipeline generation.

- Every sales-assisted customer acquired in the window has a **won** New Business opportunity
  closing on its signup date with deal value = 12 x initial MRR.
- Mid-Market/Enterprise expansions mostly run through a **won** Expansion opportunity closing on
  the expansion date with deal value = 12 x MRR added.
- Each won deal is accompanied by a geometric number of **lost** deals determined by the owning
  rep's funnel conversion (so leads per rep are similar but wins differ by skill).
- Deals still in flight at the window end are **open** with the stage reached so far.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.database.metadata import OPEN_OPPORTUNITY_STAGES
from data.generator.config import ACQUISITION_SEASONALITY, FUNNEL, SEGMENT_PARAMS, GeneratorConfig
from data.generator.entities import SalesRep

_STAGE_NAMES = ("Lead", "Qualified", "Proposal", "Negotiation")
_LOST_CYCLE_DAYS = ((5, 30), (15, 50), (25, 80), (40, 120))  # by stage at which the deal is lost
_SEGMENT_CYCLE_FACTOR = {"SMB": 0.6, "Mid-Market": 1.0, "Enterprise": 1.5}


def _stage_conversions(base: tuple[float, ...], rep: SalesRep) -> np.ndarray:
    return np.clip(np.asarray(base) * rep.stage_factor, 0.01, 0.99)


def _win_probability(base: tuple[float, ...], rep: SalesRep) -> float:
    return float(np.prod(_stage_conversions(base, rep)))


def _loss_stage_distribution(base: tuple[float, ...], rep: SalesRep) -> np.ndarray:
    """P(deal is lost at stage k | deal is lost), k = Lead..Negotiation."""
    conv = _stage_conversions(base, rep)
    reach = np.concatenate(([1.0], np.cumprod(conv)[:-1]))
    lost_at = reach * (1 - conv)
    return lost_at / lost_at.sum()


def assign_account_owners(customers: pd.DataFrame, reps: list[SalesRep], rng: np.random.Generator) -> np.ndarray:
    """Owner rep index per customer (weighted by rep win probability within the region)."""
    owners = np.empty(len(customers), dtype=int)
    for region in customers["region"].unique():
        idx = np.flatnonzero(customers["region"].to_numpy() == region)
        pool = [i for i, r in enumerate(reps) if r.region == region]
        weights = np.array([_win_probability(FUNNEL.new_business, reps[i]) for i in pool])
        owners[idx] = rng.choice(pool, size=idx.size, p=weights / weights.sum())
    return owners


def _seasonal_dates(rng: np.random.Generator, start: date, end: date, size: int) -> list[date]:
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    weights = np.array([ACQUISITION_SEASONALITY[d.month] * (0.2 if d.weekday() >= 5 else 1.0) for d in days])
    picks = rng.choice(len(days), size=size, p=weights / weights.sum())
    return [days[i] for i in picks]


def _open_stage(rng: np.random.Generator, final_stage: int, elapsed: float, cycle: float) -> int:
    progressed = int(np.floor(min(elapsed / max(cycle, 1.0), 0.999) * (final_stage + 1)))
    return int(min(final_stage, progressed))


def build_opportunities(
    config: GeneratorConfig,
    customers: pd.DataFrame,
    subscriptions: pd.DataFrame,
    reps: list[SalesRep],
    churn_date: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    owners = assign_account_owners(customers, reps, rng)
    cust_pos = {cid: i for i, cid in enumerate(customers["customer_id"])}
    window_start, window_end = config.start_date, config.end_date
    rows: list[dict[str, object]] = []

    def add(
        rep_i: int,
        customer_id: str | None,
        opp_type: str,
        segment: str,
        created: date,
        close: date | None,
        stage: str,
        furthest: str,
        value: float,
    ) -> None:
        rows.append(
            {
                "customer_id": customer_id,
                "sales_rep": reps[rep_i].name,
                "opportunity_type": opp_type,
                "segment": segment,
                "region": reps[rep_i].region,
                "created_date": created,
                "close_date": close,
                "stage": stage,
                "furthest_stage": furthest,
                "deal_value": round(float(value), 2),
                "probability": FUNNEL.stage_probability[stage],
            }
        )

    def add_lost_or_open(
        rep_i: int,
        customer_id: str | None,
        opp_type: str,
        segment: str,
        created: date,
        value: float,
        base: tuple[float, ...],
    ) -> None:
        lost_stage = int(rng.choice(4, p=_loss_stage_distribution(base, reps[rep_i])))
        lo, hi = _LOST_CYCLE_DAYS[lost_stage]
        cycle = rng.uniform(lo, hi) * _SEGMENT_CYCLE_FACTOR[segment]
        close = created + timedelta(days=round(cycle))
        if close <= window_end:
            add(rep_i, customer_id, opp_type, segment, created, close, "Lost", _STAGE_NAMES[lost_stage], value)
        else:
            stage = _STAGE_NAMES[_open_stage(rng, lost_stage, (window_end - created).days, cycle)]
            add(rep_i, customer_id, opp_type, segment, created, None, stage, stage, value)

    # ---- New business -----------------------------------------------------------------------
    in_window = customers[~customers["is_opening"]]
    initial = subscriptions[subscriptions["change_type"] == "new"].set_index("customer_id")["monthly_recurring_revenue"]
    assisted = in_window[
        in_window["segment"].isin(["Mid-Market", "Enterprise"])
        | in_window["acquisition_channel"].isin(FUNNEL.sales_assisted_channels)
    ]
    won_by_segment: dict[str, list[float]] = {s: [] for s in SEGMENT_PARAMS}
    for cust in assisted.itertuples(index=False):
        rep_i = owners[cust_pos[cust.customer_id]]
        value = 12 * float(initial[cust.customer_id])
        won_by_segment[cust.segment].append(value)
        cycle = rng.lognormal(np.log(SEGMENT_PARAMS[cust.segment].sales_cycle_median_days), 0.35)
        add(
            rep_i,
            cust.customer_id,
            "New Business",
            cust.segment,
            cust.signup_date - timedelta(days=max(7, round(cycle))),
            cust.signup_date,
            "Won",
            "Won",
            value,
        )
        n_lost = rng.negative_binomial(1, _win_probability(FUNNEL.new_business, reps[rep_i]))
        for created in _seasonal_dates(rng, window_start, window_end, n_lost):
            add_lost_or_open(
                rep_i,
                None,
                "New Business",
                cust.segment,
                created,
                value * rng.lognormal(0.0, 0.35),
                FUNNEL.new_business,
            )

    # Deals that will close after the window: open pipeline built from the recent win rate.
    recent_start = window_end - timedelta(days=90)
    recent = assisted[assisted["signup_date"] > recent_start]
    for cust in recent.itertuples(index=False):
        rep_i = owners[cust_pos[cust.customer_id]]
        cycle = rng.lognormal(np.log(SEGMENT_PARAMS[cust.segment].sales_cycle_median_days), 0.35)
        n_future = rng.poisson(cycle / 90)
        for _ in range(n_future):
            elapsed = rng.uniform(0, cycle)
            created = window_end - timedelta(days=round(elapsed))
            stage = _STAGE_NAMES[_open_stage(rng, 3, elapsed, cycle)]
            pool = won_by_segment[cust.segment]
            value = float(pool[rng.integers(len(pool))]) * rng.lognormal(0.0, 0.3)
            add(rep_i, None, "New Business", cust.segment, created, None, stage, stage, value)

    # ---- Expansion ----------------------------------------------------------------------------
    subs = subscriptions.sort_values(["customer_id", "start_date"])
    expansions = subs[(subs["change_type"] == "expansion") & (subs["start_date"] >= pd.Timestamp(window_start))]
    segment_of = customers.set_index("customer_id")["segment"]
    signup_of = customers.set_index("customer_id")["signup_date"]
    for exp_row in expansions.itertuples(index=False):
        segment = segment_of[exp_row.customer_id]
        if segment == "SMB" or rng.random() > FUNNEL.expansion_opportunity_share:
            continue
        pos = cust_pos[exp_row.customer_id]
        rep_i = owners[pos]
        value = 12 * (float(exp_row.monthly_recurring_revenue) - float(exp_row.previous_mrr))
        close = pd.Timestamp(exp_row.start_date).date()
        add(
            rep_i,
            exp_row.customer_id,
            "Expansion",
            segment,
            close - timedelta(days=int(rng.integers(20, 76))),
            close,
            "Won",
            "Won",
            value,
        )
        n_lost = rng.negative_binomial(1, _win_probability(FUNNEL.expansion, reps[rep_i]))
        alive_from = max(window_start, signup_of[exp_row.customer_id] + timedelta(days=30))
        churn = churn_date[pos]
        alive_to = window_end if np.isnat(churn) else churn.astype(object) - timedelta(days=1)
        if alive_to <= alive_from:
            continue
        for created in _seasonal_dates(rng, alive_from, alive_to, n_lost):
            add_lost_or_open(
                rep_i,
                exp_row.customer_id,
                "Expansion",
                segment,
                created,
                value * rng.lognormal(0.0, 0.35),
                FUNNEL.expansion,
            )

    opps = pd.DataFrame(rows)
    # Churned customers cannot have open expansion deals.
    churn_of = {cid: churn_date[i] for cid, i in cust_pos.items()}
    is_open = opps["stage"].isin(OPEN_OPPORTUNITY_STAGES)
    churned_owner = opps["customer_id"].map(lambda c: isinstance(c, str) and not np.isnat(churn_of[c]))
    opps = opps[~(is_open & churned_owner)]

    opps = opps.sort_values(["created_date", "sales_rep", "deal_value"], kind="stable").reset_index(drop=True)
    opps.insert(0, "opportunity_id", [f"OPP-{i + 1:06d}" for i in range(len(opps))])
    return opps
