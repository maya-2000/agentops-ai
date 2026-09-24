"""Entity creation: sales reps, company names, customer firmographics and initial contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.database.metadata import COUNTRY_REGION
from data.generator.config import (
    COMPANY_SIZE_BY_SEGMENT,
    COUNTRY_WEIGHTS,
    FUNNEL,
    INDUSTRY_WEIGHTS,
    LEGAL_SUFFIX,
    NEW_CUSTOMER_SEAT_FACTOR,
    OPENING_CHANNEL_MIX,
    OPENING_SEGMENT_MIX,
    PLAN_MIX_BY_SEGMENT,
    PLAN_PARAMS,
    SALES_TEAM,
    SEGMENT_MIX_BY_CHANNEL,
    SEGMENT_PARAMS,
)
from data.generator.events import EVENTS, GroundTruthLog

_FIRST_NAMES = (
    "Aisha",
    "Daniel",
    "Mei Ling",
    "Rahul",
    "Sofia",
    "Kenji",
    "Priya",
    "Lucas",
    "Hannah",
    "Wei Jie",
    "Olivia",
    "Arjun",
    "Chloe",
    "Mateo",
    "Nurul",
    "James",
    "Yuki",
    "Isabel",
    "Farhan",
    "Emma",
    "Gabriel",
    "Siti",
    "Noah",
    "Ananya",
)
_LAST_NAMES = (
    "Tan",
    "Rahman",
    "Fischer",
    "Nakamura",
    "Silva",
    "Kapoor",
    "Martin",
    "Lim",
    "Okafor",
    "Dubois",
    "Hughes",
    "Santos",
    "Wong",
    "Schmidt",
    "Iyer",
    "Morales",
    "Chua",
    "Bennett",
    "Sato",
    "Van Dijk",
    "Reyes",
    "Goh",
    "Clarke",
    "Menon",
)
_NAME_PREFIXES = (
    "Blue",
    "North",
    "South",
    "Silver",
    "Iron",
    "Bright",
    "Swift",
    "Clear",
    "Green",
    "Summit",
    "Harbor",
    "Pine",
    "Cedar",
    "Stone",
    "River",
    "Atlas",
    "Nova",
    "Apex",
    "Crest",
    "Lumen",
    "Vertex",
    "Orbit",
    "Pioneer",
    "Quantum",
    "Sterling",
    "Coral",
    "Maple",
    "Falcon",
    "Horizon",
    "Aurora",
    "Beacon",
    "Cobalt",
    "Delta",
    "Ember",
    "Granite",
    "Helix",
    "Juniper",
    "Kestrel",
    "Lotus",
    "Meridian",
)
_NAME_SUFFIXES = (
    "field",
    "point",
    "way",
    "line",
    "bridge",
    "gate",
    "wave",
    "path",
    "works",
    "stream",
    "peak",
    "forge",
    "leaf",
    "port",
    "mark",
    "view",
    "light",
    "shore",
    "ridge",
    "wood",
)
_INDUSTRY_NOUNS: dict[str, tuple[str, ...]] = {
    "Fintech": ("Capital", "Payments", "Finance", "Pay", "Ledger"),
    "Retail & E-commerce": ("Retail", "Commerce", "Goods", "Market", "Stores"),
    "Healthcare": ("Health", "Medical", "Care", "Clinics", "Bio"),
    "Logistics": ("Logistics", "Freight", "Shipping", "Transport", "Supply"),
    "Manufacturing": ("Industries", "Manufacturing", "Components", "Engineering", "Materials"),
    "Media & Entertainment": ("Media", "Studios", "Digital", "Publishing", "Entertainment"),
    "Education": ("Learning", "Education", "Academy", "EdTech", "Schools"),
    "Professional Services": ("Consulting", "Advisory", "Partners", "Solutions", "Group"),
}


def choice(rng: np.random.Generator, weights: dict[str, float], size: int) -> np.ndarray:
    keys = list(weights)
    probs = np.array([weights[k] for k in keys], dtype=float)
    return np.array(keys, dtype=object)[rng.choice(len(keys), size=size, p=probs / probs.sum())]


@dataclass(frozen=True)
class SalesRep:
    name: str
    region: str
    stage_factor: float  # multiplies each funnel transition probability (hidden skill)


def create_sales_reps(rng: np.random.Generator, truth: GroundTruthLog) -> list[SalesRep]:
    total = sum(SALES_TEAM.values())
    firsts = rng.permutation(len(_FIRST_NAMES))[:total]
    lasts = rng.permutation(len(_LAST_NAMES))[:total]
    names = [f"{_FIRST_NAMES[first]} {_LAST_NAMES[last]}" for first, last in zip(firsts, lasts, strict=True)]
    lo, hi = FUNNEL.rep_skill_bounds
    skills = np.clip(rng.normal(1.0, FUNNEL.rep_skill_sd, total), lo, hi)

    regions = [region for region, count in SALES_TEAM.items() for _ in range(count)]
    eligible = [i for i, r in enumerate(regions) if r not in EVENTS.e4_low_conversion_rep.excluded_regions]
    low_idx = int(rng.choice(eligible))
    skills[low_idx] = EVENTS.e4_low_conversion_rep.stage_conversion_factor

    reps = [SalesRep(n, r, float(s)) for n, r, s in zip(names, regions, skills, strict=True)]
    truth.record(
        "E4",
        injected=True,
        period=None,
        sales_rep=reps[low_idx].name,
        region=reps[low_idx].region,
        stage_conversion_factor=EVENTS.e4_low_conversion_rep.stage_conversion_factor,
    )
    return reps


def company_names(rng: np.random.Generator, countries: np.ndarray, industries: np.ndarray) -> list[str]:
    used: set[str] = set()
    names: list[str] = []
    for country, industry in zip(countries, industries, strict=True):
        nouns = _INDUSTRY_NOUNS[industry]
        while True:
            stem = (
                f"{_NAME_PREFIXES[rng.integers(len(_NAME_PREFIXES))]}"
                f"{_NAME_SUFFIXES[rng.integers(len(_NAME_SUFFIXES))]} {nouns[rng.integers(len(nouns))]}"
            )
            legal = LEGAL_SUFFIX[country]
            name = f"{legal} {stem}" if legal == "PT" else f"{stem} {legal}"
            if name not in used:
                used.add(name)
                names.append(name)
                break
    return names


def initial_contract(
    rng: np.random.Generator, segments: np.ndarray, is_new: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw plan, seats, negotiated discount and MRR for each customer's first subscription."""
    n = len(segments)
    plans = np.empty(n, dtype=object)
    seats = np.zeros(n, dtype=int)
    discounts = np.zeros(n)
    for segment, params in SEGMENT_PARAMS.items():
        idx = np.flatnonzero(segments == segment)
        if idx.size == 0:
            continue
        plans[idx] = choice(rng, PLAN_MIX_BY_SEGMENT[segment], idx.size)
        raw = rng.lognormal(np.log(params.seats_median), params.seats_sigma, idx.size)
        raw = np.where(is_new[idx], raw * NEW_CUSTOMER_SEAT_FACTOR, raw)
        plan_min = np.array([PLAN_PARAMS[p].min_seats for p in plans[idx]])
        seats[idx] = np.clip(np.round(raw), np.maximum(params.seats_min, plan_min), params.seats_max)
        discounts[idx] = rng.uniform(params.discount_low, params.discount_high, idx.size)
    mrr = mrr_for(plans, seats, discounts)
    return plans, seats, discounts, mrr


def mrr_for(plans: np.ndarray, seats: np.ndarray, discounts: np.ndarray) -> np.ndarray:
    price = np.array([PLAN_PARAMS[p].price_per_seat for p in plans])
    return np.round(seats * price * (1.0 - discounts), 2)


def build_customers(
    rng: np.random.Generator,
    opening_signups: np.ndarray,
    new_signups: pd.DataFrame,
) -> pd.DataFrame:
    """Create customer firmographics and initial contracts.

    ``new_signups`` has columns ``signup_date``, ``acquisition_channel`` and
    ``acquisition_campaign_id`` (None for non-marketing channels).
    Customers are ordered by signup date and given sequential ``customer_id`` values.
    """
    n_open = len(opening_signups)
    opening = pd.DataFrame(
        {
            "signup_date": opening_signups,
            "segment": choice(rng, OPENING_SEGMENT_MIX, n_open),
            "acquisition_channel": choice(rng, OPENING_CHANNEL_MIX, n_open),
            "acquisition_campaign_id": None,
            "is_opening": True,
        }
    )
    new = new_signups.copy()
    new["segment"] = [choice(rng, SEGMENT_MIX_BY_CHANNEL[ch], 1)[0] for ch in new["acquisition_channel"].to_numpy()]
    new["is_opening"] = False
    customers = pd.concat([opening, new], ignore_index=True)

    n = len(customers)
    customers["tiebreak"] = rng.random(n)
    customers = customers.sort_values(["signup_date", "tiebreak"], kind="stable").reset_index(drop=True)
    customers = customers.drop(columns="tiebreak")
    customers.insert(0, "customer_id", [f"CUST-{i + 1:06d}" for i in range(n)])

    segments = customers["segment"].to_numpy(dtype=object)
    customers["company_size"] = [choice(rng, COMPANY_SIZE_BY_SEGMENT[s], 1)[0] for s in segments]
    customers["country"] = choice(rng, COUNTRY_WEIGHTS, n)
    customers["region"] = customers["country"].map(COUNTRY_REGION)
    customers["industry"] = choice(rng, INDUSTRY_WEIGHTS, n)
    customers["company_name"] = company_names(
        rng, customers["country"].to_numpy(dtype=object), customers["industry"].to_numpy(dtype=object)
    )
    plans, seats, discounts, mrr = initial_contract(rng, segments, ~customers["is_opening"].to_numpy())
    customers["plan"] = plans
    customers["seats"] = seats
    customers["discount"] = discounts
    customers["mrr"] = mrr
    return customers
