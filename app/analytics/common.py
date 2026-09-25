"""Helpers shared by the analytics modules (not part of the public contract)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date
from typing import Any

from app.analytics.dimensions import Filters
from app.analytics.errors import UnsupportedDimensionError
from app.analytics.periods import Period, resolve_period
from app.config import get_settings

CUSTOMER_FILTER_COLUMNS: dict[str, str] = {
    key: f"c.{key}" for key in ("region", "country", "segment", "industry", "acquisition_channel", "customer_id")
}

PeriodSpec = str | Period | None


def business_as_of(as_of: date | None) -> date:
    return as_of or get_settings().as_of_date


def to_period(period: PeriodSpec, as_of: date | None) -> Period:
    return resolve_period(period, as_of=business_as_of(as_of))


def to_filters(filters: Filters | Mapping[str, str] | None) -> dict[str, str]:
    if filters is None:
        return {}
    if isinstance(filters, Filters):
        return filters.active()
    return Filters(**dict(filters)).active()


def filter_clause(filters: Mapping[str, str], columns: Mapping[str, str], operation: str) -> tuple[str, dict[str, Any]]:
    """``AND <column> = $f_<key>`` clauses for allow-listed columns, plus their bound values."""
    clauses, bind = [], {}
    for key, value in filters.items():
        if key not in columns:
            raise UnsupportedDimensionError(
                f"{operation} cannot be filtered by {key!r}. Supported filters: {', '.join(columns)}"
            )
        clauses.append(f"\n  AND {columns[key]} = $f_{key}")
        bind[f"f_{key}"] = value
    return "".join(clauses), bind


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion (95% by default). ``None`` if no trials."""
    if trials <= 0:
        return None
    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def pct_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or previous == 0:
        return None
    return current / previous - 1.0
