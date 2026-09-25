"""Expected values from the independent reference implementations (no production analytics code).

The numbers come from ``tests/integration/reference_kpis.py`` (``Reference``). This is the Phase 2
pandas re-implementation of every KPI, written from the definitions and sharing no code with
``app/analytics``. Forecasts and anomaly scores are checked with ``tests/reference_timeseries.py``,
the Phase 3 textbook reference. This module only maps a KPI key to the reference call, as the
Phase 2 correctness test does, and derives per-member values and monthly series from it.

Tolerances (documented in ``docs/evaluation.md``):

- ``money``: SGD 0.01 or 1e-9 relative. This is the Phase 2 KPI correctness tolerance; the two
  sides differ only by float versus DECIMAL summation order.
- ``rate``: 1e-9 absolute. Rates are ratios of exact integer counts; any real difference is at
  least one customer or deal, which is several orders of magnitude larger.
- ``count``: exact.
- ``duration``: 1e-6 hours or days. These are means of exact values.
- ``forecast``: 1e-6 relative. Naive and drift forecasts are the same arithmetic on the same
  series; only float rounding differs.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date, timedelta
from typing import TYPE_CHECKING

from evals.reference.periods import PeriodRange, month_ranges

if TYPE_CHECKING:
    from tests.integration.reference_kpis import Reference

Tolerance = str
KPI_TOLERANCE: dict[str, Tolerance] = {
    "revenue": "money",
    "mrr": "money",
    "arr": "money",
    "arpu": "money",
    "cac": "money",
    "clv": "money",
    "average_order_value": "money",
    "pipeline_value": "money",
    "customer_count": "count",
    "support_ticket_volume": "count",
    "logo_churn_rate": "rate",
    "retention_rate": "rate",
    "revenue_churn_rate": "rate",
    "nrr": "rate",
    "conversion_rate": "rate",
    "win_rate": "rate",
    "product_adoption": "rate",
    "revenue_growth": "rate",
    "sales_cycle": "duration",
    "average_resolution_time": "duration",
}


def within(actual: float | None, expected: float | None, tolerance: Tolerance) -> bool:
    """Whether ``actual`` agrees with the reference under the named tolerance."""
    if actual is None or expected is None:
        return actual is None and expected is None
    if math.isnan(expected) or math.isnan(actual):
        return False
    diff = abs(actual - expected)
    if tolerance == "money":
        return diff <= max(0.01, abs(expected) * 1e-9)
    if tolerance == "rate":
        return diff <= 1e-9
    if tolerance == "count":
        return diff == 0
    if tolerance == "duration":
        return diff <= 1e-6
    if tolerance == "forecast":
        return diff <= max(1e-6, abs(expected) * 1e-6)
    raise ValueError(f"Unknown tolerance {tolerance!r}")


def _months(start: date, end: date) -> float:
    """Whole calendar months, else days / (365.25 / 12) (the Phase 2 reference convention)."""
    if start.day == 1 and (end + timedelta(days=1)).day == 1:
        return float((end.year - start.year) * 12 + end.month - start.month + 1)
    return ((end - start).days + 1) / (365.25 / 12)


Getter = Callable[["Reference", date, date, dict[str, str]], float]

_KPIS: dict[str, Getter] = {
    "revenue": lambda r, s, e, f: r.revenue(s, e, f)["revenue"],
    "mrr": lambda r, s, e, f: r.recurring_state(e, f)["mrr"],
    "arr": lambda r, s, e, f: 12 * r.recurring_state(e, f)["mrr"],
    "arpu": lambda r, s, e, f: r.recurring_state(e, f)["mrr"] / r.recurring_state(e, f)["customers"],
    "customer_count": lambda r, s, e, f: float(r.recurring_state(e, f)["customers"]),
    "logo_churn_rate": lambda r, s, e, f: r.logo_churn(s, e, f),
    "retention_rate": lambda r, s, e, f: 1 - r.logo_churn(s, e, f),
    "revenue_churn_rate": lambda r, s, e, f: r.cohort(s, e, f)["churned_mrr"] / r.cohort(s, e, f)["opening_mrr"],
    "nrr": lambda r, s, e, f: r.nrr(s, e, f),
    "clv": lambda r, s, e, f: r.clv(s, e, _months(s, e), f),
    "cac": lambda r, s, e, f: r.marketing(s, e, f)["spend"] / r.marketing(s, e, f)["conversions"],
    "conversion_rate": lambda r, s, e, f: r.marketing(s, e, f)["conversions"] / r.marketing(s, e, f)["leads"],
    "win_rate": lambda r, s, e, f: r.win_rate(s, e, f),
    "average_order_value": lambda r, s, e, f: r.average_order_value(s, e, f),
    "sales_cycle": lambda r, s, e, f: r.sales_cycle(s, e, f),
    "pipeline_value": lambda r, s, e, f: r.pipeline(e, f),
    "support_ticket_volume": lambda r, s, e, f: float(len(r.tickets(s, e, f))),
    "average_resolution_time": lambda r, s, e, f: float(
        r.tickets(s, e, f).query("status == 'Resolved'")["resolution_time"].mean()
    ),
    "product_adoption": lambda r, s, e, f: r.adoption(s, e, f["product_feature"]),
}


def kpi(ref: Reference, key: str, period: PeriodRange, filters: dict[str, str] | None = None) -> float:
    """The reference value of a registered KPI for a period."""
    if key not in _KPIS:
        raise KeyError(f"No independent reference for KPI {key!r}")
    return float(_KPIS[key](ref, period.start, period.end, dict(filters or {})))


def dimension_members(ref: Reference, dimension: str) -> list[str]:
    """Observed members of a dimension, read from the raw reference tables."""
    tables: dict[str, tuple[str, str]] = {
        "region": ("customers", "region"),
        "country": ("customers", "country"),
        "segment": ("customers", "segment"),
        "industry": ("customers", "industry"),
        "acquisition_channel": ("marketing_campaigns", "channel"),
        "campaign": ("marketing_campaigns", "campaign_id"),
        "sales_rep": ("sales_opportunities", "sales_rep"),
        "product_feature": ("product_features", "feature_name"),
    }
    table, column = tables[dimension]
    return sorted(str(v) for v in ref.t[table][column].dropna().unique())


def member_values(
    ref: Reference, key: str, dimension: str, period: PeriodRange, eligible: Callable[[str], bool] | None = None
) -> dict[str, float]:
    """The KPI per member of ``dimension`` (members where the KPI is undefined are skipped)."""
    values: dict[str, float] = {}
    for member in dimension_members(ref, dimension):
        if eligible is not None and not eligible(member):
            continue
        try:
            value = kpi(ref, key, period, {dimension: member})
        except ZeroDivisionError:
            continue
        if not math.isnan(value):
            values[member] = value
    return values


def revenue_changes(ref: Reference, dimension: str, current: PeriodRange, previous: PeriodRange) -> dict[str, float]:
    """Revenue change (current - previous) per member of a customer dimension."""
    return {
        member: kpi(ref, "revenue", current, {dimension: member}) - kpi(ref, "revenue", previous, {dimension: member})
        for member in dimension_members(ref, dimension)
    }


def monthly_series(
    ref: Reference, metric: str, start: date, end: date, filters: dict[str, str] | None = None
) -> list[float]:
    """The monthly series of a time-series metric: flows per month, stocks at month end."""
    return [kpi(ref, metric, month, filters) for month in month_ranges(start, end)]
