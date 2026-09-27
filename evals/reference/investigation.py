"""Evaluation-side expectations for investigations (eval_v2), independent of the investigation code.

- **Periods.** Relative period specs (``last_month``, ``last_quarter``, ``previous``) are resolved from
  the as-of date with the evaluation's own calendar arithmetic (``evals.reference.periods``).
- **Expected co-movement.** The sign with which an indicator moves relative to an outcome follows
  from the KPI definitions: churn moves against revenue, retention with it, and so on. The table is
  written here from those definitions. A ``supports`` or ``contradicts`` driver is graded by the sign
  of its evidence's own change against this table, not by the direction the system reports.
- **Reference values.** The outcome's values in both periods and each member's revenue change come from
  the independent pandas reference (``evals.reference.kpis``), never from production analytics.
- **Wording.** Causal phrases and suggestion verbs are the evaluation's own patterns.
"""

from __future__ import annotations

import calendar
import re
from datetime import date
from typing import Any

from evals.reference import kpis
from evals.reference.context import EvalContext
from evals.reference.periods import parse_period, shift_months

_REVENUE = {
    "logo_churn_rate": -1,
    "revenue_churn_rate": -1,
    "nrr": 1,
    "retention_rate": 1,
    "customer_count": 1,
    "win_rate": 1,
    "average_order_value": 1,
    "sales_cycle": -1,
    "pipeline_value": 1,
    "conversion_rate": 1,
    "product_adoption": 1,
}
_CHURN = {"nrr": -1, "retention_rate": -1, "customer_count": -1, "product_adoption": -1, "support_ticket_volume": 1}
EXPECTED_SIGN: dict[str, dict[str, int]] = {
    "revenue": _REVENUE,
    "mrr": _REVENUE,
    "arr": _REVENUE,
    "revenue_growth": _REVENUE,
    "logo_churn_rate": _CHURN,
    "revenue_churn_rate": _CHURN,
    "nrr": {"logo_churn_rate": -1, "revenue_churn_rate": -1, "customer_count": 1},
    "retention_rate": {"logo_churn_rate": -1, "revenue_churn_rate": -1, "nrr": 1, "customer_count": 1},
    "customer_count": {"logo_churn_rate": -1, "nrr": 1},
    "win_rate": {"sales_cycle": -1, "pipeline_value": 1},
    "pipeline_value": {"win_rate": 1},
    "support_ticket_volume": {"tickets_per_active_customer": 1, "customer_count": 1},
    "average_resolution_time": {"support_ticket_volume": 1, "tickets": 1},
}
NON_CAUSAL_RELATIONSHIPS = frozenset({"supports", "correlates_with", "contributes_to", "contradicts", "contextualizes"})
CAUSAL_PATTERN = re.compile(
    r"\b(caused by|caused|causes|causing|because of|because|due to|led to|leads to|resulted from|resulted in|"
    r"results in|drove|drives|driven by|triggered|attributable to|responsible for|the reason (?:for|why))\b",
    re.IGNORECASE,
)
NEGATION = re.compile(r"\b(not|cannot|can't|no|does not|do not|without)\b", re.IGNORECASE)
SUGGESTION = re.compile(r"^(review|investigate|check|compare|examine|analy[sz]e|monitor|validate|assess)\b", re.I)
# Claims a management brief must not make without data (Step 35): invented priorities, impact, sentiment, markets.
UNSUPPORTED_TOPICS = re.compile(
    r"\b(customer sentiment|market share|competitor\w*|financial impact|top priority|priorit(?:y|ies) (?:is|are)|"
    r"will (?:grow|decline|increase|decrease|recover))\b",
    re.IGNORECASE,
)
# Fixed statements about the investigation process itself, not about the business (the budget stop is worded
# as the specification requires); they are not causal claims about the data.
SYSTEM_STATEMENTS = ("Investigation stopped because the analysis budget was reached.",)
REASONING_MARKERS = re.compile(r"\b(i think|let me|chain of thought|my reasoning|step by step|thought:)\b", re.I)


def _month_end(d: date) -> bool:
    return d.day == calendar.monthrange(d.year, d.month)[1]


def last_complete_month(as_of: date) -> str:
    label = f"{as_of.year}-{as_of.month:02d}"
    return label if _month_end(as_of) else shift_months(label, -1)


def last_complete_quarter(as_of: date) -> str:
    quarter = (as_of.month - 1) // 3 + 1
    year = as_of.year
    if not (as_of.month % 3 == 0 and _month_end(as_of)):
        quarter -= 1
        if quarter == 0:
            year, quarter = year - 1, 4
    return f"{year}-Q{quarter}"


def previous_label(label: str) -> str:
    """The period of the same length immediately before ``label`` (a month or a quarter)."""
    if "-Q" in label:
        year, quarter = int(label[:4]), int(label[-1])
        return f"{year - 1}-Q4" if quarter == 1 else f"{year}-Q{quarter - 1}"
    return shift_months(label, -1)


def resolve_period(spec: str | None, as_of: date) -> str | None:
    if spec is None:
        return None
    if spec == "last_month":
        return last_complete_month(as_of)
    if spec == "last_quarter":
        return last_complete_quarter(as_of)
    parse_period(spec)  # an explicit label must be valid
    return spec


def resolve_comparison(spec: str | None, period: str | None) -> str | None:
    if spec is None:
        return None
    if spec == "previous":
        assert period is not None, "a relative comparison needs a period"
        return previous_label(period)
    parse_period(spec)
    return spec


def outcome_reference(
    ctx: EvalContext, metric: str, period: str, comparison: str, filters: dict[str, str]
) -> dict[str, Any]:
    current = kpis.kpi(ctx.reference, metric, parse_period(period), filters)
    previous = kpis.kpi(ctx.reference, metric, parse_period(comparison), filters)
    return {
        "current": current,
        "previous": previous,
        "direction": "increase" if current > previous else "decrease" if current < previous else "none",
        "tolerance": kpis.KPI_TOLERANCE[metric],
    }


def top_contribution_reference(ctx: EvalContext, dimension: str, period: str, comparison: str) -> dict[str, Any]:
    """Each member's revenue change, and the leading member and share in each direction."""
    changes = kpis.revenue_changes(ctx.reference, dimension, parse_period(period), parse_period(comparison))
    declines = {m: c for m, c in changes.items() if c < 0}
    increases = {m: c for m, c in changes.items() if c > 0}
    result: dict[str, Any] = {"changes": changes}
    if declines:
        member = min(declines, key=lambda m: declines[m])
        result["decrease"] = {"member": member, "share": declines[member] / sum(declines.values())}
    if increases:
        member = max(increases, key=lambda m: increases[m])
        result["increase"] = {"member": member, "share": increases[member] / sum(increases.values())}
    return result


def within(actual: float | None, expected: float | None, tolerance: str) -> bool:
    return kpis.within(actual, expected, tolerance)


def causal_assertions(text: str) -> list[str]:
    """Sentences that assert a cause about the business (causal wording that is not negated)."""
    for statement in SYSTEM_STATEMENTS:
        text = text.replace(statement, "")
    sentences = re.split(r"(?<=[.!?;])\s+", text)
    return [s for s in sentences if CAUSAL_PATTERN.search(s) and not NEGATION.search(s)]
