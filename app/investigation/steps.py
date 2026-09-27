"""Step dependencies: named conditions and evidence-bound arguments (a closed set, evaluated in code).

A dependent step runs only when every step it depends on completed. A ``condition`` is one of the
named checks below, read from the dependencies' evidence; a ``binding`` reads one argument (a
dimension member or a feature name) from that evidence. There are no expressions, no model output
and no arbitrary code: an unknown condition or binding cannot be represented (``Literal`` types).
The bound value is still an ordinary tool argument, validated and authorised like any other.
"""

from __future__ import annotations

from typing import Any

from app.agent.findings import CONCENTRATION_SHARE
from app.evidence.models import Evidence
from app.investigation.models import StepBinding, StepCondition

SKIP_REASONS: dict[str, str] = {
    "dependency": "Skipped: an earlier step it depends on did not complete.",
    "outcome_changed": "Skipped: no change was measured, so there is nothing to decompose.",
    "concentrated": "Skipped: the change was not concentrated in one member, so no drill-down was needed.",
    "binding": "Skipped: the earlier result did not identify a value to analyse.",
}


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _change(evidence: list[Evidence]) -> float | None:
    """The headline change of a step (a total, not a dimension member)."""
    for e in evidence:
        if e.dimension_value is not None or e.status != "ok":
            continue
        for field in ("absolute_change", "percentage_change"):
            value = _number(e.attributes.get(field))
            if value is not None:
                return value
    return None


def concentrated_member(evidence: list[Evidence]) -> Evidence | None:
    """The rank-1 member of a decomposition when it accounts for at least ``CONCENTRATION_SHARE`` of the change."""
    for e in evidence:
        if e.dimension_value is None or e.status != "ok" or e.attributes.get("rank") != 1:
            continue
        share = _number(e.attributes.get("share_of_gross_decline"))
        if share is None:
            share = _number(e.attributes.get("share_of_gross_increase"))
        if share is not None and share >= CONCENTRATION_SHARE:
            return e
    return None


def condition_met(condition: StepCondition, evidence: list[Evidence]) -> bool:
    if condition == "outcome_changed":
        change = _change(evidence)
        return change is not None and change != 0
    return concentrated_member(evidence) is not None


def bind(binding: StepBinding, evidence: list[Evidence], arguments: dict[str, Any]) -> dict[str, Any] | None:
    """The step's arguments with the bound value, or ``None`` when the evidence does not provide one."""
    filters = dict(arguments.get("filters") or {})
    if binding.kind == "concentrated_member":
        member = concentrated_member(evidence)
        if member is None or member.dimension is None or member.dimension_value is None:
            return None
        filters[member.dimension] = member.dimension_value
    else:
        features = [e.dimension_value for e in evidence if e.dimension_value is not None and e.status == "ok"]
        if len(features) < binding.index:
            return None
        filters["product_feature"] = str(features[binding.index - 1])
    return {**arguments, "filters": filters}
