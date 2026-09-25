"""Deterministic response composition (the offline model's writer).

It arranges already-validated claims into the response structure. The direct answer uses the
primary claims, key findings use the other observed and calculated claims, the interpretation uses
inferences and the recommendations use recommendation claims. Claim text is used verbatim, so every
number in the response is the number the evidence holds.
"""

from __future__ import annotations

from typing import Any

from app.llm.schemas import DraftItemOutput, ResponseDraftOutput

_ANSWER_ORDER = (
    "causality_limit",
    "change",
    "concentration",
    "contribution",
    "ranking",
    "kpi_value",
    "forecast",
    "anomaly_summary",
    "fact",
)
_FINDING_ORDER = (
    "contribution",
    "bridge_component",
    "change",
    "anomaly_summary",
    "ranking",
    "anomaly",
    "kpi_value",
    "forecast",
    "forecast_quality",
    "association_data",
    "fact",
)
MAX_ANSWER_CLAIMS = 3
MAX_FINDINGS = 8
MAX_INTERPRETATION = 5
MAX_RECOMMENDATIONS = 3


def _rank(order: tuple[str, ...], kind: str) -> int:
    return order.index(kind) if kind in order else len(order)


def compose(context: dict[str, Any]) -> dict[str, Any]:
    claims = [c for c in context.get("claims", []) if c.get("support_status") != "unsupported"]
    primary = sorted(
        (c for c in claims if c.get("primary")), key=lambda c: (_rank(_ANSWER_ORDER, c["kind"]), c["claim_id"])
    )
    answer_claims = primary[:MAX_ANSWER_CLAIMS]
    if any(c["kind"] == "forecast" for c in answer_claims):
        answer_claims = [c for c in primary if c["kind"] == "forecast"][:6]
    used = {c["claim_id"] for c in answer_claims}
    findings = sorted(
        (c for c in claims if c["claim_id"] not in used and c["type"] in ("observed_fact", "calculated_result")),
        key=lambda c: (_rank(_FINDING_ORDER, c["kind"]), c["claim_id"]),
    )[:MAX_FINDINGS]
    inferences = [c for c in claims if c["type"] == "inference" and c["claim_id"] not in used][:MAX_INTERPRETATION]
    recommendations = [c for c in claims if c["type"] == "recommendation"][:MAX_RECOMMENDATIONS]
    return ResponseDraftOutput(
        answer=" ".join(c["text"] for c in answer_claims),
        answer_claim_ids=[c["claim_id"] for c in answer_claims],
        key_findings=[DraftItemOutput(text=c["text"], claim_ids=[c["claim_id"]]) for c in findings],
        interpretation=[DraftItemOutput(text=c["text"], claim_ids=[c["claim_id"]]) for c in inferences],
        recommendations=[DraftItemOutput(text=c["text"], claim_ids=[c["claim_id"]]) for c in recommendations],
    ).model_dump()
