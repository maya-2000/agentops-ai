"""Builders for Phase 4 unit tests (evidence, claims and graphs without a database)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

from app.agent.graph import business_context
from app.evidence.models import Claim, Evidence, EvidenceGraph, NumericAssertion

AUGUST = (date(2026, 8, 1), date(2026, 8, 31))
JULY = (date(2026, 7, 1), date(2026, 7, 31))
AS_OF = date(2026, 8, 31)
FEATURES = ["API Access", "Alerts", "Dashboards", "Reports", "Scheduled Exports"]


def understanding_context(question: str, as_of: date = AS_OF) -> dict[str, Any]:
    """The context the agent gives the understanding step (vocabulary only), built without a database."""
    return {"question": question, "as_of": as_of.isoformat(), **business_context(FEATURES)}


def evidence(evidence_id: str, value: Any = 100.0, **overrides: Any) -> Evidence:
    data: dict[str, Any] = {
        "evidence_id": evidence_id,
        "evidence_type": "observed",
        "statement": f"Metric for 2026-08: {value}.",
        "metric": "revenue",
        "value": value,
        "unit": "SGD",
        "period_label": "2026-08",
        "period_start": AUGUST[0],
        "period_end": AUGUST[1],
        "tool_name": "get_kpi",
        "tool_call_id": "T1",
        "operation": "get_kpi",
        "query_ids": ["Q-test"],
        "source_tables": ["daily_revenue"],
        "calculation": "SUM(revenue)",
        "execution_timestamp": datetime(2026, 9, 1, tzinfo=UTC),
    }
    data.update(overrides)
    return Evidence(**data)


def claim(claim_id: str, text: str, evidence_ids: list[str], **overrides: Any) -> Claim:
    data: dict[str, Any] = {
        "claim_id": claim_id,
        "text": text,
        "claim_type": "observed_fact",
        "evidence_ids": evidence_ids,
        "primary": True,
    }
    data.update(overrides)
    return Claim(**data)


def graph(*items: Evidence, claims: tuple[Claim, ...] = ()) -> EvidenceGraph:
    g = EvidenceGraph()
    for item in items:
        g.add_evidence(item)
    for c in claims:
        g.add_claim(c)
    return g


def assertion(evidence_id: str, value: float, field: str = "value") -> NumericAssertion:
    return NumericAssertion(evidence_id=evidence_id, field=field, value=value)
