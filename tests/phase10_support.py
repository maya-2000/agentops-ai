"""Helpers for the Phase 10 investigation tests: investigators on scripted or deterministic agents.

No live model, network or API key: the agent's understanding step uses the deterministic model
(optionally with scripted outputs, as in the Phase 5 tests); every tool call runs on a generated dataset.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

from fastapi.testclient import TestClient

from app.evidence.models import EvidenceGraph
from app.investigation import Investigation, Investigator
from app.tools import ToolRegistry
from tests.phase5_support import runner

INVESTIGATE = "/api/v1/investigations"
INVESTIGATE_STREAM = "/api/v1/investigations/stream"

REVENUE = "Why is revenue growth slowing?"
CHURN = "Why is customer churn increasing?"
SUPPORT = "Why did support tickets increase?"
SALES = "Why did win rate drop?"
BRIEF = "Give me a management brief on the current state of the business."
PRICE_CAUSE = "Did the price increase cause churn?"

# Causal wording the brief, findings, drivers and recommendations must never use (Step 12).
CAUSAL_PHRASES = ("caused by", "because of", "resulted from", "led to", "due to")
CAUSAL_WORDS = re.compile(
    r"\b(caused|causes|because|due to|led to|leads to|resulted in|resulted from|drove|driven by|triggered)\b",
    re.IGNORECASE,
)


def investigator(
    db: Any,
    script: dict[Any, list[Any]] | None = None,
    registry: ToolRegistry | None = None,
    **config: Any,
) -> Investigator:
    """An investigator on an agent whose model steps replay ``script`` (else the deterministic model)."""
    agent, _ = runner(db, script, registry, **config)
    return Investigator(agent.runtime)


def graph_of(investigation: Investigation) -> EvidenceGraph:
    """The investigation's evidence graph, rebuilt from its (sealed) evidence and claims."""
    return EvidenceGraph(
        evidence={e.evidence_id: e for e in investigation.evidence},
        claims={c.claim_id: c for c in investigation.claims},
    )


def user_texts(investigation: Investigation) -> list[str]:
    """Every user-facing text of an investigation: summary, findings, drivers, risks, recommendations, notes."""
    texts = [f.text for f in investigation.findings]
    brief = investigation.brief
    if brief is not None:
        texts.append(brief.executive_summary)
        texts += [d.statement for d in brief.drivers + brief.contradictions + brief.context]
        texts += [r.text for r in brief.risks]
        texts += [r.text for r in brief.recommendations] + [r.rationale for r in brief.recommendations]
        texts += brief.uncertainty
    if investigation.message:
        texts.append(investigation.message)
    return texts


def investigate(client: TestClient, objective: str, **body: Any) -> dict[str, Any]:
    headers = body.pop("headers", None)
    response = client.post(INVESTIGATE, json={"objective": objective, **body}, headers=headers)
    assert response.status_code == 200, response.text
    data: dict[str, Any] = response.json()
    return data


def stream_lines(client: TestClient, objective: str, **body: Any) -> Iterator[dict[str, Any]]:
    headers = body.pop("headers", None)
    with client.stream("POST", INVESTIGATE_STREAM, json={"objective": objective, **body}, headers=headers) as response:
        assert response.status_code == 200, response.read()
        assert response.headers["content-type"].startswith("application/x-ndjson")
        for line in response.iter_lines():
            if line.strip():
                yield json.loads(line)
