"""Evidence-integrity benchmark: corrupt a real agent answer and check that every corruption is detected.

The agent answers the scenario's question normally. Each mutation then produces a deliberately
broken copy of that answer, which is checked twice:

- by the production validators (Phase 4/5: ``validate_evidence``, ``validate_response``,
  evidence fingerprints);
- by the evaluator's own grounding checks (``evals.graders.answer``).

A mutation that slips past the production validators is a production finding. One that slips
past the evaluator is a defect in the benchmark itself.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from evals.graders.answer import Answer, Item
from evals.scenarios.model import Mutation


@dataclass
class MutatedAnswer:
    mutation: Mutation
    answer: Answer
    applicable: bool = True
    note: str = ""


def _primary(answer: Answer) -> str | None:
    head = answer.items[0]
    return head.claim_ids[0] if head.claim_ids else None


def _first_evidence(answer: Answer) -> str | None:
    claim = _primary(answer)
    if claim is None or not answer.claims[claim].evidence_ids:
        return None
    return answer.claims[claim].evidence_ids[0]


def _fake_evidence_id(answer: Answer) -> Answer:
    claim = _primary(answer)
    assert claim is not None
    answer.claims[claim] = answer.claims[claim].model_copy(update={"evidence_ids": ["E9999"]})
    answer.items[0].evidence_ids = ["E9999"]
    return answer


def _missing_evidence(answer: Answer) -> Answer:
    evidence_id = _first_evidence(answer)
    assert evidence_id is not None
    answer.evidence.pop(evidence_id)
    return answer


def _wrong_evidence(answer: Answer) -> Answer:
    """The answer's primary claim now cites a different evidence item that does not state its numbers."""
    claim_id = _primary(answer)
    assert claim_id is not None
    claim = answer.claims[claim_id]
    cited_values = {answer.evidence[e].value for e in claim.evidence_ids if e in answer.evidence}
    others = [
        e
        for e, item in answer.evidence.items()
        if e not in claim.evidence_ids and isinstance(item.value, (int, float)) and item.value not in cited_values
    ]
    if not others:
        raise LookupError("no other numeric evidence to point at")
    answer.claims[claim_id] = claim.model_copy(update={"evidence_ids": [others[-1]]})
    head = answer.items[0]
    answer.items[0] = Item(head.section, head.text, [claim_id], [others[-1]])
    return answer


def _mismatched_period(answer: Answer) -> Answer:
    evidence_id = _first_evidence(answer)
    assert evidence_id is not None
    old = answer.evidence[evidence_id]
    update: dict[str, Any] = {
        "period_label": "2019-01",
        "period_start": date(2019, 1, 1),
        "period_end": date(2019, 1, 31),
        "fingerprint": "",
    }
    if old.comparison_label is not None:
        update |= {
            "comparison_label": "2018-12",
            "comparison_start": date(2018, 12, 1),
            "comparison_end": date(2018, 12, 31),
        }
    moved = old.model_copy(update=update)
    answer.evidence[evidence_id] = moved.model_copy(update={"fingerprint": moved.compute_fingerprint()})
    return answer


def _mismatched_metric(answer: Answer) -> Answer:
    evidence_id = _first_evidence(answer)
    assert evidence_id is not None
    old = answer.evidence[evidence_id]
    renamed = old.model_copy(update={"metric": "invented_metric", "fingerprint": ""})
    answer.evidence[evidence_id] = renamed.model_copy(update={"fingerprint": renamed.compute_fingerprint()})
    return answer


def _fabricated_number(answer: Answer) -> Answer:
    head = answer.items[0]
    answer.items[0] = Item(
        head.section, head.text + " It also grew by 37.2% to SGD 9,876,543.", head.claim_ids, head.evidence_ids
    )
    return answer


def _missing_provenance(answer: Answer) -> Answer:
    evidence_id = _first_evidence(answer)
    assert evidence_id is not None
    old = answer.evidence[evidence_id]
    stripped = old.model_copy(update={"query_ids": [], "source_tables": [], "fingerprint": ""})
    answer.evidence[evidence_id] = stripped.model_copy(update={"fingerprint": stripped.compute_fingerprint()})
    return answer


def _causal_overstatement(answer: Answer) -> Answer:
    head = answer.items[0]
    answer.items[0] = Item(
        head.section, head.text + " This was caused by customer churn.", head.claim_ids, head.evidence_ids
    )
    return answer


def _tampered_evidence(answer: Answer) -> Answer:
    evidence_id = _first_evidence(answer)
    assert evidence_id is not None
    old = answer.evidence[evidence_id]
    value = old.value if isinstance(old.value, (int, float)) else 0.0
    answer.evidence[evidence_id] = old.model_copy(update={"value": float(value) * 1.5 + 1})  # fingerprint not resealed
    return answer


MUTATORS: dict[str, Callable[[Answer], Answer]] = {
    "fake_evidence_id": _fake_evidence_id,
    "missing_evidence": _missing_evidence,
    "wrong_evidence": _wrong_evidence,
    "mismatched_period": _mismatched_period,
    "mismatched_metric": _mismatched_metric,
    "fabricated_number": _fabricated_number,
    "missing_provenance": _missing_provenance,
    "causal_overstatement": _causal_overstatement,
    "tampered_evidence": _tampered_evidence,
}


def mutate(answer: Answer, mutation: Mutation) -> MutatedAnswer:
    clone = copy.deepcopy(answer)
    try:
        return MutatedAnswer(mutation, MUTATORS[mutation](clone))
    except (AssertionError, LookupError) as exc:
        return MutatedAnswer(mutation, clone, applicable=False, note=f"not applicable to this answer: {exc}")
