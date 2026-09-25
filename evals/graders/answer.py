"""Answer-level checks on the Phase 4 evidence model: grounding, claim support, hallucination, causality.

The checks read the agent's structured output (response items with claim and evidence IDs, claims,
evidence, tool trace). They never parse free text for numbers the way a human would. The number,
causal-wording and labelling rules reuse the Phase 4/5 validator (``validate_response``,
``validate_evidence``, ``causal_sentences``) rather than a second evidence system. On top of
them, the evaluator adds the checks the production validators cannot make, because they need
knowledge the agent does not have:

- the expected period of the question;
- evidence that came from a tool call that did not succeed;
- sources outside the approved relations;
- hidden-label text.

``Answer`` is a neutral artifact so the same checks grade real answers and the deliberately
corrupted answers of the evidence-integrity benchmark.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.agent import AgentRunResult
from app.evidence.formatting import extract_numbers
from app.evidence.models import Claim, Evidence, EvidenceGraph
from app.evidence.validation import causal_sentences, validate_evidence, validate_response
from app.llm.schemas import DraftItemOutput, ResponseDraftOutput

CUSTOMER_ID = re.compile(r"\bCUST-\d{6}\b")


@dataclass
class Item:
    section: str
    text: str
    claim_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class Answer:
    status: str
    items: list[Item]
    claims: dict[str, Claim]
    evidence: dict[str, Evidence]
    successful_calls: set[str]
    caveats: list[str] = field(default_factory=list)

    @classmethod
    def from_result(cls, result: AgentRunResult) -> Answer:
        r = result.response
        items = [Item("answer", r.answer, list(r.answer_claim_ids), list(r.answer_evidence_ids))]
        for section, entries in (
            ("key_findings", r.key_findings),
            ("interpretation", r.interpretation),
            ("recommendations", r.recommendations),
        ):
            items += [Item(section, i.text, list(i.claim_ids), list(i.evidence_ids)) for i in entries]
        return cls(
            status=result.status,
            items=items,
            claims={c.claim_id: c for c in result.claims},
            evidence={e.evidence_id: e for e in result.evidence},
            successful_calls={c.call_id for c in result.tool_trace if c.success},
            caveats=list(r.caveats),
        )

    @property
    def text(self) -> str:
        return " ".join(i.text for i in self.items)

    @property
    def cited_claim_ids(self) -> list[str]:
        return list(dict.fromkeys(c for i in self.items for c in i.claim_ids))

    def draft(self) -> ResponseDraftOutput:
        answer = self.items[0]
        by_section: dict[str, list[DraftItemOutput]] = {"key_findings": [], "interpretation": [], "recommendations": []}
        for item in self.items[1:]:
            by_section[item.section].append(DraftItemOutput(text=item.text, claim_ids=item.claim_ids))
        return ResponseDraftOutput(
            answer=answer.text,
            answer_claim_ids=answer.claim_ids,
            key_findings=by_section["key_findings"],
            interpretation=by_section["interpretation"],
            recommendations=by_section["recommendations"],
        )

    def graph(self) -> EvidenceGraph:
        return EvidenceGraph(evidence=dict(self.evidence), claims=dict(self.claims))


@dataclass
class AnswerChecks:
    """The outcome of the answer-level checks (counts are per response item or per claim)."""

    items: int = 0
    grounded_items: int = 0
    ungrounded: list[str] = field(default_factory=list)  # reasons
    claims_cited: int = 0
    claim_support: float | None = None
    unsupported_primary: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    hallucinated_sources: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    causal_sentences: list[str] = field(default_factory=list)
    validator_errors: list[str] = field(default_factory=list)
    integrity_errors: list[str] = field(default_factory=list)

    @property
    def grounding_rate(self) -> float | None:
        return None if self.items == 0 else self.grounded_items / self.items

    @property
    def hallucinations(self) -> int:
        return len(self.unsupported_numbers) + len(self.hallucinated_sources) + len(self.unsupported_claims)


_SUPPORT_WEIGHT = {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0}


def check_answer(
    answer: Answer,
    *,
    approved_relations: frozenset[str],
    expected_periods: set[str] | None = None,
    expected_metrics: set[str] | None = None,
    answerable: bool = True,
) -> AnswerChecks:
    out = AnswerChecks()
    graph = answer.graph()
    # Integrity of the evidence itself (fingerprints) and of claims against evidence (Phase 4/5).
    out.integrity_errors = [f"{e}: fingerprint mismatch" for e in graph.verify_integrity()]
    evidence_check = validate_evidence(graph, successful_call_ids=answer.successful_calls, require_primary=False)
    response_check = validate_response(answer.draft(), graph, answerable=answerable)
    out.validator_errors = [*evidence_check.errors, *response_check.errors]
    # Digits inside customer identifiers (CUST-002529) are identifiers, not stated numbers.
    identifiers = " ".join(CUSTOMER_ID.findall(answer.text))
    out.unsupported_numbers = [n for n in response_check.unsupported_numbers if n not in identifiers]

    for item in answer.items:
        if not _material(item, answer):
            continue
        out.items += 1
        reasons = _item_problems(item, answer, approved_relations, expected_periods, expected_metrics)
        if reasons:
            out.ungrounded.extend(f"{item.section}: {r}" for r in reasons)
        else:
            out.grounded_items += 1

    cited = [answer.claims[c] for c in answer.cited_claim_ids if c in answer.claims]
    out.claims_cited = len(cited)
    if cited:
        out.claim_support = sum(_SUPPORT_WEIGHT[c.support_status] for c in cited) / len(cited)
    out.unsupported_claims = [c.claim_id for c in cited if c.support_status == "unsupported"]
    out.unsupported_primary = [c.claim_id for c in cited if c.primary and c.support_status == "unsupported"]
    unknown_claims = [c for c in answer.cited_claim_ids if c not in answer.claims]
    out.hallucinated_sources += [f"claim {c} does not exist" for c in unknown_claims]

    for evidence_id in {e for i in answer.items for e in i.evidence_ids}:
        evidence = answer.evidence.get(evidence_id)
        if evidence is None:
            out.hallucinated_sources.append(f"evidence {evidence_id} does not exist")
            continue
        if evidence.tool_call_id not in answer.successful_calls:
            out.hallucinated_sources.append(f"{evidence_id} is not from a successful tool call")
        outside = set(evidence.source_tables) - approved_relations
        if outside:
            out.hallucinated_sources.append(f"{evidence_id} cites unapproved sources {sorted(outside)}")

    supported_causal = cited and all(c.causal_basis != "none" for c in cited)
    if not supported_causal:
        out.causal_sentences = causal_sentences(answer.text)
    return out


def _item_problems(
    item: Item,
    answer: Answer,
    approved: frozenset[str],
    expected_periods: set[str] | None,
    expected_metrics: set[str] | None,
) -> list[str]:
    problems: list[str] = []
    evidence_ids = list(item.evidence_ids)
    for claim_id in item.claim_ids:
        claim = answer.claims.get(claim_id)
        if claim is None:
            problems.append(f"cites unknown claim {claim_id}")
            continue
        evidence_ids += claim.evidence_ids
    evidence_ids = list(dict.fromkeys(evidence_ids))
    if not evidence_ids:
        problems.append("states content without citing evidence")
        return problems
    items = []
    for evidence_id in evidence_ids:
        evidence = answer.evidence.get(evidence_id)
        if evidence is None:
            problems.append(f"cites evidence {evidence_id} that does not exist")
            continue
        items.append(evidence)
        if evidence.tool_call_id not in answer.successful_calls:
            problems.append(f"{evidence_id} is not from a successful tool call")
        if evidence.evidence_type != "derived" and not evidence.has_provenance:
            problems.append(f"{evidence_id} lacks provenance")
        if set(evidence.source_tables) - approved:
            problems.append(f"{evidence_id} cites unapproved sources")
        if evidence.fingerprint and evidence.fingerprint != evidence.compute_fingerprint():
            problems.append(f"{evidence_id} was modified after creation")
    if item.section == "answer":
        for evidence in items:  # every item the answer rests on must be about the question's metric and period
            if expected_metrics and evidence.metric not in expected_metrics:
                problems.append(f"{evidence.evidence_id} is about {evidence.metric}, not {sorted(expected_metrics)}")
            labels = {evidence.period_label, evidence.comparison_label} - {None}
            if (
                expected_periods
                and labels
                and not labels & expected_periods
                and not _covers(evidence, expected_periods)
            ):
                problems.append(
                    f"{evidence.evidence_id} covers {sorted(str(x) for x in labels)}, not {sorted(expected_periods)}"
                )
    return problems


def _material(item: Item, answer: Answer) -> bool:
    """An item that asserts something: it cites claims or evidence, or it is a completed answer or states numbers."""
    if not item.text.strip():
        return False
    if item.claim_ids or item.evidence_ids:
        return True
    if answer.status == "completed" and item.section == "answer":
        return True
    return bool(business_numbers(item.text))


def business_numbers(text: str) -> list[str]:
    """Numbers that look like business facts: percentages, or values of 1,000 and more (not dates or small counts)."""
    return [n.raw for n in extract_numbers(text) if n.percent or abs(n.value * n.scale) >= 1000]


def _covers(evidence: Evidence, periods: set[str]) -> bool:
    """Evidence over a range (e.g. a 12-month anomaly window) covers a month inside it."""
    if evidence.period_start is None or evidence.period_end is None:
        return False
    for label in periods:
        if re.fullmatch(r"\d{4}-\d{2}", label):
            year, month = int(label[:4]), int(label[5:])
            first = evidence.period_start.year * 12 + evidence.period_start.month
            last = evidence.period_end.year * 12 + evidence.period_end.month
            if first <= year * 12 + month <= last:
                return True
    return False
