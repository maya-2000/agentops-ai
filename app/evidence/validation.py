"""Deterministic evidence and response validation.

``validate_evidence`` checks claims against evidence before any response is written:

1. every material numerical claim has evidence;
2. the evidence comes from an executed, successful tool call;
3. the evidence carries provenance (call, query IDs, source tables, calculation);
4. the evidence covers the period the claim is about (not stale);
5. the claim's numbers and direction do not contradict the evidence;
6. an observed fact is backed by observed evidence (anything else must be calculated or inference);
7. there is no causal language;
8. evidence reporting insufficient data does not support a claim;
9. claims resting on truncated SQL results say so;
10. the claim is about what its evidence is about: the claim's structured subject has the same metric,
    unit, period, comparison period, dimension, member and filters as the evidence it restates; every
    number it asserts comes from evidence about the same metric; and its text does not name a different
    registered KPI than its subject (a revenue claim never rests on MRR evidence, even with equal numbers).

``validate_response`` checks the generated draft against the claims (structure, not semantics):
cited claims exist and are supported, every number in the text appears in the cited evidence,
a statement that names a registered KPI cites a claim about that KPI, there is no causal language,
forecasts and anomalies are labelled as such, inferences are not presented as findings, truncated
results are not presented as complete, and an unsupported request gets no fabricated content.

Metric identity uses the typed KPI identifiers of the registry (``revenue``, ``mrr``, ``cac``, ...), never
string similarity. ``metrics_named`` recognises only the registry's display names and standard
abbreviations, longest match first, so "Net Revenue Retention" is ``nrr`` and not ``revenue``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, Field

from app.evidence.formatting import extract_numbers, is_small_count, number_is_supported
from app.evidence.models import Claim, Evidence, EvidenceGraph
from app.llm.schemas import DraftItemOutput, ResponseDraftOutput

_CAUSAL = re.compile(
    r"\b(caused|causes|causing|cause of|the cause|because|due to|led to|leads to|lead to|leading to|resulted in|"
    r"results in|resulted from|results from|result of|as a result|drove|drives|driven by|triggered|attributable to|"
    r"responsible for|the reason (?:for|why)|explains why|directly caused)\b",
    re.IGNORECASE,
)
# Wording guardrails for the response (Phase 5).
_INCREASE_WORDS = re.compile(r"\b(increas\w*|rose|risen|grew|grows|growing|climb\w*|jump\w*|surg\w*|up by)\b", re.I)
_DECREASE_WORDS = re.compile(r"\b(decreas\w*|declin\w*|fell|fall(?:s|ing|en)?|drop\w*|down by|shr[ai]nk\w*)\b", re.I)
_FORECAST_CERTAINTY = re.compile(
    r"\b(will (?:be|reach|hit|grow|fall|decline|increase|decrease|drop|rise|exceed|stay|remain|total)|"
    r"is going to|are going to|guarantee\w*|certainly|definitely|for sure|is certain)\b",
    re.IGNORECASE,
)
_JUDGEMENT_WORDS = re.compile(
    r"\b(bad|good|poor|terrible|awful|great|excellent|healthy|unhealthy|problem\w*|failure|failing|crisis|"
    r"alarming|worrying|concerning|disaster\w*|catastroph\w*|worse|better)\b",
    re.IGNORECASE,
)
_SUGGESTION_VERBS = re.compile(
    r"\b(review|investigate|check|consider|examine|compare|monitor|assess|validate|explore|analy[sz]e|"
    r"look into|confirm|verify)\b",
    re.IGNORECASE,
)
_DIRECTIVE_WORDS = re.compile(
    r"\b(immediately|must|urgent\w*|fix|guarantee\w*|will (?:fix|solve|resolve|restore|improve|stop|prevent)|"
    r"definitely|certainly|terminate|fire)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(not|no evidence|cannot|can't|does not|doesn't|do not|did not|without|unable|neither|nor)\b", re.IGNORECASE
)
_FORECAST_WORDS = re.compile(r"\b(forecast\w*|project\w*|predict\w*|expected)\b", re.IGNORECASE)
_ANOMALY_WORDS = re.compile(r"\b(unusual|anomal\w*|flagged|statistically)\b", re.IGNORECASE)
_TRUNCATION_WORDS = re.compile(r"\b(truncat\w*|partial|incomplete|first \d+ rows)\b", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_TOLERANCE = 1e-6
# Names that identify one registered KPI (display names from the registry and standard abbreviations).
# Ambiguous words ("churn", "tickets", "growth") are deliberately absent: they name no single KPI.
_KPI_TERMS: dict[str, tuple[str, ...]] = {
    "mrr": ("monthly recurring revenue", "mrr"),
    "arr": ("annual recurring revenue", "arr"),
    "revenue_growth": ("revenue growth",),
    "logo_churn_rate": ("churn rate - logo", "logo churn"),
    "revenue_churn_rate": ("churn rate - revenue", "revenue churn"),
    "retention_rate": ("retention rate",),
    "nrr": ("net revenue retention", "nrr"),
    "cac": ("customer acquisition cost", "cac"),
    "clv": ("customer lifetime value", "clv", "ltv"),
    "arpu": ("average revenue per user", "arpu"),
    "average_order_value": ("average order value", "aov"),
    "conversion_rate": ("conversion rate",),
    "pipeline_value": ("pipeline value",),
    "win_rate": ("win rate",),
    "sales_cycle": ("sales cycle",),
    "support_ticket_volume": ("support ticket volume", "ticket volume"),
    "average_resolution_time": ("average resolution time",),
    "product_adoption": ("product adoption",),
    "customer_count": ("active customer count", "customer count"),
    "revenue": ("revenue",),
}
_KPI_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    sorted(
        (
            (re.compile(rf"(?<![\w-]){re.escape(term)}(?![\w-])", re.IGNORECASE), key)
            for key, terms in _KPI_TERMS.items()
            for term in terms
        ),
        key=lambda item: -len(item[0].pattern),
    )
)


def metrics_named(text: str) -> set[str]:
    """The registered KPIs a text names (longest name first; a matched span is not re-read by shorter names)."""
    taken: list[tuple[int, int]] = []
    found: set[str] = set()
    for pattern, key in _KPI_PATTERNS:
        for match in pattern.finditer(text):
            if any(match.start() < end and start < match.end() for start, end in taken):
                continue
            taken.append(match.span())
            found.add(key)
    return found


def _kpi_subject(claim: Claim) -> str | None:
    metric = claim.subject.metric if claim.subject is not None else None
    return metric if metric in _KPI_TERMS else None


def causal_sentences(text: str) -> list[str]:
    """Sentences that assert causality (causal wording that is not negated)."""
    return [s for s in _SENTENCE.split(text) if _CAUSAL.search(s) and not _NEGATION.search(s)]


def _causal_supported(claims: Iterable[Claim]) -> bool:
    """Causal wording needs a causal basis on every cited claim (never inferred from wording)."""
    cited = list(claims)
    return bool(cited) and all(c.causal_basis != "none" for c in cited)


class EvidenceValidationResult(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    unsupported_claim_ids: list[str] = Field(default_factory=list)
    missing_evidence_claim_ids: list[str] = Field(default_factory=list)


def validate_evidence(
    graph: EvidenceGraph,
    *,
    successful_call_ids: Iterable[str],
    failed_tools: Sequence[str] = (),
    require_primary: bool = True,
) -> EvidenceValidationResult:
    executed = set(successful_call_ids)
    errors: list[str] = []
    warnings = [f"Tool call failed and contributed no evidence: {name}" for name in failed_tools]
    unsupported: list[str] = []
    missing: list[str] = []

    for evidence_id in graph.verify_integrity():
        errors.append(f"{evidence_id}: evidence content does not match its fingerprint (modified after creation)")
    tampered = set(graph.verify_integrity())
    for claim in graph.claims.values():
        problems = _claim_problems(claim, graph, executed)
        problems += [f"rests on modified evidence {e}" for e in claim.evidence_ids if e in tampered]
        if not claim.evidence_ids and (claim.claim_type != "recommendation" or _has_numbers(claim.text)):
            missing.append(claim.claim_id)
            problems.append("has no evidence")
        if problems:
            unsupported.append(claim.claim_id)
            errors.extend(f"{claim.claim_id}: {p}" for p in problems)

    supported_primary = [
        c
        for c in graph.claims.values()
        if c.primary and c.claim_id not in unsupported and c.support_status in ("supported", "partially_supported")
    ]
    if require_primary and not supported_primary:
        errors.append("No supported claim answers the question.")
    for evidence in graph.evidence.values():
        if evidence.status != "ok":
            warnings.append(f"{evidence.evidence_id}: tool reported {evidence.status}: {evidence.statement}")
        if evidence.truncated:
            warnings.append(f"{evidence.evidence_id}: SQL result truncated; rows are not complete.")
    return EvidenceValidationResult(
        valid=not errors,
        errors=errors,
        warnings=list(dict.fromkeys(warnings)),
        unsupported_claim_ids=unsupported,
        missing_evidence_claim_ids=missing,
    )


def _has_numbers(text: str) -> bool:
    return any(not is_small_count(n) for n in extract_numbers(text))


def _claim_problems(claim: Claim, graph: EvidenceGraph, executed: set[str]) -> list[str]:
    problems: list[str] = []
    evidence = [graph.evidence[e] for e in claim.evidence_ids if e in graph.evidence]
    if len(evidence) != len(claim.evidence_ids):
        problems.append("references unknown evidence")
    for e in evidence:
        if e.tool_call_id not in executed:
            problems.append(f"{e.evidence_id} does not come from a successful tool call")
        if not e.has_provenance:
            problems.append(f"{e.evidence_id} lacks provenance")
        if e.status != "ok" and claim.claim_type in ("observed_fact", "calculated_result"):
            problems.append(f"{e.evidence_id} reports {e.status}")
        if e.truncated and not any("truncat" in note.lower() for note in claim.limitations):
            problems.append(f"{e.evidence_id} is a truncated SQL result but the claim does not say so")
    if claim.claim_type == "observed_fact" and any(e.evidence_type != "observed" for e in evidence):
        problems.append("is marked observed but rests on non-observed evidence (should be calculated or inference)")
    if causal_sentences(claim.text) and claim.causal_basis == "none":
        problems.append("uses causal language the evidence does not establish")
    for e in evidence:
        problems += _typed_provenance_problems(e)
    problems += _subject_problems(claim, graph)
    if claim.about_period_start and claim.about_period_end and evidence:
        dated = [e for e in evidence if e.period_start and e.period_end]
        if dated and not any(
            e.period_start <= claim.about_period_end and e.period_end >= claim.about_period_start  # type: ignore[operator]
            for e in dated
        ):
            problems.append("evidence does not cover the period the claim is about (stale)")
    for assertion in claim.numeric_assertions:
        source = graph.evidence.get(assertion.evidence_id)
        actual = (
            None
            if source is None
            else source.value
            if assertion.field == "value"
            else source.attributes.get(assertion.field)
        )
        if not isinstance(actual, (int, float)) or abs(float(actual) - assertion.value) > _TOLERANCE * max(
            1.0, abs(float(actual))
        ):
            problems.append(
                f"states {assertion.value} for {assertion.evidence_id}.{assertion.field}, contradicting {actual}"
            )
    if claim.direction is not None and claim.direction_evidence is not None:
        source = graph.evidence.get(claim.direction_evidence.evidence_id)
        field = claim.direction_evidence.field
        actual = None if source is None else source.value if field == "value" else source.attributes.get(field)
        expected = None
        if isinstance(actual, (int, float)):
            expected = "increase" if actual > 0 else "decrease" if actual < 0 else "none"
        if expected != claim.direction:
            problems.append(f"direction {claim.direction} contradicts the evidence ({expected})")
    return problems


def _subject_problems(claim: Claim, graph: EvidenceGraph) -> list[str]:
    """The claim is about what its evidence is about: metric, unit, periods, dimension, member, filters."""
    subject = claim.subject
    if subject is None:
        return []
    problems: list[str] = []
    source = graph.evidence.get(subject.evidence_id)
    if source is None or subject.evidence_id not in claim.evidence_ids:
        problems.append(f"does not cite the evidence it is about ({subject.evidence_id})")
    else:
        problems += [
            f"is about {field} {claimed!r} but {source.evidence_id} reports {actual!r}"
            for field, claimed, actual in subject.mismatches(source)
        ]
    if subject.metric is not None:
        assertions = [*claim.numeric_assertions, *([claim.direction_evidence] if claim.direction_evidence else [])]
        for evidence_id in dict.fromkeys(a.evidence_id for a in assertions):
            item = graph.evidence.get(evidence_id)
            if item is not None and item.metric is not None and item.metric != subject.metric:
                problems.append(f"states a {item.metric} number ({evidence_id}) in a claim about {subject.metric}")
    kpi = _kpi_subject(claim)
    named = metrics_named(claim.text)
    if kpi is not None and named and kpi not in named:
        problems.append(f"names {', '.join(sorted(named))} but is about {kpi}")
    return problems


def _typed_provenance_problems(e: Evidence) -> list[str]:
    """Forecast and anomaly evidence must keep what makes it interpretable."""
    if e.evidence_type == "forecast":
        missing = [k for k in ("model", "cutoff_date", "horizon") if e.details.get(k) in (None, "")]
        if missing:
            return [f"{e.evidence_id} is a forecast without {', '.join(missing)}"]
        cutoff = str(e.details["cutoff_date"])
        if e.period_start is None or e.period_start.isoformat() <= cutoff:
            return [f"{e.evidence_id} is a forecast that does not lie after its cutoff"]
    if e.evidence_type == "anomaly":
        missing = [k for k in ("detector", "threshold", "direction") if e.details.get(k) in (None, "")]
        missing += [k for k in ("expected_value", "score") if k not in e.attributes]
        if missing or e.period_start is None:
            return [f"{e.evidence_id} is an anomaly result without {', '.join(missing) or 'a period'}"]
    return []


# ------------------------------------------------------------------------------------------------ response


class ResponseValidationResult(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    unsupported_numbers: list[str] = Field(default_factory=list)
    invalid_claim_refs: list[str] = Field(default_factory=list)


def validate_response(
    draft: ResponseDraftOutput,
    graph: EvidenceGraph,
    *,
    answerable: bool = True,
    max_chars: int = 4000,
) -> ResponseValidationResult:
    errors: list[str] = []
    unsupported_numbers: list[str] = []
    invalid_refs: list[str] = []
    sections: list[tuple[str, DraftItemOutput]] = [
        ("answer", DraftItemOutput(text=draft.answer, claim_ids=draft.answer_claim_ids))
    ]
    sections += [("key_findings", item) for item in draft.key_findings]
    sections += [("interpretation", item) for item in draft.interpretation]
    sections += [("recommendations", item) for item in draft.recommendations]

    total_chars = sum(len(item.text) for _, item in sections)
    if total_chars > max_chars:
        errors.append(f"Response is {total_chars} characters (limit {max_chars}).")

    if not answerable:
        if draft.key_findings or draft.interpretation or draft.recommendations:
            errors.append("An unanswered request must not contain findings or recommendations.")
        numbers = [n.raw for n in extract_numbers(draft.answer) if not is_small_count(n)]
        if numbers:
            errors.append(f"An unanswered request must not state business numbers: {numbers}")
        return ResponseValidationResult(valid=not errors, errors=errors, unsupported_numbers=numbers)

    if not draft.answer_claim_ids:
        errors.append("The answer does not cite any claim.")
    for section, item in sections:
        claims: list[Claim] = []
        for claim_id in item.claim_ids:
            claim = graph.claims.get(claim_id)
            if claim is None:
                invalid_refs.append(claim_id)
                errors.append(f"{section}: cites unknown claim {claim_id}")
            elif claim.support_status == "unsupported":
                invalid_refs.append(claim_id)
                errors.append(f"{section}: cites unsupported claim {claim_id}")
            else:
                claims.append(claim)
        if not item.claim_ids and section != "answer":
            errors.append(f"{section}: statement without a claim reference: {item.text[:80]!r}")
        errors.extend(_item_problems(section, item.text, claims, graph, unsupported_numbers))
    return ResponseValidationResult(
        valid=not errors,
        errors=errors,
        unsupported_numbers=unsupported_numbers,
        invalid_claim_refs=list(dict.fromkeys(invalid_refs)),
    )


def _item_problems(
    section: str, text: str, claims: list[Claim], graph: EvidenceGraph, unsupported_numbers: list[str]
) -> list[str]:
    problems: list[str] = []
    evidence: list[Evidence] = [e for c in claims for e in graph.get_supporting_evidence(c.claim_id)]
    allowed = [n for e in evidence for n in e.numbers()]
    allowed += [a.value for c in claims for a in c.numeric_assertions]
    for number in extract_numbers(text):
        if is_small_count(number):
            continue
        if not number_is_supported(number, allowed):
            unsupported_numbers.append(number.raw)
            problems.append(f"{section}: number {number.raw!r} does not appear in the cited evidence")
    cited_kpis = {kpi for c in claims if (kpi := _kpi_subject(c)) is not None}
    named = metrics_named(text)
    if cited_kpis and named and not named & cited_kpis:
        problems.append(
            f"{section}: names {', '.join(sorted(named))} but cites claims about {', '.join(sorted(cited_kpis))}"
        )
    if not _causal_supported(claims):
        for sentence in causal_sentences(text):
            problems.append(f"{section}: unsupported causal statement: {sentence[:100]!r}")
    kinds = {c.kind for c in claims}
    types = {c.claim_type for c in claims}
    if kinds & {"forecast"} and not _FORECAST_WORDS.search(text):
        problems.append(f"{section}: forecast presented without being labelled as a forecast")
    if kinds & {"forecast"} and _FORECAST_CERTAINTY.search(text):
        problems.append(f"{section}: forecast presented with certainty (a forecast is an estimate, not a fact)")
    if kinds & {"anomaly", "anomaly_summary"} and not _ANOMALY_WORDS.search(text):
        problems.append(f"{section}: anomaly result presented without being labelled as statistically unusual")
    if kinds & {"anomaly", "anomaly_summary"} and _JUDGEMENT_WORDS.search(text):
        problems.append(f"{section}: anomaly presented as a business judgement (unusual is not good or bad)")
    directions = {c.direction for c in claims if c.kind == "change" and c.direction in ("increase", "decrease")}
    says_up, says_down = bool(_INCREASE_WORDS.search(text)), bool(_DECREASE_WORDS.search(text))
    if directions == {"decrease"} and says_up and not says_down:
        problems.append(f"{section}: states an increase but the cited evidence shows a decrease")
    if directions == {"increase"} and says_down and not says_up:
        problems.append(f"{section}: states a decrease but the cited evidence shows an increase")
    if section == "recommendations":
        if not _SUGGESTION_VERBS.search(text):
            problems.append("recommendations: a recommendation must be framed as a suggested next step")
        if _DIRECTIVE_WORDS.search(text):
            problems.append("recommendations: a recommendation must not be a directive or promise an outcome")
    if section == "key_findings" and types and types <= {"inference", "recommendation"}:
        problems.append("key_findings: an inference or recommendation is presented as a finding")
    if section == "recommendations" and claims and "recommendation" not in types:
        problems.append("recommendations: recommendation does not cite a recommendation claim")
    if any(e.truncated for e in evidence) and not _TRUNCATION_WORDS.search(text):
        problems.append(f"{section}: truncated SQL result presented as complete")
    return problems
