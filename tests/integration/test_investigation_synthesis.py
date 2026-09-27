"""Phase 10: cross-finding validation, driver analysis, grounded recommendations and causal guardrails.

The validators are exercised on real investigations (their findings, evidence graph and plan) with one
defect introduced at a time: each defect must be removed or downgraded and recorded, never repaired.
Driver and recommendation checks run on the full generated dataset.
"""

from __future__ import annotations

import pytest

from app.database.base import Database
from app.evidence.validation import causal_sentences, recommendation_wording_problems
from app.investigation.drivers import EXPECTED_CO_MOVEMENT
from app.investigation.models import (
    Driver,
    Finding,
    FindingRelationship,
    Investigation,
    Recommendation,
)
from app.investigation.recommendations import ADVERSE_DIRECTION, MAX_RECOMMENDATIONS
from app.investigation.validation import (
    NON_CAUSAL,
    text_problems,
    validate_drivers,
    validate_findings,
    validate_recommendations,
    validate_relationships,
)
from tests.phase10_support import (
    BRIEF,
    CAUSAL_PHRASES,
    CHURN,
    REVENUE,
    SALES,
    graph_of,
    investigator,
    user_texts,
)

SIGN = {"increase": 1, "decrease": -1}


@pytest.fixture(scope="module")
def small(small_db: Database) -> Investigation:
    return investigator(small_db).investigate(REVENUE)


@pytest.fixture(scope="module")
def full(full_db: Database) -> dict[str, Investigation]:
    agent = investigator(full_db)
    return {objective: agent.investigate(objective) for objective in (REVENUE, CHURN, SALES, BRIEF)}


def successful(investigation: Investigation) -> list[str]:
    return [c.call_id for c in investigation.tool_trace if c.success]


def revalidate(investigation: Investigation, findings: list[Finding], graph: object = None) -> tuple[list, list]:
    assert investigation.plan is not None
    return validate_findings(
        findings,
        graph or graph_of(investigation),  # type: ignore[arg-type]
        investigation.plan,
        successful_call_ids=successful(investigation),
    )


# ------------------------------------------------------------------ cross-finding validation (Step 10)


def test_valid_findings_pass_unchanged(small: Investigation) -> None:
    kept, issues = revalidate(small, small.findings)
    assert kept == small.findings and not issues


def test_a_finding_citing_unknown_evidence_is_removed(small: Investigation) -> None:
    broken = small.findings[0].model_copy(update={"evidence_ids": ["E9999"]})
    kept, issues = revalidate(small, [broken])
    assert not kept and issues[0].action == "removed" and "unknown evidence" in issues[0].reason


def test_a_finding_on_modified_evidence_is_removed(small: Investigation) -> None:
    graph = graph_of(small)
    finding = small.findings[0]
    evidence = graph.evidence[finding.evidence_ids[0]]
    graph.evidence[evidence.evidence_id] = evidence.model_copy(update={"value": 123456789.0})
    kept, issues = revalidate(small, [finding], graph)
    assert not kept and "modified" in issues[0].reason


def test_a_finding_from_a_failed_step_is_removed(small: Investigation) -> None:
    finding = small.findings[0]
    assert small.plan is not None
    kept, issues = validate_findings([finding], graph_of(small), small.plan, successful_call_ids=[])
    assert not kept and "successful step" in issues[0].reason


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("metric", "win_rate"),
        ("unit", "percent"),
        ("period", "2025-01"),
        ("comparison_period", "2024-12"),
        ("breakdown", "Atlantis"),
        ("filters", {"segment": "Enterprise"}),
    ],
)
def test_a_finding_whose_identity_differs_from_its_evidence_is_removed(
    small: Investigation, field: str, value: object
) -> None:
    outcome = next(f for f in small.findings if f.primary)
    kept, issues = revalidate(small, [outcome.model_copy(update={field: value})])
    assert not kept and f"its {field}" in issues[0].reason


def test_a_comparison_outside_the_investigation_periods_is_removed(small: Investigation) -> None:
    assert small.plan is not None
    outcome = next(f for f in small.findings if f.primary)
    other = small.plan.model_copy(update={"period_label": "2026-06", "comparison_label": "2026-05"})
    kept, issues = validate_findings([outcome], graph_of(small), other, successful_call_ids=successful(small))
    assert not kept and "not the investigation's" in issues[0].reason


@pytest.mark.parametrize(
    "text",
    [
        "Revenue fell by 987,654,321 in 2026-08.",  # a number that is not in the evidence
        "Revenue declined because of churn.",  # causal wording
        "Win rate declined in 2026-08.",  # names only a KPI the evidence is not about
    ],
)
def test_a_finding_whose_text_is_not_supported_is_removed(small: Investigation, text: str) -> None:
    outcome = next(f for f in small.findings if f.primary)
    graph = graph_of(small)
    graph.claims[outcome.claim_id] = graph.claims[outcome.claim_id].model_copy(update={"text": text})
    kept, issues = revalidate(small, [outcome.model_copy(update={"text": text})], graph)
    assert not kept and issues[0].action == "removed"


def test_a_partially_supported_finding_is_downgraded_not_removed(small: Investigation) -> None:
    graph = graph_of(small)
    finding = next(f for f in small.findings if f.confidence != "low")
    graph.claims[finding.claim_id] = graph.claims[finding.claim_id].model_copy(
        update={"support_status": "partially_supported"}
    )
    kept, issues = revalidate(small, [finding], graph)
    assert kept[0].confidence == "low" and issues[0].action == "downgraded"
    assert finding.confidence != "low"  # the original is not mutated


def test_relationships_need_surviving_findings_about_the_same_periods(small: Investigation) -> None:
    outcome = next(f for f in small.findings if f.primary)
    other = next(f for f in small.findings if f is not outcome and f.comparison_period == outcome.comparison_period)
    good = FindingRelationship(
        source_finding_id=other.finding_id, target_finding_id=outcome.finding_id, relationship="supports", rule="t"
    )
    ghost = good.model_copy(update={"source_finding_id": "F999"})
    shifted = other.model_copy(update={"finding_id": "F998", "period": "2025-01"})
    elsewhere = good.model_copy(update={"source_finding_id": "F998"})
    kept, issues = validate_relationships([good, ghost, elsewhere], [outcome, other, shifted])
    assert kept == [good]
    assert {i.item_id for i in issues} == {f"F999->{outcome.finding_id}", f"F998->{outcome.finding_id}"}
    assert {"supports", "correlates_with", "contributes_to", "contradicts", "contextualizes"} == NON_CAUSAL


def test_drivers_need_a_validated_relationship_and_non_causal_wording(small: Investigation) -> None:
    brief = small.brief
    assert brief is not None and brief.drivers
    driver = brief.drivers[0]
    edges = [r for r in small.relationships if r.source_finding_id == driver.finding_ids[0]]
    kept, issues = validate_drivers([driver], small.findings, edges)
    assert kept == [driver] and not issues
    kept, issues = validate_drivers([driver], small.findings, [])
    assert not kept and "relationship to the outcome" in issues[0].reason
    causal = driver.model_copy(update={"statement": "Revenue fell because of this region."})
    kept, issues = validate_drivers([causal], small.findings, edges)
    assert not kept and "causal wording" in issues[0].reason
    orphan = driver.model_copy(update={"finding_ids": ["F999"]})
    assert not validate_drivers([orphan], small.findings, edges)[0]


# ------------------------------------------------------------------ causal guardrails (Step 12)


@pytest.mark.parametrize("phrase", CAUSAL_PHRASES)
def test_causal_phrases_are_rejected_in_findings_drivers_and_recommendations(small: Investigation, phrase: str) -> None:
    outcome = next(f for f in small.findings if f.primary)
    text = f"Revenue declined, {phrase} customer churn."
    assert causal_sentences(text)
    graph = graph_of(small)
    assert text_problems(text, [outcome.claim_id], graph)
    brief = small.brief
    assert brief is not None and brief.drivers
    driver = brief.drivers[0].model_copy(update={"statement": text})
    edges = [r for r in small.relationships if r.source_finding_id == driver.finding_ids[0]]
    assert not validate_drivers([driver], small.findings, edges)[0]
    rec = Recommendation(
        recommendation_id="R9",
        claim_id=outcome.claim_id,
        text=f"Review the churned accounts; revenue declined {phrase} churn.",
        rationale="test",
        supporting_finding_ids=[outcome.finding_id],
    )
    kept, issues = validate_recommendations([rec], small.findings, graph)
    assert not kept and "causal wording" in issues[0].reason


@pytest.mark.parametrize("phrase", CAUSAL_PHRASES)
def test_no_investigation_text_uses_causal_wording(full: dict[str, Investigation], phrase: str) -> None:
    for investigation in full.values():
        for text in user_texts(investigation):
            for sentence in causal_sentences(text):
                pytest.fail(f"causal sentence in {investigation.objective!r}: {sentence}")
            if phrase in text.lower():
                assert "not" in text.lower() or "cannot" in text.lower(), text


# ------------------------------------------------------------------ driver analysis (Step 11)


def test_drivers_are_typed_non_causal_and_rest_on_findings(full: dict[str, Investigation]) -> None:
    for investigation in full.values():
        brief = investigation.brief
        assert brief is not None
        by_id = {f.finding_id: f for f in investigation.findings}
        edges = {(r.source_finding_id, r.relationship) for r in investigation.relationships}
        for driver in [*brief.drivers, *brief.contradictions, *brief.context]:
            assert driver.relationship in NON_CAUSAL
            assert driver.finding_ids and set(driver.finding_ids) <= set(by_id)
            assert (driver.finding_ids[0], driver.relationship) in edges
            assert set(driver.evidence_ids) <= {e for i in driver.finding_ids for e in by_id[i].evidence_ids}
            order = ["low", "medium", "high"]
            assert order.index(driver.confidence) <= min(order.index(by_id[i].confidence) for i in driver.finding_ids)
            assert not causal_sentences(driver.statement)
        for rel in investigation.relationships:
            assert rel.relationship in NON_CAUSAL and rel.rule


def test_revenue_drivers_follow_the_documented_rules(full: dict[str, Investigation]) -> None:
    investigation = full[REVENUE]
    assert investigation.status == "completed"
    outcome = next(f for f in investigation.findings if f.primary)
    brief = investigation.brief
    assert brief is not None and brief.drivers
    table = EXPECTED_CO_MOVEMENT[outcome.metric or ""]
    by_id = {f.finding_id: f for f in investigation.findings}
    for driver in brief.drivers:
        first = by_id[driver.finding_ids[0]]
        if driver.relationship == "contributes_to":
            assert first.kind == "contribution" and first.direction == outcome.direction
            assert driver.share is not None and 0 < driver.share <= 1
        elif driver.relationship == "supports" and first.kind == "change":
            expected, _ = table[str(first.metric)]
            assert SIGN[str(first.direction)] * SIGN[str(outcome.direction)] == expected
        elif driver.relationship == "correlates_with":
            churn = [f for f in investigation.findings if f.metric == "logo_churn_rate" and f.kind == "change"]
            assert churn and churn[0].direction == "increase"
    for driver in brief.contradictions:
        first = by_id[driver.finding_ids[0]]
        expected, _ = table[str(first.metric)]
        assert SIGN[str(first.direction)] * SIGN[str(outcome.direction)] == -expected
    shares = [d.share or 0.0 for d in brief.drivers if d.relationship == "contributes_to"]
    assert shares == sorted(shares, reverse=True)


def test_contradictions_are_reported_and_explained(full: dict[str, Investigation]) -> None:
    for investigation in full.values():
        brief = investigation.brief
        assert brief is not None
        for driver in brief.contradictions:
            assert driver.relationship == "contradicts"
            assert any(driver.name in note for note in brief.uncertainty), driver.name


# ------------------------------------------------------------------ grounded recommendations (Step 14)


def test_recommendations_are_grounded_in_findings_and_evidence(full: dict[str, Investigation]) -> None:
    for investigation in full.values():
        brief = investigation.brief
        assert brief is not None and len(brief.recommendations) <= MAX_RECOMMENDATIONS
        by_id = {f.finding_id: f for f in investigation.findings}
        claims = {c.claim_id: c for c in investigation.claims}
        for rec in brief.recommendations:
            assert rec.claim_type == "recommendation" and rec.supporting_finding_ids
            assert set(rec.supporting_finding_ids) <= set(by_id)
            claim = claims[rec.claim_id]
            assert claim.claim_type == "recommendation" and claim.support_status != "unsupported"
            assert set(rec.evidence_ids) == {e for i in rec.supporting_finding_ids for e in by_id[i].evidence_ids}
            assert not recommendation_wording_problems(rec.text), rec.text
            assert rec.uncertainty and all(i in rec.rationale for i in rec.supporting_finding_ids)


def test_brief_recommendations_address_adverse_movements(full: dict[str, Investigation]) -> None:
    investigation = full[BRIEF]
    brief = investigation.brief
    assert brief is not None
    by_id = {f.finding_id: f for f in investigation.findings}
    for rec in brief.recommendations:
        cited = [by_id[i] for i in rec.supporting_finding_ids]
        adverse = [f for f in cited if f.kind == "change" and ADVERSE_DIRECTION.get(f.metric or "") == f.direction]
        assert adverse or any(f.kind in ("contribution", "concentration", "anomaly", "ranking") for f in cited)


def test_unsupported_recommendations_are_rejected(small: Investigation) -> None:
    graph = graph_of(small)
    outcome = next(f for f in small.findings if f.primary)
    base = Recommendation(
        recommendation_id="R1",
        claim_id=outcome.claim_id,
        text="Review the accounts in the largest declining region.",
        rationale="test",
        supporting_finding_ids=[outcome.finding_id],
    )
    assert validate_recommendations([base], small.findings, graph)[0] == [base]
    cases = {
        "no findings": base.model_copy(update={"supporting_finding_ids": []}),
        "unknown finding": base.model_copy(update={"supporting_finding_ids": ["F999"]}),
        "directive": base.model_copy(update={"text": "You must cut prices immediately to guarantee growth."}),
        "no claim": base.model_copy(update={"claim_id": "C999"}),
    }
    for name, rec in cases.items():
        kept, issues = validate_recommendations([rec], small.findings, graph)
        assert not kept and issues[0].action == "removed", name


def test_every_driver_in_the_brief_is_a_driver_model(full: dict[str, Investigation]) -> None:
    brief = full[REVENUE].brief
    assert brief is not None
    assert all(isinstance(d, Driver) for d in brief.drivers)
    assert [d.driver_id for d in brief.drivers] == [f"D{i}" for i in range(1, len(brief.drivers) + 1)]
