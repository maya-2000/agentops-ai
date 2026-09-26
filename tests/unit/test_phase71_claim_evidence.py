"""Phase 7.1 regressions: the claim/evidence identity contract and identifier-safe number extraction.

A claim is supported only by evidence about the same thing: the same metric identifier, unit, period,
comparison period, dimension, member and filters. An equal number on evidence about something else
(MRR for revenue, CLV for CAC, ...) never supports it. Identifiers such as ``CUST-002529`` are names,
not business numbers.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.evidence.formatting import extract_numbers
from app.evidence.models import ClaimSubject, Evidence
from app.evidence.validation import metrics_named, validate_evidence, validate_response
from app.llm.schemas import DraftItemOutput, ResponseDraftOutput
from tests.phase4_support import AUGUST, JULY, assertion, claim, evidence, graph

CALLS = {"T1"}


def about(e: Evidence, **overrides: Any) -> ClaimSubject:
    return ClaimSubject.of(e).model_copy(update=overrides)


def problems(*items: Evidence, text: str, subject: ClaimSubject | None, **claim_fields: Any) -> list[str]:
    c = claim("C1", text, [i.evidence_id for i in items], subject=subject, **claim_fields)
    return validate_evidence(graph(*items, claims=(c,)), successful_call_ids=CALLS).errors


# ------------------------------------------------------------------ E. identifiers are not numbers


@pytest.mark.parametrize(
    "text",
    [
        "CUST-12345",
        "CUST-002529",
        "EV-00042",
        "query_1847",
        "2026-08-31",
        "run-abc123",
        "Q-76ade5f37395",
        "INV-00731",
        "R-eval-direct-1",
        "E12 C3 T1",
        "2026-Q2 and 2026-08",
    ],
)
def test_identifiers_and_dates_are_not_business_numbers(text: str) -> None:
    assert extract_numbers(text) == []


def test_business_numbers_next_to_identifiers_are_still_checked() -> None:
    text = "Customer CUST-002529 (Mid-Market, EMEA): risk score 95 (high band), MRR SGD 790; revenue -5.20%."
    assert [n.raw for n in extract_numbers(text)] == ["95", "790", "-5.20%"]
    assert [n.raw for n in extract_numbers("Revenue was SGD 1,234,567 (+3.5%) in 2026-08.")] == ["1,234,567", "+3.5%"]


def test_the_response_validator_accepts_customer_ids_but_not_invented_numbers() -> None:
    risk = evidence(
        "E1",
        95.0,
        statement="Customer CUST-002529: risk score 95, MRR SGD 790.",
        metric=None,
        unit="points",
        attributes={"mrr": 790.0},
    )
    c = claim("C1", risk.statement, ["E1"], claim_type="calculated_result", subject=ClaimSubject.of(risk))
    g = graph(risk, claims=(c,))
    g = g.model_copy(update={"claims": {"C1": c.model_copy(update={"support_status": "supported"})}})
    ok = validate_response(ResponseDraftOutput(answer=risk.statement, answer_claim_ids=["C1"]), g)
    assert ok.valid, ok.errors
    bad = validate_response(ResponseDraftOutput(answer=f"{risk.statement} Churn risk 61%.", answer_claim_ids=["C1"]), g)
    assert not bad.valid and bad.unsupported_numbers == ["61%"]


# ------------------------------------------------------------------ F-I. claim subject vs evidence


def test_a_claim_about_its_own_evidence_is_supported() -> None:
    revenue = evidence("E1", 100.0, statement="Revenue for 2026-08: SGD 100.")
    assert not problems(
        revenue, text=revenue.statement, subject=ClaimSubject.of(revenue), numeric_assertions=[assertion("E1", 100.0)]
    )


def test_f_metric_mismatch_is_unsupported_even_when_the_numbers_match() -> None:
    mrr = evidence("E1", 100.0, metric="mrr", statement="Monthly Recurring Revenue for 2026-08: SGD 100.")
    errors = problems(mrr, text="Revenue for 2026-08: SGD 100.", subject=about(mrr, metric="revenue"))
    assert any("is about metric 'revenue' but E1 reports 'mrr'" in e for e in errors), errors


@pytest.mark.parametrize(
    ("claimed", "actual", "claim_text"),
    [
        ("revenue", "mrr", "Revenue increased by 5%."),
        ("arr", "mrr", "ARR increased by 5%."),
        ("cac", "clv", "CAC was SGD 100."),
        ("logo_churn_rate", "retention_rate", "Logo churn was 5%."),
        ("revenue", "pipeline_value", "Revenue was SGD 100."),
        ("support_ticket_volume", "average_resolution_time", "Support ticket volume was 100."),
        ("customer_count", "arpu", "Customer count was 100."),
    ],
)
def test_f_related_metrics_are_not_interchangeable(claimed: str, actual: str, claim_text: str) -> None:
    item = evidence("E1", 100.0, metric=actual)
    errors = problems(item, text=claim_text, subject=about(item, metric=claimed))
    assert any(f"is about metric '{claimed}' but E1 reports '{actual}'" in e for e in errors), errors


def test_f_a_claim_text_naming_another_kpi_than_its_subject_is_unsupported() -> None:
    """The canonical case: "Revenue increased by 5%" resting on evidence that MRR increased by 5%."""
    mrr = evidence("E1", 0.05, metric="mrr", unit="ratio", statement="Monthly Recurring Revenue changed by +5.00%.")
    errors = problems(mrr, text="Revenue increased by 5%.", subject=ClaimSubject.of(mrr))
    assert any("names revenue but is about mrr" in e for e in errors), errors
    assert not problems(mrr, text="MRR increased by 5%.", subject=ClaimSubject.of(mrr))


def test_f_a_number_asserted_from_evidence_about_another_metric_is_unsupported() -> None:
    revenue = evidence("E1", 100.0)
    mrr = evidence("E2", 100.0, metric="mrr", statement="MRR for 2026-08: SGD 100.")
    errors = problems(
        revenue,
        mrr,
        text="Revenue for 2026-08: SGD 100.",
        subject=ClaimSubject.of(revenue),
        numeric_assertions=[assertion("E2", 100.0)],
    )
    assert any("states a mrr number (E2) in a claim about revenue" in e for e in errors), errors


def test_g_period_and_comparison_mismatches_are_unsupported() -> None:
    july = evidence("E1", 100.0, period_label="2026-07", period_start=JULY[0], period_end=JULY[1])
    errors = problems(july, text="Revenue for 2026-08: SGD 100.", subject=about(july, period_label="2026-08"))
    assert any("is about period_label '2026-08' but E1 reports '2026-07'" in e for e in errors), errors
    change = evidence("E1", -5.0, comparison_label="2026-06")
    errors = problems(change, text="Revenue changed.", subject=about(change, comparison_label="2026-07"))
    assert any("comparison_label '2026-07' but E1 reports '2026-06'" in e for e in errors), errors


def test_h_dimension_member_and_filter_mismatches_are_unsupported() -> None:
    apac = evidence("E1", 100.0, dimension="region", dimension_value="APAC")
    errors = problems(apac, text="Revenue in EMEA: SGD 100.", subject=about(apac, dimension_value="EMEA"))
    assert any("dimension_value 'EMEA' but E1 reports 'APAC'" in e for e in errors), errors
    errors = problems(apac, text="Revenue by segment.", subject=about(apac, dimension="segment"))
    assert any("dimension 'segment' but E1 reports 'region'" in e for e in errors), errors
    smb = evidence("E1", 100.0, filters={"segment": "SMB"})
    errors = problems(smb, text="Revenue: SGD 100.", subject=about(smb, filters={}))
    assert any("is about filters {} but E1 reports {'segment': 'SMB'}" in e for e in errors), errors


def test_i_unit_mismatch_is_unsupported() -> None:
    hours = evidence("E1", 40.0, metric="average_resolution_time", unit="hours")
    errors = problems(hours, text="Average resolution time: 40.", subject=about(hours, unit="days"))
    assert any("is about unit 'days' but E1 reports 'hours'" in e for e in errors), errors


def test_the_subject_evidence_must_be_cited() -> None:
    revenue, other = evidence("E1", 100.0), evidence("E2", 100.0)
    c = claim("C1", "Revenue for 2026-08: SGD 100.", ["E2"], subject=ClaimSubject.of(revenue))
    errors = validate_evidence(graph(revenue, other, claims=(c,)), successful_call_ids=CALLS).errors
    assert any("does not cite the evidence it is about (E1)" in e for e in errors), errors


def test_a_response_naming_another_metric_than_its_claims_is_rejected() -> None:
    mrr = evidence("E1", 100.0, metric="mrr", statement="Monthly Recurring Revenue for 2026-08: SGD 100.")
    c = claim("C1", mrr.statement, ["E1"], subject=ClaimSubject.of(mrr), support_status="supported")
    g = graph(mrr, claims=(c,))
    bad = validate_response(ResponseDraftOutput(answer="Revenue for 2026-08: SGD 100.", answer_claim_ids=["C1"]), g)
    assert any("names revenue but cites claims about mrr" in e for e in bad.errors), bad.errors
    good = validate_response(ResponseDraftOutput(answer="MRR for 2026-08: SGD 100.", answer_claim_ids=["C1"]), g)
    assert good.valid, good.errors
    # An item may name more than one metric, as long as it names one its claims are about.
    both = ResponseDraftOutput(
        answer="MRR for 2026-08 was SGD 100.",
        answer_claim_ids=["C1"],
        key_findings=[DraftItemOutput(text="The revenue decline coincided with MRR of SGD 100.", claim_ids=["C1"])],
    )
    assert validate_response(both, g).valid


@pytest.mark.parametrize(
    ("text", "named"),
    [
        ("Revenue for 2026-08", {"revenue"}),
        ("Monthly Recurring Revenue for 2026-08", {"mrr"}),
        ("Net Revenue Retention was 98%", {"nrr"}),
        ("Churn Rate - Revenue for 2026-08", {"revenue_churn_rate"}),
        ("Churn Rate - Logo and revenue", {"logo_churn_rate", "revenue"}),
        ("Average Revenue Per User (Account)", {"arpu"}),
        ("ARR and MRR", {"arr", "mrr"}),
        ("CAC vs CLV", {"cac", "clv"}),
        ("Support ticket volume and average resolution time", {"support_ticket_volume", "average_resolution_time"}),
        ("Active customer count", {"customer_count"}),
        ("Pipeline value and win rate", {"pipeline_value", "win_rate"}),
        ("Tickets rose and churn fell", set()),  # ambiguous words name no single KPI
    ],
)
def test_metric_names_resolve_to_registry_identifiers_longest_first(text: str, named: set[str]) -> None:
    assert metrics_named(text) == named


def test_claims_built_by_the_agent_carry_their_subject() -> None:
    from app.agent.findings import build_claims
    from app.agent.request import ValidatedRequest
    from app.analytics.periods import Period
    from app.llm.schemas import Intent

    revenue = evidence("E1", 100.0, statement="Revenue for 2026-08: SGD 100.", evidence_type="observed")
    g = build_claims(
        graph(revenue),
        ValidatedRequest(
            intent=Intent.KPI_LOOKUP,
            metric="revenue",
            period=Period(start=AUGUST[0], end=AUGUST[1], label="2026-08"),
        ),
    )
    (built,) = g.claims.values()
    assert built.subject == ClaimSubject.of(revenue) and built.subject.metric == "revenue"
    assert validate_evidence(g, successful_call_ids=CALLS).valid
