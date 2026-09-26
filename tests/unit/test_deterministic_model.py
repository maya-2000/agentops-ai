"""The deterministic offline model: rule-based understanding, intent playbooks and claim composition.

These rules parse language and choose tools. They never produce a business number, which the
composition tests check by requiring response text to be copied verbatim from claims.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest

from app.agent.request import validate_understanding
from app.llm.deterministic.composition import compose
from app.llm.deterministic.planning import plan
from app.llm.deterministic.understanding import understand
from app.llm.schemas import PlanOutput, ResponseDraftOutput, UnderstandingOutput
from app.tools import ToolRegistry
from tests.phase4_support import AS_OF, understanding_context

COVERAGE = (date(2024, 9, 1), date(2026, 8, 31))
REGISTRY = ToolRegistry()


def _understand(question: str) -> dict[str, Any]:
    output = understand(understanding_context(question))
    UnderstandingOutput.model_validate(output)  # the offline model obeys the same output contract
    return output


# ---- understanding -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What was revenue last month?", {"intent": "kpi_lookup", "metric": "revenue", "period": "last_month"}),
        (
            "How did MRR change compared with the previous month?",
            {"intent": "period_comparison", "metric": "mrr", "period": None, "comparison_period": "previous_month"},
        ),
        (
            "Which segment contributed most to the revenue decline last month?",
            {"intent": "dimensional_comparison", "dimensions": ["segment"], "analysis_type": "contribution"},
        ),
        (
            "Why did revenue decline last month?",
            {"intent": "revenue_investigation", "metric": "revenue", "analysis_type": "change"},
        ),
        ("Forecast revenue for the next 3 months", {"intent": "forecast", "metric": "revenue", "horizon": 3}),
        ("What will MRR be next month?", {"intent": "forecast", "metric": "mrr", "horizon": 1}),
        # Phase 8: an "N-month" forecast names its horizon (it previously fell back to one month).
        ("What is our 3-month revenue forecast?", {"intent": "forecast", "metric": "revenue", "horizon": 3}),
        ("Give me a six month MRR outlook", {"intent": "forecast", "metric": "mrr", "horizon": 6}),
        ("Revenue 2 months ahead", {"intent": "forecast", "metric": "revenue", "horizon": 2}),
        (
            "Were there any unusual movements in revenue last month?",
            {"intent": "anomaly_detection", "metric": "revenue"},
        ),
        (
            "How did support tickets change last month?",
            {"intent": "period_comparison", "metric": "support_ticket_volume"},
        ),
        (
            "Which segment had the highest churn last month?",
            {"intent": "dimensional_comparison", "metric": "logo_churn_rate", "analysis_type": "highest"},
        ),
        ("Which channel has the lowest CAC?", {"metric": "cac", "dimensions": ["acquisition_channel"]}),
        ("What caused churn?", {"intent": "customer_investigation", "analysis_type": "causal"}),
        ("Which customers are at risk?", {"intent": "customer_investigation", "analysis_type": "risk"}),
        ("Show cohort retention", {"intent": "customer_investigation", "analysis_type": "cohort"}),
        (
            "How did revenue and support tickets change last month?",
            {"intent": "mixed_investigation", "metric": "revenue"},
        ),
        ("Revenue in Q2 2026 vs Q1 2026", {"period": "2026-Q2", "comparison_period": "2026-Q1"}),
        (
            "How did revenue change from July to August?",
            {"intent": "period_comparison", "period": "2026-08", "comparison_period": "2026-07"},
        ),
        ("Was churn in August higher than in July?", {"intent": "period_comparison", "comparison_period": "2026-07"}),
        ("What was revenue in May?", {"period": "2026-05"}),
        ("What was revenue over the last three months?", {"period": "trailing_3_months"}),
        ("What was revenue churn last month?", {"metric": "revenue_churn_rate"}),
        ("What is net revenue retention?", {"metric": "nrr"}),
        ("What was the win rate last quarter?", {"metric": "win_rate", "period": "last_quarter"}),
    ],
)
def test_understanding(question: str, expected: dict[str, Any]) -> None:
    output = _understand(question)
    assert {key: output[key] for key in expected} == expected


@pytest.mark.parametrize(
    "question",
    [
        "What is Apple's stock price?",
        "What is the weather in Singapore?",
        "Delete all customers",
        "Drop the revenue table",
        "Tell me a joke",
        "How are our competitors doing?",
    ],
)
def test_out_of_scope_and_write_requests_are_unsupported(question: str) -> None:
    output = _understand(question)
    assert output["intent"] == "unsupported" and output["unsupported_reason"]
    assert output["metric"] is None


def test_filters_come_from_the_supplied_vocabulary() -> None:
    question = "What was churn for Enterprise customers in Singapore last month?"
    output = _understand(question)
    assert {(f["dimension"], f["value"]) for f in output["filters"]} == {
        ("country", "Singapore"),
        ("segment", "Enterprise"),
    }
    assert output["ambiguities"] and not output["material_ambiguity"]  # shared segment/plan name, noted
    context = understanding_context(question)
    context["dimension_values"] = {
        k: [v for v in vs if v != "Singapore"] for k, vs in context["dimension_values"].items()
    }
    assert "country" not in {f["dimension"] for f in understand(context)["filters"]}


def test_relative_months_resolve_against_the_business_date() -> None:
    later = understand(understanding_context("What was revenue in September?", as_of=date(2027, 1, 31)))
    assert later["period"] == "2026-09"
    assert _understand("What was revenue in September?")["period"] == "2025-09"


# ---- planning ------------------------------------------------------------------------------------


def _request(question: str) -> dict[str, Any]:
    validation = validate_understanding(
        UnderstandingOutput.model_validate(_understand(question)), as_of=AS_OF, coverage=COVERAGE
    )
    assert validation.outcome == "valid", validation.message
    assert validation.request is not None
    return validation.request.model_dump(mode="json")


PLANNED_QUESTIONS = [
    "What was revenue last month?",
    "How did MRR change compared with the previous month?",
    "Which segment contributed most to the revenue decline last month?",
    "Why did revenue decline last month?",
    "Forecast revenue for the next 3 months",
    "Were there any unusual movements in revenue last month?",
    "Were there any anomalies last month?",
    "How did support tickets change last month?",
    "Why did support tickets increase last month?",
    "Which segment had the highest churn last month?",
    "Which channel has the lowest CAC?",
    "What caused churn?",
    "Which customers are at risk?",
    "Show cohort retention",
    "How did revenue and support tickets change last month?",
    "What was the win rate last quarter?",
    "How is marketing performing?",
    "What is adoption of Dashboards?",
    "How is feature adoption?",
    "What was churn for Enterprise customers in Singapore last month?",
]


@pytest.mark.parametrize("question", PLANNED_QUESTIONS)
def test_playbooks_use_registered_tools_with_valid_arguments(question: str) -> None:
    request = _request(question)
    output = PlanOutput.model_validate(plan({"request": request, "iteration": 1, "remaining_tool_calls": 12}))
    assert 1 <= len(output.steps) <= 12
    for step in output.steps:
        arguments = json.loads(step.arguments_json)
        REGISTRY.validate_arguments(step.tool_name, arguments)  # raises on an invalid plan
        assert step.purpose


def test_plan_arguments_use_the_validated_dates() -> None:
    request = _request("Why did revenue decline last month?")
    first = json.loads(PlanOutput.model_validate(plan({"request": request})).steps[0].arguments_json)
    assert (first["start_date"], first["end_date"]) == ("2026-08-01", "2026-08-31")
    assert (first["comparison_start_date"], first["comparison_end_date"]) == ("2026-07-01", "2026-07-31")


def test_playbook_is_not_silently_truncated_to_the_budget() -> None:
    request = _request("Why did revenue decline last month?")
    output = PlanOutput.model_validate(plan({"request": request, "iteration": 1, "remaining_tool_calls": 2}))
    assert len(output.steps) > 2  # the graph rejects over-budget plans explicitly


def _decomposition(dimension: str, member: str, share: float, rank: int = 1, **extra: Any) -> dict[str, Any]:
    return {
        "operation": "analyze_revenue.decompose_revenue_change",
        "dimension": dimension,
        "dimension_value": member,
        "filters": {},
        "attributes": {"rank": rank, "share_of_gross_decline": share},
        **extra,
    }


def _follow_up(evidence: list[dict[str, Any]], executed: list[dict[str, Any]] | None = None) -> PlanOutput:
    context = {
        "request": _request("Why did revenue decline last month?"),
        "iteration": 2,
        "evidence": evidence,
        "executed_steps": executed or [],
    }
    return PlanOutput.model_validate(plan(context))


def test_follow_up_drills_into_the_concentrated_member_read_from_evidence() -> None:
    output = _follow_up([_decomposition("region", "Region-X", 0.7), _decomposition("country", "Country-Y", 0.62)])
    assert not output.sufficient and len(output.steps) == 3
    for step in output.steps:
        arguments = json.loads(step.arguments_json)
        assert arguments["filters"] == {"country": "Country-Y"}  # the name comes from evidence, not the code
        REGISTRY.validate_arguments(step.tool_name, arguments)


@pytest.mark.parametrize(
    ("evidence", "executed"),
    [
        ([_decomposition("country", "Country-Y", 0.3)], None),  # not concentrated
        ([_decomposition("country", "Country-Y", 0.9, rank=2)], None),  # not the largest contributor
        ([_decomposition("segment", "Segment-Z", 0.9)], None),  # drill-down is geographic only
        ([_decomposition("country", "Country-Y", 0.9)], [{"arguments": {"filters": {"country": "Country-Y"}}}]),
        ([], None),
    ],
)
def test_follow_up_stops_when_there_is_nothing_to_drill_into(
    evidence: list[dict[str, Any]], executed: list[dict[str, Any]] | None
) -> None:
    output = _follow_up(evidence, executed)
    assert output.sufficient and not output.steps


def test_follow_up_is_only_for_revenue_investigations() -> None:
    context = {
        "request": _request("How did support tickets change last month?"),
        "iteration": 2,
        "evidence": [_decomposition("country", "Country-Y", 0.9)],
    }
    assert PlanOutput.model_validate(plan(context)).sufficient


# ---- composition ---------------------------------------------------------------------------------


def _claim(claim_id: str, kind: str, claim_type: str = "calculated_result", primary: bool = False) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "type": claim_type,
        "kind": kind,
        "text": f"Claim text {claim_id}.",
        "primary": primary,
        "support_status": "supported",
        "evidence_ids": ["E1"],
    }


def test_composition_places_claims_by_type_and_copies_their_text() -> None:
    claims = [
        _claim("C1", "kpi_value", "observed_fact", primary=True),
        _claim("C2", "change", primary=True),
        _claim("C3", "contribution"),
        _claim("C4", "concentration", "inference"),
        _claim("C5", "recommendation", "recommendation"),
        {**_claim("C6", "fact"), "support_status": "unsupported"},
    ]
    draft = ResponseDraftOutput.model_validate(compose({"claims": claims}))
    assert draft.answer_claim_ids == ["C2", "C1"]  # a change answers before a single value
    assert draft.answer == "Claim text C2. Claim text C1."
    assert [i.claim_ids for i in draft.key_findings] == [["C3"]]
    assert [i.claim_ids for i in draft.interpretation] == [["C4"]]
    assert [i.claim_ids for i in draft.recommendations] == [["C5"]]
    every_text = [draft.answer, *(i.text for i in draft.key_findings + draft.interpretation + draft.recommendations)]
    assert all(text.startswith("Claim text") for text in every_text)
    assert "C6" not in json.dumps(draft.model_dump())


def test_forecast_answers_list_every_forecast_month() -> None:
    claims = [_claim(f"C{i}", "forecast", primary=True) for i in range(1, 5)]
    claims.append(_claim("C5", "forecast_quality"))
    draft = ResponseDraftOutput.model_validate(compose({"claims": claims}))
    assert draft.answer_claim_ids == ["C1", "C2", "C3", "C4"]
    assert [i.claim_ids for i in draft.key_findings] == [["C5"]]


def test_composition_without_claims_is_empty() -> None:
    draft = ResponseDraftOutput.model_validate(compose({"claims": []}))
    assert draft.answer == "" and not draft.answer_claim_ids
