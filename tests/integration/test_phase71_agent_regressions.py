"""Phase 7.1 regressions through the real LangGraph agent on the full generated dataset.

Each fixed failure is exercised end to end: question -> understanding -> validated request -> plan ->
tool execution -> evidence -> claims (with their structured subject) -> claim validation -> response
validation. Expected members and values are never written into the tests. They come from direct
calls to the deterministic analytics the tools wrap, so each test checks the agent reported what the
analytics layer computed.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from app.agent import AgentConfig, AgentRunner, AgentRunResult
from app.analytics.kpis import calculate_kpi
from app.analytics.revenue import decompose_revenue_change
from app.analytics.sales import rep_performance
from app.database.base import Database
from app.llm.deterministic import DeterministicLLM
from tests.phase4_support import AS_OF
from tests.phase6_support import service

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def agent(full_db: Database) -> AgentRunner:
    return AgentRunner(full_db, llm=DeterministicLLM(), config=AgentConfig(), as_of=AS_OF)


def calls(result: AgentRunResult) -> list[tuple[str, dict[str, Any]]]:
    return [(c.tool_name, dict(c.input)) for c in result.tool_trace]


def primary(result: AgentRunResult) -> Any:
    claims = [c for c in result.claims if c.primary and c.support_status == "supported"]
    assert claims, [c.text for c in result.claims]
    return claims[-1] if len(claims) > 1 else claims[0]


def completed(result: AgentRunResult) -> None:
    """Answered, validated, and every claim resting on evidence about the same subject."""
    assert result.status == "completed", (result.status, result.response.answer)
    assert result.response.answer_claim_ids
    by_id = {e.evidence_id: e for e in result.evidence}
    for claim in result.claims:
        if claim.subject is not None:
            source = by_id[claim.subject.evidence_id]
            assert claim.subject.evidence_id in claim.evidence_ids
            assert (claim.subject.metric, claim.subject.period_label) == (source.metric, source.period_label)
    assert not [e for e in result.security_events if e.event_type in ("plan_rejected", "argument_rejected")]


# ------------------------------------------------------------------ A. explicit comparison periods


@pytest.mark.parametrize(
    ("question", "tool", "period", "comparison"),
    [
        (
            "How did support tickets change in July compared with May?",
            "analyze_support",
            ("2026-07-01", "2026-07-31"),
            ("2026-05-01", "2026-05-31"),
        ),
        (
            "How did revenue change in July compared with June?",
            "analyze_revenue",
            ("2026-07-01", "2026-07-31"),
            ("2026-06-01", "2026-06-30"),
        ),
        (
            "How did revenue in July compare with the previous month?",
            "analyze_revenue",
            ("2026-07-01", "2026-07-31"),
            ("2026-06-01", "2026-06-30"),
        ),
        (
            "Compare revenue in 2026-07 with 2026-05.",
            "analyze_revenue",
            ("2026-07-01", "2026-07-31"),
            ("2026-05-01", "2026-05-31"),
        ),
        (
            "Compare revenue from 2026-07-01 to 2026-07-31 with 2026-05-01 to 2026-05-31.",
            "analyze_revenue",
            ("2026-07-01", "2026-07-31"),
            ("2026-05-01", "2026-05-31"),
        ),
        (
            "How did revenue change last month?",
            "analyze_revenue",
            ("2026-08-01", "2026-08-31"),
            ("2026-07-01", "2026-07-31"),
        ),
    ],
)
def test_a_the_tool_receives_the_comparison_the_question_asked_for(
    agent: AgentRunner, question: str, tool: str, period: tuple[str, str], comparison: tuple[str, str]
) -> None:
    result = agent.run(question)
    completed(result)
    name, args = calls(result)[0]
    assert name == tool
    assert (args["start_date"], args["end_date"]) == period
    assert (args["comparison_start_date"], args["comparison_end_date"]) == comparison
    change = primary(result)
    assert change.subject is not None and change.subject.comparison_label == comparison[0][:7]


def test_a_last_month_lookup_is_unchanged(agent: AgentRunner) -> None:
    result = agent.run("What was revenue last month?")
    completed(result)
    assert calls(result) == [("get_kpi", {"kpi": "revenue", "start_date": "2026-08-01", "end_date": "2026-08-31"})]


# ------------------------------------------------------------------ B. level vs change rankings


def _decomposition(db: Database) -> Any:
    return decompose_revenue_change(db, "region", "2026-08", "2026-07", as_of=AS_OF)


@pytest.mark.parametrize(
    ("question", "summary_key"),
    [
        ("Which region had the largest revenue decline last month?", "largest_decline"),
        ("Which region had the largest percentage decline in revenue last month?", "largest_percentage_decline"),
        ("Which region had the largest revenue increase last month?", "largest_increase"),
    ],
)
def test_b_change_rankings_name_the_member_the_decomposition_names(
    agent: AgentRunner, full_db: Database, question: str, summary_key: str
) -> None:
    result = agent.run(question)
    completed(result)
    ((tool, args),) = calls(result)
    assert (tool, args["operation"], args["dimension"]) == ("analyze_revenue", "decompose_revenue_change", "region")
    assert (args["comparison_start_date"], args["comparison_end_date"]) == ("2026-07-01", "2026-07-31")
    expected = _decomposition(full_db).summary[summary_key]
    claim = primary(result)
    assert claim.kind == "ranking"
    if expected is None:
        assert claim.text.startswith("No region had a revenue")
        return
    assert claim.subject is not None and claim.subject.dimension_value == expected
    assert claim.subject.metric == "revenue" and claim.subject.comparison_label == "2026-07"
    assert expected in result.response.answer


@pytest.mark.parametrize(("question", "pick"), [("highest", max), ("lowest", min)])
def test_b_level_rankings_rank_revenue_levels(agent: AgentRunner, full_db: Database, question: str, pick: Any) -> None:
    result = agent.run(f"Which region had the {question} revenue last month?")
    completed(result)
    ((tool, args),) = calls(result)
    assert (tool, args["kpi"], args["dimension"]) == ("get_kpi", "revenue", "region")
    rows = calculate_kpi(full_db, "revenue", period="2026-08", dimension="region", as_of=AS_OF).breakdown
    expected = pick(rows, key=lambda r: r.value or 0).dimension_value
    claim = primary(result)
    assert claim.subject is not None and claim.subject.dimension_value == expected
    assert claim.subject.comparison_label is None  # a level, not a change


# ------------------------------------------------------------------ C. CAC by channel


def test_c_cac_by_channel_end_to_end(agent: AgentRunner, full_db: Database) -> None:
    result = agent.run("Which marketing channel had the highest CAC last quarter?")
    completed(result)
    assert result.understanding is not None and result.understanding.intent.value == "dimensional_comparison"
    assert result.request is not None and result.request.dimensions == ["acquisition_channel"]
    ((tool, args),) = calls(result)
    assert (tool, args["kpi"], args["dimension"]) == ("get_kpi", "cac", "acquisition_channel")
    channels = [e for e in result.evidence if e.dimension == "acquisition_channel" and e.dimension_value]
    breakdown = calculate_kpi(full_db, "cac", period="2026-Q2", dimension="acquisition_channel", as_of=AS_OF).breakdown
    assert {e.dimension_value for e in channels} == {r.dimension_value for r in breakdown}
    top = max(breakdown, key=lambda r: r.value or 0)
    claim = primary(result)
    assert claim.subject is not None
    assert (claim.subject.metric, claim.subject.dimension, claim.subject.dimension_value) == (
        "cac",
        "acquisition_channel",
        top.dimension_value,
    )
    assert top.dimension_value in result.response.answer


def test_c_cac_by_channel_through_mcp_matches_the_direct_analytics(full_db: Database) -> None:
    outcome = service(full_db).call(
        "agentops_get_kpi", {"kpi": "cac", "period": "2026-Q2", "dimension": "acquisition_channel"}
    )
    assert not outcome.is_error
    breakdown = calculate_kpi(full_db, "cac", period="2026-Q2", dimension="acquisition_channel", as_of=AS_OF).breakdown
    rows = {r["dimension_value"]: r["value"] for r in outcome.payload["result"]["breakdown"]}
    assert rows == pytest.approx({r.dimension_value: r.value for r in breakdown})


# ------------------------------------------------------------------ D. sales reps


@pytest.mark.parametrize(
    ("question", "lowest"),
    [
        ("Which sales rep had the lowest win rate in Q2 2026?", True),
        ("Which sales rep had the highest win rate in Q2 2026?", False),
        ("Which sales rep had the best conversion in Q2 2026?", False),
    ],
)
def test_d_rep_questions_are_answered_from_rep_performance(
    agent: AgentRunner, full_db: Database, question: str, lowest: bool
) -> None:
    result = agent.run(question)
    completed(result)
    assert calls(result) == [
        ("analyze_sales", {"operation": "rep_performance", "start_date": "2026-04-01", "end_date": "2026-06-30"})
    ]
    ranked = sorted((r for r in rep_performance(full_db, "2026-Q2", as_of=AS_OF).data if r.rank), key=lambda r: r.rank)
    expected = ranked[-1] if lowest else ranked[0]
    claim = primary(result)
    assert claim.subject is not None
    assert (claim.subject.metric, claim.subject.dimension_value) == ("win_rate", expected.sales_rep)
    assert expected.sales_rep in result.response.answer and not result.security_events[1:]


def test_d_rep_performance_overview(agent: AgentRunner) -> None:
    result = agent.run("How are the sales reps performing? Show the win rate by rep for Q2 2026.")
    completed(result)
    assert calls(result)[0][1]["operation"] == "rep_performance"


def test_d_unavailable_and_invalid_rep_requests_are_declined_without_running_tools(agent: AgentRunner) -> None:
    pipeline = agent.run("Which sales rep has the largest pipeline?")
    assert pipeline.status == "unsupported_request" and not pipeline.tool_trace
    assert "not available" in pipeline.response.answer
    assert not [e for e in pipeline.security_events if e.severity.value in ("HIGH", "CRITICAL")]
    before_data = agent.run("Which sales rep had the lowest win rate in 2019?")
    assert before_data.status == "insufficient_evidence" and not before_data.tool_trace


def test_d_rep_names_come_only_from_the_allowed_operation(agent: AgentRunner) -> None:
    result = agent.run("Which sales rep had the lowest win rate in Q2 2026?")
    named = [e for e in result.evidence if e.dimension == "sales_rep" and e.dimension_value]
    assert named and all(e.operation == "analyze_sales.rep_performance" for e in named)


# ------------------------------------------------------------------ E. customer identifiers


def test_e_the_customer_risk_answer_validates_with_customer_ids(agent: AgentRunner) -> None:
    result = agent.run("Which customers are most at risk of churning?")
    completed(result)
    customers = [e for e in result.evidence if e.dimension == "customer_id"]
    assert customers and all(e.dimension_value and e.dimension_value.startswith("CUST-") for e in customers)
    text = " ".join([result.response.answer, *(i.text for i in result.response.key_findings)])
    assert "CUST-" in text  # the IDs are shown, and the response validator accepted them
    assert not [e for e in result.security_events if e.event_type == "output_validation_failed"]


def test_period_coverage_is_still_enforced(agent: AgentRunner) -> None:
    assert agent.run("How did revenue change in July compared with May 2019?").status == "insufficient_evidence"
    assert date(2026, 5, 1) < AS_OF
