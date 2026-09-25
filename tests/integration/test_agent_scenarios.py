"""End-to-end agent scenarios on the full generated dataset with the deterministic offline model.

Every number the agent states is checked against an independent, direct call to the Phase 2/3
layer that produced it. The expected answers are never written into the test. They come from the
same deterministic analytics the agent's tools wrap, so the tests check that the agent reports
those results faithfully, completely and with the right status.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from app.agent import AgentConfig, AgentRunner, AgentRunResult
from app.analytics.customers import churn_by_dimension
from app.analytics.kpis import KPIService
from app.analytics.revenue import decompose_revenue_change
from app.anomalies import AnomalyService
from app.database.base import Database
from app.evidence.formatting import format_percent, format_value
from app.evidence.models import EvidenceGraph
from app.evidence.validation import causal_sentences, validate_response
from app.forecasting import ForecastService
from app.llm.deterministic import DeterministicLLM
from app.llm.schemas import DraftItemOutput, ResponseDraftOutput
from tests.phase4_support import AS_OF

pytestmark = pytest.mark.slow

LAST_MONTH = "2026-08"
PREVIOUS_MONTH = "2026-07"


@pytest.fixture(scope="module")
def runner(full_db: Database) -> AgentRunner:
    return AgentRunner(full_db, llm=DeterministicLLM(), config=AgentConfig(), as_of=AS_OF)


@pytest.fixture(scope="module")
def kpis(full_db: Database) -> KPIService:
    return KPIService(full_db, as_of=AS_OF)


def _text(result: AgentRunResult) -> str:
    r = result.response
    return " ".join([r.answer, *(i.text for i in r.key_findings + r.interpretation + r.recommendations)])


def _assert_grounded(result: AgentRunResult) -> None:
    """Common guarantees for every answered scenario."""
    response = result.response
    assert response.answer and response.evidence and response.answer_evidence_ids
    cited = {e.evidence_id for e in response.evidence}
    assert set(response.answer_evidence_ids) <= cited
    for item in response.key_findings + response.interpretation + response.recommendations:
        assert item.evidence_ids and set(item.evidence_ids) <= cited
    for evidence in result.evidence:
        assert evidence.has_provenance and evidence.query_ids and evidence.source_tables
    successful = {c.call_id for c in result.tool_trace if c.success}
    assert {e.tool_call_id for e in result.evidence} <= successful
    assert not causal_sentences(_text(result))
    assert result.total_tool_calls <= 12 and len(result.plans) <= 2
    # Re-validate the delivered text: every number in it must be found in the cited evidence.
    graph_claims = {c.claim_id: c for c in result.claims}
    draft = ResponseDraftOutput(
        answer=response.answer,
        answer_claim_ids=response.answer_claim_ids,
        key_findings=[DraftItemOutput(text=i.text, claim_ids=i.claim_ids) for i in response.key_findings],
        interpretation=[DraftItemOutput(text=i.text, claim_ids=i.claim_ids) for i in response.interpretation],
        recommendations=[DraftItemOutput(text=i.text, claim_ids=i.claim_ids) for i in response.recommendations],
    )
    graph = EvidenceGraph(evidence={e.evidence_id: e for e in result.evidence}, claims=graph_claims)
    assert validate_response(draft, graph).valid


# ---- 1. KPI lookup -------------------------------------------------------------------------------


def test_scenario_revenue_last_month(runner: AgentRunner, kpis: KPIService) -> None:
    result = runner.run("What was revenue last month?")
    assert result.status == "completed" and [c.tool_name for c in result.tool_trace] == ["get_kpi"]
    direct = kpis.calculate_kpi("revenue", {"period": "last_month"})
    assert result.request is not None and result.request.period.label == LAST_MONTH  # type: ignore[union-attr]
    assert format_value(direct.value, direct.unit) in result.response.answer
    assert LAST_MONTH in result.response.answer
    _assert_grounded(result)


# ---- 2. period comparison ------------------------------------------------------------------------


def test_scenario_mrr_change(runner: AgentRunner, kpis: KPIService) -> None:
    result = runner.run("How did MRR change compared with the previous month?")
    assert result.status == "completed"
    current = kpis.calculate_kpi("mrr", {"period": "last_month"})
    previous = kpis.calculate_kpi("mrr", {"period": "previous_month"})
    assert current.value is not None and previous.value is not None
    answer = result.response.answer
    assert format_value(current.value, "SGD") in answer and format_value(previous.value, "SGD") in answer
    assert format_value(abs(current.value - previous.value), "SGD") in answer
    assert any("latest complete month" in a for a in result.response.assumptions)
    _assert_grounded(result)


# ---- 3. dimensional contribution -----------------------------------------------------------------


def test_scenario_segment_contribution(runner: AgentRunner, full_db: Database) -> None:
    result = runner.run("Which segment contributed most to the revenue decline last month?")
    assert result.status == "completed"
    direct = decompose_revenue_change(full_db, "segment", "last_month", "previous_month", as_of=AS_OF)
    top = direct.data[0]  # rows are ordered from the largest decline
    assert top.share_of_gross_decline is not None
    assert top.dimension_value in result.response.answer
    assert format_percent(top.share_of_gross_decline) in result.response.answer
    assert result.response.interpretation and result.response.recommendations
    _assert_grounded(result)


# ---- 4. multi-step investigation -----------------------------------------------------------------


def test_scenario_why_did_revenue_decline(runner: AgentRunner, full_db: Database) -> None:
    result = runner.run("Why did revenue decline last month?")
    assert result.status == "completed"
    assert len(result.plans) == 2 and 6 <= result.total_tool_calls <= 12  # initial playbook + evidence-driven follow-up
    by_country = decompose_revenue_change(full_db, "country", "last_month", "previous_month", as_of=AS_OF)
    by_region = decompose_revenue_change(full_db, "region", "last_month", "previous_month", as_of=AS_OF)
    concentrated = [r for r in (by_country.data[0], by_region.data[0]) if (r.share_of_gross_decline or 0) >= 0.5]
    drilled = [s.arguments.get("filters") for s in result.plans[1].steps]
    expected_member = concentrated[0].dimension_value
    assert all(f == {concentrated[0].dimension: expected_member} for f in drilled)  # the member came from evidence
    text = _text(result)
    assert expected_member in text
    assert "concentrated in" in text or "coincided with" in text  # inference wording, not causal wording
    assert result.response.interpretation and result.response.recommendations
    assert any("statistically unusual" in c or "anomaly" in c.lower() for c in result.response.caveats)
    _assert_grounded(result)


# ---- 5. forecast ---------------------------------------------------------------------------------


def test_scenario_forecast(runner: AgentRunner, full_db: Database) -> None:
    result = runner.run("Forecast revenue for the next 3 months")
    assert result.status == "completed" and [c.tool_name for c in result.tool_trace] == ["forecast_metric"]
    direct = ForecastService(full_db, as_of=AS_OF).forecast(metric="revenue", horizon=3)
    assert len(direct.forecast_points) == 3
    for point in direct.forecast_points:
        assert point.period in result.response.answer
        assert format_value(point.predicted_value, "SGD") in result.response.answer
    assert result.response.answer.count("Forecast") >= 3
    assert any("estimates" in c for c in result.response.caveats)
    _assert_grounded(result)


# ---- 6. anomaly detection ------------------------------------------------------------------------


def test_scenario_anomaly_check(runner: AgentRunner, full_db: Database) -> None:
    result = runner.run("Were there any unusual movements in revenue last month?")
    assert result.status == "completed"
    detectors = sorted(c.input.get("detector", "") for c in result.tool_trace)
    assert detectors == ["forecast_residual", "rolling_zscore"]
    service = AnomalyService(full_db, as_of=AS_OF)
    for detector in detectors:
        report = service.detect("revenue", end_date=AS_OF, detector=detector)
        flagged = {a.period for a in report.anomalies}
        for period in flagged:
            assert period in result.response.answer
        if LAST_MONTH not in flagged:
            assert f"{LAST_MONTH} (" not in result.response.answer
    assert "statistically unusual" in result.response.answer
    _assert_grounded(result)


# ---- 7. support ----------------------------------------------------------------------------------


def test_scenario_support_tickets(runner: AgentRunner, kpis: KPIService) -> None:
    result = runner.run("How did support tickets change last month?")
    assert result.status == "completed"
    current = kpis.calculate_kpi("support_ticket_volume", {"period": "last_month"})
    previous = kpis.calculate_kpi("support_ticket_volume", {"period": "previous_month"})
    answer = result.response.answer
    assert format_value(current.value, "tickets") in answer and format_value(previous.value, "tickets") in answer
    assert LAST_MONTH in answer and PREVIOUS_MONTH in answer
    _assert_grounded(result)


# ---- 8. customers --------------------------------------------------------------------------------


def test_scenario_highest_churn_segment(runner: AgentRunner, full_db: Database) -> None:
    result = runner.run("Which segment had the highest churn last month?")
    assert result.status == "completed"
    direct = churn_by_dimension(full_db, "segment", "last_month", as_of=AS_OF)
    top = max((r for r in direct.data if r.logo_churn_rate is not None), key=lambda r: r.logo_churn_rate or 0.0)
    assert top.dimension_value in result.response.answer
    assert format_percent(top.logo_churn_rate) in result.response.answer
    _assert_grounded(result)


# ---- 9. unsupported ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question", ["What is Apple's stock price?", "What is the weather in Singapore?", "Delete all customers"]
)
def test_scenario_unsupported(runner: AgentRunner, question: str) -> None:
    result = runner.run(question)
    assert result.status == "unsupported_request" and not result.tool_trace and not result.evidence
    response = result.response
    assert not response.key_findings and not response.interpretation and not response.recommendations
    assert "Northwind Cloud" in response.answer


# ---- 10. insufficient evidence -------------------------------------------------------------------


def test_scenario_what_caused_churn(runner: AgentRunner) -> None:
    result = runner.run("What caused churn?")
    assert result.status == "insufficient_evidence"
    text = _text(result)
    assert "does not establish" in text and not causal_sentences(text)
    assert any("does not establish causes" in c for c in result.response.caveats)
    assert any("associations" in c.lower() for c in result.response.caveats)
    assert result.response.evidence  # the observed associations are still shown, with their evidence


@pytest.mark.parametrize(
    ("question", "message"),
    [
        ("What was revenue in 2030?", "after the latest available data"),
        ("What was revenue in March 2023?", "before the data begins"),
        ("What will revenue be in 2027?", "horizon is not supported"),
    ],
)
def test_scenario_outside_the_data(runner: AgentRunner, question: str, message: str) -> None:
    result = runner.run(question)
    assert result.status == "insufficient_evidence" and not result.tool_trace
    assert message in result.response.answer


# ---- determinism and performance -----------------------------------------------------------------


def test_runs_are_reproducible(runner: AgentRunner) -> None:
    first = runner.run("Why did revenue decline last month?")
    second = runner.run("Why did revenue decline last month?")
    assert first.response.answer == second.response.answer
    assert [i.text for i in first.response.key_findings] == [i.text for i in second.response.key_findings]
    assert first.run_id != second.run_id


def test_performance_budget(runner: AgentRunner) -> None:
    budgets: dict[str, float] = {
        "What was revenue last month?": 5.0,
        "Why did revenue decline last month?": 15.0,
        "Forecast revenue for the next 3 months": 15.0,
    }
    timings: dict[str, Any] = {}
    for question, budget in budgets.items():
        start = time.perf_counter()
        runner.run(question)
        timings[question] = elapsed = time.perf_counter() - start
        assert elapsed < budget, timings
