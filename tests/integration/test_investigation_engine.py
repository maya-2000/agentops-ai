"""Phase 10: the investigation engine end to end on a generated dataset (deterministic model, no network).

objective -> screen -> understand -> plan -> execute (every call through ``SecuredToolExecutor``) ->
evidence -> claims -> cross-finding validation -> drivers -> recommendations -> decision brief.
Expected values are never written into the tests: they come from the deterministic analytics the
tools wrap, or from the investigation's own evidence.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from app.analytics.revenue import decompose_revenue_change
from app.database.base import Database
from app.investigation import engine
from app.investigation.brief import TRIMMED, brief_chars
from app.investigation.engine import CANCELLED_MESSAGE, DEADLINE_MESSAGE, Investigator
from app.investigation.models import FINDING_LABELS, STOP_MESSAGE, Investigation
from app.investigation.steps import SKIP_REASONS
from app.investigation.validation import text_problems
from app.llm.base import LLMTask
from app.security.authorization import SQL_TOOL
from tests.phase4_support import AS_OF
from tests.phase5_support import raising, with_handler
from tests.phase10_support import (
    BRIEF,
    CAUSAL_WORDS,
    PRICE_CAUSE,
    REVENUE,
    SUPPORT,
    graph_of,
    investigator,
    user_texts,
)

PROGRESS_KEYS = {"stage", "label", "steps", "step_id", "title", "area", "tool_name", "status", "duration_ms"}


@pytest.fixture(scope="module")
def agent(small_db: Database) -> Investigator:
    return investigator(small_db)


@pytest.fixture(scope="module")
def revenue(agent: Investigator) -> Investigation:
    return agent.investigate(REVENUE)


# ------------------------------------------------------------------ lifecycle and execution


def test_a_revenue_investigation_runs_its_plan_through_the_secured_executor(revenue: Investigation) -> None:
    assert revenue.status == "completed" and revenue.template == "revenue" and revenue.message is None
    assert revenue.investigation_id.startswith("I-") and revenue.completed_at is not None
    assert revenue.plan is not None and [s.step_id for s in revenue.steps] == [s.step_id for s in revenue.plan.steps]
    ran = [s for s in revenue.steps if s.status in ("completed", "failed")]
    assert len(revenue.tool_trace) == len(ran) >= 5
    by_step = {s.step_id: s for s in revenue.plan.steps}
    for call in revenue.tool_trace:
        assert call.step_id in by_step and call.tool_name == by_step[call.step_id].tool_name
        assert call.tool_name != SQL_TOOL and call.success
    usage = revenue.budget.usage
    assert usage.tool_calls == len(revenue.tool_trace) <= revenue.budget.max_tool_calls
    assert usage.sql_calls == 0 and usage.model_calls == 1 and not usage.exhausted
    assert revenue.efficiency.duplicate_tool_calls == 0
    assert revenue.efficiency.tool_calls == len(revenue.tool_trace)
    denied = [e for e in revenue.security_events if e.decision == "deny"]
    assert not denied, [e.event_type for e in denied]


def test_every_finding_is_traceable_to_evidence(revenue: Investigation) -> None:
    graph = graph_of(revenue)
    assert not graph.verify_integrity()
    trace_ids = {c.call_id for c in revenue.tool_trace}
    assert revenue.findings
    for finding in revenue.findings:
        claim = graph.claims[finding.claim_id]
        assert claim.support_status in ("supported", "partially_supported")
        assert finding.label == FINDING_LABELS[claim.claim_type] and finding.label != "recommended"
        assert finding.evidence_ids and set(finding.evidence_ids) <= set(graph.evidence)
        assert {graph.evidence[e].tool_call_id for e in finding.evidence_ids} <= trace_ids
        about = graph.evidence[claim.subject.evidence_id] if claim.subject else graph.evidence[finding.evidence_ids[0]]
        assert (finding.metric, finding.unit, finding.period, finding.comparison_period) == (
            about.metric,
            about.unit,
            about.period_label,
            about.comparison_label,
        )
        assert (finding.dimension, finding.breakdown, finding.filters) == (
            about.dimension,
            about.dimension_value,
            about.filters,
        )
        assert set(finding.step_ids) <= {s.step_id for s in revenue.steps}
        assert not text_problems(finding.text, [finding.claim_id], graph), finding.text


def test_numbers_come_from_the_deterministic_analytics(revenue: Investigation, small_db: Database) -> None:
    """The outcome and the regional contributions are the analytics layer's numbers, not the model's."""
    assert revenue.plan is not None
    rows = decompose_revenue_change(small_db, "region", "2026-08", "2026-07", as_of=AS_OF).data
    graph = graph_of(revenue)
    outcome = next(f for f in revenue.findings if f.primary)
    assert (outcome.metric, outcome.period, outcome.comparison_period) == ("revenue", "2026-08", "2026-07")
    about = graph.evidence[graph.claims[outcome.claim_id].subject.evidence_id]  # type: ignore[union-attr]
    assert about.attributes["absolute_change"] == pytest.approx(sum(r.absolute_change for r in rows), abs=0.01)
    regional = [
        d
        for d in revenue.brief.drivers  # type: ignore[union-attr]
        if d.relationship == "contributes_to" and d.name.startswith("Region ")
    ]
    for driver in regional:
        finding = revenue.finding(driver.finding_ids[0])
        assert finding is not None and finding.dimension == "region"
        row = next(r for r in rows if r.dimension_value == finding.breakdown)
        expected = row.share_of_gross_decline if outcome.direction == "decrease" else row.share_of_gross_increase
        assert driver.share == pytest.approx(expected)


def test_the_brief_summary_and_texts_are_validated_and_non_causal(revenue: Investigation) -> None:
    brief = revenue.brief
    assert brief is not None and brief.complete
    graph = graph_of(revenue)
    cited = [revenue.finding(i).claim_id for i in brief.summary_finding_ids if revenue.finding(i)]  # type: ignore[union-attr]
    assert cited and not text_problems(brief.executive_summary, cited, graph)
    assert "not established causes" in brief.executive_summary
    for text in user_texts(revenue):
        for sentence in text.split(". "):
            if CAUSAL_WORDS.search(sentence):
                assert "not" in sentence.lower() or "cannot" in sentence.lower(), sentence
    assert set(brief.key_finding_ids) <= {f.finding_id for f in revenue.findings}
    assert brief_chars(brief, {f.finding_id: f for f in revenue.findings}) <= revenue.budget.max_output_chars


def test_progress_reports_stages_and_steps_never_reasoning(agent: Investigator) -> None:
    events: list[dict[str, Any]] = []
    result = agent.investigate(REVENUE, on_progress=events.append)
    stages = [e["stage"] for e in events]
    assert stages[:4] == ["started", "understanding", "planning", "plan"]
    assert stages[-3:] == ["validating", "synthesizing", "finished"]
    assert stages.count("step_started") == stages.count("step_finished") == len(result.tool_trace)
    for event in events:
        assert set(event) <= PROGRESS_KEYS, set(event) - PROGRESS_KEYS
        assert isinstance(event["label"], str) and len(event["label"]) < 200
    plan = next(e for e in events if e["stage"] == "plan")
    assert [s["step_id"] for s in plan["steps"]] == [s.step_id for s in result.steps]
    assert all(set(s) == {"step_id", "title", "area", "tool_name", "depends_on"} for s in plan["steps"])


def test_a_failing_observer_never_changes_the_investigation(agent: Investigator, revenue: Investigation) -> None:
    def broken(_: dict[str, Any]) -> None:
        raise RuntimeError("observer failed")

    result = agent.investigate(REVENUE, on_progress=broken)
    assert result.status == "completed"
    assert [f.text for f in result.findings] == [f.text for f in revenue.findings]


def test_investigations_are_deterministic_and_independent(agent: Investigator, revenue: Investigation) -> None:
    again = agent.investigate(REVENUE)
    assert again.investigation_id != revenue.investigation_id
    assert [f.text for f in again.findings] == [f.text for f in revenue.findings]
    assert again.brief is not None and revenue.brief is not None
    assert again.brief.executive_summary == revenue.brief.executive_summary


def test_the_investigation_id_is_validated(agent: Investigator) -> None:
    assert agent.investigate(REVENUE, investigation_id="I-custom-1").investigation_id == "I-custom-1"
    with pytest.raises(ValueError):
        agent.investigate(REVENUE, investigation_id="../../etc/passwd")


# ------------------------------------------------------------------ dependencies, conditions, bindings, reuse


def test_the_drill_down_is_bound_to_the_concentrated_region(revenue: Investigation) -> None:
    assert revenue.plan is not None
    drill = revenue.plan.steps[-1]
    record = next(r for r in revenue.steps if r.step_id == drill.step_id)
    source = next(r for r in revenue.steps if r.step_id == drill.depends_on[0])
    graph = graph_of(revenue)
    top = [graph.evidence[e] for e in source.evidence_ids if graph.evidence[e].attributes.get("rank") == 1]
    if record.status == "skipped":
        assert record.reason == SKIP_REASONS["concentrated"]
        return
    assert record.status == "completed" and top
    assert record.arguments["filters"] == {"region": top[0].dimension_value}


def test_feature_steps_are_bound_to_the_adoption_result(agent: Investigator) -> None:
    result = agent.investigate(SUPPORT)
    assert result.template == "product_support" and result.plan is not None
    bound = [s for s in result.plan.steps if s.binding is not None]
    source = next(r for r in result.steps if r.step_id == bound[0].depends_on[0])
    graph = graph_of(result)
    features = [graph.evidence[e].dimension_value for e in source.evidence_ids if graph.evidence[e].dimension_value]
    for step in bound:
        record = next(r for r in result.steps if r.step_id == step.step_id)
        assert step.binding is not None
        assert record.arguments["filters"]["product_feature"] == features[step.binding.index - 1]


def test_a_failed_step_skips_its_dependents_and_is_reported(small_db: Database) -> None:
    registry = with_handler("analyze_revenue", raising(RuntimeError("revenue analytics unavailable")))
    result = investigator(small_db, registry=registry).investigate(REVENUE)
    records = {r.step_id: r for r in result.steps}
    assert result.plan is not None
    for step in result.plan.steps:
        record = records[step.step_id]
        if step.tool_name == "analyze_revenue" and not step.depends_on:
            assert record.status == "failed" and record.reason and "unavailable" not in record.reason
        elif step.depends_on:
            assert record.status == "skipped" and record.reason == SKIP_REASONS["dependency"]
        else:
            assert record.status == "completed"
    assert result.status == "insufficient_evidence"  # the outcome could not be measured
    assert not any(f.primary for f in result.findings)
    assert result.brief is not None and not result.brief.drivers and not result.brief.recommendations
    assert any(u.startswith("Not analysed:") for u in result.brief.uncertainty)
    # The exception detail stays in the developer trace; user-facing texts carry fixed messages only.
    assert all("unavailable" not in text for text in user_texts(result))


def test_unmet_conditions_skip_the_step_without_a_call(small_db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "condition_met", lambda condition, evidence: False)
    result = investigator(small_db).investigate(REVENUE)
    assert result.plan is not None
    for step in result.plan.steps:
        record = next(r for r in result.steps if r.step_id == step.step_id)
        if step.condition == "outcome_changed":
            assert record.status == "skipped" and record.reason == SKIP_REASONS["outcome_changed"]
        elif step.condition == "concentrated":
            assert record.status == "skipped"  # its dependency was skipped
    called = {c.step_id for c in result.tool_trace}
    assert not called & {s.step_id for s in result.plan.steps if s.condition}


def test_an_identical_call_is_reused_not_run_twice(small_db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    original = engine.plan_investigation

    def with_duplicate(template: Any, request: Any, *, as_of: Any) -> Any:
        plan = original(template, request, as_of=as_of)
        again = plan.steps[0].model_copy(update={"step_id": f"S{len(plan.steps) + 1}", "title": "Measure again"})
        return plan.model_copy(update={"steps": [*plan.steps, again]})

    monkeypatch.setattr(engine, "plan_investigation", with_duplicate)
    result = investigator(small_db).investigate(REVENUE)
    last = result.steps[-1]
    assert last.status == "reused" and last.reused_from == "S1" and last.call_id == result.steps[0].call_id
    assert last.evidence_ids == result.steps[0].evidence_ids
    assert result.efficiency.steps_reused == 1 and result.efficiency.duplicate_tool_calls == 0
    assert len(result.tool_trace) == len(result.steps) - 1 - result.efficiency.steps_skipped


# ------------------------------------------------------------------ budgets (enforced in code)


def test_the_tool_call_budget_stops_the_investigation_safely(small_db: Database) -> None:
    result = investigator(small_db, max_investigation_tool_calls=3).investigate(REVENUE)
    assert result.status == "budget_exhausted" and result.message == STOP_MESSAGE
    assert len(result.tool_trace) == 3 and "tool_calls" in result.budget.exhausted
    not_run = [r for r in result.steps if r.status == "not_run"]
    assert not_run and all("budget was reached" in (r.reason or "") for r in not_run)
    brief = result.brief
    assert brief is not None and not brief.complete
    assert not brief.drivers and not brief.recommendations and not brief.contradictions
    assert any(STOP_MESSAGE in u for u in brief.uncertainty)
    assert any(e.event_type == "budget_exceeded" for e in result.security_events)


def test_the_step_budget_stops_the_investigation(small_db: Database) -> None:
    result = investigator(small_db, max_investigation_steps=2).investigate(REVENUE)
    assert result.status == "budget_exhausted" and result.budget.steps_run == 2
    assert "steps" in result.budget.exhausted and len(result.tool_trace) == 2


def test_the_evidence_budget_stops_the_investigation(small_db: Database) -> None:
    result = investigator(small_db, max_investigation_evidence=10).investigate(REVENUE)
    assert result.status == "budget_exhausted" and "evidence" in result.budget.exhausted
    assert any(r.status == "not_run" for r in result.steps)


def test_the_runtime_budget_stops_the_investigation(small_db: Database) -> None:
    from app.tools import TOOL_DEFINITIONS

    real = next(d for d in TOOL_DEFINITIONS if d.name == "get_kpi").handler

    def slow(*args: Any) -> Any:
        time.sleep(0.15)
        return real(*args)

    registry = with_handler("get_kpi", slow)
    result = investigator(small_db, registry=registry, max_investigation_seconds=0.3).investigate(REVENUE)
    assert result.status == "budget_exhausted" and result.message == STOP_MESSAGE
    assert result.budget.exhausted and any(r.status == "not_run" for r in result.steps)


@pytest.mark.parametrize("objective", [BRIEF, REVENUE])
def test_the_brief_is_capped_at_the_output_budget(small_db: Database, objective: str) -> None:
    result = investigator(small_db, max_investigation_output_chars=1000).investigate(objective)
    brief = result.brief
    assert brief is not None and TRIMMED in brief.uncertainty
    assert result.budget.output_chars == brief_chars(brief, {f.finding_id: f for f in result.findings})
    assert result.budget.output_chars <= 1000
    assert brief.executive_summary and brief.key_finding_ids
    full = investigator(small_db).investigate(objective)
    assert result.budget.output_chars < full.budget.output_chars


# ------------------------------------------------------------------ cancellation and deadline


def test_a_cancelled_investigation_stops_before_its_next_step(small_db: Database) -> None:
    cancel = threading.Event()

    def on_progress(event: dict[str, Any]) -> None:
        if event["stage"] == "step_finished":
            cancel.set()

    result = investigator(small_db).investigate(REVENUE, cancel=cancel, on_progress=on_progress)
    assert result.status == "cancelled" and result.message == CANCELLED_MESSAGE
    assert len(result.tool_trace) == 1
    assert not result.findings and not result.evidence and not result.claims and result.brief is None
    assert all(r.status in ("completed", "not_run") for r in result.steps)


def test_a_deadline_stops_the_investigation(small_db: Database) -> None:
    result = investigator(small_db).investigate(REVENUE, deadline_seconds=0)
    assert result.status == "cancelled" and result.message == DEADLINE_MESSAGE
    assert not result.tool_trace and not result.findings


# ------------------------------------------------------------------ controlled outcomes


def test_a_causal_question_returns_insufficient_evidence(agent: Investigator) -> None:
    result = agent.investigate(PRICE_CAUSE)
    assert result.status == "insufficient_evidence"
    assert result.message and "cannot establish the cause" in result.message
    assert result.brief is not None
    assert any("cannot establish" in u for u in result.brief.uncertainty)
    assert any("Pricing changes" in u for u in result.brief.uncertainty)
    for text in user_texts(result):
        assert "price increase caused" not in text.lower()


def test_an_unsupported_objective_runs_no_tools(agent: Investigator) -> None:
    result = agent.investigate("What will the weather be in Paris tomorrow?")
    assert result.status == "unsupported" and not result.tool_trace and result.plan is None


def test_an_incomplete_current_quarter_asks_for_clarification(agent: Investigator) -> None:
    result = agent.investigate("Why is revenue down this quarter?")
    assert result.status == "insufficient_evidence" and not result.tool_trace
    assert result.message and "last complete quarter" in result.message


def test_the_latest_quarter_is_compared_with_the_previous_quarter(agent: Investigator) -> None:
    result = agent.investigate("Why did revenue change in the latest quarter?")
    assert result.plan is not None
    assert (result.plan.period_label, result.plan.comparison_label) == ("2026-Q2", "2026-Q1")


def test_unusable_model_output_fails_the_investigation_safely(small_db: Database) -> None:
    bad = investigator(small_db, {LLMTask.UNDERSTAND: ["not json", "{}", {"intent": "nope"}, "still not json"]})
    result = bad.investigate(REVENUE)
    assert result.status == "failed" and not result.tool_trace
    assert result.message == "The objective could not be interpreted."
    assert result.errors and result.errors[0].code == "llm_output_invalid"


def test_a_management_brief_has_only_supported_sections(agent: Investigator) -> None:
    result = agent.investigate(BRIEF)
    assert result.status == "completed" and result.template == "management_brief"
    brief = result.brief
    assert brief is not None and brief.sections
    by_id = {f.finding_id: f for f in result.findings}
    for section in brief.sections:
        assert section.finding_ids and all(by_id[i].area == section.area for i in section.finding_ids)
    assert {s.area for s in brief.sections} == {f.area for f in result.findings}
    assert any("does not explain causes" in u for u in brief.uncertainty)


def test_the_objective_premise_is_checked_against_the_evidence(agent: Investigator) -> None:
    result = agent.investigate(SUPPORT)  # "increase"
    outcome = next((f for f in result.findings if f.primary), None)
    assert result.brief is not None and outcome is not None
    premise = [u for u in result.brief.uncertainty if u.startswith("The objective describes")]
    if outcome.direction == "decrease":
        assert premise and "decline" in premise[0]
    else:
        assert not premise
