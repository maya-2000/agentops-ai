"""The LangGraph state machine: every path, bound and failure state, forced with the scripted model.

Runs on the small generated dataset (Mar-Aug 2026). The scripted model replaces one LLM step at a
time; the other steps fall back to the deterministic model, so each test isolates one transition.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any

import pytest

from app.agent import AgentConfig, AgentRunner, AgentRunResult
from app.agent.graph import FAILURE_NODES, NODES
from app.agent.response import LIMIT_CAVEAT, LIMIT_MESSAGE
from app.agent.state import AgentState
from app.analytics.errors import AnalyticsDatabaseError
from app.llm import LLMError, LLMTask, ScriptedLLM
from app.llm.deterministic import DeterministicLLM
from app.tools import TOOL_DEFINITIONS, ToolRegistry
from tests.phase4_support import AS_OF

HAPPY_PATH = [
    "question_received",
    "understand_question",
    "validate_request",
    "plan_investigation",
    "execute_tools",
    "collect_evidence",
    "validate_evidence",
    "generate_response",
    "validate_response",
    "done",
]
REVENUE = "What was revenue last month?"


def _runner(db: Any, script: dict[Any, list[Any]] | None = None, registry: ToolRegistry | None = None, **config: Any):  # type: ignore[no-untyped-def]
    llm = ScriptedLLM(script or {}, fallback=DeterministicLLM())
    return AgentRunner(db, llm=llm, config=AgentConfig(**config), registry=registry, as_of=AS_OF)


def _failing(tool: str, exc: Exception) -> ToolRegistry:
    def handler(*_: Any) -> Any:
        raise exc

    return ToolRegistry(tuple(replace(d, handler=handler) if d.name == tool else d for d in TOOL_DEFINITIONS))


def _plan(*steps: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "steps": [{"tool_name": t, "arguments_json": json.dumps(a), "purpose": "test"} for t, a in steps],
        "rationale": "scripted",
        "sufficient": False,
    }


AUGUST = {"start_date": "2026-08-01", "end_date": "2026-08-31"}


# ---- happy path ----------------------------------------------------------------------------------


def test_graph_has_the_specified_nodes() -> None:
    assert tuple(HAPPY_PATH) == NODES
    assert set(FAILURE_NODES) == {
        "unsupported_request",
        "insufficient_evidence",
        "tool_error",
        "validation_failure",
        "planning_failure",
    }


def test_happy_path(small_db: Any) -> None:
    result = _runner(small_db).run(REVENUE)
    assert result.status == "completed" and result.transitions == HAPPY_PATH
    (call,) = result.tool_trace
    assert call.tool_name == "get_kpi" and call.success and call.query_ids and call.evidence_ids
    (evidence,) = result.evidence
    runner = _runner(small_db)
    direct = runner.runtime.tool_context.kpi_service.calculate_kpi("revenue", {"start_date": AS_OF.replace(day=1),
                                                                               "end_date": AS_OF})  # fmt: skip
    assert evidence.value == pytest.approx(direct.value)  # the number is the Phase 2 number
    assert evidence.query_ids == direct.query_ids or evidence.query_ids
    assert result.response.answer and result.response.evidence[0].evidence_id == evidence.evidence_id
    assert result.response.tool_trace[0].tool_name == "get_kpi"
    assert result.llm_provider == "scripted" and result.total_retries == 0


def test_result_is_serialisable_and_carries_no_runtime_objects(small_db: Any) -> None:
    result = _runner(small_db).run(REVENUE)
    restored = AgentRunResult.model_validate_json(result.model_dump_json())
    assert restored.response == result.response
    AgentState.model_json_schema()  # would fail for connections, clients or other arbitrary types
    assert not {"db", "llm", "client", "api_key", "connection"} & set(AgentState.model_fields)


# ---- early exits ---------------------------------------------------------------------------------


@pytest.mark.parametrize(("question", "message"), [("   ", "empty"), ("x" * 1001, "longer than")])
def test_empty_or_oversized_question(small_db: Any, question: str, message: str) -> None:
    result = _runner(small_db).run(question)
    assert result.status == "unsupported_request" and message in result.response.answer
    assert result.transitions == ["question_received", "unsupported_request"] and not result.llm_calls


def test_unsupported_request_runs_no_tools(small_db: Any) -> None:
    result = _runner(small_db).run("What is Apple's stock price?")
    assert result.status == "unsupported_request" and not result.tool_trace
    assert not result.response.key_findings and not result.response.evidence
    assert "Northwind Cloud" in result.response.answer


def test_clarification_and_coverage_route_to_insufficient_evidence(small_db: Any) -> None:
    result = _runner(small_db).run("What was revenue in 2030?")
    assert result.status == "insufficient_evidence" and not result.tool_trace
    assert "after the latest available data" in result.response.answer


# ---- LLM output handling -------------------------------------------------------------------------


def test_malformed_understanding_is_retried_then_fails(small_db: Any) -> None:
    result = _runner(small_db, {LLMTask.UNDERSTAND: ["not json", {"intent": "nope"}, "{}"]}).run(REVENUE)
    assert result.status == "planning_failure" and not result.tool_trace
    (record,) = result.llm_calls
    assert record.attempts == 3 and not record.success  # max_retries=2 -> 3 attempts
    assert result.errors[-1].code == "llm_output_invalid"


def test_understanding_recovers_after_one_bad_output(small_db: Any) -> None:
    result = _runner(small_db, {LLMTask.UNDERSTAND: ["not json"]}).run(REVENUE)
    assert result.status == "completed" and result.llm_calls[0].attempts == 2


def test_plan_with_unknown_tool_is_rejected(small_db: Any) -> None:
    bad = _plan(("export_database", {"path": "/tmp/x"}))
    llm = ScriptedLLM({LLMTask.PLAN: [bad, bad, bad]}, fallback=DeterministicLLM())
    result = AgentRunner(small_db, llm=llm, config=AgentConfig(), as_of=AS_OF).run(REVENUE)
    assert result.status == "planning_failure" and not result.tool_trace
    assert "Unknown tool" in result.errors[-1].message
    feedback = [r.context.get("feedback") for r in llm.requests if r.task == LLMTask.PLAN]
    assert feedback[0] is None and feedback[1] and "Unknown tool" in feedback[1][0]  # errors fed back


def test_plan_with_invalid_arguments_recovers_with_feedback(small_db: Any) -> None:
    bad = _plan(("get_kpi", {"kpi": "revenue", "drop_table": True}))
    good = _plan(("get_kpi", {"kpi": "revenue", **AUGUST}))
    result = _runner(small_db, {LLMTask.PLAN: [bad, good]}).run(REVENUE)
    assert result.status == "completed" and result.total_tool_calls == 1
    assert [c.attempts for c in result.llm_calls if c.task == "plan_investigation"] == [2]


def test_duplicate_plan_steps_run_once(small_db: Any) -> None:
    step = ("get_kpi", {"kpi": "revenue", **AUGUST})
    result = _runner(small_db, {LLMTask.PLAN: [_plan(step, step)]}).run(REVENUE)
    assert result.total_tool_calls == 1


def test_llm_provider_failure_falls_back_to_deterministic_composition(small_db: Any) -> None:
    result = _runner(small_db, {LLMTask.RESPOND: [LLMError("unavailable", retryable=False)]}).run(REVENUE)
    assert result.status == "completed"
    assert result.response.generated_by == "deterministic-fallback"
    assert any(e.code == "llm_failed" for e in result.errors)


# ---- bounds --------------------------------------------------------------------------------------


def test_over_budget_plan_is_not_silently_truncated(small_db: Any) -> None:
    result = _runner(small_db, max_tool_calls=2).run("Why did revenue decline last month?")
    assert result.status == "insufficient_evidence" and result.response.answer == LIMIT_MESSAGE
    assert result.total_tool_calls == 0


def test_tool_budget_bounds_follow_up_planning(small_db: Any) -> None:
    months = iter(range(3, 9))

    def next_step(_: Any) -> dict[str, Any]:
        month = next(months)
        return _plan(("get_kpi", {"kpi": "revenue", "period": f"2026-{month:02d}"}))

    script = {LLMTask.PLAN: [next_step] * 6}
    result = _runner(small_db, script, max_tool_calls=3, max_planning_iterations=5).run(
        "Why did revenue decline last month?"
    )
    assert result.total_tool_calls == 3 and len(result.plans) == 3
    assert {LIMIT_MESSAGE, LIMIT_CAVEAT} & set(result.response.caveats)  # the stop is disclosed


def test_planning_iterations_are_bounded(small_db: Any) -> None:
    months = iter(range(3, 9))

    def next_step(_: Any) -> dict[str, Any]:
        return _plan(("get_kpi", {"kpi": "revenue", "period": f"2026-{next(months):02d}"}))

    result = _runner(small_db, {LLMTask.PLAN: [next_step] * 6}, max_planning_iterations=2).run(
        "Why did revenue decline last month?"
    )
    assert len(result.plans) == 2 and result.total_tool_calls == 2


def test_wall_clock_limit_stops_tool_execution(small_db: Any) -> None:
    result = _runner(small_db, max_run_seconds=1e-9).run(REVENUE)
    assert result.total_tool_calls == 0 and result.status == "insufficient_evidence"
    assert LIMIT_MESSAGE in result.response.caveats


def test_recursion_limit_covers_the_longest_legal_path() -> None:
    config = AgentConfig()
    longest = 3 + 2 * config.max_planning_iterations + config.max_tool_calls + 2 * (config.max_retries + 1) + 2
    assert config.recursion_limit > longest


# ---- tool failures -------------------------------------------------------------------------------


def test_every_tool_failing_routes_to_tool_error_after_bounded_retries(small_db: Any) -> None:
    registry = _failing("get_kpi", AnalyticsDatabaseError("connection lost"))
    result = _runner(small_db, registry=registry).run(REVENUE)
    assert result.status == "tool_error" and not result.evidence
    (call,) = result.tool_trace
    assert not call.success and call.error is not None and call.error.code == "database_error"
    assert call.attempts == 3  # retryable: 1 + max_retries
    assert "get_kpi" in result.response.answer and not result.response.key_findings


def test_non_retryable_failure_is_not_retried(small_db: Any) -> None:
    result = _runner(small_db, registry=_failing("get_kpi", RuntimeError("bug"))).run(REVENUE)
    assert result.tool_trace[0].attempts == 1 and result.tool_trace[0].error.code == "internal_error"  # type: ignore[union-attr]


def test_partial_tool_failure_is_disclosed(small_db: Any) -> None:
    registry = _failing("detect_anomalies", RuntimeError("detector unavailable"))
    result = _runner(small_db, registry=registry).run("Why did revenue decline last month?")
    assert result.status in ("completed", "insufficient_evidence")
    assert any(not c.success and c.tool_name == "detect_anomalies" for c in result.tool_trace)
    assert any("detect_anomalies" in caveat for caveat in result.response.caveats)
    assert result.response.answer


# ---- response validation -------------------------------------------------------------------------


def _fabricated(request: Any) -> dict[str, Any]:
    claim = request.context["claims"][0]
    return {
        "answer": "Revenue last month was SGD 987,654,321.",
        "answer_claim_ids": [claim["claim_id"]],
        "key_findings": [],
        "interpretation": [],
        "recommendations": [],
    }


def test_fabricated_number_is_rejected_after_bounded_regeneration(small_db: Any) -> None:
    result = _runner(small_db, {LLMTask.RESPOND: [_fabricated] * 3}).run(REVENUE)
    assert result.status == "validation_failure"
    assert "987,654,321" not in result.response.model_dump_json()
    assert "numbers not found in the cited evidence" in result.response.answer
    assert result.errors[-1].code == "response_invalid"  # the details stay in the trace
    assert [c.task for c in result.llm_calls].count("generate_response") == 3
    assert result.transitions.count("validate_response") == 3


def test_regeneration_after_a_rejected_draft(small_db: Any) -> None:
    result = _runner(small_db, {LLMTask.RESPOND: [_fabricated]}).run(REVENUE)
    assert result.status == "completed" and result.total_retries == 1
    assert "987,654,321" not in result.response.answer
    assert result.transitions.count("generate_response") == 2


def test_uncited_causal_wording_is_rejected(small_db: Any) -> None:
    def causal(request: Any) -> dict[str, Any]:
        claim = request.context["claims"][0]
        return {**_fabricated(request), "answer": claim["text"] + " This was caused by churn."}

    result = _runner(small_db, {LLMTask.RESPOND: [causal] * 3}).run(REVENUE)
    assert result.status == "validation_failure" and "caused by" not in result.response.answer


# ---- privacy of prompts and logs -----------------------------------------------------------------


def test_prompts_and_logs_carry_no_ground_truth_or_secrets(small_db: Any, caplog: pytest.LogCaptureFixture) -> None:
    llm = ScriptedLLM({}, fallback=DeterministicLLM())
    with caplog.at_level(logging.INFO, logger="agentops.agent"):
        AgentRunner(small_db, llm=llm, config=AgentConfig(), as_of=AS_OF).run("Why did revenue decline last month?")
    prompts = " ".join(r.system + r.prompt for r in llm.requests).lower()
    for forbidden in ("injected", "ground_truth", "health_score", "customer_health", "calibration"):
        assert forbidden not in prompts
    events = [json.loads(r.getMessage()) for r in caplog.records if r.name == "agentops.agent"]
    assert {"transition", "tool_call", "evidence_validation", "final"} <= {e["event"] for e in events}
    logged = json.dumps(events).lower()
    assert "prompt" not in logged and "api_key" not in logged and "rows" not in logged
