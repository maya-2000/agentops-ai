"""Shared execution: the agent (LangGraph) and MCP both run tools through ``app/security/execution.py``.

The evaluation's executor probe (``evals.runners.shared.instrument_executor``) records every call
the shared ``SecuredToolExecutor`` handles. Each control is exercised on both entry points with
the same failing or hostile condition: authorization, the execution deadline, retries, output
validation, budget charging and security events. The probe only wraps the real executor, so an
entry point that bypassed it would show up here with no records.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import pytest

from app.agent.config import AgentConfig
from app.agent.graph import AgentRuntime
from app.analytics.errors import AnalyticsDatabaseError
from app.llm.base import LLMTask
from app.llm.deterministic import DeterministicLLM
from app.mcp.registry import MCP_TOOL_SPECS
from app.security.authorization import AuthorizationContext
from app.security.budget import BudgetUsage
from app.security.execution import SecuredToolExecutor
from app.tools import handlers
from app.tools.base import ToolContext
from app.tools.registry import ToolRegistry
from evals.graders.tools import grade_shared
from evals.runners.shared import ExecutorRecord, SharedObservation, instrument_executor
from evals.scenarios.model import EvaluationScenario
from tests.phase4_support import AS_OF
from tests.phase5_support import plan, raising, runner, with_handler
from tests.phase6_support import call, mcp_config, server, structured

REVENUE = "What was revenue last month?"
KPI = ("agentops_get_kpi", {"kpi": "revenue", "period": "2026-08"})
DROP = ("agentops_run_safe_sql", {"sql": "DROP TABLE customers"})
SQL_INTENT = next(s.intent for s in MCP_TOOL_SPECS if s.tool == "run_safe_sql")  # as MCP authorizes it


def both_paths(
    db: Any,
    *,
    registry: ToolRegistry | None = None,
    agent_plan: tuple[tuple[str, dict[str, Any]], ...] = (),
    mcp_calls: tuple[tuple[str, dict[str, Any]], ...] = (KPI,),
    **limits: Any,
) -> tuple[Any, list[dict[str, Any]], dict[str, list[ExecutorRecord]]]:
    records: list[ExecutorRecord] = []
    entry = ["agent"]
    script = {LLMTask.PLAN: [plan(*agent_plan)] * 3} if agent_plan else None
    with instrument_executor(records, entry):
        agent, _ = runner(db, script, registry=registry, **limits)
        result = agent.run(REVENUE)
        entry[0] = "mcp"
        target = server(db, mcp_config(AgentConfig(**limits)), registry=registry)
        outcomes = [structured(call(target, name, arguments)) for name, arguments in mcp_calls]
    by_entry = {e: [r for r in records if r.entry_point == e] for e in ("agent", "mcp")}
    return result, outcomes, by_entry


def test_an_allowed_call_gets_every_control_on_both_paths(small_db: Any) -> None:
    result, outcomes, records = both_paths(small_db)
    assert result.status == "completed" and outcomes[0]["status"] == "ok"
    for entry in ("agent", "mcp"):
        (record,) = [r for r in records[entry] if r.tool == "get_kpi"]
        assert record.allowed and record.deadline_applied and record.budget_charged, entry
        assert record.attempts == 1 and "tool_authorized" in record.events, entry


def test_retries_follow_the_same_policy_on_both_paths(small_db: Any) -> None:
    registry = with_handler("get_kpi", raising(AnalyticsDatabaseError("connection reset")))
    _, outcomes, records = both_paths(small_db, registry=registry, max_retries=1)
    assert outcomes[0]["error"]["category"] == "TOOL_FAILURE"
    for entry in ("agent", "mcp"):
        (record,) = records[entry]
        assert record.code == "database_error" and record.attempts == 2, entry  # 1 + max_retries
        assert record.events.count("retry") == 1 and record.budget_charged, entry


def test_output_validation_rejects_the_same_bad_output_on_both_paths(small_db: Any) -> None:
    def no_provenance(ctx: ToolContext, parsed: Any) -> Any:
        return replace(handlers.get_kpi(ctx, parsed), query_ids=[])

    _, outcomes, records = both_paths(small_db, registry=with_handler("get_kpi", no_provenance))
    assert outcomes[0]["error"]["category"] == "VALIDATION_FAILURE"
    for entry in ("agent", "mcp"):
        (record,) = records[entry]
        assert record.code == "invalid_tool_output" and "tool_output_rejected" in record.events, entry


def test_the_deadline_discards_a_late_result_on_both_paths(small_db: Any) -> None:
    def slow(ctx: ToolContext, parsed: Any) -> Any:
        time.sleep(0.15)
        return handlers.get_kpi(ctx, parsed)

    _, outcomes, records = both_paths(small_db, registry=with_handler("get_kpi", slow), tool_timeout_seconds=0.05)
    assert outcomes[0]["error"]["category"] == "TIMEOUT"
    for entry in ("agent", "mcp"):
        (record,) = records[entry]
        assert record.deadline_applied and record.code == "timeout" and "timeout" in record.events, entry


def test_authorization_denies_the_same_call_on_both_paths(small_db: Any) -> None:
    result, outcomes, records = both_paths(
        small_db, agent_plan=(("run_safe_sql", {"sql": "DROP TABLE customers"}),), mcp_calls=(DROP,)
    )
    # The agent refuses the plan before execution; nothing reaches the tools.
    assert result.status == "planning_failure" and not [r for r in records["agent"] if r.allowed]
    assert {"plan_rejected", "sql_rejected"} & {e.event_type for e in result.security_events}
    (mcp,) = records["mcp"]
    assert not mcp.allowed and not mcp.budget_charged and "sql_rejected" in mcp.events
    assert outcomes[0]["error"]["category"] == "UNSAFE_QUERY"
    # The agent's own executor, asked the same thing directly, decides exactly as MCP did.
    runtime = AgentRuntime(small_db, DeterministicLLM(), AgentConfig(), ToolRegistry(), AS_OF)
    direct = runtime.executor.execute(
        run_id="R-test",
        call_id="T1",
        tool_name="run_safe_sql",
        arguments={"sql": "DROP TABLE customers"},
        purpose="test",
        context=AuthorizationContext(intent=SQL_INTENT),
        budget=runtime.run_budget,
        usage=BudgetUsage(),
    )
    assert not direct.decision.allowed and direct.result.error is not None
    assert direct.result.error.code == mcp.code


def test_budgets_are_enforced_by_the_executor_on_both_paths(small_db: Any) -> None:
    sql = ("agentops_run_safe_sql", {"sql": "SELECT COUNT(*) AS n FROM customers"})
    _, outcomes, records = both_paths(small_db, mcp_calls=(sql,), max_sql_calls=0)
    (mcp,) = records["mcp"]
    assert not mcp.allowed and mcp.code == "budget_exceeded" and not mcp.budget_charged
    assert outcomes[0]["error"]["category"] == "RESOURCE_LIMIT"
    runtime = AgentRuntime(small_db, DeterministicLLM(), AgentConfig(max_sql_calls=0), ToolRegistry(), AS_OF)
    direct = runtime.executor.execute(
        run_id="R-test",
        call_id="T1",
        tool_name="run_safe_sql",
        arguments={"sql": "SELECT COUNT(*) AS n FROM customers"},
        purpose="test",
        context=AuthorizationContext(intent=SQL_INTENT),
        budget=runtime.run_budget,
        usage=BudgetUsage(),
    )
    assert direct.result.error is not None and direct.result.error.code == "budget_exceeded"


def test_the_probe_restores_the_executor_and_reports_a_bypass(scenarios: dict[str, EvaluationScenario]) -> None:
    original = SecuredToolExecutor.execute
    with pytest.raises(RuntimeError), instrument_executor([], ["agent"]):
        assert SecuredToolExecutor.execute is not original
        raise RuntimeError("boom")
    assert SecuredToolExecutor.execute is original
    scenario = scenarios["shared_execution_paths"]
    allowed = ExecutorRecord("agent", "get_kpi", True, None, True, True, 1, ["tool_authorized"])
    bypass = grade_shared(scenario, SharedObservation([allowed], "completed", [], 0.0))
    assert "shared.mcp.uses_executor" in {f.check for f in bypass.failures}
    no_deadline = ExecutorRecord("mcp", "get_kpi", True, None, False, True, 1, ["tool_authorized"])
    missing = grade_shared(scenario, SharedObservation([allowed, no_deadline], "completed", [], 0.0))
    assert "shared.mcp.deadline" in {f.check for f in missing.failures}
