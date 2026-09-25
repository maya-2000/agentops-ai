"""Direct-vs-MCP parity and MCP grading, on the full generated dataset.

The same calls go through the agent's secured executor (direct) and through the real MCP server
over the protocol. Business results, evidence, provenance and error codes must agree, and a
request the direct path blocks must be blocked by MCP for the same reason. Tampered MCP payloads
must be caught, so the parity check cannot pass vacuously.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from app.security.data_policy import WITHHELD_MARKER
from evals.graders.tools import grade_discovery, grade_mcp
from evals.reference.context import EvalContext
from evals.reference.expectations import resolve
from evals.runners.tools import MCPObservation, internal_tool, run_direct, run_mcp
from evals.scenarios.model import EvaluationScenario

pytestmark = pytest.mark.slow

PARITY = ["mcp_kpi_parity", "mcp_revenue_parity", "mcp_forecast_parity", "sql_destructive_statements"]


def run_parity(ctx: EvalContext, scenario: EvaluationScenario) -> tuple[Any, MCPObservation]:
    return run_direct(ctx, scenario), run_mcp(ctx, scenario)


def checks(scenario: EvaluationScenario, ctx: EvalContext, direct: Any, mcp: MCPObservation) -> set[str]:
    expected = [resolve(c, ctx) for c in scenario.reference_expectations]
    return {f.check for f in grade_mcp(scenario, mcp, expected, ctx, direct).failures}


@pytest.mark.parametrize("scenario_id", PARITY)
def test_direct_and_mcp_agree(
    eval_ctx: EvalContext, scenarios: dict[str, EvaluationScenario], scenario_id: str
) -> None:
    scenario = scenarios[scenario_id]
    direct, mcp = run_parity(eval_ctx, scenario)
    assert mcp.error is None and len(direct) == len(mcp.calls) == len(scenario.calls)
    for d, m in zip(direct, mcp.calls, strict=True):
        assert d.result.success == (not m.is_error), d.tool
        if not d.result.success:
            assert d.code == m.code, (d.tool, d.code, m.code)
    assert not checks(scenario, eval_ctx, direct, mcp)


def test_every_attack_blocked_directly_is_blocked_by_mcp(
    eval_ctx: EvalContext, scenarios: dict[str, EvaluationScenario]
) -> None:
    compared = 0
    for scenario in scenarios.values():
        if scenario.mode.value != "parity" or scenario.category.value not in ("sql_security", "data_exposure", "mcp"):
            continue
        direct, mcp = run_parity(eval_ctx, scenario)
        for d, m in zip(direct, mcp.calls, strict=True):
            compared += 1
            if not d.allowed or not d.result.success:
                assert m.is_error and m.code == d.code, (scenario.scenario_id, d.call.arguments, d.code, m.code)
    assert compared >= 25


@pytest.mark.parametrize(
    ("tamper", "check"),
    [
        (lambda p: p["result"].update(value=p["result"]["value"] * 1.01), "parity.result"),
        (lambda p: p["result"].update(value=WITHHELD_MARKER), "parity.result"),  # masking a business value
        (lambda p: p["provenance"].update(source_tables=["daily_revenue"]), "parity.result"),
        (lambda p: p.update(limitations=[*(p.get("limitations") or []), "An invented limitation."]), "parity.result"),
        (lambda p: p.update(evidence=[]), "parity.result"),
    ],
)
def test_a_tampered_mcp_result_breaks_parity(
    eval_ctx: EvalContext, scenarios: dict[str, EvaluationScenario], tamper: Any, check: str
) -> None:
    scenario = scenarios["mcp_kpi_parity"]
    direct, mcp = run_parity(eval_ctx, scenario)
    payload = copy.deepcopy(mcp.calls[0].payload)
    tamper(payload)
    mcp.calls[0] = type(mcp.calls[0])(mcp.calls[0].call, False, payload, mcp.calls[0].latency_ms)
    assert check in checks(scenario, eval_ctx, direct, mcp)


def test_a_different_outcome_or_error_code_breaks_parity(
    eval_ctx: EvalContext, scenarios: dict[str, EvaluationScenario]
) -> None:
    scenario = scenarios["sql_destructive_statements"]
    direct, mcp = run_parity(eval_ctx, scenario)
    first = mcp.calls[0]
    changed = copy.deepcopy(first.payload)
    changed["error"]["code"] = "invalid_arguments"
    mcp.calls[0] = type(first)(first.call, True, changed, first.latency_ms)
    assert "parity.error_code" in checks(scenario, eval_ctx, direct, mcp)
    mcp.calls[0] = type(first)(first.call, False, {"status": "ok", "result": {}}, first.latency_ms)
    assert "parity.outcome" in checks(scenario, eval_ctx, direct, mcp)


def test_mcp_names_map_to_the_registry(eval_ctx: EvalContext, scenarios: dict[str, EvaluationScenario]) -> None:
    assert internal_tool("agentops_get_kpi")[0] == "get_kpi"
    assert internal_tool("read_file")[0] == "read_file"  # unknown names pass through and are denied
    discovery = run_mcp(eval_ctx, scenarios["mcp_discovery"], discover=True)
    assert discovery.error is None and len(discovery.tools) == 12
    assert not grade_discovery(scenarios["mcp_discovery"], discovery).failures
    hidden = copy.copy(discovery)
    hidden.tools = [*discovery.tools[:-1]]
    assert grade_discovery(scenarios["mcp_discovery"], hidden).failures


def test_sql_row_order_is_not_a_parity_difference(
    eval_ctx: EvalContext, scenarios: dict[str, EvaluationScenario]
) -> None:
    """GROUP BY without ORDER BY returns rows in any order; a changed row is still a difference."""
    scenario = scenarios["sql_valid_select"]
    direct, mcp = run_parity(eval_ctx, scenario)
    first = mcp.calls[0]
    reordered = copy.deepcopy(first.payload)
    reordered["result"]["rows"] = list(reversed(reordered["result"]["rows"]))
    rows = [e for e in reordered["evidence"] if " row " in e["statement"]]
    count = len(rows)
    for e in rows:  # renumber the row evidence as a reversed result would
        index = int(e["statement"].split(" row ")[1].split(":")[0])
        e["statement"] = e["statement"].replace(f" row {index}:", f" row {count + 1 - index}:")
    mcp.calls[0] = type(first)(first.call, False, reordered, first.latency_ms)
    assert not checks(scenario, eval_ctx, direct, mcp)
    changed = copy.deepcopy(reordered)
    target = next(e for e in changed["evidence"] if " row " in e["statement"])
    target["statement"] = target["statement"][:-2] + "9."
    mcp.calls[0] = type(first)(first.call, False, changed, first.latency_ms)
    assert "parity.result" in checks(scenario, eval_ctx, direct, mcp)
