"""Runtime ground-truth leakage: while the benchmark drives the production system, no hidden label,
withheld customer field or planted secret reaches the model, the tools, MCP or any output, and the
production run never opens the ground-truth file.

The evaluation reads the labels once, when its context is opened, before any production call.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from app.llm.deterministic import DeterministicLLM
from evals.engine import canary_secret
from evals.graders.common import CANARY_SECRET
from evals.reference.context import EvalContext
from evals.runners.agent import run_agent
from evals.runners.tools import run_mcp
from evals.scenarios.model import EvaluationScenario

pytestmark = pytest.mark.slow

_OPENED: list[str] = []
_RECORDING = [False]


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if _RECORDING[0] and event == "open" and args:
        _OPENED.append(str(args[0]))


sys.addaudithook(_audit)  # hooks cannot be removed; recording is switched on only while production runs

AGENT_SCENARIOS = [
    "kpi_revenue_last_month",
    "inv_revenue_decline_why",
    "inv_customers_churning_why",
    "cust_highest_churn_segment",
    "an_unusual_revenue",
    "pi_hidden_health",
    "pi_read_seed_file",
    "pi_system_prompt",
    "pi_api_key",
    "exp_agent_customer_names",
]
MCP_SCENARIOS = ["mcp_rep_win_rate", "mcp_campaign_cac", "exp_mcp_customer_risk", "sql_filesystem_reads"]


def specific_markers(ctx: EvalContext) -> list[str]:
    """Event names, descriptions and expected signals (sentences, not generic words)."""
    return [m for m in ctx.labels.leak_markers() if " " in m]


def test_no_label_secret_or_withheld_value_reaches_the_model_or_the_answer(
    eval_ctx: EvalContext, scenarios: dict[str, EvaluationScenario]
) -> None:
    markers = [m.lower() for m in specific_markers(eval_ctx)]
    names = eval_ctx.withheld_values["company_name"]
    ground_truth = str(eval_ctx.source.ground_truth_path)
    for scenario_id in AGENT_SCENARIOS:
        _OPENED.clear()
        with canary_secret():
            _RECORDING[0] = True
            try:
                obs = run_agent(eval_ctx, scenarios[scenario_id], DeterministicLLM)
            finally:
                _RECORDING[0] = False
        assert obs.result is not None, scenario_id
        assert not [p for p in _OPENED if Path(p).name == Path(ground_truth).name], scenario_id
        requests = "\n".join(r.system + r.prompt + json.dumps(r.context, default=str) for r in obs.llm.requests)
        outputs = obs.result.model_dump_json() + json.dumps(obs.logs, default=str)
        echoed = "injected_events" in (scenarios[scenario_id].question or "")  # the attacker's own words
        for surface, text in (("model request", requests), ("run output", outputs)):
            lowered = text.lower()
            leaked = [m[:50] for m in markers if m in lowered]
            assert not leaked, (scenario_id, surface, leaked)
            assert echoed or "injected_events" not in lowered, (scenario_id, surface)
            assert CANARY_SECRET not in text, (scenario_id, surface)
            assert not [n for n in names if n in text], (scenario_id, surface)


def test_no_label_or_withheld_value_reaches_mcp_responses(
    eval_ctx: EvalContext, scenarios: dict[str, EvaluationScenario]
) -> None:
    markers = [m.lower() for m in specific_markers(eval_ctx)]
    names = eval_ctx.withheld_values["company_name"]
    for scenario_id in MCP_SCENARIOS:
        _OPENED.clear()
        with canary_secret():
            _RECORDING[0] = True
            try:
                obs = run_mcp(eval_ctx, scenarios[scenario_id])
            finally:
                _RECORDING[0] = False
        assert obs.error is None and obs.calls, scenario_id
        assert not [p for p in _OPENED if "injected_events" in p], scenario_id
        text = json.dumps([c.payload for c in obs.calls], default=str) + json.dumps(obs.logs, default=str)
        assert not [m for m in markers if m in text.lower()], scenario_id
        assert "injected_events" not in text.lower() and CANARY_SECRET not in text, scenario_id
        assert not [n for n in names if n in text], scenario_id


def test_the_labels_are_only_read_when_the_context_opens(eval_ctx: EvalContext) -> None:
    """The context holds the parsed labels; the production handle is a plain read-only database."""
    assert eval_ctx.labels.events and eval_ctx.db.read_only
    assert not hasattr(eval_ctx.db, "labels") and not hasattr(eval_ctx.db, "ground_truth_path")
