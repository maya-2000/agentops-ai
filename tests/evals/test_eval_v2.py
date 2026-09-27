"""The investigation benchmark (eval_v2, Phase 10): dataset, grader, CLI, critical suite and multi-seed.

eval_v1 is untouched: it keeps its own model, schema version and loader. The grader is tested by
corrupting a real investigation one defect at a time: every corruption must be detected by the check
that owns it, so a passing benchmark is not a vacuous one.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from app.agent.records import ToolCallRecord
from app.investigation import Investigation
from evals.engine import EngineConfig
from evals.graders.investigation import grade_investigation
from evals.investigation_benchmark import run_investigation_scenarios
from evals.reference import investigation as ref
from evals.reference.context import DataSource, EvalContext, generated_source
from evals.run import main
from evals.runners.investigation import InvestigationObservation, run_scenario
from evals.scenarios.investigation import (
    INVESTIGATION_SCHEMA_VERSION,
    InvestigationCategory,
    InvestigationDataset,
    InvestigationMode,
    InvestigationScenario,
    load_investigation_dataset,
    select_investigations,
)
from evals.scenarios.loader import dataset_path, load_dataset
from evals.scenarios.model import SCHEMA_VERSION

# Numbers a scenario may contain: budgets, HTTP status, cancellation points (never a business value).
STRUCTURAL_KEYS = {"limits", "max_tool_calls", "http_status", "cancel_after_steps", "deadline_seconds", "min_findings"}


@pytest.fixture(scope="module")
def v2() -> InvestigationDataset:
    return load_investigation_dataset()


@pytest.fixture(scope="module")
def by_id(v2: InvestigationDataset) -> dict[str, InvestigationScenario]:
    return {s.scenario_id: s for s in v2.scenarios}


# ------------------------------------------------------------------ the dataset


def test_eval_v1_is_unchanged_and_separate(v2: InvestigationDataset) -> None:
    v1 = load_dataset("eval_v1")
    assert v1.manifest.schema_version == SCHEMA_VERSION == "1.0" and len(v1.scenarios) == 89
    assert v2.manifest.dataset_version == "eval_v2" and v2.manifest.schema_version == INVESTIGATION_SCHEMA_VERSION
    with pytest.raises(ValueError):
        load_dataset("eval_v2")  # the v1 model does not accept investigation scenarios
    with pytest.raises(ValueError):
        InvestigationDataset.model_validate(json.loads(dataset_path("eval_v1").read_text(encoding="utf-8")))


def test_the_dataset_covers_every_category_and_mode(v2: InvestigationDataset) -> None:
    assert len(v2.scenarios) >= 50
    categories = Counter(s.category for s in v2.scenarios)
    assert set(categories) == set(InvestigationCategory) and len(InvestigationCategory) == 18
    assert {s.mode for s in v2.scenarios} == set(InvestigationMode)
    assert len({s.scenario_id for s in v2.scenarios}) == len(v2.scenarios)


def test_suites_have_the_documented_sizes(v2: InvestigationDataset) -> None:
    critical = select_investigations(v2.scenarios, suite="critical")
    multi_seed = select_investigations(v2.scenarios, suite="multi_seed")
    assert 10 <= len(critical) <= 15 and 10 <= len(multi_seed) <= 15
    assert {InvestigationCategory.SECURITY, InvestigationCategory.BUDGET} <= {s.category for s in critical}


def test_the_step_28_criteria_are_all_exercised(v2: InvestigationDataset) -> None:
    references = {r.kind for s in v2.scenarios for r in s.references}
    assert references == {"outcome_change", "top_contribution"}
    tags = {t for s in v2.scenarios for t in s.tags}
    assert {"budget", "sql", "ask_compat"} <= tags
    objectives = " ".join(s.objective.lower() for s in v2.scenarios)
    for attack in ("forever", "every available tool", "1,000 times", "ignore the previous limits"):
        assert attack in objectives
    assert any(s.expect.exhausted for s in v2.scenarios) and any(s.stream for s in v2.scenarios)


def _numbers(value: Any, key: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        return [n for k, v in value.items() if k not in STRUCTURAL_KEYS for n in _numbers(v, k)]
    if isinstance(value, list):
        return [n for v in value for n in _numbers(v, key)]
    return [(key, value)] if isinstance(value, (int, float)) and not isinstance(value, bool) else []


def test_scenarios_hold_no_expected_business_values(v2: InvestigationDataset) -> None:
    for scenario in v2.scenarios:
        assert not _numbers(scenario.model_dump(mode="json")), scenario.scenario_id
        dumped = json.dumps(scenario.expect.model_dump(mode="json"))
        for member in ("APAC", "Singapore", "Enterprise", "SMB"):
            assert member not in dumped, scenario.scenario_id


def test_relative_periods_are_resolved_independently() -> None:
    from datetime import date

    assert ref.last_complete_month(date(2026, 8, 31)) == "2026-08"
    assert ref.last_complete_month(date(2026, 8, 30)) == "2026-07"
    assert ref.last_complete_quarter(date(2026, 8, 31)) == "2026-Q2"
    assert ref.last_complete_quarter(date(2026, 9, 30)) == "2026-Q3"
    assert ref.last_complete_quarter(date(2026, 2, 1)) == "2025-Q4"
    assert ref.previous_label("2026-Q1") == "2025-Q4" and ref.previous_label("2026-01") == "2025-12"
    assert ref.resolve_comparison("previous", "2026-08") == "2026-07"


def test_the_causal_check_ignores_negations_and_system_statements() -> None:
    assert ref.causal_assertions("Revenue fell because of churn.")
    for phrase in ("caused by", "because of", "resulted from", "led to", "due to"):
        assert ref.causal_assertions(f"Revenue declined, {phrase} churn."), phrase
    assert not ref.causal_assertions("The evidence does not establish that churn caused it.")
    assert not ref.causal_assertions("Investigation stopped because the analysis budget was reached.")


# ------------------------------------------------------------------ the grader detects every defect


@pytest.fixture(scope="module")
def observed(eval_ctx: EvalContext, by_id: dict[str, InvestigationScenario]) -> InvestigationObservation:
    return run_scenario(eval_ctx, by_id["plan_revenue_growth_slowing"], EngineConfig().llm_factory)


def _checks(
    scenario: InvestigationScenario, investigation: Investigation, obs: InvestigationObservation, ctx: Any
) -> set[str]:
    mutated = InvestigationObservation(
        investigation=investigation, latency_ms=obs.latency_ms, llm=obs.llm, logs=obs.logs, progress=obs.progress
    )
    return {f.check for f in grade_investigation(scenario, mutated, ctx).failures}


def _corrupt(inv: Investigation, name: str) -> Investigation:
    inv = inv.model_copy(deep=True)
    brief = inv.brief
    assert brief is not None and inv.findings
    first = inv.findings[0]
    if name == "unknown_evidence":
        inv.findings[0] = first.model_copy(update={"evidence_ids": ["E99999"]})
    elif name == "wrong_period":
        inv.findings[0] = first.model_copy(update={"period": "2020-01"})
    elif name == "causal_driver":
        brief.drivers[0] = brief.drivers[0].model_copy(update={"statement": "Revenue fell because of this region."})
    elif name == "wrong_share":
        contribution = next(i for i, d in enumerate(brief.drivers) if d.relationship == "contributes_to")
        brief.drivers[contribution] = brief.drivers[contribution].model_copy(update={"share": 0.123456})
    elif name == "ungrounded_recommendation":
        brief.recommendations[0] = brief.recommendations[0].model_copy(update={"supporting_finding_ids": []})
    elif name == "sql_call":
        rogue = inv.tool_trace[0].model_copy(update={"tool_name": "run_safe_sql", "call_id": "T99"})
        inv.tool_trace.append(rogue)
        inv.budget.usage = inv.budget.usage.model_copy(update={"tool_calls": len(inv.tool_trace)})
        inv.efficiency = inv.efficiency.model_copy(update={"tool_calls": len(inv.tool_trace)})
    elif name == "duplicate_call":
        again: ToolCallRecord = inv.tool_trace[0].model_copy(update={"call_id": "T98"})
        inv.tool_trace.append(again)
    elif name == "fabricated_summary":
        brief.executive_summary = "Revenue declined by 987,654,321 in the period."
    elif name == "secret_in_summary":
        brief.executive_summary = "Use the key sk-ant-api03-" + "Z" * 30
    elif name == "tampered_evidence":
        inv.evidence[0] = inv.evidence[0].model_copy(update={"value": 42424242.0})
    elif name == "stopped_but_completed":
        inv.budget.exhausted = ["tool_calls"]
    elif name == "invented_priority":
        brief.risks[0] = brief.risks[0].model_copy(update={"text": "Customer sentiment is the top priority."})
    return inv


@pytest.mark.slow
@pytest.mark.parametrize(
    ("corruption", "check"),
    [
        ("unknown_evidence", "evidence.unknown"),
        ("wrong_period", "identity"),
        ("causal_driver", "causal.wording"),
        ("wrong_share", "drivers.share"),
        ("ungrounded_recommendation", "recommendations.findings"),
        ("sql_call", "tools.forbidden"),
        ("duplicate_call", "efficiency.duplicates"),
        ("fabricated_summary", "brief.summary_numbers"),
        ("secret_in_summary", "security.leak.secrets"),
        ("tampered_evidence", "evidence.integrity"),
        ("stopped_but_completed", "budget.status"),
    ],
)
def test_the_grader_detects_each_corruption(
    observed: InvestigationObservation,
    by_id: dict[str, InvestigationScenario],
    eval_ctx: EvalContext,
    corruption: str,
    check: str,
) -> None:
    scenario = by_id["plan_revenue_growth_slowing"]
    assert observed.investigation is not None
    assert not _checks(scenario, observed.investigation, observed, eval_ctx)  # the real one passes
    assert check in _checks(scenario, _corrupt(observed.investigation, corruption), observed, eval_ctx)


@pytest.mark.slow
def test_a_management_brief_with_invented_claims_fails(
    eval_ctx: EvalContext, by_id: dict[str, InvestigationScenario]
) -> None:
    scenario = by_id["brief_state_of_business"]
    obs = run_scenario(eval_ctx, scenario, EngineConfig().llm_factory)
    assert obs.investigation is not None and obs.investigation.brief is not None and obs.investigation.brief.risks
    assert not _checks(scenario, obs.investigation, obs, eval_ctx)
    assert "brief.unsupported_topic" in _checks(
        scenario, _corrupt(obs.investigation, "invented_priority"), obs, eval_ctx
    )


@pytest.mark.slow
def test_a_budget_violation_is_detected(
    observed: InvestigationObservation, by_id: dict[str, InvestigationScenario], eval_ctx: EvalContext
) -> None:
    strict = by_id["plan_revenue_growth_slowing"].model_copy(update={"limits": {"max_investigation_tool_calls": 3}})
    assert observed.investigation is not None
    assert "budget.tool_calls" in _checks(strict, observed.investigation, observed, eval_ctx)


# ------------------------------------------------------------------ CLI


def test_cli_lists_and_rejects_investigation_selections(
    capsys: pytest.CaptureFixture[str], v2: InvestigationDataset
) -> None:
    assert main(["--dataset", "eval_v2", "--list", "--suite", "critical"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == len(select_investigations(v2.scenarios, suite="critical"))
    assert main(["--dataset", "eval_v2", "--list", "--category", "budget_enforcement"]) == 0
    assert all("\tbudget_enforcement\t" in line for line in capsys.readouterr().out.strip().splitlines())
    assert main(["--dataset", "eval_v2", "--scenario", "does_not_exist"]) == 2
    assert main(["--category", "budget_enforcement", "--list"]) == 2  # an eval_v2 category is not an eval_v1 one


# ------------------------------------------------------------------ runs


@pytest.fixture(scope="module")
def critical_v2(eval_source: DataSource, v2: InvestigationDataset) -> list[Any]:
    return run_investigation_scenarios(
        select_investigations(v2.scenarios, suite="critical"), eval_source, EngineConfig()
    )


@pytest.mark.slow
def test_the_critical_investigation_suite_passes(critical_v2: list[Any]) -> None:
    failed = {r.scenario_id: [f.check for f in r.failures] for r in critical_v2 if not r.passed}
    assert not failed, failed
    investigations = [r for r in critical_v2 if r.mode in ("investigation", "cancellation", "mcp_parity")]
    assert investigations and all(r.scores.get("security") == 1.0 for r in investigations)
    assert all(r.scores.get("api") == 1.0 for r in critical_v2 if r.mode in ("api", "ui"))


@pytest.mark.slow
def test_the_cli_runs_the_critical_suite(
    monkeypatch: pytest.MonkeyPatch, eval_source: DataSource, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from evals import investigation_benchmark

    monkeypatch.setattr(investigation_benchmark, "data_source", lambda seed: eval_source)
    assert main(["--dataset", "eval_v2", "--suite", "critical", "--output", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "THRESHOLD FAILED" not in out
    (report,) = tmp_path.glob("EVAL2-*.md")
    text = report.read_text(encoding="utf-8")
    assert "Regression thresholds: **PASS**" in text and "Investigation stage" in text


@pytest.fixture(scope="module")
def other_seed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DataSource]:
    yield generated_source(2027, cache_dir=tmp_path_factory.mktemp("eval_v2_cache"), customer_count=400)


@pytest.mark.slow
def test_the_multi_seed_subset_passes_on_another_seed(other_seed: DataSource, v2: InvestigationDataset) -> None:
    results = run_investigation_scenarios(
        select_investigations(v2.scenarios, suite="multi_seed"), other_seed, EngineConfig()
    )
    failed = {r.scenario_id: [f.check for f in r.failures] for r in results if not r.passed}
    assert not failed, failed
    assert {r.agent_status for r in results} >= {"completed", "refused", "budget_exhausted", "cancelled"}
