"""The critical regression suite, reproducibility, the run summary and the reports.

Everything runs in deterministic mode on the session's full generated dataset: no network, no
model and no API key.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from evals.benchmark import RunOptions, run_scenarios, summarize
from evals.engine import EngineConfig, canary_secret
from evals.graders.common import CANARY_ENV, CANARY_SECRET
from evals.metrics.aggregate import overall_metrics
from evals.metrics.thresholds import check_thresholds
from evals.reference.context import DataSource, EvalContext
from evals.reports.models import EvaluationResult, EvaluationRunSummary
from evals.reports.writer import render_markdown, write_reports
from evals.scenarios.loader import select
from evals.scenarios.model import EvaluationDataset
from tests.phase7_support import fingerprint

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def summary(
    critical_results: list[EvaluationResult], eval_dataset: EvaluationDataset, eval_source: DataSource
) -> EvaluationRunSummary:
    options = RunOptions(suite="critical")
    return summarize(critical_results, options=options, dataset=eval_dataset, source=eval_source, config=EngineConfig())


# ------------------------------------------------------------------ the critical suite


def test_every_critical_scenario_passes(critical_results: list[EvaluationResult]) -> None:
    failed = {r.scenario_id: [f.check for f in r.failures] for r in critical_results if not r.passed}
    assert not failed, failed
    assert 10 <= len(critical_results) <= 15
    assert {r.mode for r in critical_results} >= {"agent", "parity", "mcp_discovery", "shared_execution"}


def test_critical_thresholds_pass(critical_results: list[EvaluationResult]) -> None:
    checks = check_thresholds(overall_metrics(critical_results), results=critical_results, suite="critical")
    assert checks and all(c.passed for c in checks), [c for c in checks if not c.passed]
    assert "critical_suite_pass_rate" in {c.name for c in checks}


def test_deterministic_runs_are_reproducible(
    critical_results: list[EvaluationResult], eval_dataset: EvaluationDataset, eval_source: DataSource
) -> None:
    again = run_scenarios(select(eval_dataset.scenarios, suite="critical"), eval_source, EngineConfig())
    assert fingerprint(again) == fingerprint(critical_results)


def test_production_latency_is_measured_apart_from_evaluation_overhead(
    critical_results: list[EvaluationResult],
) -> None:
    for r in critical_results:
        assert r.latency_ms > 0 and r.evaluation_overhead_ms >= 0, r.scenario_id
    assert sum(r.tool_calls for r in critical_results) > 0  # resource usage is recorded per scenario


# ------------------------------------------------------------------ the summary and reports


def test_summary_records_what_is_needed_to_reproduce_the_run(summary: EvaluationRunSummary) -> None:
    assert summary.run_id.startswith("EVAL-") and summary.timestamp
    assert summary.git_commit and summary.dataset_version == "eval_v1" and summary.dataset_schema_version
    assert (
        summary.data["seed"] == 42 and summary.data["origin"] == "generated" and summary.data["as_of"] == "2026-08-31"
    )
    config = summary.configuration
    assert config["mode"] == "deterministic" and config["deterministic"] is True
    assert config["provider"] == "deterministic" and config["model"] == "deterministic-rules-v1"
    assert config["temperature"] is None and config["suite"] == "critical" and config["agent_limits"]
    assert summary.total_scenarios == summary.passed and summary.thresholds_passed
    assert set(summary.by_difficulty) <= {"easy", "medium", "hard", "adversarial"}
    assert summary.performance["production_latency_ms"] and summary.performance["evaluation_overhead_ms"]["count"]


def test_reports_round_trip_and_hold_no_secrets_customer_names_or_labels(
    summary: EvaluationRunSummary, eval_ctx: EvalContext, tmp_path: Path
) -> None:
    json_path, md_path = write_reports(summary, tmp_path)
    assert EvaluationRunSummary.model_validate_json(json_path.read_text(encoding="utf-8")) == summary
    markdown = md_path.read_text(encoding="utf-8")
    for heading in (
        "## Overall",
        "## Results by category",
        "## Results by difficulty",
        "## Security and data exposure",
    ):
        assert heading in markdown
    assert "## MCP" in markdown and "## Regression thresholds" in markdown and "not a production" in markdown
    both = json_path.read_text(encoding="utf-8") + markdown
    assert CANARY_SECRET not in both
    names = eval_ctx.withheld_values["company_name"]
    assert not [n for n in names if n in both]
    for marker in eval_ctx.labels.leak_markers():
        if " " in marker:
            assert marker not in both, marker
    for word in ("production ready", "100% reliable"):
        assert word not in both.lower()


def test_a_report_containing_the_canary_is_never_written(summary: EvaluationRunSummary, tmp_path: Path) -> None:
    poisoned = summary.model_copy(deep=True)
    poisoned.results[0].details["answer"] = f"key={CANARY_SECRET}"
    with pytest.raises(RuntimeError, match="canary"):
        write_reports(poisoned, tmp_path / "out")
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


def test_failures_are_rendered_with_expected_and_actual(summary: EvaluationRunSummary) -> None:
    failing = summary.model_copy(deep=True)
    result = failing.results[0]
    result.status = "failed"
    result.failures = [_failure(result.scenario_id)]
    text = render_markdown(failing)
    assert f"### {result.scenario_id}" in text and "PARAMETER_ERROR" in text and 'Expected `"2026-08"`' in text


def _failure(scenario_id: str) -> Any:
    from evals.reports.models import Failure, FailureCategory

    return Failure(
        scenario_id=scenario_id,
        category=FailureCategory.PARAMETER_ERROR,
        check="parameter.period",
        message="period = '2026-07', expected '2026-08'.",
        expected="2026-08",
        actual="2026-07",
        tool="get_kpi",
    )


def test_the_canary_is_planted_only_for_the_run() -> None:
    assert os.environ.get(CANARY_ENV) != CANARY_SECRET
    with canary_secret():
        assert os.environ[CANARY_ENV] == CANARY_SECRET
    assert os.environ.get(CANARY_ENV) != CANARY_SECRET
