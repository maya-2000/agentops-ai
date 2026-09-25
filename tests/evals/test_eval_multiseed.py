"""Multi-seed robustness: the representative subset on a dataset generated for another seed.

Expectations are resolved from that dataset's own references and labels, so nothing written for
seed 42 is reused. The dataset is generated small (400 customers) to keep the test fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.benchmark import RunOptions, run_multi_seed, run_scenarios
from evals.engine import EngineConfig
from evals.reference import context as reference_context
from evals.reference.context import DataSource, EvalContext, generated_source
from evals.reference.expectations import resolve
from evals.reports.models import EvaluationResult
from evals.scenarios.loader import select
from evals.scenarios.model import EvaluationDataset

pytestmark = pytest.mark.slow

SEED = 7


@pytest.fixture(scope="module")
def other_seed(tmp_path_factory: pytest.TempPathFactory) -> DataSource:
    return generated_source(SEED, cache_dir=tmp_path_factory.mktemp("eval_cache"), customer_count=400)


@pytest.fixture(scope="module")
def other_results(other_seed: DataSource, eval_dataset: EvaluationDataset) -> list[EvaluationResult]:
    return run_scenarios(select(eval_dataset.scenarios, suite="multi_seed"), other_seed, EngineConfig())


def test_the_generated_source_is_separate_and_cached(other_seed: DataSource) -> None:
    assert other_seed.seed == SEED and other_seed.origin == "generated"
    assert other_seed.db_path.exists() and other_seed.ground_truth_path.exists()
    assert other_seed.db_path.parent.name == f"seed-{SEED}-400"
    mtime = other_seed.db_path.stat().st_mtime
    again = generated_source(SEED, cache_dir=other_seed.db_path.parent.parent, customer_count=400)
    assert again == other_seed and again.db_path.stat().st_mtime == mtime  # reused, not regenerated


def test_expectations_come_from_the_other_seeds_data(
    other_seed: DataSource, eval_ctx: EvalContext, eval_dataset: EvaluationDataset
) -> None:
    ctx = EvalContext.open(other_seed)
    try:
        assert ctx.labels.random_seed == SEED
        checks = [c for s in select(eval_dataset.scenarios, suite="multi_seed") for c in s.reference_expectations]
        kpi_checks = [c for c in checks if c.kind == "kpi_value"]
        assert kpi_checks
        for check in kpi_checks:
            assert resolve(check, ctx).values["value"] != resolve(check, eval_ctx).values["value"]
    finally:
        ctx.close()


def test_the_subset_runs_without_evaluation_errors(other_results: list[EvaluationResult]) -> None:
    assert other_results and not [r.scenario_id for r in other_results if r.status == "error"]
    security = [r for r in other_results if r.scores.get("security") is not None]
    assert security and all(r.scores["security"] == 1.0 for r in security)
    failed = {r.scenario_id: [f.check for f in r.failures] for r in other_results if not r.passed}
    assert not failed, failed


def test_multi_seed_rows_are_reported_per_seed(
    monkeypatch: pytest.MonkeyPatch, other_seed: DataSource, eval_dataset: EvaluationDataset
) -> None:
    monkeypatch.setattr("evals.benchmark.data_source", lambda seed: other_seed)
    rows = run_multi_seed(RunOptions(multi_seeds=[SEED]), eval_dataset.scenarios, EngineConfig())
    subset = select(eval_dataset.scenarios, suite="multi_seed")
    assert [r["scenario_id"] for r in rows] == [s.scenario_id for s in subset]
    assert all(r["seed"] == SEED and r["status"] in ("passed", "failed") for r in rows)
    assert set(rows[0]) == {
        "seed",
        "scenario_id",
        "status",
        "numerical",
        "security",
        "mcp",
        "failures",
        "not_applicable",
    }


def test_the_default_cache_is_git_ignored() -> None:
    ignored = (Path(reference_context.PROJECT_ROOT) / ".gitignore").read_text(encoding="utf-8")
    assert ".eval_cache/" in ignored and "reports/evaluation/" in ignored
    assert reference_context.DEFAULT_CACHE.name == ".eval_cache"
