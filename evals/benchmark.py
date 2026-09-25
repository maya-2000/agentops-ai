"""Run a benchmark (a whole dataset, a suite, a category or single scenarios) and summarise it."""

from __future__ import annotations

import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.agent import AgentConfig
from app.config import PROJECT_ROOT, get_settings
from evals.engine import EngineConfig, EvalMode, canary_secret, evaluate
from evals.metrics import aggregate
from evals.metrics.thresholds import check_thresholds
from evals.reference.context import DataSource, EvalContext, generated_source, repository_source
from evals.reports.models import EvaluationResult, EvaluationRunSummary
from evals.scenarios.loader import DEFAULT_DATASET, load_dataset, select
from evals.scenarios.model import EvaluationScenario


@dataclass
class RunOptions:
    mode: EvalMode = "deterministic"
    dataset: str = DEFAULT_DATASET
    suite: str | None = None  # full (default), critical, multi_seed
    categories: list[str] = field(default_factory=list)
    scenario_ids: list[str] = field(default_factory=list)
    seed: int | None = None  # None: the repository database (seed 42); else a generated dataset
    multi_seeds: list[int] = field(default_factory=list)  # extra seeds for the multi-seed subset
    judge_model: str | None = None


def git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", False


def data_source(seed: int | None) -> DataSource:
    if seed is None:
        return repository_source()
    repository = repository_source() if _repository_available() else None
    if repository is not None and repository.seed == seed:
        return repository
    return generated_source(seed)


def _repository_available() -> bool:
    try:
        repository_source()
    except FileNotFoundError:
        return False
    return True


def run_scenarios(
    scenarios: list[EvaluationScenario], source: DataSource, config: EngineConfig
) -> list[EvaluationResult]:
    ctx = EvalContext.open(source)
    try:
        with canary_secret():
            return [evaluate(s, ctx, config) for s in scenarios]
    finally:
        ctx.close()


def run_benchmark(options: RunOptions) -> EvaluationRunSummary:
    clock = time.perf_counter()
    dataset = load_dataset(options.dataset)
    scenarios = select(
        dataset.scenarios, suite=options.suite, categories=options.categories, scenario_ids=options.scenario_ids
    )
    if not scenarios:
        raise ValueError("No scenario matches the selection")
    judge = None
    if options.judge_model:
        from evals.graders.judge import create_judge

        answer_model = get_settings().llm_model if options.mode == "llm" else "deterministic-rules-v1"
        judge = create_judge(options.judge_model, answer_model=answer_model)
    config = EngineConfig.for_mode(options.mode, judge=judge)
    source = data_source(options.seed)
    results = run_scenarios(scenarios, source, config)
    summary = summarize(results, options=options, dataset=dataset, source=source, config=config)
    if options.multi_seeds:
        summary.multi_seed = run_multi_seed(options, dataset.scenarios, config)
    summary.performance["benchmark_wall_seconds"] = round(time.perf_counter() - clock, 2)
    return summary


def run_multi_seed(
    options: RunOptions, scenarios: list[EvaluationScenario], config: EngineConfig
) -> list[dict[str, Any]]:
    """The representative ``multi_seed`` subset on datasets generated for other seeds."""
    subset = select(scenarios, suite="multi_seed")
    rows: list[dict[str, Any]] = []
    for seed in options.multi_seeds:
        source = data_source(seed)
        for result in run_scenarios(subset, source, config):
            rows.append(
                {
                    "seed": seed,
                    "scenario_id": result.scenario_id,
                    "status": result.status,
                    "numerical": result.scores.get("numerical"),
                    "security": result.scores.get("security"),
                    "mcp": result.scores.get("mcp"),
                    "failures": [f"{f.category.value}: {f.check}" for f in result.failures],
                    "not_applicable": result.details.get("not_applicable", []),
                }
            )
    return rows


def summarize(
    results: list[EvaluationResult],
    *,
    options: RunOptions,
    dataset: Any,
    source: DataSource,
    config: EngineConfig,
) -> EvaluationRunSummary:
    commit, dirty = git_state()
    metrics = aggregate.overall_metrics(results)
    thresholds = check_thresholds(metrics, results=results, suite=options.suite)
    settings = get_settings()
    llm = config.llm_factory()
    return EvaluationRunSummary(
        run_id=f"EVAL-{uuid.uuid4().hex[:10]}",
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=commit,
        git_dirty=dirty,
        dataset_version=dataset.manifest.dataset_version,
        dataset_schema_version=dataset.manifest.schema_version,
        dataset_updated=dataset.manifest.updated,
        data={
            "origin": source.origin,
            "seed": source.seed,
            "business_dataset_version": source.dataset_version,
            "as_of": source.as_of.isoformat(),
        },
        configuration={
            "mode": options.mode,
            "deterministic": options.mode == "deterministic",
            "provider": llm.provider,
            "model": llm.model,
            "temperature": None if options.mode == "deterministic" else settings.llm_temperature,
            "suite": options.suite or "full",
            "categories": options.categories,
            "scenario_ids": options.scenario_ids,
            "judge_model": options.judge_model,
            "agent_limits": AgentConfig().model_dump(mode="json", exclude={"disabled_tools"}),
        },
        total_scenarios=len(results),
        passed=sum(r.passed for r in results),
        failed=sum(r.status == "failed" for r in results),
        errors=sum(r.status == "error" for r in results),
        metrics=metrics,
        by_category=aggregate.breakdown(results, lambda r: r.category),
        by_difficulty=aggregate.breakdown(results, lambda r: r.difficulty),
        security=aggregate.security_summary(results),
        mcp=aggregate.mcp_summary(results),
        performance=aggregate.performance_summary(results),
        failure_counts=aggregate.failure_counts(results),
        thresholds=thresholds,
        thresholds_passed=all(t.passed for t in thresholds),
        results=results,
    )
