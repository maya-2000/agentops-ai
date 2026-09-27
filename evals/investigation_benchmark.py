"""The investigation benchmark (eval_v2): run investigation scenarios against the production system.

For each scenario: run it through the production path of its mode (``evals/runners/investigation.py``)
and grade the observation deterministically (``evals/graders/investigation.py``). Production latency (the
investigation or the API call) is measured apart from grading. Multi-seed runs repeat the ``multi_seed``
subset on datasets generated for other seeds; every expectation is resolved from that dataset's own
reference, so nothing measured on seed 42 is reused.
"""

from __future__ import annotations

import time
import traceback
import uuid
from datetime import UTC, datetime
from typing import Any

from app.agent import AgentConfig
from evals.benchmark import RunOptions, data_source, git_state
from evals.engine import EngineConfig, canary_secret
from evals.graders.investigation import grade_investigation
from evals.metrics import aggregate
from evals.metrics.investigation import check_thresholds, investigation_metrics, performance_summary
from evals.reference.context import DataSource, EvalContext
from evals.reports.models import EvaluationResult, EvaluationRunSummary, Failure, FailureCategory
from evals.runners.investigation import run_scenario
from evals.scenarios.investigation import (
    InvestigationDataset,
    InvestigationScenario,
    load_investigation_dataset,
    select_investigations,
)


def evaluate_investigation(scenario: InvestigationScenario, ctx: EvalContext, config: EngineConfig) -> EvaluationResult:
    clock = time.perf_counter()
    base: dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "category": scenario.category.value,
        "difficulty": scenario.difficulty.value,
        "mode": scenario.mode.value,
        "latency_class": scenario.latency_class,
    }
    try:
        obs = run_scenario(ctx, scenario, config.llm_factory)
        grade = grade_investigation(scenario, obs, ctx)
    except Exception as exc:  # a defect in the benchmark itself: visible, never silent
        return EvaluationResult(
            **base,
            status="error",
            failures=[
                Failure(
                    scenario_id=scenario.scenario_id,
                    category=FailureCategory.UNKNOWN,
                    check="evaluation",
                    message=f"The evaluation raised {type(exc).__name__}: {str(exc)[:300]}",
                    actual=traceback.format_exc(limit=3)[-800:],
                )
            ],
            evaluation_overhead_ms=round((time.perf_counter() - clock) * 1000, 3),
        )
    total = (time.perf_counter() - clock) * 1000
    inv = obs.investigation
    body = obs.api.body if obs.api is not None else {}
    run = body.get("run") or {}
    trace = inv.tool_trace if inv is not None else []
    status = inv.status if inv is not None else body.get("status")
    return EvaluationResult(
        **base,
        status="passed" if not grade.failures else "failed",
        scores=grade.scores,
        failures=grade.failures,
        latency_ms=round(obs.latency_ms, 3),
        evaluation_overhead_ms=round(max(0.0, total - obs.latency_ms), 3),
        tool_calls=len(trace) if inv is not None else int(run.get("tool_calls") or 0),
        successful_calls=sum(1 for c in trace if c.success),
        failed_calls=sum(1 for c in trace if not c.success),
        duplicate_calls=int(grade.details.get("duplicate_calls", 0)),
        sql_calls=inv.budget.usage.sql_calls if inv is not None else 0,
        response_chars=inv.budget.output_chars if inv is not None else 0,
        agent_status=str(status) if status is not None else None,
        refused=status in ("refused", "unsupported") if status is not None else None,
        tool_trace=grade.trace[:20],
        evidence_ids=[e.evidence_id for e in inv.evidence][:40] if inv is not None else [],
        security_events=sorted(set(grade.events)),
        details=grade.details,
    )


def run_investigation_scenarios(
    scenarios: list[InvestigationScenario], source: DataSource, config: EngineConfig
) -> list[EvaluationResult]:
    ctx = EvalContext.open(source)
    try:
        with canary_secret():
            return [evaluate_investigation(s, ctx, config) for s in scenarios]
    finally:
        ctx.close()


def run_investigation_benchmark(options: RunOptions) -> EvaluationRunSummary:
    clock = time.perf_counter()
    dataset = load_investigation_dataset(options.dataset)
    scenarios = select_investigations(
        dataset.scenarios, suite=options.suite, categories=options.categories, scenario_ids=options.scenario_ids
    )
    if not scenarios:
        raise ValueError("No scenario matches the selection")
    config = EngineConfig.for_mode(options.mode)
    source = data_source(options.seed)
    results = run_investigation_scenarios(scenarios, source, config)
    summary = summarize_investigations(results, options=options, dataset=dataset, source=source, config=config)
    if options.multi_seeds:
        summary.multi_seed = run_investigation_multi_seed(options, dataset.scenarios, config)
    summary.performance["benchmark_wall_seconds"] = round(time.perf_counter() - clock, 2)
    return summary


def run_investigation_multi_seed(
    options: RunOptions, scenarios: list[InvestigationScenario], config: EngineConfig
) -> list[dict[str, Any]]:
    subset = select_investigations(scenarios, suite="multi_seed")
    rows: list[dict[str, Any]] = []
    for seed in options.multi_seeds:
        for result in run_investigation_scenarios(subset, data_source(seed), config):
            rows.append(
                {
                    "seed": seed,
                    "scenario_id": result.scenario_id,
                    "status": result.status,
                    "agent_status": result.agent_status,
                    "numerical": result.scores.get("numerical"),
                    "security": result.scores.get("security"),
                    "failures": [f"{f.category.value}: {f.check}" for f in result.failures],
                    "not_applicable": [],
                }
            )
    return rows


def summarize_investigations(
    results: list[EvaluationResult],
    *,
    options: RunOptions,
    dataset: InvestigationDataset,
    source: DataSource,
    config: EngineConfig,
) -> EvaluationRunSummary:
    commit, dirty = git_state()
    metrics = investigation_metrics(results)
    thresholds = check_thresholds(metrics, results=results, suite=options.suite)
    llm = config.llm_factory()
    security = [r for r in results if r.category in ("security", "refusal")]
    parity = [r for r in results if r.mode == "mcp_parity"]
    return EvaluationRunSummary(
        run_id=f"EVAL2-{uuid.uuid4().hex[:10]}",
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
            "suite": options.suite or "full",
            "categories": options.categories,
            "scenario_ids": options.scenario_ids,
            "investigation_limits": {
                k: v for k, v in AgentConfig().model_dump(mode="json").items() if k.startswith("max_investigation")
            },
        },
        total_scenarios=len(results),
        passed=sum(r.passed for r in results),
        failed=sum(r.status == "failed" for r in results),
        errors=sum(r.status == "error" for r in results),
        metrics=metrics,
        by_category=aggregate.breakdown(results, lambda r: r.category),
        by_difficulty=aggregate.breakdown(results, lambda r: r.difficulty),
        security={"scenarios": len(security), "passed": sum(r.passed for r in security)},
        mcp={"scenarios": len(parity), "passed": sum(r.passed for r in parity)},
        performance=performance_summary(results),
        failure_counts=aggregate.failure_counts(results),
        thresholds=thresholds,
        thresholds_passed=all(t.passed for t in thresholds),
        results=results,
    )
