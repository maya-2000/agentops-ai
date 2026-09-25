"""Metric aggregation, latency statistics, tolerances and regression thresholds, on synthetic results."""

from __future__ import annotations

from typing import Any, cast

import pytest

from evals.metrics.aggregate import (
    breakdown,
    failure_counts,
    latency_stats,
    mcp_summary,
    mean,
    overall_metrics,
    percentile,
    performance_summary,
    rate,
    security_summary,
)
from evals.metrics.thresholds import THRESHOLDS, check_thresholds
from evals.reference.kpis import KPI_TOLERANCE, within
from evals.reports.models import EvaluationResult, Failure, FailureCategory


def result(
    scenario_id: str = "s",
    *,
    category: str = "kpi",
    mode: str = "agent",
    failures: tuple[FailureCategory, ...] = (),
    check: str = "x",
    scores: dict[str, float | None] | None = None,
    **fields: Any,
) -> EvaluationResult:
    return EvaluationResult(
        scenario_id=scenario_id,
        category=category,
        difficulty=fields.pop("difficulty", "easy"),
        mode=mode,
        latency_class=fields.pop("latency_class", "simple_kpi"),
        status=fields.pop("status", "failed" if failures else "passed"),
        scores=scores or {},
        failures=[Failure(scenario_id=scenario_id, category=c, check=check, message="m") for c in failures],
        **fields,
    )


# ------------------------------------------------------------------ primitives


def test_percentiles_interpolate_linearly() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 0.5) == 30.0
    assert percentile(values, 0.95) == pytest.approx(48.0)
    assert percentile([5.0], 0.95) == 5.0
    assert percentile([], 0.5) is None
    stats = latency_stats([3.0, 1.0, 2.0])
    assert stats == {"count": 3, "mean": 2.0, "p50": 2.0, "p95": pytest.approx(2.9), "max": 3.0}
    assert latency_stats([])["mean"] is None


def test_means_skip_not_applicable_scores_and_rates_handle_empty_denominators() -> None:
    assert mean([1.0, None, 0.0]) == 0.5
    assert mean([None, None]) is None
    assert rate(1, 3) == 0.3333
    assert rate(0, 0) is None


@pytest.mark.parametrize(
    ("tolerance", "expected", "inside", "outside"),
    [
        ("money", 1_000_000.0, 1_000_000.009, 1_000_000.02),
        ("money", 1e12, 1e12 + 900.0, 1e12 + 1100.0),  # 1e-9 relative dominates large values
        ("rate", 0.25, 0.25 + 5e-10, 0.25 + 5e-9),
        ("count", 42.0, 42.0, 43.0),
        ("duration", 30.5, 30.5 + 5e-7, 30.5 + 5e-6),
        ("forecast", 1_000_000.0, 1_000_000.5, 1_000_002.0),
    ],
)
def test_documented_tolerances(tolerance: Any, expected: float, inside: float, outside: float) -> None:
    assert within(inside, expected, tolerance)
    assert within(expected - (inside - expected), expected, tolerance)
    assert not within(outside, expected, tolerance)


def test_tolerance_edge_cases() -> None:
    assert within(None, None, "money") and not within(None, 1.0, "money") and not within(1.0, None, "rate")
    assert not within(float("nan"), 1.0, "rate")
    with pytest.raises(ValueError):
        within(1.0, 1.0, cast(Any, "percent"))
    assert set(KPI_TOLERANCE.values()) <= {"money", "rate", "count", "duration"}


# ------------------------------------------------------------------ aggregation


def test_refusal_precision_recall_and_false_refusal() -> None:
    results = [
        result("refused_correctly", refused=True, details={"should_refuse": True}),
        result("missed", refused=False, details={"should_refuse": True}, failures=(FailureCategory.REFUSAL_ERROR,)),
        result("false_refusal", refused=True, details={"should_refuse": False}),
        result("answered_1", refused=False, details={"should_refuse": False}),
        result("answered_2", refused=False, details={"should_refuse": False}),
        result("answered_3", refused=False, details={"should_refuse": False}),
        result("not_scored", refused=False, details={"should_refuse": None}),
    ]
    metrics = overall_metrics(results)
    assert metrics["refusal_precision"] == 0.5
    assert metrics["refusal_recall"] == 0.5
    assert metrics["false_refusal_rate"] == 0.25


def test_hallucination_causality_grounding_and_parity_rates() -> None:
    results = [
        result(
            "a", scores={"hallucination": 1.0, "causality": 1.0}, details={"grounding": {"items": 4, "grounded": 4}}
        ),
        result(
            "b",
            scores={"hallucination": 0.0, "causality": 0.0},
            failures=(FailureCategory.HALLUCINATION, FailureCategory.CAUSALITY_ERROR),
            details={"grounding": {"items": 4, "grounded": 2}},
        ),
        result("c", scores={"hallucination": None}),  # not an answer: excluded from the rate
        result("p1", mode="parity", category="mcp"),
        result("p2", mode="parity", category="mcp", failures=(FailureCategory.MCP_ERROR,), check="parity.result"),
        result("p3", mode="parity", category="mcp", failures=(FailureCategory.MCP_ERROR,), check="mcp.provenance"),
    ]
    metrics = overall_metrics(results)
    assert metrics["hallucination_rate"] == 0.5
    assert metrics["unsupported_causal_claim_rate"] == 0.5
    assert metrics["evidence_grounding_rate"] == 0.75
    assert metrics["mcp_parity_rate"] == 0.6667  # only parity checks count against parity
    assert metrics["pass_rate"] == pytest.approx(3 / 6, abs=1e-4)


def test_security_and_exposure_counts() -> None:
    results = [
        result("pi", category="prompt_injection", scores={"security": 1.0}),
        result("sql", category="sql_security", mode="parity", scores={"security": 1.0}),
        result(
            "leak",
            category="data_exposure",
            scores={"security": 0.0},
            failures=(FailureCategory.SECURITY_ERROR, FailureCategory.DATA_EXPOSURE_ERROR),
        ),
        result("kpi", scores={"security": None}),
    ]
    metrics = overall_metrics(results)
    assert metrics["security_block_rate"] == 0.6667
    assert metrics["critical_security_failures"] == 1.0
    assert metrics["data_exposure_failures"] == 1.0
    summary = security_summary(results)
    assert summary["scenarios"] == 3 and summary["passed"] == 2
    assert len(summary["security_errors"]) == 1 and len(summary["data_exposure_errors"]) == 1


def test_breakdowns_failure_counts_and_performance() -> None:
    results = [
        result("a", difficulty="easy", latency_ms=10.0, evaluation_overhead_ms=5.0, tool_calls=2, sql_calls=1),
        result("b", difficulty="hard", latency_ms=30.0, failures=(FailureCategory.NUMERICAL_ERROR,)),
        result("c", mode="parity", category="mcp", latency_class="mcp_invocation", latency_ms=8.0, tool_calls=3),
        result("d", status="error", failures=(FailureCategory.UNKNOWN,), latency_ms=999.0),
    ]
    by_difficulty = breakdown(results, lambda r: r.difficulty)
    assert by_difficulty["hard"] == {"total": 1, "passed": 0, "pass_rate": 0.0, "failures": {"NUMERICAL_ERROR": 1}}
    assert failure_counts(results) == {"NUMERICAL_ERROR": 1, "UNKNOWN": 1}
    perf = performance_summary(results)
    assert perf["production_latency_ms"]["simple_kpi"]["count"] == 2  # the errored run is excluded
    assert perf["production_latency_all_ms"]["max"] == 30.0
    assert perf["evaluation_overhead_ms"]["max"] == 5.0  # reported apart from production latency
    assert perf["resource_usage"]["tool_calls"] == 5 and perf["resource_usage"]["sql_calls"] == 1
    mcp = mcp_summary(results)
    assert mcp["parity_scenarios"] == 1 and mcp["parity_calls"] == 3 and mcp["parity_rate"] == 1.0


# ------------------------------------------------------------------ thresholds


def passing_metrics() -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for t in THRESHOLDS:
        metrics[t.metric] = t.value
    return metrics


def test_thresholds_pass_at_their_limits_and_fail_beyond() -> None:
    assert all(c.passed for c in check_thresholds(passing_metrics(), results=[], suite=None))
    for threshold in THRESHOLDS:
        metrics = passing_metrics()
        if threshold.comparison == ">=" or (threshold.comparison == "==" and threshold.value > 0):
            metrics[threshold.metric] = threshold.value - 0.01
        else:
            metrics[threshold.metric] = threshold.value + 1
        failed = [c.name for c in check_thresholds(metrics, results=[], suite=None) if not c.passed]
        assert failed == [threshold.metric], threshold.metric


def test_partial_runs_skip_composition_dependent_thresholds() -> None:
    full = {c.name for c in check_thresholds(passing_metrics(), results=[], suite=None)}
    partial = {c.name for c in check_thresholds(passing_metrics(), results=[], suite="multi_seed")}
    assert partial < full
    assert {"critical_security_failures", "data_exposure_failures", "mcp_parity_rate"} <= partial
    unmeasured = passing_metrics() | {"numerical_accuracy": None}
    assert "numerical_accuracy" not in {c.name for c in check_thresholds(unmeasured, results=[], suite=None)}


def test_the_critical_suite_requires_every_scenario_to_pass() -> None:
    ok = check_thresholds(passing_metrics(), results=[result("a"), result("b")], suite="critical")
    assert ok[-1].name == "critical_suite_pass_rate" and ok[-1].passed and ok[-1].actual == 1.0
    failed = [result("a"), result("b", failures=(FailureCategory.INTENT_ERROR,))]
    check = check_thresholds(passing_metrics(), results=failed, suite="critical")[-1]
    assert not check.passed and check.actual == 0.5 and "b" in check.rationale


def test_thresholds_are_documented_and_make_no_reliability_claim() -> None:
    assert {t.metric for t in THRESHOLDS} >= {
        "critical_security_failures",
        "data_exposure_failures",
        "security_block_rate",
        "mcp_parity_rate",
        "hallucination_rate",
        "evidence_grounding_rate",
        "numerical_accuracy",
        "false_refusal_rate",
    }
    for t in THRESHOLDS:
        assert t.comparison in (">=", "<=", "==") and len(t.rationale) > 20
        lowered = t.rationale.lower()
        assert "production ready" not in lowered and "100% reliable" not in lowered
    zero_tolerance = {t.metric: t.value for t in THRESHOLDS if t.metric.endswith("failures")}
    assert zero_tolerance == {"critical_security_failures": 0, "data_exposure_failures": 0}
