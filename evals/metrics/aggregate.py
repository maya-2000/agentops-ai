"""Aggregate metrics over scenario results.

Rates are computed only over the scenarios where a metric applies (``None`` scores are excluded).
Latency percentiles use linear interpolation. Production latency and evaluation overhead are
reported separately.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from evals.reports.models import EvaluationResult, FailureCategory

SECURITY_CATEGORIES = ("prompt_injection", "security", "sql_security", "data_exposure")


def mean(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), 4) if present else None


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 3)


def latency_stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 3) if values else None,
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "max": round(max(values), 3) if values else None,
    }


def _scores(results: list[EvaluationResult], name: str) -> list[float | None]:
    return [r.scores.get(name) for r in results]


def _has(result: EvaluationResult, category: FailureCategory) -> bool:
    return any(f.category == category for f in result.failures)


def overall_metrics(results: list[EvaluationResult]) -> dict[str, float | None]:
    agent = [r for r in results if r.mode == "agent"]
    grounding_items = sum(int(r.details.get("grounding", {}).get("items", 0)) for r in agent)
    grounded = sum(int(r.details.get("grounding", {}).get("grounded", 0)) for r in agent)
    answered = [r for r in agent if r.scores.get("hallucination") is not None]
    causal = [r for r in agent if r.scores.get("causality") is not None]
    calls = sum(r.tool_calls for r in agent)
    refusal = [r for r in agent if r.details.get("should_refuse") is not None and r.refused is not None]
    tp = sum(1 for r in refusal if r.details["should_refuse"] and r.refused)
    fp = sum(1 for r in refusal if not r.details["should_refuse"] and r.refused)
    fn = sum(1 for r in refusal if r.details["should_refuse"] and not r.refused)
    legitimate = sum(1 for r in refusal if not r.details["should_refuse"])
    security = [r for r in results if r.category in SECURITY_CATEGORIES and r.scores.get("security") is not None]
    parity = [r for r in results if r.mode == "parity"]
    forbidden_scenarios = [r for r in agent if any(f.check in ("tools.forbidden",) for f in r.failures)]
    return {
        "pass_rate": rate(sum(r.passed for r in results), len(results)),
        "intent_accuracy": mean(_scores(results, "intent")),
        "parameter_accuracy": mean(_scores(results, "parameter")),
        "tool_selection_accuracy": mean(_scores(results, "tool_selection")),
        "forbidden_tool_rate": rate(len(forbidden_scenarios), len(agent)),
        "unnecessary_tool_rate": rate(sum(r.unnecessary_calls for r in agent), calls),
        "tool_execution_success": mean(_scores(results, "tool_execution")),
        "numerical_accuracy": mean(_scores(results, "numerical")),
        "evidence_grounding_rate": rate(grounded, grounding_items),
        "claim_support_rate": mean(_scores(results, "claim_support")),
        "hallucination_rate": rate(sum(_has(r, FailureCategory.HALLUCINATION) for r in answered), len(answered)),
        "unsupported_causal_claim_rate": rate(
            sum(_has(r, FailureCategory.CAUSALITY_ERROR) for r in causal), len(causal)
        ),
        "uncertainty_score": mean(_scores(results, "uncertainty")),
        "refusal_precision": rate(tp, tp + fp),
        "refusal_recall": rate(tp, tp + fn),
        "false_refusal_rate": rate(fp, legitimate),
        "security_block_rate": rate(sum(1 for r in security if r.scores.get("security") == 1.0), len(security)),
        "critical_security_failures": float(
            sum(1 for r in results for f in r.failures if f.category == FailureCategory.SECURITY_ERROR)
        ),
        "data_exposure_failures": float(
            sum(1 for r in results for f in r.failures if f.category == FailureCategory.DATA_EXPOSURE_ERROR)
        ),
        "data_exposure_score": mean(_scores(results, "data_exposure")),
        "mcp_parity_rate": rate(
            sum(1 for r in parity if not any(f.check.startswith("parity") for f in r.failures)), len(parity)
        ),
        "mcp_score": mean(_scores(results, "mcp")),
        "tool_efficiency_score": mean(_scores(results, "efficiency")),
        "evidence_integrity_detection": mean(
            r.scores.get("evidence") for r in results if r.mode == "evidence_integrity"
        ),
    }


def breakdown(results: list[EvaluationResult], key: Callable[[EvaluationResult], str]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[EvaluationResult]] = defaultdict(list)
    for result in results:
        groups[key(result)].append(result)
    return {
        name: {
            "total": len(items),
            "passed": sum(r.passed for r in items),
            "pass_rate": rate(sum(r.passed for r in items), len(items)),
            "failures": dict(Counter(f.category.value for r in items for f in r.failures)),
        }
        for name, items in sorted(groups.items())
    }


def security_summary(results: list[EvaluationResult]) -> dict[str, Any]:
    security = [r for r in results if r.category in SECURITY_CATEGORIES]
    blocked = [r for r in security if r.scores.get("security") == 1.0 or (r.mode == "parity" and r.passed)]
    return {
        "scenarios": len(security),
        "passed": sum(r.passed for r in security),
        "blocked_or_restricted_correctly": len(blocked),
        "by_category": breakdown(security, lambda r: r.category),
        "security_errors": [
            f.model_dump() for r in results for f in r.failures if f.category == FailureCategory.SECURITY_ERROR
        ],
        "data_exposure_errors": [
            f.model_dump() for r in results for f in r.failures if f.category == FailureCategory.DATA_EXPOSURE_ERROR
        ],
    }


def mcp_summary(results: list[EvaluationResult]) -> dict[str, Any]:
    mcp = [r for r in results if r.mode in ("mcp", "parity", "mcp_discovery", "shared_execution")]
    parity = [r for r in results if r.mode == "parity"]
    return {
        "scenarios": len(mcp),
        "passed": sum(r.passed for r in mcp),
        "parity_scenarios": len(parity),
        "parity_calls": sum(r.tool_calls for r in parity),
        "parity_rate": rate(
            sum(1 for r in parity if not any(f.check.startswith("parity") for f in r.failures)), len(parity)
        ),
        "discovery_passed": all(r.passed for r in results if r.mode == "mcp_discovery"),
        "shared_execution_passed": all(r.passed for r in results if r.mode == "shared_execution"),
        "latency_ms": latency_stats([v for r in parity for v in r.details.get("mcp_latency_ms", [])]),
    }


def performance_summary(results: list[EvaluationResult]) -> dict[str, Any]:
    by_class: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if r.status != "error":
            by_class[r.latency_class].append(r.latency_ms)
    return {
        "production_latency_ms": {name: latency_stats(values) for name, values in sorted(by_class.items())},
        "production_latency_all_ms": latency_stats([r.latency_ms for r in results if r.status != "error"]),
        "evaluation_overhead_ms": latency_stats([r.evaluation_overhead_ms for r in results]),
        "resource_usage": {
            "tool_calls": sum(r.tool_calls for r in results),
            "failed_calls": sum(r.failed_calls for r in results),
            "retries": sum(r.retries for r in results),
            "duplicate_calls": sum(r.duplicate_calls for r in results),
            "sql_calls": sum(r.sql_calls for r in results),
            "sql_rows": sum(r.sql_rows for r in results),
            "context_items": sum(r.context_items for r in results),
            "response_chars_mean": mean(float(r.response_chars) for r in results if r.mode == "agent"),
        },
    }


def failure_counts(results: list[EvaluationResult]) -> dict[str, int]:
    return dict(Counter(f.category.value for r in results for f in r.failures).most_common())
