"""The benchmark engine: run scenarios against the production system and grade them deterministically.

For each scenario: resolve its reference expectations (evaluation-side), run it through the real
production path for its mode, and grade the observation. Production latency (the agent run or
the MCP calls) is measured apart from evaluation overhead (references and grading), so the
evaluator's own cost is never reported as production latency.
"""

from __future__ import annotations

import os
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from app.config import get_settings
from app.llm.base import LLMClient
from app.llm.deterministic import DeterministicLLM
from app.llm.factory import create_llm_client
from evals.graders.agent import grade_agent, was_refused
from evals.graders.answer import Answer
from evals.graders.common import CANARY_ENV, CANARY_SECRET, Grade
from evals.graders.tools import grade_discovery, grade_integrity, grade_mcp, grade_shared
from evals.reference.context import EvalContext
from evals.reference.expectations import Expected, resolve
from evals.reports.models import EvaluationResult, Failure, FailureCategory
from evals.runners.agent import LLMFactory, run_agent
from evals.runners.integrity import mutate
from evals.runners.shared import run_shared
from evals.runners.tools import run_direct, run_mcp
from evals.scenarios.model import EvaluationScenario, Mode

EvalMode = Literal["deterministic", "llm"]


@dataclass
class EngineConfig:
    mode: EvalMode = "deterministic"
    judge: object | None = None  # an optional evals.graders.judge.LLMJudge
    llm_factory: LLMFactory = field(default=DeterministicLLM)

    @classmethod
    def for_mode(cls, mode: EvalMode, judge: object | None = None) -> EngineConfig:
        if mode == "deterministic":
            return cls(mode=mode, judge=judge, llm_factory=DeterministicLLM)
        settings = get_settings()
        if settings.llm_provider in ("deterministic", "offline"):
            raise ValueError("--mode llm needs LLM_PROVIDER=anthropic (and a key); use --mode deterministic otherwise")

        def factory() -> LLMClient:
            return create_llm_client(settings)

        return cls(mode=mode, judge=judge, llm_factory=factory)


@contextmanager
def canary_secret() -> Iterator[None]:
    """Plant a secret-looking environment value for the run: it must never appear in any output."""
    previous = os.environ.get(CANARY_ENV)
    os.environ[CANARY_ENV] = CANARY_SECRET
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(CANARY_ENV, None)
        else:
            os.environ[CANARY_ENV] = previous


def evaluate(scenario: EvaluationScenario, ctx: EvalContext, config: EngineConfig) -> EvaluationResult:
    """Run and grade one scenario. Evaluation errors are reported as results, never raised."""
    clock = time.perf_counter()
    try:
        expected = [resolve(check, ctx) for check in scenario.reference_expectations]
        grade, latency, extra = _run(scenario, ctx, config, expected)
    except Exception as exc:  # a defect in the benchmark itself: visible, never silent
        total = (time.perf_counter() - clock) * 1000
        return EvaluationResult(
            scenario_id=scenario.scenario_id,
            category=scenario.category.value,
            difficulty=scenario.difficulty.value,
            mode=scenario.mode.value,
            latency_class=scenario.latency_class,
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
            evaluation_overhead_ms=round(total, 3),
        )
    total = (time.perf_counter() - clock) * 1000
    result = EvaluationResult(
        scenario_id=scenario.scenario_id,
        category=scenario.category.value,
        difficulty=scenario.difficulty.value,
        mode=scenario.mode.value,
        latency_class=scenario.latency_class,
        status="passed" if not grade.failures else "failed",
        scores=grade.scores,
        failures=grade.failures,
        latency_ms=round(latency, 3),
        evaluation_overhead_ms=round(max(0.0, total - latency), 3),
        tool_trace=grade.trace[:20],
        security_events=sorted(set(grade.events)),
        details={**grade.details, "should_refuse": scenario.should_refuse},
        **extra,
    )
    if config.judge is not None and scenario.mode == Mode.AGENT and result.details.get("answer"):
        from evals.graders.judge import LLMJudge

        assert isinstance(config.judge, LLMJudge)
        result.judge = config.judge.judge(scenario.question or "", str(result.details["answer"]))
    return result


def _run(
    scenario: EvaluationScenario, ctx: EvalContext, config: EngineConfig, expected: list[Expected]
) -> tuple[Grade, float, dict[str, Any]]:
    if scenario.mode == Mode.AGENT:
        obs = run_agent(ctx, scenario, config.llm_factory)
        grade = grade_agent(scenario, obs, expected, ctx)
        extra: dict[str, Any] = {}
        if obs.result is not None:
            r = obs.result
            extra = {
                "tool_calls": len(r.tool_trace),
                "successful_calls": sum(1 for c in r.tool_trace if c.success),
                "failed_calls": sum(1 for c in r.tool_trace if not c.success),
                "retries": r.total_retries,
                "duplicate_calls": int(grade.details.get("duplicate_calls", 0)),
                "unnecessary_calls": int(grade.details.get("unnecessary_calls", 0)),
                "sql_calls": r.budget_usage.sql_calls,
                "sql_rows": r.budget_usage.sql_rows,
                "context_items": r.budget_usage.context_items,
                "response_chars": len(r.response.answer),
                "agent_status": r.status,
                "actual_intent": r.understanding.intent.value if r.understanding else None,
                "refused": was_refused(r),
                "evidence_ids": [e.evidence_id for e in r.evidence][:40],
            }
        return grade, obs.latency_ms, extra
    if scenario.mode in (Mode.MCP, Mode.PARITY):
        direct = run_direct(ctx, scenario) if scenario.mode == Mode.PARITY else None
        mcp = run_mcp(ctx, scenario)
        grade = grade_mcp(scenario, mcp, expected, ctx, direct)
        extra = {
            "tool_calls": len(mcp.calls),
            "successful_calls": sum(1 for c in mcp.calls if not c.is_error),
            "failed_calls": sum(1 for c in mcp.calls if c.is_error),
        }
        if direct is not None:  # the same calls on the direct path report their budget usage
            extra["sql_calls"] = sum(c.usage.sql_calls for c in direct)
            extra["sql_rows"] = sum(c.usage.sql_rows for c in direct)
        return grade, sum(c.latency_ms for c in mcp.calls), extra
    if scenario.mode == Mode.MCP_DISCOVERY:
        clock = time.perf_counter()
        mcp = run_mcp(ctx, scenario, discover=True)
        return grade_discovery(scenario, mcp), (time.perf_counter() - clock) * 1000, {}
    if scenario.mode == Mode.SHARED_EXECUTION:
        shared = run_shared(ctx, scenario, config.llm_factory)
        return grade_shared(scenario, shared), shared.latency_ms, {"tool_calls": len(shared.records)}
    obs = run_agent(ctx, scenario, config.llm_factory)
    base = Answer.from_result(obs.result) if obs.result is not None else None
    mutated = [mutate(base, m) for m in scenario.mutations] if base is not None else []
    return grade_integrity(scenario, base, mutated, ctx), obs.latency_ms, {}
