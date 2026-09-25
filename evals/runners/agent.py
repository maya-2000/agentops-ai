"""Run a question through the production LangGraph agent (``AgentRunner``) and record what happened.

The agent is built exactly as in production: the same runtime, tools, policies and shared
executor. It receives only the read-only database and the business as-of date. The only
evaluation-side additions are:

- a transparent ``RecordingLLM`` proxy around the model client, so the leakage check can see
  the model requests;
- for adversarial scenarios, a scripted plan standing in for a compromised model
  (``adversarial_plan``). The rest of the run still uses the configured model.

Production latency is the wall time of ``AgentRunner.run`` only. Grading is timed separately.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.agent import AgentConfig, AgentRunner, AgentRunResult
from app.llm.base import LLMClient, LLMTask
from app.llm.scripted import ScriptedLLM
from evals.reference.context import EvalContext
from evals.runners.capture import RecordingLLM, capture_logs
from evals.scenarios.model import EvaluationScenario

LLMFactory = Callable[[], LLMClient]


@dataclass
class AgentObservation:
    result: AgentRunResult | None
    latency_ms: float
    llm: RecordingLLM
    logs: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def agent_config(scenario: EvaluationScenario) -> AgentConfig:
    return AgentConfig(**scenario.limits)


def _adversarial_plan(scenario: EvaluationScenario) -> dict[str, Any]:
    assert scenario.adversarial_plan is not None
    steps = [
        {"tool_name": call.tool, "arguments_json": json.dumps(call.arguments), "purpose": "adversarial"}
        for call in scenario.adversarial_plan
    ]
    return {"steps": steps, "rationale": "A compromised planner proposes these calls.", "sufficient": False}


def run_agent(ctx: EvalContext, scenario: EvaluationScenario, llm_factory: LLMFactory) -> AgentObservation:
    assert scenario.question is not None
    base: LLMClient = llm_factory()
    if scenario.adversarial_plan is not None:
        base = ScriptedLLM({LLMTask.PLAN: [_adversarial_plan(scenario)]}, fallback=base)
    llm = RecordingLLM(base)
    runner = AgentRunner(ctx.db, llm=llm, config=agent_config(scenario), as_of=ctx.as_of)
    with capture_logs() as logs:
        clock = time.perf_counter()
        try:
            result = runner.run(scenario.question)
            error = None
        except Exception as exc:  # the agent must never raise; recorded as a failure, not hidden
            result, error = None, f"{type(exc).__name__}: {str(exc)[:200]}"
        latency = (time.perf_counter() - clock) * 1000
    return AgentObservation(result=result, latency_ms=latency, llm=llm, logs=list(logs), error=error)
