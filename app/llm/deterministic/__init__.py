"""The deterministic offline model: same interface as a network LLM, no network, no key, no randomness.

It reads the structured request context (not the prompt text) and returns JSON for the task:
rule-based question understanding, intent playbooks for planning, and claim-based composition for
responses. Its output goes through exactly the same validation as a network model's output.
"""

from __future__ import annotations

import json

from app.llm.base import LLMRequest, LLMResponse, LLMTask
from app.llm.deterministic.composition import compose
from app.llm.deterministic.planning import plan
from app.llm.deterministic.understanding import understand


class DeterministicLLM:
    provider = "deterministic"
    model = "deterministic-rules-v1"

    def generate(self, request: LLMRequest) -> LLMResponse:
        if request.task == LLMTask.UNDERSTAND:
            output = understand(request.context)
        elif request.task == LLMTask.PLAN:
            output = plan(request.context)
        else:
            output = compose(request.context)
        return LLMResponse(
            task=request.task,
            content=json.dumps(output, sort_keys=True),
            provider=self.provider,
            model=self.model,
            stop_reason="end_turn",
        )


__all__ = ["DeterministicLLM"]
