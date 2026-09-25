"""A scripted test double: returns pre-written outputs per task, in order, and records every request.

It exercises the same parsing, validation, retry and failure paths as a real model, without any
intelligence and without a network. Tests use it to force specific transitions (an unknown tool in a
plan, malformed JSON, a draft that cites a number no tool produced, a provider error, ...).
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from app.llm.base import LLMError, LLMRequest, LLMResponse, LLMTask

ScriptItem = str | dict[str, Any] | Exception | Callable[[LLMRequest], str | dict[str, Any]]


class ScriptedLLM:
    provider = "scripted"

    def __init__(self, script: Mapping[LLMTask | str, Iterable[ScriptItem]], *, fallback: Any | None = None):
        self._script: dict[LLMTask, deque[ScriptItem]] = {LLMTask(k): deque(v) for k, v in script.items()}
        self._fallback = fallback  # an LLMClient used when a task has no scripted item left
        self.model = "scripted"
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        queue = self._script.get(request.task)
        if not queue:
            if self._fallback is not None:
                response: LLMResponse = self._fallback.generate(request)
                return response
            raise LLMError(f"No scripted output left for task {request.task.value}")
        item = queue.popleft()
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(request)
        content = item if isinstance(item, str) else json.dumps(item)
        return LLMResponse(task=request.task, content=content, provider=self.provider, model=self.model)
