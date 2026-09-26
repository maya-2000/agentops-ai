"""The API adds little to the agent's own time (generous bounds: a regression guard, not a benchmark).

Measured numbers are documented in docs/api.md ("Performance").
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import pytest

from app.agent.runner import AgentRunner
from app.api.presenter import build_response
from tests.phase8_support import ASK, api_client

pytestmark = pytest.mark.slow

QUESTIONS = [
    "What was revenue in July 2026 compared with June 2026?",
    "Which region had the largest revenue decline?",
    "Which acquisition channel has the highest CAC?",
    "Are there any unusual trends in support tickets?",
]
REPEATS = 5


def _median_ms(fn: Any) -> float:
    samples = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def test_api_overhead_is_small(full_db: Any) -> None:
    with api_client(full_db) as client:
        runner: AgentRunner = client.app.state.service._runner  # type: ignore[attr-defined]
        overheads = []
        for question in QUESTIONS:
            runner.run(question)
            client.post(ASK, json={"question": question})  # warm-up
            direct = _median_ms(lambda q=question: runner.run(q))
            api = _median_ms(lambda q=question: client.post(ASK, json={"question": q}))
            overheads.append(api - direct)
    assert statistics.median(overheads) < 25.0, overheads


def test_progress_reporting_and_presentation_are_cheap(full_db: Any) -> None:
    with api_client(full_db) as client:
        runner: AgentRunner = client.app.state.service._runner  # type: ignore[attr-defined]
        for question in QUESTIONS:
            runner.run(question)
            invoked = _median_ms(lambda q=question: runner.run(q))
            streamed = _median_ms(lambda q=question: runner.run(q, on_progress=lambda _: None))
            assert streamed < invoked * 1.3 + 10, (question, invoked, streamed)
            result = runner.run(question)
            presented = _median_ms(lambda r=result: build_response(r).model_dump_json())
            assert presented < 20.0, (question, presented)
