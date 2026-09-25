"""Fixtures for the evaluation-framework tests.

The benchmark runs on the session's full generated dataset (the ``full_dataset`` fixture, seed 42),
with the ground-truth file that generation wrote next to it. Nothing is read from the repository
database, so the tests need no pre-built database, network, model or API key.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from data.generator.generate import GenerationResult
from evals.reference.context import DataSource, EvalContext
from evals.scenarios.loader import load_dataset, select
from evals.scenarios.model import EvaluationDataset, EvaluationScenario
from tests.phase7_support import source_for

if TYPE_CHECKING:
    from evals.reports.models import EvaluationResult


@pytest.fixture(scope="session")
def eval_dataset() -> EvaluationDataset:
    return load_dataset()


@pytest.fixture(scope="session")
def scenarios(eval_dataset: EvaluationDataset) -> dict[str, EvaluationScenario]:
    return {s.scenario_id: s for s in eval_dataset.scenarios}


@pytest.fixture(scope="session")
def eval_source(full_dataset: GenerationResult) -> DataSource:
    return source_for(full_dataset)


@pytest.fixture(scope="session")
def eval_ctx(eval_source: DataSource) -> Iterator[EvalContext]:
    ctx = EvalContext.open(eval_source)
    yield ctx
    ctx.close()


@pytest.fixture(scope="session")
def critical_results(eval_dataset: EvaluationDataset, eval_source: DataSource) -> list[EvaluationResult]:
    """The critical regression suite, run once per session on the full dataset (deterministic mode)."""
    from evals.benchmark import run_scenarios
    from evals.engine import EngineConfig

    return run_scenarios(select(eval_dataset.scenarios, suite="critical"), eval_source, EngineConfig())
