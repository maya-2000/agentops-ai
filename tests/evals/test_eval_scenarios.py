"""The scenario model, the versioned eval_v1 dataset and scenario selection."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from evals.reference.context import REPOSITORY_GROUND_TRUTH
from evals.reference.labels import HiddenLabels
from evals.scenarios.loader import load_dataset, select
from evals.scenarios.model import (
    SCHEMA_VERSION,
    Category,
    Difficulty,
    EvaluationDataset,
    EvaluationScenario,
    Mode,
    ReferenceCheck,
)

REQUIRED_CATEGORIES = {
    "kpi",
    "revenue",
    "customers",
    "sales",
    "marketing",
    "support",
    "product",
    "cohorts",
    "risk",
    "forecast",
    "anomaly",
    "investigation",
    "insufficient_evidence",
    "unsupported",
    "security",
    "prompt_injection",
    "sql_security",
    "data_exposure",
    "mcp",
}


def scenario(**fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "scenario_id": "kpi_example",
        "category": "kpi",
        "difficulty": "easy",
        "question": "What was revenue last month?",
        "dataset_version": "eval_v1",
    }
    return base | fields


def dataset(*items: dict[str, Any], version: str = "eval_v1", schema: str = SCHEMA_VERSION) -> dict[str, Any]:
    manifest = {"dataset_version": version, "schema_version": schema, "updated": "2026-09-25", "description": "t"}
    return {"manifest": manifest, "scenarios": list(items)}


# ------------------------------------------------------------------ the eval_v1 dataset


def test_dataset_is_versioned_and_large_enough(eval_dataset: EvaluationDataset) -> None:
    assert eval_dataset.manifest.dataset_version == "eval_v1"
    assert eval_dataset.manifest.schema_version == SCHEMA_VERSION
    assert len(eval_dataset.scenarios) >= 50
    assert len({s.scenario_id for s in eval_dataset.scenarios}) == len(eval_dataset.scenarios)
    assert all(s.dataset_version == "eval_v1" for s in eval_dataset.scenarios)


def test_every_required_category_and_difficulty_is_covered(eval_dataset: EvaluationDataset) -> None:
    categories = Counter(s.category.value for s in eval_dataset.scenarios)
    assert set(categories) >= REQUIRED_CATEGORIES, REQUIRED_CATEGORIES - set(categories)
    assert {s.difficulty for s in eval_dataset.scenarios} == set(Difficulty)
    assert {s.mode for s in eval_dataset.scenarios} == set(Mode)


def test_every_hidden_event_is_covered_through_observable_signals(eval_dataset: EvaluationDataset) -> None:
    tagged = {t.split(":", 1)[1] for s in eval_dataset.scenarios for t in s.tags if t.startswith("event:")}
    assert tagged == {f"E{i}" for i in range(1, 8)}
    # Events are checked structurally (a reference or observable-manifestation check), never by name.
    checked = {c.event for s in eval_dataset.scenarios for c in s.reference_expectations if c.event}
    assert checked, "at least one scenario checks an event's observable manifestation"


def test_security_benchmarks_are_broad(eval_dataset: EvaluationDataset) -> None:
    injections = {s.question for s in eval_dataset.scenarios if s.category == Category.PROMPT_INJECTION}
    assert len(injections) >= 10
    sql = " ".join(
        str(c.arguments.get("sql", ""))
        for s in eval_dataset.scenarios
        if s.category in (Category.SQL_SECURITY, Category.DATA_EXPOSURE)
        for c in s.calls
    ).upper()
    for keyword in ("DROP", "DELETE", "UPDATE", "ATTACH", "COPY", "INSTALL", "LOAD", "READ_CSV", "GLOB", "/ETC/"):
        assert keyword in sql, keyword
    assert "HEALTH_SCORE" in sql and "NOT_A_COLUMN" in sql  # unknown columns / hidden state
    assert max(len(str(c.arguments.get("sql", ""))) for s in eval_dataset.scenarios for c in s.calls) > 4000
    tags = {t for s in eval_dataset.scenarios for t in s.tags}
    assert {"tool_escalation", "budget", "compromised_model", "data_exposure_false_positive", "false_positive"} <= tags
    mcp_tools = {c.tool for s in eval_dataset.scenarios for c in s.calls}
    assert {"read_file", "run_shell", "agentops_execute_python"} <= mcp_tools  # unauthorized MCP tools


def test_suites_have_the_documented_sizes(eval_dataset: EvaluationDataset) -> None:
    critical = select(eval_dataset.scenarios, suite="critical")
    multi_seed = select(eval_dataset.scenarios, suite="multi_seed")
    assert 10 <= len(critical) <= 15
    assert 5 <= len(multi_seed) <= 15
    critical_categories = {s.category for s in critical}
    assert {Category.KPI, Category.PROMPT_INJECTION, Category.SQL_SECURITY, Category.MCP} <= critical_categories


def test_scenarios_hold_no_expected_business_values_or_answers(eval_dataset: EvaluationDataset) -> None:
    """Expectations are named reference checks: values and answers are resolved at run time, never written down."""
    assert not {"value", "expected", "expected_value", "answer", "member"} & set(ReferenceCheck.model_fields)
    labels = HiddenLabels.load(REPOSITORY_GROUND_TRUTH)
    checked = 0
    for s in eval_dataset.scenarios:
        events = {c.event for c in s.reference_expectations if c.event and c.kind != "kpi_value"}
        text = s.model_dump_json()
        for event_id in events:
            observable = labels.observable(event_id)
            answers = [observable.country, observable.segment, observable.channel, observable.campaign_id]
            answers += [observable.sales_rep, *observable.ticket_categories]
            for answer in filter(None, answers):
                checked += 1
                assert answer not in text, (s.scenario_id, event_id, answer)
        for marker in labels.leak_markers():  # event names and descriptions (not the generic file names)
            assert " " not in marker or marker not in text, (s.scenario_id, marker)
    assert checked >= 8


# ------------------------------------------------------------------ the model's validation


def test_valid_scenario_parses() -> None:
    parsed = EvaluationScenario.model_validate(scenario())
    assert parsed.mode == Mode.AGENT and parsed.efficiency_budget == 1


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"question": None}, "needs a question"),
        ({"mode": "parity", "question": None}, "needs calls"),
        ({"mode": "evidence_integrity"}, "need mutations"),
        ({"category": "prompt_injection"}, "security expectation"),
        ({"unknown_field": 1}, "Extra inputs"),
        ({"scenario_id": "Bad-ID"}, "pattern"),
        ({"difficulty": "trivial"}, "difficulty"),
        (
            {
                "mode": "parity",
                "calls": [{"tool": "agentops_get_kpi"}],
                "call_expectations": [{"outcome": "ok"}, {"outcome": "ok"}],
            },
            "one call expectation per call",
        ),
    ],
)
def test_invalid_scenarios_are_rejected(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        EvaluationScenario.model_validate(scenario(**fields))


def test_dataset_rejects_duplicates_version_mismatch_and_schema_drift() -> None:
    with pytest.raises(ValidationError, match="Duplicate scenario IDs"):
        EvaluationDataset.model_validate(dataset(scenario(), scenario()))
    with pytest.raises(ValidationError, match="another dataset version"):
        EvaluationDataset.model_validate(dataset(scenario(dataset_version="eval_v2")))
    with pytest.raises(ValidationError, match="Schema version"):
        EvaluationDataset.model_validate(dataset(scenario(), schema="0.9"))
    with pytest.raises(ValidationError):
        EvaluationDataset.model_validate(dataset(scenario(), version="v1"))


def test_loader_checks_the_declared_version(tmp_path: Path) -> None:
    path = tmp_path / "eval_v9.json"
    path.write_text(json.dumps(dataset(scenario())), encoding="utf-8")
    assert load_dataset(path=path).manifest.dataset_version == "eval_v1"
    with pytest.raises(FileNotFoundError, match="Unknown dataset"):
        load_dataset("eval_v999")


# ------------------------------------------------------------------ selection


def test_selection_by_suite_category_and_id(eval_dataset: EvaluationDataset) -> None:
    items = eval_dataset.scenarios
    assert len(select(items)) == len(select(items, suite="full")) == len(items)
    kpis = select(items, categories=["kpi"])
    assert kpis and all(s.category == Category.KPI for s in kpis)
    one = select(items, scenario_ids=["kpi_revenue_last_month"])
    assert [s.scenario_id for s in one] == ["kpi_revenue_last_month"]
    assert select(items, suite="critical", categories=["kpi"]) == [s for s in kpis if "critical" in s.suites]
    with pytest.raises(ValueError, match="Unknown scenarios"):
        select(items, scenario_ids=["does_not_exist"])
