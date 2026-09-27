"""The eval_v2 investigation scenario (Phase 10): what to investigate and what the system should do.

eval_v1 (``evals/scenarios/model.py``) is unchanged: it evaluates single questions, MCP calls and
evidence integrity. eval_v2 evaluates multi-step investigations with its own typed model, so neither
dataset constrains the other.

As in eval_v1, a scenario never contains an expected business number or conclusion. Periods are
relative (``last_month``, ``last_quarter``) or explicit labels. Numerical expectations are named
reference checks, resolved at run time from the independent reference implementation, so the
benchmark stays correct on another seed.

Execution modes:

- ``investigation``: an objective to ``Investigator`` on the agent's runtime (in process).
- ``api``: ``POST /api/v1/investigations`` and ``/investigations/stream`` through the real FastAPI app.
- ``ui``: the API response through the Streamlit page's view models (what the page renders).
- ``cancellation``: an investigation cancelled after some steps, or given no time.
- ``mcp_parity``: each executed step whose tool the MCP server exposes is repeated through MCP; the
  business results must match. Discovery checks that MCP gained no investigation capability.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from evals.scenarios.loader import DATASETS_DIR
from evals.scenarios.model import Difficulty, LatencyClass, LeakKind, Strict

INVESTIGATION_SCHEMA_VERSION = "2.0"
INVESTIGATION_DATASETS = ("eval_v2",)


class InvestigationCategory(StrEnum):
    PLANNING = "investigation_planning"
    STEP_SELECTION = "step_selection"
    TOOL_SELECTION = "tool_selection"
    EVIDENCE_GROUNDING = "evidence_grounding"
    PERIOD = "period_correctness"
    COMPARISON = "comparison_correctness"
    DRIVERS = "driver_identification"
    RECOMMENDATIONS = "recommendation_grounding"
    CAUSAL = "causal_language_safety"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    REFUSAL = "refusal"
    SECURITY = "security"
    BUDGET = "budget_enforcement"
    CANCELLATION = "cancellation"
    MANAGEMENT_BRIEF = "management_brief"
    API = "investigation_api"
    UI = "ui_transformation"
    MCP_PARITY = "mcp_parity"


class InvestigationMode(StrEnum):
    INVESTIGATION = "investigation"
    API = "api"
    UI = "ui"
    CANCELLATION = "cancellation"
    MCP_PARITY = "mcp_parity"


ALL_LEAKS: tuple[LeakKind, ...] = ("system_prompt", "secrets", "ground_truth", "file_contents", "withheld_fields")


class StepPattern(Strict):
    """A plan step the objective needs: its tool and, optionally, the operation, KPI, metric or dimension."""

    tool: str
    operation: str | None = None
    kpi: str | None = None
    metric: str | None = None
    dimension: str | None = None


class InvestigationReference(Strict):
    """A check whose expected value comes from the independent reference implementation at run time.

    - ``outcome_change``: the outcome finding's current and comparison values equal the reference KPI
      in the plan's two periods, and its direction is the reference direction.
    - ``top_contribution``: the largest ``contributes_to`` driver of ``dimension`` is the member with the
      largest change in the outcome's direction, and its share is that member's share of the gross change.
    """

    kind: Literal["outcome_change", "top_contribution"]
    metric: str | None = None
    dimension: str | None = None


class InvestigationExpectation(Strict):
    statuses: list[str]  # acceptable final statuses
    template: str | None = None
    required_steps: list[StepPattern] = Field(default_factory=list)
    forbidden_steps: list[StepPattern] = Field(default_factory=list)  # steps the objective makes unnecessary
    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=lambda: ["run_safe_sql"])
    period: str | None = None  # "last_month", "last_quarter" or a label (YYYY-MM / YYYY-Qn)
    comparison_period: str | None = None  # "previous" (same length, immediately before) or a label
    filters: dict[str, str] | None = None
    outcome_metric: str | None = None
    min_findings: int = 0
    require_outcome: bool | None = None
    require_drivers: bool | None = None
    require_recommendations: bool | None = None
    require_sections: bool | None = None
    uncertainty_contains: list[str] = Field(default_factory=list)
    message_contains: list[str] = Field(default_factory=list)
    max_tool_calls: int | None = None
    exhausted: list[str] = Field(default_factory=list)  # budgets that must be reported as exhausted
    refusal_kind: str | None = None  # API: policy / invalid_input / out_of_scope
    http_status: int = 200  # API
    error_code: str | None = None  # API error code for a rejected request
    must_not_leak: list[LeakKind] = Field(default_factory=lambda: list(ALL_LEAKS))


class InvestigationScenario(Strict):
    scenario_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    category: InvestigationCategory
    difficulty: Difficulty
    mode: InvestigationMode = InvestigationMode.INVESTIGATION
    description: str = ""
    objective: str
    limits: dict[str, Any] = Field(default_factory=dict)  # AgentConfig overrides (investigation budgets)
    cancel_after_steps: int | None = None  # cancellation mode: cancel after this many finished steps
    deadline_seconds: float | None = None  # cancellation mode: the investigation's deadline
    expect: InvestigationExpectation
    references: list[InvestigationReference] = Field(default_factory=list)
    stream: bool = False  # api mode: also run the streaming endpoint
    latency_class: LatencyClass = "complex_investigation"
    suites: list[Literal["critical", "multi_seed"]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    dataset_version: str

    @model_validator(mode="after")
    def _consistent(self) -> InvestigationScenario:
        if (
            self.mode == InvestigationMode.CANCELLATION
            and self.cancel_after_steps is None
            and (self.deadline_seconds is None)
        ):
            raise ValueError(f"{self.scenario_id}: a cancellation scenario needs cancel_after_steps or a deadline")
        if self.expect.exhausted and "budget_exhausted" not in self.expect.statuses:
            raise ValueError(f"{self.scenario_id}: an exhausted budget implies the budget_exhausted status")
        if self.expect.http_status != 200 and self.mode != InvestigationMode.API:
            raise ValueError(f"{self.scenario_id}: only API scenarios expect an HTTP error")
        return self


class InvestigationManifest(Strict):
    dataset_version: str = Field(pattern=r"^eval_v\d+$")
    schema_version: str
    updated: str
    description: str


class InvestigationDataset(Strict):
    manifest: InvestigationManifest
    scenarios: list[InvestigationScenario]

    @model_validator(mode="after")
    def _unique(self) -> InvestigationDataset:
        ids = [s.scenario_id for s in self.scenarios]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"Duplicate scenario IDs: {', '.join(duplicates)}")
        wrong = [s.scenario_id for s in self.scenarios if s.dataset_version != self.manifest.dataset_version]
        if wrong:
            raise ValueError(f"Scenarios with another dataset version: {', '.join(wrong)}")
        if self.manifest.schema_version != INVESTIGATION_SCHEMA_VERSION:
            raise ValueError(f"Schema version {self.manifest.schema_version} is not {INVESTIGATION_SCHEMA_VERSION}")
        return self


def is_investigation_dataset(version: str) -> bool:
    return version in INVESTIGATION_DATASETS


def load_investigation_dataset(version: str = "eval_v2", path: Path | None = None) -> InvestigationDataset:
    source = path or DATASETS_DIR / f"{version}.json"
    if not source.is_file():
        raise FileNotFoundError(f"Unknown investigation dataset {version!r}")
    dataset = InvestigationDataset.model_validate(json.loads(source.read_text(encoding="utf-8")))
    if path is None and dataset.manifest.dataset_version != version:
        raise ValueError(f"{version}.json declares {dataset.manifest.dataset_version}")
    return dataset


def select_investigations(
    scenarios: Iterable[InvestigationScenario],
    *,
    suite: str | None = None,
    categories: Iterable[str] = (),
    scenario_ids: Iterable[str] = (),
) -> list[InvestigationScenario]:
    chosen = list(scenarios)
    wanted_ids = set(scenario_ids)
    if wanted_ids:
        missing = sorted(wanted_ids - {s.scenario_id for s in chosen})
        if missing:
            raise ValueError(f"Unknown scenarios: {', '.join(missing)}")
        chosen = [s for s in chosen if s.scenario_id in wanted_ids]
    wanted = {InvestigationCategory(c) for c in categories}
    if wanted:
        chosen = [s for s in chosen if s.category in wanted]
    if suite not in (None, "full"):
        chosen = [s for s in chosen if suite in s.suites]
    return chosen
