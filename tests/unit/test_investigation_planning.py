"""Phase 10: the investigation model, templates, planner and step dependencies (no database).

Plans are data: each step names one allow-listed tool, arguments that its input model accepts, the
validated intent it is authorised under, and dependencies, conditions and bindings from closed sets.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from app.agent.request import ValidatedRequest
from app.analytics.periods import previous_period, resolve_period
from app.evidence.models import Evidence
from app.investigation import FINAL_STATUSES, Investigation
from app.investigation.models import (
    STOP_MESSAGE,
    AnalysisStep,
    FindingRelationship,
    InvestigationBudgetReport,
    StepBinding,
)
from app.investigation.planner import asks_for_brief, plan_investigation, select_template
from app.investigation.steps import SKIP_REASONS, bind, concentrated_member, condition_met
from app.investigation.templates import DEFAULT_OUTCOMES, TEMPLATE_TITLES, build_plan
from app.llm.schemas import Intent
from app.security.authorization import INTENT_TOOL_PERMISSIONS, SQL_TOOL
from app.tools import ToolRegistry
from app.tools.base import ToolResult
from tests.phase4_support import AS_OF

TEMPLATES = tuple(TEMPLATE_TITLES)
AUGUST = resolve_period("2026-08", as_of=AS_OF)
JULY = previous_period(AUGUST)
REASONING = ("i think", "let me", "reasoning", "chain of thought", "because", "step by step")


def plan(template: str, **filters: str) -> Any:
    return build_plan(template, period=AUGUST, comparison=JULY, filters=filters)  # type: ignore[arg-type]


# ------------------------------------------------------------------ templates


@pytest.mark.parametrize("template", TEMPLATES)
def test_every_step_is_an_allow_listed_tool_authorised_for_its_intent(template: str) -> None:
    registry = ToolRegistry()
    analysis = plan(template)
    assert analysis.steps and analysis.title == TEMPLATE_TITLES[template]
    for step in analysis.steps:
        assert step.tool_name in registry.names and step.tool_name != SQL_TOOL
        assert step.tool_name in INTENT_TOOL_PERMISSIONS[step.authorized_as], (step.step_id, step.tool_name)
        assert "sql" not in step.arguments and "query" not in step.arguments


@pytest.mark.parametrize("template", TEMPLATES)
def test_every_step_has_arguments_its_tool_accepts(template: str) -> None:
    registry = ToolRegistry()
    for step in plan(template).steps:
        definition = registry.get(step.tool_name)
        assert definition is not None
        if step.binding is None:  # a bound step gets its missing filter from evidence at run time
            definition.input_model.model_validate(step.arguments)


@pytest.mark.parametrize("template", TEMPLATES)
def test_steps_are_ordered_with_dependencies_on_earlier_steps(template: str) -> None:
    steps = plan(template).steps
    ids = [s.step_id for s in steps]
    assert ids == [f"S{i}" for i in range(1, len(ids) + 1)]
    for index, step in enumerate(steps):
        assert set(step.depends_on) <= set(ids[:index])
        if step.binding is not None:
            assert step.binding.source_step in step.depends_on
        if step.condition is not None:
            assert step.depends_on


@pytest.mark.parametrize("template", TEMPLATES)
def test_step_titles_are_short_actions_not_reasoning(template: str) -> None:
    for step in plan(template).steps:
        assert 3 < len(step.title) <= 90
        assert not any(word in step.title.lower() for word in REASONING), step.title


@pytest.mark.parametrize("template", TEMPLATES)
def test_plans_fit_the_default_investigation_budget(template: str) -> None:
    from app.agent.config import AgentConfig

    config = AgentConfig()
    assert len(plan(template).steps) <= min(config.max_investigation_steps, config.max_investigation_tool_calls)


def test_the_revenue_plan_decomposes_then_drills_into_a_concentrated_region() -> None:
    steps = plan("revenue").steps
    decompositions = [s for s in steps if s.arguments.get("operation") == "decompose_revenue_change"]
    region = next(s for s in decompositions if s.arguments.get("dimension") == "region")
    assert region.condition == "outcome_changed" and region.depends_on == ["S1"]
    drill = steps[-1]
    assert drill.condition == "concentrated" and drill.arguments["dimension"] == "country"
    assert drill.binding == StepBinding(kind="concentrated_member", source_step=region.step_id)


def test_a_filtered_dimension_is_not_decomposed_again() -> None:
    steps = plan("revenue", region="EMEA").steps
    dimensions = [s.arguments.get("dimension") for s in steps if s.tool_name == "analyze_revenue"]
    assert "region" not in dimensions and "country" not in dimensions
    assert all(
        s.arguments.get("filters", {}).get("region") == "EMEA" for s in steps if s.tool_name == "analyze_revenue"
    )


def test_the_management_brief_covers_every_area_and_a_forecast() -> None:
    steps = plan("management_brief").steps
    assert {s.area for s in steps} >= {"revenue", "customers", "sales", "product", "support", "anomalies", "forecast"}
    forecast = next(s for s in steps if s.tool_name == "forecast_metric")
    assert forecast.authorized_as == Intent.FORECAST


def test_product_support_binds_the_first_two_features() -> None:
    steps = plan("product_support").steps
    bound = [s for s in steps if s.binding is not None]
    assert [b.binding.index for b in bound if b.binding] == [1, 2]
    assert all(b.binding and b.binding.kind == "feature" for b in bound)


@pytest.mark.parametrize(
    ("template", "outcome"),
    [
        ("customer", "customer_count"),
        ("customer", "retention_rate"),
        ("customer", "revenue_churn_rate"),
        ("customer", "logo_churn_rate"),
        ("customer", "nrr"),
        ("product_support", "average_resolution_time"),
        ("product_support", "support_ticket_volume"),
        ("sales", "cac"),
        ("sales", "win_rate"),
    ],
)
def test_each_template_measures_the_outcome_it_explains(template: str, outcome: str) -> None:
    steps = build_plan(template, period=AUGUST, comparison=JULY, outcome_metric=outcome).steps  # type: ignore[arg-type]
    measured = [s for s in steps if s.tool_name == "get_kpi" and s.arguments.get("kpi") == outcome]
    assert len(measured) == 1 and "comparison_start_date" in measured[0].arguments


def test_product_adoption_is_measured_per_feature() -> None:
    request_ = ValidatedRequest(intent=Intent.PRODUCT_ANALYSIS, metric="product_adoption")
    analysis = plan_investigation("product_support", request_, as_of=AS_OF)
    assert analysis.outcome_metric == "product_adoption"
    bound = [s for s in analysis.steps if s.binding is not None]
    assert bound and all(s.arguments["kpi"] == "product_adoption" for s in bound)


# ------------------------------------------------------------------ planner


def request(**fields: Any) -> ValidatedRequest:
    return ValidatedRequest(**{"intent": Intent.REVENUE_INVESTIGATION, "metric": "revenue", **fields})


@pytest.mark.parametrize(
    ("objective", "fields", "expected"),
    [
        ("Why is revenue growth slowing?", {}, "revenue"),
        (
            "Why is churn increasing?",
            {"intent": Intent.CUSTOMER_INVESTIGATION, "metric": "logo_churn_rate"},
            "customer",
        ),
        ("Why did win rate drop?", {"intent": Intent.SALES_ANALYSIS, "metric": "win_rate"}, "sales"),
        ("Why did CAC rise?", {"intent": Intent.MARKETING_ANALYSIS, "metric": "cac"}, "sales"),
        (
            "Why did support tickets increase?",
            {"intent": Intent.SUPPORT_ANALYSIS, "metric": "support_ticket_volume"},
            "product_support",
        ),
        ("Why is adoption falling?", {"intent": Intent.PRODUCT_ANALYSIS, "metric": None}, "product_support"),
        ("Investigate revenue and churn", {"intent": Intent.MIXED_INVESTIGATION, "metric": "revenue"}, "general"),
        ("Give me a management brief", {}, "management_brief"),
        ("What should management investigate next?", {"intent": Intent.UNSUPPORTED}, "management_brief"),
    ],
)
def test_template_selection(objective: str, fields: dict[str, Any], expected: str) -> None:
    assert select_template(objective, request(**fields)) == expected


def test_no_template_without_a_validated_request_or_brief_cue() -> None:
    assert select_template("Why is revenue growth slowing?", None) is None
    assert select_template("Tell me a joke", request(intent=Intent.UNSUPPORTED, metric=None)) is None
    assert asks_for_brief("Summarise the state of the business") and not asks_for_brief("Why did revenue fall?")


def test_default_periods_are_recorded_as_assumptions() -> None:
    analysis = plan_investigation("revenue", request(), as_of=AS_OF)
    assert (analysis.period_label, analysis.comparison_label) == ("2026-08", "2026-07")
    assert any("latest complete month" in a for a in analysis.assumptions)
    assert any("previous period" in a for a in analysis.assumptions)


def test_validated_periods_and_filters_are_used() -> None:
    q2 = resolve_period("2026-Q2", as_of=AS_OF)
    analysis = plan_investigation("revenue", request(period=q2, filters={"segment": "SMB"}), as_of=AS_OF)
    assert (analysis.period_label, analysis.comparison_label) == ("2026-Q2", "2026-Q1")
    assert not any("latest complete month" in a for a in analysis.assumptions)
    assert all(s.arguments.get("filters", {}).get("segment") == "SMB" for s in analysis.steps[:1])


@pytest.mark.parametrize("template", TEMPLATES)
def test_the_outcome_metric_defaults_per_template(template: str) -> None:
    analysis = plan_investigation(template, None, as_of=AS_OF)  # type: ignore[arg-type]
    assert analysis.outcome_metric == DEFAULT_OUTCOMES[template]  # type: ignore[index]


def test_a_template_metric_becomes_the_outcome() -> None:
    analysis = plan_investigation("customer", request(metric="nrr", intent=Intent.CUSTOMER_INVESTIGATION), as_of=AS_OF)
    assert analysis.outcome_metric == "nrr"
    unrelated = plan_investigation("customer", request(metric="win_rate"), as_of=AS_OF)
    assert unrelated.outcome_metric == DEFAULT_OUTCOMES["customer"]


# ------------------------------------------------------------------ closed conditions and bindings


def evidence(
    evidence_id: str, *, member: str | None = None, dimension: str | None = None, **attributes: Any
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        evidence_type="calculated",
        statement="test",
        metric="revenue",
        value=1.0,
        dimension=dimension,
        dimension_value=member,
        attributes=attributes,
        tool_name="analyze_revenue",
        tool_call_id="T1",
        operation="test",
        execution_timestamp=datetime.now(UTC),
    )


def test_outcome_changed_reads_the_headline_change() -> None:
    assert condition_met("outcome_changed", [evidence("E1", absolute_change=-10.0)])
    assert not condition_met("outcome_changed", [evidence("E1", absolute_change=0.0)])
    assert not condition_met("outcome_changed", [evidence("E1", member="APAC", dimension="region", absolute_change=-5)])
    assert not condition_met("outcome_changed", [])


def test_concentration_needs_a_rank_one_member_with_most_of_the_change() -> None:
    top = evidence("E2", member="APAC", dimension="region", rank=1, share_of_gross_decline=0.8)
    assert concentrated_member([evidence("E1", absolute_change=-1), top]) is top
    assert condition_met("concentrated", [top])
    spread = evidence("E2", member="APAC", dimension="region", rank=1, share_of_gross_decline=0.3)
    assert not condition_met("concentrated", [spread])
    second = evidence("E3", member="EMEA", dimension="region", rank=2, share_of_gross_decline=0.9)
    assert concentrated_member([second]) is None


def test_bindings_fill_one_filter_from_evidence_or_skip() -> None:
    top = evidence("E2", member="APAC", dimension="region", rank=1, share_of_gross_decline=0.8)
    binding = StepBinding(kind="concentrated_member", source_step="S2")
    bound = bind(binding, [top], {"dimension": "country", "filters": {"segment": "SMB"}})
    assert bound == {"dimension": "country", "filters": {"segment": "SMB", "region": "APAC"}}
    assert bind(binding, [], {"dimension": "country"}) is None
    features = [evidence("E1", member="dashboards"), evidence("E2", member="exports")]
    assert bind(StepBinding(kind="feature", source_step="S1", index=2), features, {})["filters"] == {  # type: ignore[index]
        "product_feature": "exports"
    }
    assert bind(StepBinding(kind="feature", source_step="S1", index=3), features, {}) is None
    assert set(SKIP_REASONS) == {"dependency", "outcome_changed", "concentrated", "binding"}


@pytest.mark.parametrize(
    "bad",
    [
        {"condition": "__import__('os').system('id')"},
        {"condition": "lambda: True"},
        {"binding": {"kind": "python", "source_step": "S1"}},
        {"binding": {"kind": "feature", "source_step": "S1", "index": 100}},
        {"authorized_as": "admin"},
        {"area": "filesystem"},
    ],
)
def test_conditions_bindings_and_intents_are_closed_sets(bad: dict[str, Any]) -> None:
    base: dict[str, Any] = {
        "step_id": "S1",
        "title": "Measure",
        "area": "revenue",
        "tool_name": "get_kpi",
        "authorized_as": Intent.KPI_LOOKUP,
    }
    with pytest.raises(ValidationError):
        AnalysisStep(**{**base, **bad})


@pytest.mark.parametrize("relationship", ["causes", "caused_by", "explains", "drives"])
def test_causal_relationship_types_cannot_be_represented(relationship: str) -> None:
    with pytest.raises(ValidationError):
        FindingRelationship(source_finding_id="F1", target_finding_id="F2", relationship=relationship, rule="x")  # type: ignore[arg-type]


# ------------------------------------------------------------------ the investigation model


def test_raw_tool_results_are_never_serialised() -> None:
    investigation = Investigation(
        investigation_id="I-test",
        objective="x",
        created_at=datetime.now(UTC),
        budget=InvestigationBudgetReport(
            max_steps=1, max_tool_calls=1, max_seconds=1.0, max_evidence=10, max_output_chars=1000
        ),
        tool_results=[
            ToolResult(
                call_id="T1",
                tool_name="get_kpi",
                success=True,
                status="ok",
                message="withheld",
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
                execution_time_ms=1.0,
            )
        ],
    )
    dumped = investigation.model_dump_json()
    assert "tool_results" not in dumped and "withheld" not in dumped
    assert "withheld" not in repr(investigation)


def test_final_statuses_and_the_stop_message() -> None:
    assert set(FINAL_STATUSES) == {
        "completed",
        "insufficient_evidence",
        "budget_exhausted",
        "refused",
        "unsupported",
        "failed",
        "cancelled",
    }
    assert STOP_MESSAGE == "Investigation stopped because the analysis budget was reached."
    assert date(2026, 8, 31) == AS_OF
