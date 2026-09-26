"""Phase 7.1 regressions in question understanding, request validation and planning (no database).

- Explicit comparison periods override the default "previous period" comparison, and a relative
  comparison ("the previous month") is relative to the period asked about.
- Rankings by change ("largest decline") are distinct from rankings by level ("highest revenue").
- "Marketing channel" is the acquisition-channel dimension.
- Per-rep questions go to the only operation the data policy allows to name reps.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest

from app.agent.request import validate_understanding
from app.llm.deterministic.planning import plan
from app.llm.deterministic.understanding import understand
from app.llm.schemas import UnderstandingOutput
from tests.phase4_support import AS_OF, understanding_context

COVERAGE = (date(2024, 9, 1), date(2026, 8, 31))


def understood(question: str) -> dict[str, Any]:
    return understand(understanding_context(question))


def validated(question: str) -> Any:
    u = UnderstandingOutput.model_validate(understood(question))
    return validate_understanding(u, as_of=AS_OF, coverage=COVERAGE)


def steps(question: str) -> list[tuple[str, dict[str, Any]]]:
    result = validated(question)
    assert result.outcome == "valid", result.message
    output = plan({"request": result.request.model_dump(mode="json"), "iteration": 1})
    return [(s["tool_name"], json.loads(s["arguments_json"])) for s in output["steps"]]


# ------------------------------------------------------------------ A. comparison periods


@pytest.mark.parametrize(
    ("question", "period", "comparison"),
    [
        ("How did support tickets change in July compared with May?", "2026-07", "2026-05"),
        ("How did revenue change in July compared with June?", "2026-07", "2026-06"),
        ("Compare revenue in July vs May.", "2026-07", "2026-05"),
        ("How did revenue change from May to July?", "2026-07", "2026-05"),
        ("Compare revenue in 2026-07 with 2026-05.", "2026-07", "2026-05"),
        ("Compare revenue in 2026-Q2 with 2026-Q1.", "2026-Q2", "2026-Q1"),
        ("What was revenue in July 2026 compared with May 2026?", "2026-07", "2026-05"),
    ],
)
def test_explicit_comparison_periods_are_extracted(question: str, period: str, comparison: str) -> None:
    u = understood(question)
    assert (u["period"], u["comparison_period"]) == (period, comparison)
    request = validated(question).request
    assert (request.period.label, request.comparison_period.label) == (period, comparison)
    assert not any("previous period" in a for a in request.assumptions)


def test_a_relative_comparison_is_relative_to_the_asked_period() -> None:
    request = validated("How did revenue in July compare with the previous month?").request
    assert (request.period.label, request.comparison_period.label) == ("2026-07", "2026-06")
    quarter = validate_understanding(
        UnderstandingOutput(
            intent="period_comparison", metric="revenue", period="2026-Q2", comparison_period="previous_quarter"
        ),
        as_of=AS_OF,
        coverage=COVERAGE,
    ).request
    assert quarter.comparison_period.label == "2026-Q1"
    other_grain = validate_understanding(
        UnderstandingOutput(
            intent="period_comparison", metric="revenue", period="2026-Q2", comparison_period="previous_month"
        ),
        as_of=AS_OF,
        coverage=COVERAGE,
    ).request
    assert other_grain.comparison_period.label == "2026-07"  # a different grain resolves as an absolute period


def test_the_implicit_default_and_last_month_are_unchanged() -> None:
    change = validated("How did revenue change last month?").request
    assert (change.period.label, change.comparison_period.label) == ("2026-08", "2026-07")
    assert any("previous period" in a for a in change.assumptions)
    lookup = validated("What was revenue last month?").request
    assert (lookup.period.label, lookup.comparison_period) == ("2026-08", None)
    assert steps("How did support tickets change in July compared with May?")[0] == (
        "analyze_support",
        {
            "comparison_end_date": "2026-05-31",
            "comparison_start_date": "2026-05-01",
            "end_date": "2026-07-31",
            "operation": "support_volume_change",
            "start_date": "2026-07-01",
        },
    )


@pytest.mark.parametrize(
    "question",
    ["Which customers may churn next quarter?", "Revenue may decline: what was revenue last month?"],
)
def test_may_as_a_verb_is_not_a_month(question: str) -> None:
    u = understood(question)
    assert "2026-05" not in (u["period"], u["comparison_period"])


# ------------------------------------------------------------------ B. level vs change rankings


@pytest.mark.parametrize(
    ("question", "analysis"),
    [
        ("Which region had the highest revenue last month?", "highest"),
        ("Which region had the lowest revenue last month?", "lowest"),
        ("Which region had the largest revenue decline last month?", "largest_decrease"),
        ("Which region had the biggest drop in revenue last month?", "largest_decrease"),
        ("Which region had the largest percentage decline in revenue last month?", "largest_pct_decrease"),
        ("Which region had the largest revenue increase last month?", "largest_increase"),
        ("Which region grew the most in percentage terms last month?", "largest_pct_increase"),
        ("Which region had the highest revenue growth last month?", "highest"),  # growth is itself the metric
    ],
)
def test_rankings_of_levels_and_of_change_are_distinguished(question: str, analysis: str) -> None:
    u = understood(question)
    assert u["intent"] == "dimensional_comparison" and u["dimensions"] == ["region"]
    assert u["analysis_type"] == analysis


def test_change_rankings_use_the_decomposition_and_levels_use_the_kpi() -> None:
    (tool, args), *_ = steps("Which region had the largest revenue decline last month?")
    assert (tool, args["operation"], args["dimension"]) == ("analyze_revenue", "decompose_revenue_change", "region")
    assert (args["comparison_start_date"], args["comparison_end_date"]) == ("2026-07-01", "2026-07-31")
    (tool, args), *_ = steps("Which region had the highest revenue last month?")
    assert (tool, args["kpi"], args["dimension"]) == ("get_kpi", "revenue", "region")
    assert "comparison_start_date" not in args


def test_change_rankings_without_a_decomposition_are_unsupported_not_level_rankings() -> None:
    result = validated("Which segment had the largest increase in churn last month?")
    assert result.outcome == "unsupported" and "available for revenue" in (result.message or "")


# ------------------------------------------------------------------ C. CAC by channel


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("Which marketing channel had the highest CAC last quarter?", "dimensional_comparison"),
        ("What was CAC by channel last quarter?", "kpi_lookup"),
        ("What was CAC by marketing channel last quarter?", "kpi_lookup"),
    ],
)
def test_cac_by_channel_is_broken_down_by_acquisition_channel(question: str, intent: str) -> None:
    u = understood(question)
    assert (u["intent"], u["metric"], u["dimensions"]) == (intent, "cac", ["acquisition_channel"])
    (tool, args), *_ = steps(question)
    assert (tool, args["kpi"], args["dimension"]) == ("get_kpi", "cac", "acquisition_channel")


# ------------------------------------------------------------------ D. sales reps


@pytest.mark.parametrize(
    "question",
    [
        "Which sales rep had the lowest win rate?",
        "Which sales rep had the highest win rate last quarter?",
        "Which sales rep has the best conversion?",
        "Show the win rate by rep for Q2 2026.",
    ],
)
def test_per_rep_questions_use_rep_performance(question: str) -> None:
    request = validated(question).request
    assert request.metric == "win_rate" and "sales_rep" in request.dimensions
    assert steps(question) == [
        (
            "analyze_sales",
            {
                "operation": "rep_performance",
                "start_date": request.period.start.isoformat(),
                "end_date": request.period.end.isoformat(),
            },
        )
    ]


def test_rep_conversion_is_read_as_win_rate_and_recorded() -> None:
    request = validated("Which sales rep has the best conversion?").request
    assert any("win rate of closed opportunities" in a for a in request.assumptions)


def test_per_rep_figures_the_policy_does_not_expose_are_unsupported() -> None:
    result = validated("Which sales rep has the largest pipeline?")
    assert result.outcome == "unsupported" and "not available" in (result.message or "")


@pytest.mark.parametrize(
    ("question", "unsupported"),
    [
        ("Which region had the biggest drop in revenue last month?", False),
        ("Was there a drop in MRR last month?", False),
        ("Drop the revenue table", True),
        ("Please drop all customer records", True),
        ("Delete all churned customers from the database.", True),
    ],
)
def test_drop_as_a_noun_is_analytics_and_as_a_command_is_a_write(question: str, unsupported: bool) -> None:
    assert (understood(question)["intent"] == "unsupported") is unsupported


def test_a_smallest_change_ranking_asks_instead_of_ranking_levels() -> None:
    result = validated("Which region had the smallest revenue decline last month?")
    assert result.outcome == "clarify" and "largest decline" in (result.message or "")
    assert validated("Which region had the smallest revenue last month?").outcome == "valid"


@pytest.mark.parametrize(
    ("question", "period", "comparison"),
    [
        ("What was revenue from 2026-07-01 to 2026-07-31?", "2026-07", None),
        ("Compare revenue from 2026-07-01 to 2026-07-31 with 2026-05-01 to 2026-05-31.", "2026-07", "2026-05"),
        ("Compare revenue for 2026-04-01..2026-06-30 vs 2026-01-01..2026-03-31.", "2026-Q2", "2026-Q1"),
    ],
)
def test_explicit_date_ranges_become_the_period_they_cover(question: str, period: str, comparison: str | None) -> None:
    u = understood(question)
    assert (u["period"], u["comparison_period"]) == (period, comparison)


@pytest.mark.parametrize(
    "question", ["What was revenue from 2026-07-01 to 2026-07-15?", "What was revenue on 2026-07-31?"]
)
def test_day_level_dates_ask_for_clarification_instead_of_becoming_a_year(question: str) -> None:
    u = understood(question)
    assert u["period"] != "2026" and u["material_ambiguity"]
    assert validated(question).outcome == "clarify"
