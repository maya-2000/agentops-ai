"""Phase 9 regression: a question about one day is never answered with a month's value.

KPIs are reported for whole months, quarters and years. "What was revenue on 3 March 2026?" used
to return March's monthly revenue silently. A day-level date now asks for clarification, and says
that the KPI is not available for a single day. Month-level phrasings are unchanged, and so are the
Phase 7.1 comparison periods.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from app.agent.request import validate_understanding
from app.llm.deterministic.understanding import understand
from app.llm.schemas import UnderstandingOutput
from tests.phase4_support import AS_OF, understanding_context

COVERAGE = (date(2024, 9, 1), AS_OF)


def understood(question: str) -> dict[str, Any]:
    output: dict[str, Any] = understand(understanding_context(question))
    return output


def outcome(question: str) -> str:
    u = UnderstandingOutput.model_validate(understood(question))
    return validate_understanding(u, as_of=AS_OF, coverage=COVERAGE).outcome


DAY_LEVEL = [
    "What was revenue on 3 March 2026?",
    "What was revenue on March 3, 2026?",
    "What was revenue on March 3?",
    "What was revenue on March 3rd?",
    "What was revenue on the 3rd of March 2026?",
    "What was revenue on 2026-03-03?",
    "What was revenue on 03/04/2026?",  # ambiguous order (3 April or 4 March): still one day
    "How many support tickets were opened on 15 June 2026?",
    "What was revenue on May 5?",
]
MONTH_LEVEL = [
    ("What was revenue in March 2026?", "2026-03"),
    ("What was revenue in 2026-03?", "2026-03"),
    ("What was revenue during March?", "2026-03"),
    ("What was revenue in March?", "2026-03"),
    ("What was revenue in May 2026?", "2026-05"),
    ("Which 3 regions had the highest revenue in March 2026?", "2026-03"),
]


@pytest.mark.parametrize("question", DAY_LEVEL)
def test_a_single_day_asks_for_clarification(question: str) -> None:
    u = understood(question)
    assert u["material_ambiguity"], question
    assert any("single day" in a and "months, quarters or years" in a for a in u["ambiguities"])
    assert outcome(question) == "clarify"


@pytest.mark.parametrize(("question", "period"), MONTH_LEVEL)
def test_months_stay_months(question: str, period: str) -> None:
    u = understood(question)
    assert u["period"] == period and not u["material_ambiguity"], (question, u["ambiguities"])
    assert outcome(question) == "valid"


@pytest.mark.parametrize(
    ("question", "period", "comparison"),
    [
        ("What was revenue in July compared with June?", "2026-07", "2026-06"),
        ("What was revenue in July 2026 compared with June 2026?", "2026-07", "2026-06"),
        ("Compare revenue from May to July", "2026-07", "2026-05"),
        ("What was revenue from 2026-07-01 to 2026-07-31?", "2026-07", None),
    ],
)
def test_phase71_comparison_periods_are_unchanged(question: str, period: str, comparison: str | None) -> None:
    u = understood(question)
    assert (u["period"], u["comparison_period"]) == (period, comparison)
    assert not u["material_ambiguity"]


@pytest.mark.parametrize(
    "question",
    [
        "Will the top 3 may accounts renew in July?",  # "may" as a verb next to a number
        "What was revenue for the last 3 months?",
        "How did revenue in March compare with 3 months earlier?",
        "Forecast revenue for the next 3 months",
    ],
)
def test_numbers_near_months_that_are_not_days(question: str) -> None:
    assert not any("single day" in a for a in understood(question)["ambiguities"]), question


@pytest.mark.slow
def test_the_agent_never_substitutes_the_month_for_a_day(full_db: Any) -> None:
    from app.agent.runner import AgentRunner

    agent = AgentRunner(full_db, as_of=AS_OF)
    day = agent.run("What was revenue on 3 March 2026?")
    month = agent.run("What was revenue in March 2026?")
    assert month.status == "completed" and month.evidence
    assert day.status == "insufficient_evidence" and not day.evidence and not day.tool_trace
    assert "single day" in day.response.answer and "months" in day.response.answer
    assert ".." not in day.response.answer  # one full stop, not the message's and the wrapper's
    monthly_value = month.evidence[0].display_value
    assert monthly_value and monthly_value not in day.response.answer
