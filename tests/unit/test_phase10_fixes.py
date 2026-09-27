"""Regressions for the fixes found while building Phase 10 (they affect /ask as well as investigations)."""

from __future__ import annotations

import pytest

from app.evidence.formatting import extract_numbers, format_number, format_value, number_is_supported
from app.llm.deterministic.understanding import understand
from tests.phase4_support import understanding_context


@pytest.mark.parametrize("value", [29470.0, 12345.678, -56285.78, 1000.0, 999.5, 3.14159, 0.5, 250_000_000.0])
def test_numbers_are_never_written_in_exponent_notation(value: float) -> None:
    text = format_number(value)
    assert "e+" not in text and "e-" not in text, text
    (parsed,) = extract_numbers(f"The value was {text}.")
    assert number_is_supported(parsed, [value]), (value, text)


def test_unitless_values_use_the_same_format() -> None:
    assert format_value(29470.0, None) == format_number(29470.0) == "29,470"


@pytest.mark.parametrize("question", ["Why is revenue down this quarter?", "What was churn in the current quarter?"])
def test_an_incomplete_current_quarter_asks_for_a_complete_period(question: str) -> None:
    result = understand(understanding_context(question))
    assert result["material_ambiguity"] is True
    assert any("last complete quarter" in a for a in result["ambiguities"])


@pytest.mark.parametrize(
    ("question", "period"),
    [
        ("Why did revenue change in the latest quarter?", "last_quarter"),
        ("What was revenue in the most recent quarter?", "last_quarter"),
        ("What was revenue in the latest month?", "last_month"),
        ("What was churn in the most recent month?", "last_month"),
    ],
)
def test_latest_and_most_recent_periods_mean_the_last_complete_one(question: str, period: str) -> None:
    result = understand(understanding_context(question))
    assert result["period"] == period and not result["material_ambiguity"]
