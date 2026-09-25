"""Deterministic request validation: the understood question is checked against the catalogue and data coverage."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from app.agent.request import validate_understanding
from app.llm.schemas import UnderstandingOutput
from tests.phase4_support import AS_OF

COVERAGE = (date(2024, 9, 1), date(2026, 8, 31))


def _validate(**fields: Any) -> Any:
    return validate_understanding(UnderstandingOutput.model_validate(fields), as_of=AS_OF, coverage=COVERAGE)


def test_last_month_is_august_2026() -> None:
    result = _validate(intent="kpi_lookup", metric="revenue", period="last_month")
    assert result.outcome == "valid" and result.request is not None
    assert (result.request.period.start, result.request.period.end) == (date(2026, 8, 1), date(2026, 8, 31))
    assert result.request.metric_name and not result.request.assumptions


def test_defaults_are_applied_and_recorded() -> None:
    result = _validate(intent="period_comparison", metric="mrr")
    request = result.request
    assert request is not None
    assert request.period.start == date(2026, 8, 1) and request.comparison_period.start == date(2026, 7, 1)
    assert any("latest complete month" in a for a in request.assumptions)
    assert any("previous period" in a for a in request.assumptions)
    forecast = _validate(intent="forecast", metric="revenue").request
    assert forecast is not None and forecast.horizon == 1 and forecast.period is None
    assert any("next month" in a for a in forecast.assumptions)


def test_explicit_comparison_is_kept() -> None:
    request = _validate(
        intent="period_comparison", metric="revenue", period="2026-Q2", comparison_period="2026-Q1"
    ).request
    assert request is not None
    assert request.comparison_period.start == date(2026, 1, 1) and request.comparison_period.end == date(2026, 3, 31)


def test_causal_questions_are_flagged() -> None:
    request = _validate(intent="customer_investigation", metric="logo_churn_rate", analysis_type="causal").request
    assert request is not None and request.causal_question


@pytest.mark.parametrize(
    ("fields", "outcome", "message"),
    [
        ({"intent": "unsupported", "unsupported_reason": "Stock prices are out of scope."}, "unsupported", "Stock"),
        ({"intent": "kpi_lookup", "metric": "stock_price"}, "unsupported", "not available"),
        ({"intent": "kpi_lookup", "metric": "revenue", "dimensions": ["star_sign"]}, "unsupported", "dimension"),
        ({"intent": "kpi_lookup", "metric": "revenue", "filters": [{"dimension": "planet", "value": "Mars"}]},
         "unsupported", "filter"),
        ({"intent": "kpi_lookup", "metric": "revenue", "filters": [{"dimension": "segment", "value": "Galactic"}]},
         "insufficient", "No data exists"),
        ({"intent": "kpi_lookup"}, "clarify", "which metric"),
        ({"intent": "kpi_lookup", "metric": "revenue", "material_ambiguity": True, "ambiguities": ["plan or segment"]},
         "clarify", "plan or segment"),
        ({"intent": "kpi_lookup", "metric": "revenue", "period": "next_decade"}, "clarify", "period"),
        ({"intent": "kpi_lookup", "metric": "revenue", "period": "2030"}, "insufficient", "after the latest"),
        ({"intent": "kpi_lookup", "metric": "revenue", "period": "2023-03"}, "insufficient", "before the data begins"),
        ({"intent": "period_comparison", "metric": "revenue", "period": "2026-08", "comparison_period": "2023-08"},
         "insufficient", "Comparison period"),
        ({"intent": "forecast", "metric": "revenue", "horizon": 16}, "insufficient", "horizon"),
        ({"intent": "forecast", "metric": "win_rate", "horizon": 3}, "insufficient", "no monthly series"),
        ({"intent": "anomaly_detection", "metric": "cac"}, "insufficient", "no monthly series"),
    ],
)  # fmt: skip
def test_invalid_requests_get_a_typed_outcome(fields: dict[str, Any], outcome: str, message: str) -> None:
    result = _validate(**fields)
    assert result.outcome == outcome and result.request is None
    assert message in (result.message or "")
