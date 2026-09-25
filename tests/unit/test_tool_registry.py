"""The allow-listed tool registry: catalogue, argument validation and typed error handling (no database)."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from typing import Any

import pytest
from pydantic import BaseModel

from app.analytics.errors import AnalyticsDatabaseError, InvalidFilterValueError, InvalidRequestError
from app.tools import TOOL_DEFINITIONS, ToolContext, ToolRegistry, ToolRequest
from app.tools.base import ToolOutput
from app.tools.sql_safety import UnsafeSQLError

EXPECTED_TOOLS = {
    "get_kpi",
    "analyze_revenue",
    "analyze_customers",
    "analyze_sales",
    "analyze_marketing",
    "analyze_support",
    "analyze_product",
    "get_cohort_analysis",
    "get_customer_risk",
    "forecast_metric",
    "detect_anomalies",
    "run_safe_sql",
}
REGISTRY = ToolRegistry()


def test_exactly_the_twelve_tools_are_registered() -> None:
    assert set(REGISTRY.names) == EXPECTED_TOOLS and len(REGISTRY.names) == 12
    assert REGISTRY.get("send_email") is None


@pytest.mark.parametrize("definition", TOOL_DEFINITIONS, ids=lambda d: d.name)
def test_every_tool_is_documented_and_strict(definition: Any) -> None:
    for text in (
        definition.description,
        definition.when_to_use,
        definition.not_for,
        definition.output_description,
        definition.limitations,
    ):
        assert len(text) > 20
    assert definition.deterministic
    assert definition.input_schema["additionalProperties"] is False
    entry = definition.catalog_entry()
    assert entry["tool_name"] == definition.name
    assert entry["input_schema"]["properties"] == definition.input_schema["properties"]
    json.dumps(entry)  # the catalogue goes into prompts, so it must be plain JSON


def test_catalogue_contains_no_ground_truth_or_generator_detail() -> None:
    text = json.dumps(REGISTRY.catalog()).lower()
    for forbidden in ("injected", "ground truth", "ground_truth", "health_score", "hidden", "calibration"):
        assert forbidden not in text


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("get_kpi", {"kpi": "revenue", "period": "last_month"}),
        (
            "get_kpi",
            {"kpi": "mrr", "start_date": "2026-08-01", "end_date": "2026-08-31", "comparison_period": "2026-07"},
        ),
        ("analyze_revenue", {"operation": "decompose_revenue_change", "dimension": "segment", "period": "2026-08"}),
        ("analyze_customers", {"operation": "churn_by_dimension", "dimension": "segment", "min_customers": 10}),
        ("analyze_sales", {"operation": "pipeline_summary"}),
        ("analyze_marketing", {"operation": "campaign_performance", "channel": "Paid Search"}),
        ("analyze_support", {"operation": "support_volume_change", "period": "last_month"}),
        ("analyze_product", {"operation": "adoption_trend", "feature": "Dashboards"}),
        ("get_cohort_analysis", {"max_months": 6}),
        ("get_customer_risk", {"min_band": "high", "limit": 5}),
        ("forecast_metric", {"metric": "revenue", "horizon": 3}),
        ("detect_anomalies", {"metric": "mrr", "detector": "forecast_residual", "end_date": "2026-08-31"}),
        ("run_safe_sql", {"sql": "SELECT segment FROM customers", "max_rows": 5}),
    ],
)
def test_valid_arguments_are_canonicalised(tool: str, arguments: dict[str, Any]) -> None:
    canonical = REGISTRY.validate_arguments(tool, arguments)
    assert set(canonical) == set(arguments)  # defaults are not added; nothing is dropped


@pytest.mark.parametrize(
    ("tool", "arguments", "message"),
    [
        ("launch_rocket", {}, "Unknown tool"),
        ("get_kpi", {"kpi": "revenue", "sql": "DROP TABLE customers"}, "Extra inputs"),
        ("get_kpi", {"period": "last_month"}, "kpi"),
        ("get_kpi", {"kpi": "revenue", "start_date": "2026-08-01"}, "together"),
        (
            "get_kpi",
            {"kpi": "revenue", "period": "last_month", "start_date": "2026-08-01", "end_date": "2026-08-31"},
            "either",
        ),
        ("analyze_revenue", {"operation": "delete_everything"}, "operation"),
        ("analyze_revenue", {"operation": "decompose_revenue_change"}, "requires: dimension"),
        ("analyze_revenue", {"operation": "revenue_bridge", "dimension": "segment"}, "does not accept: dimension"),
        ("analyze_customers", {"operation": "churn_summary", "min_customers": 0}, "min_customers"),
        ("analyze_product", {"operation": "adoption_trend"}, "requires: feature"),
        ("get_customer_risk", {"limit": 1000}, "limit"),
        ("get_customer_risk", {"min_band": "critical"}, "min_band"),
        ("detect_anomalies", {"metric": "revenue", "detector": "magic"}, "detector"),
        ("forecast_metric", {"metric": "revenue", "confidence_level": 0.2}, "confidence_level"),
    ],
)
def test_invalid_arguments_are_rejected(tool: str, arguments: dict[str, Any], message: str) -> None:
    with pytest.raises(InvalidRequestError, match=message):
        REGISTRY.validate_arguments(tool, arguments)


# ---- execution boundary --------------------------------------------------------------------------


class _Result(BaseModel):
    value: float


class _NoDatabase:
    """Stands in for the Database: these tests never reach a query."""


def _context() -> ToolContext:
    return ToolContext(db=_NoDatabase(), as_of=date(2026, 8, 31), sql_row_limit=10)  # type: ignore[arg-type]


def _registry(handler: Any) -> ToolRegistry:
    base = next(d for d in TOOL_DEFINITIONS if d.name == "get_kpi")
    return ToolRegistry((replace(base, handler=handler),))


def _run(registry: ToolRegistry, arguments: dict[str, Any] | None = None, tool: str = "get_kpi") -> Any:
    request = ToolRequest(call_id="T1", tool_name=tool, arguments=arguments or {"kpi": "revenue"}, purpose="test")
    return registry.execute(request, _context())


def test_successful_call_carries_result_and_provenance() -> None:
    def handler(_: ToolContext, args: Any) -> ToolOutput:
        assert args.kpi == "revenue"
        return ToolOutput(
            result=_Result(value=1.0),
            status="ok",
            source_tables=["daily_revenue"],
            query_ids=["Q1"],
            calculation="SUM(revenue)",
        )

    result = _run(_registry(handler))
    assert result.success and result.status == "ok" and result.result_type == "_Result"
    assert result.query_ids == ["Q1"] and result.source_tables == ["daily_revenue"] and result.query_id == "Q1"
    assert result.execution_time_ms >= 0 and result.finished_at >= result.started_at
    assert result.arguments == {"kpi": "revenue"}


def _raise(exc: Exception) -> Any:
    def handler(_: ToolContext, __: Any) -> ToolOutput:
        raise exc

    return handler


@pytest.mark.parametrize(
    ("exc", "code", "retryable"),
    [
        (AnalyticsDatabaseError("connection lost"), "database_error", True),
        (InvalidFilterValueError("segment 'Galactic' does not exist"), "invalid_filter_value", False),
        (UnsafeSQLError("Only SELECT"), "unsafe_sql", False),
        (RuntimeError("unexpected"), "internal_error", False),
    ],
)
def test_failures_become_typed_errors_not_fabricated_results(exc: Exception, code: str, retryable: bool) -> None:
    result = _run(_registry(_raise(exc)), {"kpi": "revenue"})
    assert not result.success and result.status == "error" and result.result is None
    assert result.error is not None and result.error.code == code and result.error.retryable is retryable
    assert not result.query_ids


def test_unknown_tool_and_invalid_arguments_never_reach_a_handler() -> None:
    registry = _registry(_raise(AssertionError("handler must not run")))
    unknown = _run(registry, tool="drop_database")
    assert unknown.error is not None and unknown.error.code == "unknown_tool"
    invalid = _run(registry, {"kpi": "revenue", "unexpected": 1})
    assert invalid.error is not None and invalid.error.code == "invalid_arguments"
    inconsistent = _run(registry, {"kpi": "revenue", "start_date": "2026-08-01"})
    assert inconsistent.error is not None and inconsistent.error.code == "invalid_arguments"


def test_tool_result_serialises() -> None:
    def handler(_: ToolContext, __: Any) -> ToolOutput:
        return ToolOutput(result=_Result(value=2.5), status="ok", source_tables=["t"], query_ids=["Q"], calculation="c")

    payload = json.loads(_run(_registry(handler)).model_dump_json())
    assert payload["result"] == {"value": 2.5} and payload["tool_name"] == "get_kpi"
