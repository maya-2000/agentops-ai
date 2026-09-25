"""MCP input and output schemas: the Phase 4 models are the input schemas; malformed input never reaches a tool."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from app.mcp.registry import MCP_TOOL_SPECS, MCPToolRegistry
from app.mcp.schemas import MCPToolOutput
from app.tools.base import ToolContext
from app.tools.registry import TOOL_DEFINITIONS, ToolRegistry
from tests.phase6_support import EXPECTED_TOOLS, VALID_CALLS, mcp_config, service

REGISTRY = MCPToolRegistry(ToolRegistry(), mcp_config().limits)
TOOLS = {t.name: t for t in REGISTRY.tools()}
OUTPUT_VALIDATOR = Draft202012Validator(MCPToolOutput.model_json_schema(mode="serialization"))


class Spy:
    """A tool registry whose handlers only record that business logic was reached."""

    def __init__(self) -> None:
        self.calls: list[str] = []

        def handler_for(name: str) -> Any:
            def handler(context: ToolContext, parsed: Any) -> Any:
                self.calls.append(name)
                raise RuntimeError("business logic reached")

            return handler

        self.registry = ToolRegistry(tuple(replace(d, handler=handler_for(d.name)) for d in TOOL_DEFINITIONS))


def rejected(db: Any, name: str, arguments: Any) -> dict[str, Any]:
    """Call through the adapter with spy handlers; assert the call was rejected before any handler ran."""
    spy = Spy()
    outcome = service(db, registry=spy.registry).call(name, arguments)
    assert spy.calls == [], f"{name} reached business logic with {arguments!r}"
    payload = outcome.payload
    assert payload["status"] == "error" and payload["error"] is not None
    assert OUTPUT_VALIDATOR.is_valid(payload), "error outputs conform to the output schema too"
    return payload["error"]


# ------------------------------------------------------------------------------------------ schema shape


@pytest.mark.parametrize("name", EXPECTED_TOOLS)
def test_input_schema_is_the_phase4_model_schema(name: str) -> None:
    spec = next(s for s in MCP_TOOL_SPECS if s.name == name)
    definition = next(d for d in TOOL_DEFINITIONS if d.name == spec.tool)
    schema = TOOLS[name].input_schema
    assert schema == definition.input_model.model_json_schema()
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("name", EXPECTED_TOOLS)
def test_output_schema_is_the_typed_envelope(name: str) -> None:
    schema = TOOLS[name].output_schema
    assert schema is not None
    Draft202012Validator.check_schema(schema)
    for field in ("status", "result", "evidence", "provenance", "warnings", "limitations", "query_id", "tool_name"):
        assert field in schema["properties"], field
    assert set(schema["required"]) >= {"tool_name", "request_id", "status", "output_kind"}


def test_required_and_optional_fields_match_the_documented_interfaces() -> None:
    required = {name: set(TOOLS[name].input_schema.get("required", [])) for name in EXPECTED_TOOLS}
    assert required["agentops_get_kpi"] == {"kpi"}
    assert required["agentops_forecast_metric"] == {"metric"}
    assert required["agentops_detect_anomalies"] == {"metric"}
    assert required["agentops_run_safe_sql"] == {"sql"}
    assert required["agentops_get_customer_risk"] == set() and required["agentops_get_cohort_analysis"] == set()
    for name in EXPECTED_TOOLS[1:7]:
        assert required[name] == {"operation"}, name
    optional = set(TOOLS["agentops_forecast_metric"].input_schema["properties"]) - required["agentops_forecast_metric"]
    assert optional >= {"horizon", "cutoff_date", "filters"}
    assert set(TOOLS["agentops_detect_anomalies"].input_schema["properties"]) >= {"start_date", "end_date", "detector"}


def test_enum_constraints_are_in_the_schema() -> None:
    revenue_ops = TOOLS["agentops_analyze_revenue"].input_schema["properties"]["operation"]
    assert "revenue_change" in revenue_ops["enum"] and "mrr_series" in revenue_ops["enum"]
    detector = TOOLS["agentops_detect_anomalies"].input_schema["properties"]["detector"]
    assert detector["enum"] == ["rolling_zscore", "iqr", "forecast_residual"]
    band = TOOLS["agentops_get_customer_risk"].input_schema["properties"]["min_band"]
    assert band["enum"] == ["low", "medium", "high"]


# ------------------------------------------------------------------------------------------ rejection


@pytest.mark.parametrize("name", [n for n in EXPECTED_TOOLS if TOOLS[n].input_schema.get("required")])
def test_missing_required_fields_are_rejected_before_business_logic(small_db: Any, name: str) -> None:
    required = TOOLS[name].input_schema["required"]
    arguments = {k: v for k, v in VALID_CALLS[name].items() if k not in required}
    assert rejected(small_db, name, arguments)["category"] == "INVALID_ARGUMENT"


@pytest.mark.parametrize("name", EXPECTED_TOOLS)
def test_unknown_fields_are_rejected_before_business_logic(small_db: Any, name: str) -> None:
    error = rejected(small_db, name, {**VALID_CALLS[name], "admin": True})
    assert error["category"] == "INVALID_ARGUMENT"
    assert "admin" in (error["detail"] or "")


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("agentops_analyze_revenue", {"operation": "drop_everything", "period": "2026-08"}),
        ("agentops_analyze_sales", {"operation": "rep_salaries"}),
        ("agentops_detect_anomalies", {"metric": "revenue", "detector": "magic"}),
        ("agentops_detect_anomalies", {"metric": "revenue", "transform": "cube"}),
        ("agentops_analyze_product", {"operation": "adoption_trend", "feature": "Reports", "grain": "week"}),
        ("agentops_get_customer_risk", {"min_band": "catastrophic"}),
    ],
)
def test_values_outside_an_enum_are_rejected(small_db: Any, name: str, arguments: dict[str, Any]) -> None:
    assert rejected(small_db, name, arguments)["category"] == "INVALID_ARGUMENT"


@pytest.mark.parametrize(
    "dates",
    [
        {"start_date": "2026-13-01", "end_date": "2026-13-31"},
        {"start_date": "yesterday", "end_date": "today"},
        {"start_date": "2026-08-31", "end_date": "2026-08-01"},
        {"start_date": "2026-08-01"},
        {"start_date": "1899-01-01", "end_date": "1899-01-31"},
        {"start_date": "2026-08-01", "end_date": "2026-08-31", "period": "2026-08"},
    ],
)
def test_invalid_dates_are_rejected(small_db: Any, dates: dict[str, Any]) -> None:
    assert rejected(small_db, "agentops_get_kpi", {"kpi": "revenue", **dates})["category"] == "INVALID_ARGUMENT"


@pytest.mark.parametrize(
    "filters",
    [
        {"segment": "Galactic"},
        {"favourite_colour": "blue"},
        {"region": "Atlantis"},
        {"segment": "SMB", "region": "APAC", "plan": "Pro", "country": "SG", "industry": "Retail", "channel": "x"},
    ],
)
def test_invalid_filters_are_rejected(small_db: Any, filters: dict[str, str]) -> None:
    error = rejected(small_db, "agentops_get_kpi", {"kpi": "revenue", "period": "2026-08", "filters": filters})
    assert error["category"] in ("INVALID_ARGUMENT", "UNSAFE_QUERY")


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("agentops_forecast_metric", {"metric": "revenue", "horizon": 0}),
        ("agentops_forecast_metric", {"metric": "revenue", "horizon": 7}),
        ("agentops_forecast_metric", {"metric": "revenue", "horizon": "three"}),
        ("agentops_forecast_metric", {"metric": "revenue", "confidence_level": 1.5}),
        ("agentops_get_kpi", {"kpi": "revenue", "period": "the_good_old_days"}),
        ("agentops_get_kpi", {"kpi": "revenue", "dimension": "favourite_colour"}),
        ("agentops_get_customer_risk", {"limit": -1}),
        ("agentops_run_safe_sql", {"sql": "SELECT 1", "max_rows": 0}),
    ],
)
def test_invalid_values_are_rejected(small_db: Any, name: str, arguments: dict[str, Any]) -> None:
    assert rejected(small_db, name, arguments)["category"] in ("INVALID_ARGUMENT", "UNSUPPORTED_REQUEST")


@pytest.mark.parametrize(
    "arguments",
    [
        {"kpi": 42},
        {"kpi": ["revenue"]},
        {"kpi": {"$ne": None}},
        {"kpi": "revenue", "filters": ["segment", "Enterprise"]},
        {"kpi": "revenue", "filters": {"segment": 5}},
        {"kpi": "revenue", "filters": {"segment": {"nested": "x"}}},
        {"kpi": None},
        {"kpi": "revenue", "period": 202608},
        ["kpi", "revenue"],
        "kpi=revenue",
    ],
)
def test_malformed_inputs_are_rejected(small_db: Any, arguments: Any) -> None:
    assert rejected(small_db, "agentops_get_kpi", arguments)["category"] == "INVALID_ARGUMENT"


def test_valid_inputs_do_reach_the_tool(small_db: Any) -> None:
    spy = Spy()
    outcome = service(small_db, registry=spy.registry).call("agentops_get_kpi", {"kpi": "revenue", "period": "2026-08"})
    assert spy.calls == ["get_kpi"]  # the spy then fails, which is reported safely
    assert outcome.output.error is not None and outcome.output.error.category == "INTERNAL_ERROR"


@pytest.mark.parametrize("name", EXPECTED_TOOLS)
def test_successful_outputs_conform_to_the_output_schema(full_db: Any, name: str) -> None:
    payload = service(full_db).call(name, VALID_CALLS[name]).payload
    assert payload["status"] == "ok", payload.get("error")
    errors = sorted(OUTPUT_VALIDATOR.iter_errors(payload), key=str)
    assert not errors, errors[0].message if errors else ""
    assert MCPToolOutput.model_validate(payload).tool_name == name
