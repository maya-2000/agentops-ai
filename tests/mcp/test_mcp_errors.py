"""The MCP error model: every category, the fixed client-safe messages and error sanitisation."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any

import mcp.types as types
import pytest

import app.mcp.adapters as adapters
from app.agent.config import AgentConfig
from app.analytics.errors import AnalyticsDatabaseError, QueryTimeoutAnalyticsError
from app.mcp.errors import MESSAGES, UNSUPPORTED_KPI_MESSAGE, mcp_category, mcp_error
from app.security.errors import _CODE_CATEGORIES
from app.tools import handlers
from app.tools.base import ToolContext
from tests.phase5_support import raising, with_handler
from tests.phase6_support import call, mcp_config, server, service, structured

SECRET = "sk-ant-api03-SECRETSECRETSECRET1234567890"
LEAKS = (SECRET, "/home/", "/usr/lib", "Traceback", 'File "', "duckdb.", "IOException", "RuntimeError", "line ")
KPI = {"kpi": "revenue", "period": "2026-08"}


def assert_safe(payload: dict[str, Any]) -> None:
    text = json.dumps(payload)
    for leak in LEAKS:
        assert leak not in text, f"leaked {leak!r}"


def failing(db: Any, handler: Any, config: Any = None, tool: str = "get_kpi") -> Any:
    return service(db, config, registry=with_handler(tool, handler)).call(f"agentops_{tool}", KPI)


def test_the_specified_messages() -> None:
    assert UNSUPPORTED_KPI_MESSAGE == "The requested KPI is not supported."
    assert MESSAGES["UNSAFE_QUERY"] == "The requested operation was rejected by the data-access policy."
    assert MESSAGES["UNAUTHORIZED_TOOL"] == "The requested operation was rejected by the data-access policy."
    assert MESSAGES["TOOL_FAILURE"] == "The analysis could not be completed because the analytics tool failed."
    assert MESSAGES["RESOURCE_LIMIT"] == "The analysis exceeded the configured execution limit."
    assert MESSAGES["TIMEOUT"] == "The analysis exceeded the configured execution limit."
    assert MESSAGES["INTERNAL_ERROR"] == "The analysis could not be completed."
    assert MESSAGES["VALIDATION_FAILURE"] == "The analysis could not be completed."


def test_every_phase5_code_maps_to_an_mcp_category_and_unknown_codes_fail_closed() -> None:
    for code in _CODE_CATEGORIES:
        assert mcp_category(code) in MESSAGES
    assert mcp_category("something_new") == "INTERNAL_ERROR"
    assert mcp_category(None) == "INTERNAL_ERROR"
    assert mcp_error("unsafe_sql", detail="Table secret_table").detail is None, "no detail for security rejections"


def test_invalid_argument(small_db: Any) -> None:
    out = service(small_db).call("agentops_get_kpi", {"kpi": "revenue", "start_date": "2026-02-30", "end_date": "x"})
    error = out.output.error
    assert error is not None and error.category == "INVALID_ARGUMENT" and error.detail
    assert error.message == MESSAGES["INVALID_ARGUMENT"]


def test_unsupported_kpi_and_metric(small_db: Any) -> None:
    kpi = service(small_db).call("agentops_get_kpi", {"kpi": "happiness"}).output.error
    assert kpi is not None and kpi.category == "UNSUPPORTED_REQUEST" and kpi.message == UNSUPPORTED_KPI_MESSAGE
    metric = service(small_db).call("agentops_forecast_metric", {"metric": "happiness"}).output.error
    assert metric is not None and metric.category == "UNSUPPORTED_REQUEST"


def test_unauthorized_tool(small_db: Any) -> None:
    error = service(small_db).call("agentops_delete_customers", {}).output.error
    assert error is not None and error.category == "UNAUTHORIZED_TOOL" and error.code == "unknown_tool"


def test_unsafe_query(small_db: Any) -> None:
    out = service(small_db).call("agentops_run_safe_sql", {"sql": "DROP TABLE customers"})
    assert out.output.error is not None and out.output.error.category == "UNSAFE_QUERY"
    assert out.output.error.detail is None, "the policy detail stays in the audit trail"


def test_resource_limit_for_budget_and_request_size(small_db: Any) -> None:
    budget = service(small_db, mcp_config(AgentConfig(max_sql_calls=0))).call(
        "agentops_run_safe_sql", {"sql": "SELECT COUNT(*) AS n FROM customers"}
    )
    assert budget.output.error is not None and budget.output.error.category == "RESOURCE_LIMIT"
    assert budget.output.error.code == "budget_exceeded"
    size = service(small_db, mcp_config(max_request_bytes=1024)).call("agentops_get_kpi", {"kpi": "x" * 2000})
    assert size.output.error is not None and size.output.error.category == "RESOURCE_LIMIT"
    assert size.output.error.code == "request_too_large"


def test_tool_failure_is_retried_then_sanitised(small_db: Any) -> None:
    detail = f"duckdb.IOException: could not open /home/user/agentops-ai/database/x.duckdb; api_key={SECRET}"
    outcome = failing(small_db, raising(AnalyticsDatabaseError(detail)))
    error = outcome.output.error
    assert error is not None and error.category == "TOOL_FAILURE" and error.retryable
    assert error.message == "The analysis could not be completed because the analytics tool failed."
    assert [e.event_type for e in outcome.events].count("retry") == AgentConfig().max_retries
    assert_safe(outcome.payload)


def test_timeout_from_the_database_deadline(small_db: Any) -> None:
    outcome = failing(small_db, raising(QueryTimeoutAnalyticsError("Query interrupted: the execution deadline passed")))
    assert outcome.output.error is not None and outcome.output.error.category == "TIMEOUT"
    assert "timeout" in [e.event_type for e in outcome.events]


def test_timeout_after_the_tool_deadline_discards_the_result(small_db: Any) -> None:
    def slow(ctx: ToolContext, parsed: Any) -> Any:
        time.sleep(0.3)
        return handlers.get_kpi(ctx, parsed)

    outcome = failing(small_db, slow, mcp_config(AgentConfig(tool_timeout_seconds=0.1)))
    assert outcome.output.error is not None and outcome.output.error.category == "TIMEOUT"
    assert outcome.output.result is None and outcome.output.evidence == []


def test_validation_failure_when_tool_output_breaks_the_contract(small_db: Any) -> None:
    def no_provenance(ctx: ToolContext, parsed: Any) -> Any:
        return replace(handlers.get_kpi(ctx, parsed), query_ids=[])

    outcome = failing(small_db, no_provenance)
    assert outcome.output.error is not None and outcome.output.error.category == "VALIDATION_FAILURE"
    assert outcome.output.result is None and outcome.output.evidence == []
    assert "tool_output_rejected" in [e.event_type for e in outcome.events]


def test_internal_error_hides_everything(small_db: Any) -> None:
    message = f'Traceback (most recent call last):\n  File "/home/user/agentops-ai/app/x.py", line 3\nKEY={SECRET}'
    outcome = failing(small_db, raising(RuntimeError(message)))
    error = outcome.output.error
    assert error is not None and error.category == "INTERNAL_ERROR"
    assert error.message == "The analysis could not be completed." and error.detail is None
    assert_safe(outcome.payload)


def test_adapter_failures_fail_closed(small_db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_: Any, **__: Any) -> Any:
        raise RuntimeError(f"/home/user/agentops-ai/app/evidence/builder.py exploded {SECRET}")

    monkeypatch.setattr(adapters, "build_evidence", broken)
    outcome = service(small_db).call("agentops_get_kpi", KPI)
    assert outcome.output.error is not None and outcome.output.error.category == "INTERNAL_ERROR"
    assert_safe(outcome.payload)


def test_oversized_responses_are_trimmed_with_metadata(full_db: Any) -> None:
    outcome = service(full_db, mcp_config(max_response_bytes=16384)).call(
        "agentops_get_cohort_analysis", {"max_months": 6}
    )
    out = outcome.output
    assert out.error is None and out.truncated and out.result is None
    assert out.provenance is not None and out.provenance.query_ids, "provenance is never dropped"
    assert out.evidence and any("Response size limit" in w for w in out.warnings)
    assert len(json.dumps(outcome.payload)) <= 16384
    assert "response_truncated" in [e.event_type for e in outcome.events]


def test_responses_that_cannot_fit_become_resource_limit_errors(small_db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    real = adapters._size
    monkeypatch.setattr(adapters, "_size", lambda payload: 10**9 if payload.get("status") != "error" else real(payload))
    error = service(small_db).call("agentops_get_kpi", KPI).output.error
    assert error is not None and error.category == "RESOURCE_LIMIT" and error.code == "response_too_large"


def test_errors_travel_as_mcp_tool_errors_not_protocol_failures(small_db: Any) -> None:
    registry = with_handler("get_kpi", raising(RuntimeError(f"boom {SECRET} /home/user/x")))
    result = call(server(small_db, registry=registry), "agentops_get_kpi", KPI)
    assert result.is_error
    payload = structured(result)
    assert payload["error"]["category"] == "INTERNAL_ERROR"
    block = result.content[0]
    assert isinstance(block, types.TextContent) and json.loads(block.text) == payload
    assert_safe(payload)
    assert SECRET not in block.text
