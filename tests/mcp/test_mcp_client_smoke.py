"""Protocol integration: a local MCP client discovers, inspects and calls the server, in process and over
stdio, plus the two end-to-end workflows (KPI and forecast). This is a protocol test, not an evaluation."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
from jsonschema import Draft202012Validator
from mcp import Client, StdioServerParameters

from app.config import PROJECT_ROOT
from app.tools.base import ToolContext, ToolRequest
from app.tools.registry import ToolRegistry
from tests.phase4_support import AS_OF
from tests.phase6_support import EXPECTED_TOOLS, server, structured, with_client

SMOKE_CALLS = {
    "agentops_get_kpi": {"kpi": "mrr", "period": "2026-08"},
    "agentops_analyze_revenue": {"operation": "revenue_change", "period": "2026-08", "comparison_period": "2026-07"},
    "agentops_forecast_metric": {"metric": "revenue", "horizon": 2},
    "agentops_detect_anomalies": {"metric": "revenue"},
}


def test_client_smoke_in_process(full_db: Any) -> None:
    async def fn(client: Client) -> dict[str, Any]:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        results = {name: await client.call_tool(name, args) for name, args in SMOKE_CALLS.items()}
        invalid = [
            await client.call_tool("agentops_get_kpi", {"kpi": "not_a_kpi"}),
            await client.call_tool("agentops_forecast_metric", {"metric": "revenue", "horizon": 60}),
            await client.call_tool("agentops_run_safe_sql", {"sql": "DROP TABLE customers"}),
            await client.call_tool("read_file", {"path": "data/seeds/injected_events.json"}),
        ]
        return {"tools": tools, "results": results, "invalid": invalid}

    got = with_client(server(full_db), fn)
    # discovery and schema inspection
    assert list(got["tools"]) == list(EXPECTED_TOOLS)
    for name, arguments in SMOKE_CALLS.items():
        schema = got["tools"][name].input_schema
        Draft202012Validator.check_schema(schema)
        assert not list(Draft202012Validator(schema).iter_errors(arguments)), "the smoke inputs satisfy the schema"
    # structured responses (the client has already validated them against each tool's output schema)
    for name, result in got["results"].items():
        payload = structured(result)
        assert not result.is_error and payload["status"] == "ok" and payload["tool_name"] == name
        assert payload["evidence"] and payload["provenance"]["query_ids"] and payload["query_id"]
        assert payload["provenance"]["source_tables"] and payload["provenance"]["calculation"]
    # invalid requests fail safely, as tool errors rather than protocol failures
    categories = [structured(r)["error"]["category"] for r in got["invalid"]]
    assert all(r.is_error for r in got["invalid"])
    assert categories == ["UNSUPPORTED_REQUEST", "INVALID_ARGUMENT", "UNSAFE_QUERY", "UNAUTHORIZED_TOOL"]


def test_client_smoke_over_stdio(small_dataset: Any) -> None:
    """The real transport: ``python -m app.mcp`` as a subprocess, driven by the SDK's stdio client."""
    db_path = Path(small_dataset.config.db_path)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp", "--log-level", "WARNING"],
        cwd=str(PROJECT_ROOT),
        env={"DATABASE_URL": f"duckdb:///{db_path}", "PYTHONPATH": str(PROJECT_ROOT)},
    )

    async def main() -> tuple[list[str], Any, Any]:
        async with Client(params, read_timeout_seconds=60) as client:
            names = [t.name for t in (await client.list_tools()).tools]
            ok = await client.call_tool("agentops_get_kpi", {"kpi": "customer_count", "period": "2026-08"})
            bad = await client.call_tool("agentops_run_safe_sql", {"sql": "SELECT * FROM read_csv('/etc/passwd')"})
            return names, ok, bad

    names, ok, bad = anyio.run(main)
    assert names == list(EXPECTED_TOOLS)
    assert not ok.is_error and structured(ok)["evidence"][0]["metric"] == "customer_count"
    assert bad.is_error and structured(bad)["error"]["category"] == "UNSAFE_QUERY"


def test_e2e_kpi_workflow(full_db: Any, caplog: pytest.LogCaptureFixture) -> None:
    """Client -> discover -> get_kpi -> result -> evidence -> provenance -> audit trace."""
    arguments = {"kpi": "revenue", "start_date": "2026-08-01", "end_date": "2026-08-31"}

    async def fn(client: Client) -> Any:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert "kpi" in tools["agentops_get_kpi"].input_schema["required"]
        return await client.call_tool("agentops_get_kpi", arguments)

    with caplog.at_level(logging.INFO):
        payload = structured(with_client(server(full_db), fn))

    # result: exactly the deterministic Phase 2 KPI, via the Phase 4 tool
    context = ToolContext(db=full_db, as_of=AS_OF, sql_row_limit=200)
    direct = ToolRegistry().execute(ToolRequest(call_id="T1", tool_name="get_kpi", arguments=arguments), context)
    assert direct.result is not None
    value = payload["result"]["value"]
    assert payload["status"] == "ok" and value == direct.result.model_dump()["value"]
    assert payload["result"]["key"] == "revenue" and payload["result"]["unit"]

    # evidence: the same number, with its period, tool and query
    [evidence] = payload["evidence"]
    assert evidence["value"] == value and evidence["evidence_type"] == "observed"
    assert (evidence["period_start"], evidence["period_end"]) == ("2026-08-01", "2026-08-31")
    assert evidence["tool_name"] == "get_kpi" and evidence["query_ids"] == payload["provenance"]["query_ids"]

    # provenance
    provenance = payload["provenance"]
    assert provenance["source_tables"] and "daily_revenue" in provenance["source_tables"]
    assert provenance["arguments"] == arguments and provenance["source_layer"] == "phase2_kpi"
    assert provenance["dataset_version"] == full_db.dataset_version

    # audit trace: the same run ID across the MCP audit record and the Phase 5 security events
    run_id = payload["request_id"]
    audit = [json.loads(r.message) for r in caplog.records if r.name == "agentops.mcp"]
    call = next(a for a in audit if a.get("event") == "tool_call")
    assert call["request_id"] == run_id and call["evidence_ids"] == ["E1"]
    assert call["query_ids"] == provenance["query_ids"] and call["authorization"] == "allowed"
    security = [json.loads(r.message) for r in caplog.records if r.name == "agentops.security"]
    assert [s["event_type"] for s in security if s["run_id"] == run_id] == ["tool_authorized"]
    events = [a["event"] for a in audit]
    assert events == ["server_started", "tool_call", "server_stopped"]


def test_e2e_forecast_workflow(full_db: Any) -> None:
    """Client -> forecast_metric -> forecast metadata -> evidence -> provenance."""

    async def fn(client: Client) -> Any:
        return await client.call_tool("agentops_forecast_metric", {"metric": "mrr", "horizon": 3})

    payload = structured(with_client(server(full_db), fn))
    assert payload["status"] == "ok" and payload["output_kind"] == "forecast"

    forecast = payload["forecast"]
    assert forecast["metric"] == "mrr" and forecast["horizon"] == 3 and forecast["cutoff_date"] == "2026-08-31"
    assert forecast["model"] and forecast["interval_method"] and forecast["confidence_level"] > 0.5
    assert [p["period"] for p in forecast["points"]] == ["2026-09", "2026-10", "2026-11"]
    assert forecast["backtest"]["mae"] is not None and forecast["backtest"]["fold_count"] > 0
    assert any("model predictions" in w for w in payload["warnings"])

    evidence = payload["evidence"]
    assert [e["evidence_type"] for e in evidence] == ["forecast"] * 3
    for item, point in zip(evidence, forecast["points"], strict=True):
        assert item["value"] == point["predicted_value"] and item["period_label"] == point["period"]
        assert item["details"]["model"] == forecast["model"] and item["details"]["cutoff_date"] == "2026-08-31"
        assert item["attributes"]["lower_bound"] == point["lower_bound"]
        assert "Forecast" in item["statement"]

    provenance = payload["provenance"]
    assert provenance["tool"] == "forecast_metric" and provenance["source_layer"] == "phase3_forecasting"
    assert provenance["query_ids"] and provenance["calculation"] and provenance["source_tables"]
