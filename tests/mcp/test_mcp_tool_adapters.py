"""The MCP adapters over the real tools: every tool works, results equal direct execution, evidence and
provenance survive, forecast and anomaly metadata are preserved, and customer-level data is policed."""

from __future__ import annotations

from typing import Any

import pytest

from app.evidence.models import Evidence
from app.mcp.registry import MCP_TOOL_SPECS
from app.security.data_policy import WITHHELD_MARKER, default_exposure_policy, names_hidden_state
from app.tools.base import ToolContext, ToolRequest
from app.tools.registry import ToolRegistry
from tests.phase4_support import AS_OF
from tests.phase6_support import EXPECTED_TOOLS, VALID_CALLS, service

VOLATILE = {"query_id", "query_ids", "operation_id", "tool_run_id", "execution_timestamp", "execution_time_ms", "sql"}
SPEC = {s.name: s for s in MCP_TOOL_SPECS}


@pytest.fixture(scope="module")
def outputs(full_db: Any) -> dict[str, dict[str, Any]]:
    svc = service(full_db)
    return {name: svc.call(name, args).payload for name, args in VALID_CALLS.items()}


def _stable(value: Any) -> Any:
    """Drop per-execution identifiers and timestamps so two executions can be compared."""
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [_stable(v) for v in value]
    return value


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in _keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _keys(v)}
    return set()


@pytest.mark.parametrize("name", EXPECTED_TOOLS)
def test_every_tool_works_through_mcp(outputs: dict[str, dict[str, Any]], name: str) -> None:
    out = outputs[name]
    assert out["status"] == "ok" and out["error"] is None, out.get("error")
    assert out["tool_name"] == name and out["request_id"].startswith("R-")
    assert out["result"] and out["result_type"] and out["query_id"]
    assert out["evidence"], "every successful call carries evidence"
    assert out["output_kind"] == SPEC[name].output_kind


@pytest.mark.parametrize("name", EXPECTED_TOOLS)
def test_results_equal_direct_tool_execution(full_db: Any, outputs: dict[str, dict[str, Any]], name: str) -> None:
    """No duplicate business logic: MCP returns exactly what the Phase 4 tool computes."""
    spec = SPEC[name]
    context = ToolContext(db=full_db, as_of=AS_OF, sql_row_limit=200)
    request = ToolRequest(call_id="T1", tool_name=spec.tool, arguments=VALID_CALLS[name])
    direct = ToolRegistry().execute(request, context)
    assert direct.success and direct.result is not None
    expected = _stable(direct.result.model_dump(mode="json"))
    actual = _stable(outputs[name]["result"])
    if name == "agentops_get_customer_risk":  # the only difference: masked company names
        for row in expected["data"]:
            row["company_name"] = WITHHELD_MARKER
    assert actual == expected


@pytest.mark.parametrize("name", EXPECTED_TOOLS)
def test_evidence_is_preserved_with_provenance(outputs: dict[str, dict[str, Any]], name: str) -> None:
    out = outputs[name]
    spec = SPEC[name]
    for i, item in enumerate(out["evidence"], start=1):
        evidence = Evidence.model_validate(item)
        assert evidence.evidence_id == f"E{i}"
        assert evidence.fingerprint == evidence.compute_fingerprint(), "evidence arrives untampered"
        assert evidence.tool_name == spec.tool and evidence.tool_call_id == "T1"
        assert evidence.statement and evidence.calculation
        assert set(evidence.query_ids) <= set(out["provenance"]["query_ids"])
        assert set(evidence.source_tables) <= set(out["provenance"]["source_tables"])
        assert evidence.input_arguments == out["provenance"]["arguments"]
        if evidence.evidence_type != "derived":
            assert evidence.has_provenance


@pytest.mark.parametrize("name", EXPECTED_TOOLS)
def test_provenance_is_preserved(full_db: Any, outputs: dict[str, dict[str, Any]], name: str) -> None:
    provenance = outputs[name]["provenance"]
    approved = default_exposure_policy().approved_relations
    assert provenance["tool"] == SPEC[name].tool and provenance["call_id"] == "T1"
    assert provenance["source_layer"].startswith("phase")
    assert provenance["query_ids"] and outputs[name]["query_id"] == provenance["query_ids"][-1]
    assert provenance["source_tables"] and set(provenance["source_tables"]) <= approved
    assert provenance["calculation"] and provenance["executed_at"]
    assert provenance["dataset_version"] == full_db.dataset_version
    assert provenance["as_of"] == AS_OF.isoformat() and provenance["toolset_version"] == SPEC[name].version
    for key, value in VALID_CALLS[name].items():
        assert provenance["arguments"][key] == value, "the canonical arguments that ran"


def test_internal_query_text_is_not_returned(outputs: dict[str, dict[str, Any]]) -> None:
    for name, out in outputs.items():
        keys = _keys(out["result"])
        if name == "agentops_run_safe_sql":
            assert out["result"]["sql"].upper().startswith("SELECT")  # the caller's own query, as validated
            keys -= {"sql"}
        assert "sql" not in keys, name
        queries = (out["result"].get("provenance") or {}).get("queries", [])
        assert all(q["query_id"] and q["lineage"]["source_tables"] for q in queries), name


def test_forecast_metadata_is_preserved_and_labelled(outputs: dict[str, dict[str, Any]]) -> None:
    out = outputs["agentops_forecast_metric"]
    forecast, result = out["forecast"], out["result"]
    assert forecast["kind"] == "forecast" and out["output_kind"] == "forecast"
    assert forecast["metric"] == "revenue" and forecast["horizon"] == 3
    assert forecast["cutoff_date"] == AS_OF.isoformat() and forecast["model"] == result["model"]
    assert forecast["forecast_period_start"] == "2026-09-01" and forecast["forecast_period_end"] == "2026-11-30"
    predicted = [p["predicted_value"] for p in forecast["points"]]
    assert predicted == [p["predicted_value"] for p in result["forecast_points"]]
    assert all(p["lower_bound"] <= p["predicted_value"] <= p["upper_bound"] for p in forecast["points"])
    assert forecast["interval_available"] and forecast["confidence_level"] == result["confidence_level"]
    assert forecast["backtest"]["mae"] is not None and forecast["backtest"]["model"] == result["model"]
    assert forecast["baseline"] is not None
    assert {e["evidence_type"] for e in out["evidence"]} == {"forecast"}
    assert any("model predictions" in w for w in out["warnings"])


def test_anomaly_metadata_is_preserved_without_reinterpretation(outputs: dict[str, dict[str, Any]]) -> None:
    out = outputs["agentops_detect_anomalies"]
    view, result = out["anomalies"], out["result"]
    assert view["kind"] == "anomaly" and out["output_kind"] == "anomaly"
    assert view["metric"] == "support_ticket_volume" and view["detector"] == result["detector"] == "rolling_zscore"
    assert view["scored_periods"] == len(result["results"])
    assert [a["period"] for a in view["flagged"]] == [a["period"] for a in result["anomalies"]]
    assert view["flagged"], "the dataset contains flagged support-volume months"
    for flagged, original in zip(view["flagged"], result["anomalies"], strict=True):
        for key in ("observed_value", "expected_value", "score", "detector", "severity", "direction", "threshold"):
            assert flagged[key] == original[key], key
    assert any(e["evidence_type"] == "anomaly" for e in out["evidence"])
    assert any("not business judgements" in w for w in out["warnings"])


def test_customer_risk_uses_observable_signals_only(outputs: dict[str, dict[str, Any]]) -> None:
    out = outputs["agentops_get_customer_risk"]
    rows = out["result"]["data"]
    assert 0 < len(rows) <= 5
    assert all(row["company_name"] == WITHHELD_MARKER for row in rows)
    assert not [k for k in _keys(out) if names_hidden_state(k)], "no hidden health or generator state"
    assert any("not churn probabilities" in w for w in out["warnings"])


def test_sql_results_report_rows_and_truncation(full_db: Any) -> None:
    out = service(full_db).call("agentops_run_safe_sql", {"sql": "SELECT customer_id FROM customers", "max_rows": 5})
    result = out.output.result
    assert result is not None and result["row_count"] == 5 and result["truncated"] is True
    assert any("truncated" in w for w in out.output.warnings)


def test_no_data_statuses_pass_through_without_error(small_db: Any) -> None:
    """Six months of history cannot support a forecast: a labelled status, not an error or a made-up number."""
    outcome = service(small_db).call("agentops_forecast_metric", {"metric": "revenue", "horizon": 3})
    out = outcome.output
    assert not outcome.is_error and out.error is None
    assert out.status == "insufficient_history"
    assert out.forecast is not None and out.forecast.status == "insufficient_history" and out.forecast.points == []
    assert any("insufficient_history" in w for w in out.warnings)
    assert [e.evidence_type for e in out.evidence] == ["derived"]
