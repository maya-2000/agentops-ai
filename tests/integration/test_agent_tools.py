"""Every registered tool on the full dataset: the numbers are the Phase 2/3 numbers, with provenance.

Each tool is called through the registry, which is the only way the agent calls it, and its
result is compared with a direct call to the underlying Phase 2/3 function.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.analytics import cohorts, customers, revenue, risk
from app.analytics.common import pct_change
from app.analytics.executor import QueryRunner
from app.analytics.kpis import KPIService
from app.anomalies import AnomalyService
from app.database.base import Database
from app.evidence import EvidenceGraph
from app.evidence.builder import build_evidence
from app.forecasting import ForecastService
from app.tools import ToolContext, ToolRegistry, ToolRequest
from app.tools.results import KPIComparison, SQLResult
from tests.phase4_support import AS_OF

pytestmark = pytest.mark.slow
REGISTRY = ToolRegistry()
LAST_MONTH = {"start_date": "2026-08-01", "end_date": "2026-08-31"}
COMPARISON = {"comparison_start_date": "2026-07-01", "comparison_end_date": "2026-07-31"}


@pytest.fixture(scope="module")
def context(full_db: Database) -> ToolContext:
    return ToolContext(db=full_db, as_of=AS_OF, sql_row_limit=50)


def _call(context: ToolContext, tool: str, **arguments: Any) -> Any:
    return REGISTRY.execute(ToolRequest(call_id="T1", tool_name=tool, arguments=arguments, purpose="test"), context)


def _data(result: Any) -> list[dict[str, Any]]:
    return [row.model_dump(mode="json") for row in result.data]


# ---- every tool succeeds with provenance and yields evidence ---------------------------------------

TOOL_CALLS: list[tuple[str, dict[str, Any]]] = [
    ("get_kpi", {"kpi": "revenue", **LAST_MONTH}),
    ("analyze_revenue", {"operation": "decompose_revenue_change", "dimension": "region", **LAST_MONTH, **COMPARISON}),
    ("analyze_customers", {"operation": "churn_summary", **LAST_MONTH}),
    ("analyze_sales", {"operation": "sales_performance", "dimension": "segment", **LAST_MONTH}),
    ("analyze_marketing", {"operation": "channel_performance", **LAST_MONTH}),
    ("analyze_support", {"operation": "support_by_dimension", "dimension": "ticket_category", **LAST_MONTH}),
    ("analyze_product", {"operation": "feature_adoption", **LAST_MONTH}),
    ("get_cohort_analysis", {"max_months": 6}),
    ("get_customer_risk", {"min_band": "high", "limit": 10}),
    ("forecast_metric", {"metric": "mrr", "horizon": 2}),
    ("detect_anomalies", {"metric": "support_ticket_volume", "end_date": "2026-08-31"}),
    ("run_safe_sql", {"sql": "SELECT segment, COUNT(*) AS customers FROM customers GROUP BY segment ORDER BY 1"}),
]


@pytest.mark.parametrize(("tool", "arguments"), TOOL_CALLS, ids=[t for t, _ in TOOL_CALLS])
def test_tool_returns_typed_result_with_provenance(context: ToolContext, tool: str, arguments: dict[str, Any]) -> None:
    result = _call(context, tool, **arguments)
    assert result.success and result.error is None, result.message
    assert result.status == "ok" and result.result is not None and result.result_type
    assert result.query_ids and result.source_tables and result.calculation
    assert result.execution_time_ms > 0
    graph = EvidenceGraph()
    evidence = build_evidence(result, graph)
    assert evidence, f"{tool} produced no evidence"
    for item in evidence:
        assert item.tool_call_id == "T1" and item.tool_name == tool
        assert item.has_provenance and set(item.query_ids) <= set(result.query_ids)


def test_the_catalogue_is_covered() -> None:
    assert {t for t, _ in TOOL_CALLS} == set(REGISTRY.names)


# ---- numbers equal the direct Phase 2/3 calls -----------------------------------------------------


def test_get_kpi_matches_the_kpi_service(context: ToolContext, full_db: Database) -> None:
    direct = KPIService(full_db, as_of=AS_OF).calculate_kpi("revenue", {"period": "last_month"})
    tool = _call(context, "get_kpi", kpi="revenue", period="last_month").result
    assert tool.value == direct.value and tool.components == direct.components


def test_get_kpi_comparison_uses_phase2_arithmetic(context: ToolContext, full_db: Database) -> None:
    service = KPIService(full_db, as_of=AS_OF)
    current = service.calculate_kpi("mrr", {"period": "last_month"})
    previous = service.calculate_kpi("mrr", {"period": "previous_month"})
    tool = _call(context, "get_kpi", kpi="mrr", period="last_month", comparison_period="previous_month")
    result = tool.result
    assert isinstance(result, KPIComparison)
    assert result.current.value == current.value and result.comparison.value == previous.value
    assert result.absolute_change == pytest.approx(current.value - previous.value)  # type: ignore[operator]
    assert result.percentage_change == pct_change(current.value, previous.value)
    assert result.direction == ("decline" if current.value < previous.value else "increase")  # type: ignore[operator]
    assert len(tool.query_ids) == len(current.query_ids) + len(previous.query_ids)


def test_revenue_decomposition_matches(context: ToolContext, full_db: Database) -> None:
    direct = revenue.decompose_revenue_change(full_db, "segment", "last_month", "previous_month", as_of=AS_OF)
    tool = _call(
        context,
        "analyze_revenue",
        operation="decompose_revenue_change",
        dimension="segment",
        period="last_month",
        comparison_period="previous_month",
    )
    assert _data(tool.result) == _data(direct) and tool.result.summary == direct.summary


def test_churn_by_dimension_matches(context: ToolContext, full_db: Database) -> None:
    direct = customers.churn_by_dimension(full_db, "region", "last_month", as_of=AS_OF)
    tool = _call(context, "analyze_customers", operation="churn_by_dimension", dimension="region", period="last_month")
    assert _data(tool.result) == _data(direct)


def test_cohorts_and_risk_match(context: ToolContext, full_db: Database) -> None:
    assert _data(_call(context, "get_cohort_analysis", max_months=6).result) == _data(
        cohorts.cohort_retention(full_db, max_months=6, as_of=AS_OF)
    )
    direct = risk.score_customer_risk(full_db, as_of=AS_OF, min_band="high", limit=10)
    assert _data(_call(context, "get_customer_risk", min_band="high", limit=10).result) == _data(direct)


def test_forecast_matches_the_forecast_service(context: ToolContext, full_db: Database) -> None:
    direct = ForecastService(full_db, as_of=AS_OF).forecast(metric="revenue", horizon=3)
    tool = _call(context, "forecast_metric", metric="revenue", horizon=3).result
    assert [p.model_dump() for p in tool.forecast_points] == [p.model_dump() for p in direct.forecast_points]
    assert tool.model == direct.model and tool.cutoff_date == direct.cutoff_date


def test_anomalies_match_the_anomaly_service(context: ToolContext, full_db: Database) -> None:
    direct = AnomalyService(full_db, as_of=AS_OF).detect("revenue", end_date=AS_OF, detector="forecast_residual")
    tool = _call(context, "detect_anomalies", metric="revenue", end_date="2026-08-31", detector="forecast_residual")
    assert [(r.period, r.score, r.is_anomaly) for r in tool.result.results] == [
        (r.period, r.score, r.is_anomaly) for r in direct.results
    ]


def test_sql_tool_matches_a_direct_query(context: ToolContext, full_db: Database) -> None:
    sql = "SELECT COUNT(*) AS n FROM customers WHERE segment = $segment"
    direct = QueryRunner(full_db, "direct").run(sql, {"segment": "SMB"})
    tool = _call(context, "run_safe_sql", sql=sql, parameters={"segment": "SMB"})
    assert isinstance(tool.result, SQLResult) and tool.result.rows == [[direct.rows[0][0]]]
    assert tool.result.parameters == {"segment": "SMB"} and not tool.result.truncated


# ---- typed errors ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "arguments", "code"),
    [
        ("get_kpi", {"kpi": "stock_price"}, "unsupported_kpi"),
        ("get_kpi", {"kpi": "revenue", "filters": {"segment": "Galactic"}}, "invalid_filter_value"),
        ("get_kpi", {"kpi": "revenue", "filters": {"planet": "Mars"}}, "unsupported_dimension"),
        (
            "analyze_revenue",
            {"operation": "decompose_revenue_change", "dimension": "star_sign"},
            "invalid_request",
        ),
        ("forecast_metric", {"metric": "win_rate", "horizon": 3}, "unsupported_metric"),
        ("forecast_metric", {"metric": "revenue", "horizon": 24}, "invalid_horizon"),
    ],
)
def test_invalid_requests_become_typed_errors(
    context: ToolContext, tool: str, arguments: dict[str, Any], code: str
) -> None:
    result = _call(context, tool, **arguments)
    assert not result.success and result.result is None
    assert result.error is not None and result.error.code == code and not result.error.retryable


# ---- SQL safety at execution time --------------------------------------------------------------


def test_sql_truncation_is_flagged(context: ToolContext) -> None:
    tool = _call(context, "run_safe_sql", sql="SELECT customer_id FROM customers ORDER BY customer_id", max_rows=5)
    result = tool.result
    assert result.truncated and result.row_count == 5 and len(result.rows) == 5
    assert any("Truncated" in note for note in tool.limitations)
    evidence = build_evidence(tool, EvidenceGraph())
    assert evidence and all(e.truncated for e in evidence)
    assert "TRUNCATED" in evidence[0].statement


def test_sql_row_limit_cannot_be_raised_by_the_caller(context: ToolContext) -> None:
    result = _call(context, "run_safe_sql", sql="SELECT customer_id FROM customers", max_rows=100_000).result
    assert result.max_rows == context.sql_row_limit and result.row_count == context.sql_row_limit


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM customers",
        "UPDATE customers SET segment = 'SMB'",
        "DROP TABLE customers",
        "CREATE TABLE x AS SELECT customer_id FROM customers",
        "COPY customers TO 'customers.csv'",
        "ATTACH 'other.duckdb' AS other",
        "SELECT * FROM read_json_auto('data/seeds/injected_events.json')",
        "SELECT sales_rep FROM sales_opportunities",
    ],
)
def test_unsafe_sql_is_refused_before_execution(context: ToolContext, full_db: Database, sql: str) -> None:
    before = QueryRunner(full_db, "count").run("SELECT COUNT(*) FROM customers").rows[0][0]
    result = _call(context, "run_safe_sql", sql=sql)
    assert not result.success and result.error is not None and result.error.code == "unsafe_sql"
    assert not result.query_ids
    assert QueryRunner(full_db, "count").run("SELECT COUNT(*) FROM customers").rows[0][0] == before


def test_the_connection_itself_is_read_only(full_db: Database) -> None:
    """Defence in depth: even a statement that bypassed validation could not write."""
    with pytest.raises(Exception, match=r"(?i)read.only"):
        QueryRunner(full_db, "write_probe").run("CREATE TABLE agent_probe (x INTEGER)")
