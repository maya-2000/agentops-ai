"""SQL attacks against run_safe_sql: every statement that is not a bounded, read-only, allow-listed SELECT fails."""

from __future__ import annotations

import time
from datetime import date
from typing import Any

import pytest

from app.analytics.errors import QueryTimeoutAnalyticsError
from app.analytics.executor import QueryRunner
from app.database import metadata
from app.database.deadline import QueryTimeoutError, execution_deadline, remaining_seconds
from app.security.data_policy import APPROVED_TABLES, APPROVED_VIEWS, default_exposure_policy
from app.tools import ToolContext, ToolRegistry, ToolRequest
from app.tools.sql_safety import SQLComplexityLimits, UnsafeSQLError, validate_sql

LIMIT = 50


def _reject(sql: str, match: str, params: dict[str, Any] | None = None, **limits: Any) -> None:
    with pytest.raises(UnsafeSQLError, match=match):
        validate_sql(sql, params or {}, LIMIT, limits=SQLComplexityLimits(**limits))


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT segment, COUNT(*) AS n FROM customers GROUP BY segment",
        "WITH s AS (SELECT customer_id FROM subscriptions WHERE status = 'active') "
        "SELECT COUNT(*) AS n FROM s JOIN customers AS c ON c.customer_id = s.customer_id",
        "SELECT region, SUM(revenue) AS r FROM daily_revenue WHERE date >= $start GROUP BY region",
        "SELECT date_part('year', date) AS y, median(revenue) AS m FROM daily_revenue GROUP BY y",
        "SELECT month, revenue FROM v_monthly_revenue ORDER BY month DESC LIMIT 12",
    ],
)
def test_bounded_read_only_selects_are_allowed(sql: str) -> None:
    validated = validate_sql(sql, {"start": date(2026, 1, 1)}, LIMIT)
    assert validated.sql.upper().rstrip().split("LIMIT")[-1].strip().isdigit()


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO customers (customer_id) VALUES ('x')",
        "UPDATE customers SET segment = 'SMB'",
        "DELETE FROM customers",
        "DROP TABLE customers",
        "ALTER TABLE customers RENAME TO c2",
        "CREATE TABLE t AS SELECT segment FROM customers",
        "CREATE VIEW v AS SELECT segment FROM customers",
        "TRUNCATE customers",
        "ATTACH 'other.duckdb' AS other",
        "DETACH other",
        "COPY customers TO 'customers.csv'",
        "EXPORT DATABASE 'dump'",
        "IMPORT DATABASE 'dump'",
        "INSTALL httpfs",
        "LOAD httpfs",
        "CALL pragma_version()",
        "SET threads = 64",
        "PRAGMA enable_profiling",
        "SUMMARIZE customers",
        "DESCRIBE customers",
        "MERGE INTO customers USING daily_revenue ON TRUE WHEN MATCHED THEN DELETE",
    ],
)
def test_writes_ddl_and_commands_are_rejected(sql: str) -> None:
    with pytest.raises(UnsafeSQLError):
        validate_sql(sql, {}, LIMIT)


@pytest.mark.parametrize(
    ("sql", "match"),
    [
        ("SELECT * FROM read_csv_auto('data/seeds/x.csv')", "not allowed"),
        ("SELECT * FROM read_json_auto('data/seeds/injected_events.json')", "not allowed"),
        ("SELECT * FROM read_parquet('data/seeds/parquet/*.parquet')", "not allowed"),
        ("SELECT read_text('.env') AS t FROM customers", "not allowed"),
        ("SELECT * FROM 'data/seeds/injected_events.json'", "not allowed"),
        ('SELECT * FROM "data/seeds/injected_events.json"', "not allowed"),
        ("SELECT * FROM 's3://bucket/file.parquet'", "not allowed"),
        ("SELECT * FROM 'https://example.com/data.csv'", "not allowed"),
        ("SELECT * FROM glob('**/*.json')", "not allowed"),
        ("SELECT getenv('ANTHROPIC_API_KEY') AS k FROM customers", "not allowed"),
        ("SELECT current_setting('home_directory') AS h FROM customers", "not allowed"),
        ("SELECT current_database() AS d FROM customers", "not allowed"),
        ("SELECT version() AS v FROM customers", "not allowed"),
        ("SELECT http_get('https://example.com') AS r FROM customers", "not allowed"),
        ("SELECT * FROM duckdb_settings()", "not allowed"),
        ("SELECT * FROM information_schema.columns", "Qualified"),
        ("SELECT * FROM pg_catalog.pg_tables", "Qualified"),
        ("SELECT * FROM sqlite_master", "not allowed"),
        ("SELECT * FROM range(1000000000)", "not allowed"),
        ("SELECT generate_series(1, 1000000000) AS n FROM customers", "not allowed"),
        ("SELECT my_extension_function(segment) AS x FROM customers", "allowlist"),
    ],
)
def test_filesystem_network_system_and_unknown_functions_are_rejected(sql: str, match: str) -> None:
    _reject(sql, match)


@pytest.mark.parametrize(
    ("sql", "match"),
    [
        ("SELECT segment FROM secret_table", "not allowed"),
        ("SELECT segment FROM customers UNION ALL SELECT name FROM hidden_events", "not allowed"),
        ("SELECT segment FROM customers WHERE customer_id IN (SELECT id FROM ground_truth)", "not allowed"),
        ("SELECT health_score FROM customers", "Unknown column"),
        ("SELECT password FROM customers", "Unknown column"),
        ("SELECT company_name FROM customers", "withheld"),
        ("SELECT sales_rep, deal_value FROM sales_opportunities", "PII"),
        ("SELECT * FROM customers", "withheld"),
        ("SELECT * FROM sales_opportunities", "PII"),
    ],
)
def test_tables_and_columns_outside_the_exposure_policy_are_rejected(sql: str, match: str) -> None:
    _reject(sql, match)


def test_complexity_limits() -> None:
    _reject("SELECT segment FROM customers" + " " * 5000, "longer than", max_length=4000)
    _reject("SELECT c.segment FROM customers c, subscriptions s", "explicit ON")
    _reject("SELECT c.segment FROM customers c CROSS JOIN usage_events u", "explicit ON")
    joins = " ".join(f"JOIN subscriptions s{i} ON s{i}.customer_id = c.customer_id" for i in range(5))
    _reject(f"SELECT c.segment FROM customers c {joins}", "joins", max_joins=4)
    nested = "SELECT segment FROM customers WHERE customer_id IN (" * 5 + "SELECT customer_id FROM customers" + ")" * 5
    _reject(nested, "nested", max_nesting_depth=3)
    ctes = ", ".join(f"c{i} AS (SELECT segment FROM customers)" for i in range(6))
    _reject(f"WITH {ctes} SELECT segment FROM c0", "CTEs", max_ctes=4)
    _reject("WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t) SELECT n FROM t", "Recursive")
    params = {f"p{i}": i for i in range(25)}
    where = " OR ".join(f"seats = $p{i}" for i in range(25))
    _reject(f"SELECT plan FROM subscriptions WHERE {where}", "parameters", params, max_parameters=20)
    _reject("SELECT segment FROM customers WHERE country = $c", "longer than", {"c": "x" * 500})


def test_injection_through_values_and_comments_is_inert() -> None:
    validated = validate_sql(
        "SELECT segment FROM customers WHERE country = $c", {"c": "x'; DROP TABLE customers; --"}, 5
    )
    assert "DROP" not in validated.sql.upper() and validated.parameters["c"].startswith("x'")
    _reject("SELECT segment FROM customers; DROP TABLE customers", "Exactly one")
    commented = validate_sql("SELECT segment FROM customers -- */ DROP TABLE customers; /*", {}, 5)
    assert commented.sql == "SELECT segment FROM customers LIMIT 6"  # no comments; LIMIT = row limit + 1


def test_policy_is_explicit_and_fails_closed() -> None:
    policy = default_exposure_policy()
    schema = {t.name for t in metadata.TABLES} | {v.name for v in metadata.VIEWS}
    assert schema >= APPROVED_TABLES | APPROVED_VIEWS
    assert policy.approved_relations == APPROVED_TABLES | APPROVED_VIEWS  # nothing else is exposed
    assert "company_name" not in policy.exposed_columns["customers"]
    assert "sales_rep" not in policy.exposed_columns["sales_opportunities"]
    for columns in policy.exposed_columns.values():
        assert not {c for c in columns if "health" in c or "latent" in c or "injected" in c}


# ---- execution: bounded rows, timeouts, read-only ----------------------------------------------------------


@pytest.fixture(scope="module")
def context(small_db: Any) -> ToolContext:
    return ToolContext(db=small_db, as_of=date(2026, 8, 31), sql_row_limit=20, sql_timeout_seconds=5.0)


def _sql(context: ToolContext, **arguments: Any) -> Any:
    request = ToolRequest(call_id="T1", tool_name="run_safe_sql", arguments=arguments)
    return ToolRegistry().execute(request, context)


def test_excessive_rows_are_bounded_and_flagged(context: ToolContext) -> None:
    result = _sql(context, sql="SELECT customer_id FROM customers ORDER BY customer_id")
    assert result.success and result.result.truncated and result.result.row_count == 20
    assert any("Truncated" in note for note in result.limitations)


def test_long_running_query_is_interrupted(small_db: Any) -> None:
    heavy = (
        "SELECT SUM(a.api_calls * b.sessions * c.active_users) AS x FROM usage_events a "
        "JOIN usage_events b ON a.api_calls >= 0 JOIN usage_events c ON c.api_calls >= 0"
    )
    started = time.perf_counter()
    with pytest.raises(QueryTimeoutAnalyticsError), execution_deadline(0.2):
        QueryRunner(small_db, "heavy").run(heavy)
    assert time.perf_counter() - started < 5
    assert QueryRunner(small_db, "after").run("SELECT 1 AS ok").rows == [(1,)]  # the connection stays usable


def test_sql_tool_timeout_is_a_controlled_failure(small_db: Any) -> None:
    ctx = ToolContext(db=small_db, as_of=date(2026, 8, 31), sql_row_limit=20, sql_timeout_seconds=0.2)
    heavy = (
        "SELECT SUM(a.api_calls * b.sessions * c.active_users) AS x FROM usage_events a "
        "JOIN usage_events b ON a.api_calls >= 0 JOIN usage_events c ON c.api_calls >= 0"
    )
    result = _sql(ctx, sql=heavy)
    assert not result.success and result.error.code == "timeout" and not result.error.retryable
    assert result.result is None


def test_expired_deadline_refuses_to_start(small_db: Any) -> None:
    with execution_deadline(0.0):
        assert (remaining_seconds() or 0) <= 0
        with pytest.raises(QueryTimeoutError):
            small_db.query("SELECT 1")
    assert remaining_seconds() is None
    with execution_deadline(5.0), execution_deadline(60.0):
        assert (remaining_seconds() or 99) <= 5.0  # nested deadlines keep the earliest


def test_database_connection_is_read_only(small_db: Any) -> None:
    for statement in (
        "CREATE TABLE probe (x INTEGER)",
        "DELETE FROM customers",
        "INSERT INTO customers SELECT * FROM customers",
    ):
        with pytest.raises(Exception, match=r"(?i)read.only"):
            QueryRunner(small_db, "write_probe").run(statement)
