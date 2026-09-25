"""Minimum SQL safety of the run_safe_sql tool (formal security testing follows in Phase 5)."""

from __future__ import annotations

import pytest

from app.tools.sql_safety import UnsafeSQLError, default_policy, validate_sql

LIMIT = 50


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT segment, COUNT(*) AS n FROM customers GROUP BY segment",
        "SELECT c.segment, SUM(r.revenue) AS revenue FROM daily_revenue AS r JOIN customers AS c "
        "ON c.customer_id = r.customer_id WHERE r.date >= $start GROUP BY c.segment ORDER BY revenue DESC",
        "WITH sg AS (SELECT customer_id FROM customers WHERE country = $country) SELECT COUNT(*) AS n FROM sg",
        "SELECT month, revenue FROM v_monthly_revenue WHERE region = 'APAC'",
        "SELECT region FROM customers UNION ALL SELECT region FROM daily_revenue",
        "SELECT opportunity_id, stage, deal_value FROM sales_opportunities LIMIT 10",
    ],
)
def test_select_is_allowed(sql: str) -> None:
    validated = validate_sql(sql, {"start": "2026-01-01", "country": "Singapore"}, LIMIT)
    assert validated.source_tables
    assert "LIMIT" in validated.sql.upper()


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("INSERT INTO customers (customer_id) VALUES ('x')", "Only SELECT"),
        ("UPDATE customers SET segment = 'SMB'", "Only SELECT"),
        ("DELETE FROM customers", "Only SELECT"),
        ("DROP TABLE customers", "Only SELECT"),
        ("ALTER TABLE customers ADD COLUMN x INT", "Only SELECT"),
        ("CREATE TABLE t AS SELECT * FROM customers", "Only SELECT"),
        ("ATTACH 'other.duckdb' AS other", "Only SELECT"),
        ("COPY customers TO 'out.csv'", "Only SELECT"),
        ("PRAGMA database_list", "Only SELECT"),
        ("INSTALL httpfs", "Only SELECT"),
        ("SET threads = 1", "Only SELECT"),
        ("DESCRIBE customers", "Only SELECT"),
        ("SELECT 1; DROP TABLE customers", "Exactly one"),
        ("SELECT * FROM read_csv('/etc/passwd')", "not allowed"),
        ("SELECT * FROM 'data/seeds/injected_events.json'", "not allowed"),
        ("SELECT read_text('/etc/passwd') AS t FROM customers", "not allowed"),
        ("SELECT getenv('HOME') AS h FROM customers", "not allowed"),
        ("SELECT * FROM glob('*')", "not allowed"),
        ("SELECT * FROM duckdb_tables()", "not allowed"),
        ("SELECT * FROM range(10)", "Table functions"),
        ("SELECT * FROM information_schema.tables", "Qualified"),
        ("SELECT * FROM main.customers", "Qualified"),
        ("SELECT * FROM secret_table", "not allowed"),
        ("SELECT * INTO copy_t FROM customers", "INTO"),
        ("SELECT password FROM customers", "Unknown column"),
        ("SELECT sales_rep FROM sales_opportunities", "PII"),
        ("SELECT * FROM sales_opportunities", "PII"),
        ("SELECT customer_id FROM customers WHERE country = ?", "named"),
        ("SELECT customer_id FROM customers LIMIT $n", "literal"),
        ("SELECT 1 AS x", "at least one"),
        ("", "Empty"),
        ("SELEC customer_id FROM", "parsed"),
    ],
)
def test_unsafe_sql_is_rejected(sql: str, message: str) -> None:
    with pytest.raises(UnsafeSQLError, match=message):
        validate_sql(sql, {"n": 5}, LIMIT)


def test_row_limit_is_capped_at_the_limit_plus_one() -> None:
    assert validate_sql("SELECT customer_id FROM customers", {}, LIMIT).sql.endswith(f"LIMIT {LIMIT + 1}")
    excessive = validate_sql("SELECT customer_id FROM customers LIMIT 100000", {}, LIMIT)
    assert excessive.sql.endswith(f"LIMIT {LIMIT + 1}")
    small = validate_sql("SELECT customer_id FROM customers LIMIT 3", {}, LIMIT)
    assert small.sql.endswith("LIMIT 3")


def test_parameters_are_bound_not_formatted() -> None:
    validated = validate_sql(
        "SELECT customer_id FROM customers WHERE country = $country", {"country": "x' OR 1=1 --"}, 5
    )
    assert "OR 1=1" not in validated.sql and "$country" in validated.sql
    assert validated.parameters == {"country": "x' OR 1=1 --"}
    with pytest.raises(UnsafeSQLError, match="Missing values"):
        validate_sql("SELECT customer_id FROM customers WHERE country = $country", {}, 5)
    with pytest.raises(UnsafeSQLError, match="scalar"):
        validate_sql("SELECT customer_id FROM customers WHERE country = $country", {"country": ["a", "b"]}, 5)


def test_policy_covers_business_tables_and_withholds_pii() -> None:
    policy = default_policy()
    assert {"customers", "daily_revenue", "v_monthly_mrr"} <= set(policy.allowed_tables)
    assert "sales_rep" not in policy.allowed_columns["sales_opportunities"]
    assert "sales_rep" in policy.pii_columns["sales_opportunities"]
