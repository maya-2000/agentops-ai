"""Lineage foundation: identifiers, table extraction and query-result provenance."""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import pytest

from app.database.duckdb_backend import DuckDBDatabase
from app.database.lineage import DatasetInfo, LineageRecord, extract_source_tables, new_query_id, new_tool_run_id


def test_identifier_formats_are_unique() -> None:
    ids = {new_query_id() for _ in range(1000)}
    assert len(ids) == 1000
    assert all(re.fullmatch(r"Q-[0-9a-f]{12}", i) for i in ids)
    assert re.fullmatch(r"T-[0-9a-f]{12}", new_tool_run_id())


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT * FROM customers", ["customers"]),
        (
            "SELECT c.region, SUM(d.revenue) FROM daily_revenue d JOIN customers c USING (customer_id) GROUP BY 1",
            ["customers", "daily_revenue"],
        ),
        ("WITH m AS (SELECT * FROM v_monthly_mrr) SELECT * FROM m", ["v_monthly_mrr"]),
        (
            "SELECT * FROM support_tickets WHERE customer_id IN (SELECT customer_id FROM customers)",
            ["customers", "support_tickets"],
        ),
        ("this is not sql (((", []),
    ],
)
def test_extract_source_tables(sql: str, expected: list[str]) -> None:
    assert extract_source_tables(sql) == expected


def test_dataset_info_from_manifest(tmp_path: Path) -> None:
    manifest = {
        "dataset_name": "Northwind Cloud",
        "dataset_version": "1.0.0",
        "random_seed": 42,
        "generation_timestamp": "2026-09-24T00:00:00+00:00",
        "period_start": "2024-09-01",
        "period_end": "2026-08-31",
        "currency": "SGD",
        "schema_version": "1.0.0",
        "tables": {},
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(manifest))
    info = DatasetInfo.from_manifest(path)
    assert info.dataset_version == "1.0.0" and info.currency == "SGD"


def test_query_result_carries_lineage(tmp_path: Path) -> None:
    db_file = tmp_path / "t.duckdb"
    con = duckdb.connect(str(db_file))
    con.execute("CREATE TABLE customers (customer_id VARCHAR, segment VARCHAR)")
    con.execute("INSERT INTO customers VALUES ('C1', 'SMB'), ('C2', 'Enterprise')")
    con.close()

    with DuckDBDatabase(db_file, dataset_version="9.9.9") as db:
        result = db.query(
            "SELECT segment, COUNT(*) AS n FROM customers GROUP BY 1 ORDER BY 1",
            tool_run_id="T-abc",
            calculation="count of customers by segment",
        )
    assert result.row_count == 2 and result.columns == ["segment", "n"]
    assert result.to_records()[0] == {"segment": "Enterprise", "n": 1}
    lineage: LineageRecord = result.lineage
    assert lineage.dataset_version == "9.9.9"
    assert lineage.query_id == result.query_id
    assert lineage.tool_run_id == "T-abc"
    assert lineage.source_tables == ["customers"]
    assert lineage.calculation == "count of customers by segment"
    assert lineage.execution_timestamp.tzinfo is not None


def test_default_connection_is_read_only(tmp_path: Path) -> None:
    db_file = tmp_path / "ro.duckdb"
    duckdb.connect(str(db_file)).execute("CREATE TABLE t (x INT)").close()
    with DuckDBDatabase(db_file) as db, pytest.raises(duckdb.Error):
        db.query("INSERT INTO t VALUES (1)")


def test_missing_database_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"data\.generator\.generate"):
        DuckDBDatabase(tmp_path / "missing.duckdb")
