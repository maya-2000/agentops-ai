"""Dataset manifest and content checksums, computed from the loaded database."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from app.database.metadata import SCHEMA_VERSION, TABLES
from data.generator.config import CURRENCY, DATASET_NAME, DATASET_VERSION, GeneratorConfig

MANIFEST_FILENAME = "dataset_manifest.json"
CHECKSUMS_FILENAME = "dataset_checksums.json"


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> Any:
    row = con.execute(sql).fetchone()
    return row[0] if row else None


def build_manifest(config: GeneratorConfig, db_path: Path) -> dict[str, Any]:
    """Build the manifest in the fixed structural template; every count is queried, not assumed."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {t.name: {"row_count": int(_scalar(con, f'SELECT COUNT(*) FROM "{t.name}"'))} for t in TABLES}
        return {
            "dataset_name": DATASET_NAME,
            "dataset_version": DATASET_VERSION,
            "random_seed": config.seed,
            "generation_timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "period_start": config.start_date.isoformat(),
            "period_end": config.end_date.isoformat(),
            "currency": CURRENCY,
            "customer_count": int(_scalar(con, "SELECT COUNT(*) FROM customers")),
            "region_count": int(_scalar(con, "SELECT COUNT(DISTINCT region) FROM customers")),
            "country_count": int(_scalar(con, "SELECT COUNT(DISTINCT country) FROM customers")),
            "segment_count": int(_scalar(con, "SELECT COUNT(DISTINCT segment) FROM customers")),
            "plan_count": int(_scalar(con, "SELECT COUNT(DISTINCT plan) FROM subscriptions")),
            "sales_rep_count": int(_scalar(con, "SELECT COUNT(DISTINCT sales_rep) FROM sales_opportunities")),
            "tables": tables,
            "schema_version": SCHEMA_VERSION,
        }
    finally:
        con.close()


def table_checksums(db_path: Path) -> dict[str, dict[str, Any]]:
    """Order-independent content fingerprint per table (for reproducibility checks)."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        out: dict[str, dict[str, Any]] = {}
        for table in TABLES:
            cols = ", ".join(f'"{c}"' for c in table.column_names)
            order = ", ".join(f'"{c}"' for c in table.primary_key)
            digest = _scalar(
                con,
                f"SELECT md5(string_agg(CAST(ROW({cols}) AS VARCHAR), '|' ORDER BY {order})) FROM \"{table.name}\"",
            )
            count = _scalar(con, f'SELECT COUNT(*) FROM "{table.name}"')
            out[table.name] = {"row_count": int(count), "md5": digest}
        return out
    finally:
        con.close()


def dataset_fingerprint(checksums: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(checksums, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
