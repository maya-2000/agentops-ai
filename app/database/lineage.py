"""Data-lineage foundation.

Every future analytical result (SQL query, KPI calculation, forecast, anomaly) will carry a
``LineageRecord`` so the Phase 4 evidence layer can trace a claim back to:

- the dataset version it was computed on (from the dataset manifest),
- the query and tool run that produced it,
- the source tables read, and
- the calculation applied.

Phase 1 provides only these structures and helpers; the evidence store is built later.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, date, datetime
from pathlib import Path

import sqlglot
from pydantic import BaseModel, Field
from sqlglot import exp


class DatasetInfo(BaseModel):
    """Identity of the dataset a result was computed on (subset of the manifest)."""

    dataset_name: str
    dataset_version: str
    schema_version: str
    generation_timestamp: str | None = None
    random_seed: int | None = None
    period_start: date
    period_end: date
    currency: str

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> DatasetInfo:
        with manifest_path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)
        return cls.model_validate(manifest)


class LineageRecord(BaseModel):
    """Provenance attached to a query result or derived calculation."""

    dataset_version: str
    query_id: str
    tool_run_id: str | None = None
    source_tables: list[str] = Field(default_factory=list)
    calculation: str | None = None
    execution_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


def new_query_id() -> str:
    """Globally unique query identifier (``Q-`` prefix, 12 hex chars)."""
    return f"Q-{uuid.uuid4().hex[:12]}"


def new_tool_run_id() -> str:
    """Globally unique tool-run identifier (``T-`` prefix, 12 hex chars)."""
    return f"T-{uuid.uuid4().hex[:12]}"


def extract_source_tables(sql: str, dialect: str = "duckdb") -> list[str]:
    """Return the physical tables/views referenced by a query, excluding CTE names.

    Unparseable SQL returns an empty list rather than raising: lineage extraction must never
    be the reason a query fails. (SQL *validation* is a separate, later guardrail.)
    """
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except sqlglot.errors.ParseError:
        return []
    tables: set[str] = set()
    for statement in statements:
        if statement is None:
            continue
        cte_names = {cte.alias_or_name for cte in statement.find_all(exp.CTE)}
        for table in statement.find_all(exp.Table):
            name = table.name
            if name and name not in cte_names:
                tables.add(name)
    return sorted(tables)
