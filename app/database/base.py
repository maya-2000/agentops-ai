"""Backend-independent database interface.

Everything above the database layer depends on the ``Database`` protocol, never on a
specific driver, so DuckDB (default) can later be swapped for PostgreSQL via DATABASE_URL.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from app.database.lineage import LineageRecord


class QueryResult(BaseModel):
    """Result of a read-only query, including lineage for future evidence tracing."""

    query_id: str
    sql: str
    execution_time_ms: float
    row_count: int
    columns: list[str]
    rows: list[tuple[Any, ...]]
    lineage: LineageRecord

    def to_records(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, row, strict=True)) for row in self.rows]


@runtime_checkable
class Database(Protocol):
    """Read-only access to the business database."""

    dataset_version: str

    def query(
        self,
        sql: str,
        params: list[Any] | None = None,
        *,
        tool_run_id: str | None = None,
        calculation: str | None = None,
    ) -> QueryResult: ...

    def list_tables(self) -> list[str]: ...

    def close(self) -> None: ...
