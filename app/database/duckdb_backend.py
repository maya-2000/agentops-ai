"""DuckDB implementation of the ``Database`` protocol (read-only by default)."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

import duckdb

from app.database.base import QueryResult
from app.database.deadline import QueryTimeoutError, remaining_seconds
from app.database.lineage import LineageRecord, extract_source_tables, new_query_id


class DuckDBDatabase:
    """Business database backed by a DuckDB file.

    The connection is opened ``read_only=True`` unless explicitly requested otherwise, so the
    engine itself refuses writes. Only the data loader opens a writable connection.

    Queries honour the active ``execution_deadline``: a query is not started after the deadline
    and is interrupted (``connection.interrupt``) when the deadline passes during execution.
    """

    def __init__(self, path: Path, *, dataset_version: str = "unknown", read_only: bool = True):
        if read_only and not path.exists():
            raise FileNotFoundError(
                f"DuckDB database not found at {path}. Build it with: python -m data.generator.generate"
            )
        self.path = path
        self.dataset_version = dataset_version
        self.read_only = read_only
        self._con = duckdb.connect(str(path), read_only=read_only)

    def query(
        self,
        sql: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
        *,
        tool_run_id: str | None = None,
        calculation: str | None = None,
    ) -> QueryResult:
        query_id = new_query_id()
        remaining = remaining_seconds()
        if remaining is not None and remaining <= 0:
            raise QueryTimeoutError("Query not started: the execution deadline has passed")
        timer = threading.Timer(remaining, self._con.interrupt) if remaining is not None else None
        started = time.perf_counter()
        try:
            if timer is not None:
                timer.start()
            cursor = self._con.execute(sql, params or [])
            rows = cursor.fetchall()
        except duckdb.InterruptException as exc:
            raise QueryTimeoutError("Query interrupted: the execution deadline passed") from exc
        finally:
            if timer is not None:
                timer.cancel()
        elapsed_ms = (time.perf_counter() - started) * 1000
        columns = [d[0] for d in cursor.description] if cursor.description else []
        lineage = LineageRecord(
            dataset_version=self.dataset_version,
            query_id=query_id,
            tool_run_id=tool_run_id,
            source_tables=extract_source_tables(sql),
            calculation=calculation,
        )
        return QueryResult(
            query_id=query_id,
            sql=sql,
            execution_time_ms=round(elapsed_ms, 3),
            row_count=len(rows),
            columns=columns,
            rows=rows,
            lineage=lineage,
        )

    def list_tables(self) -> list[str]:
        rows = self._con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
        return [r[0] for r in rows]

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Raw connection for internal data-layer code (validation, loading). Not for agents."""
        return self._con

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> DuckDBDatabase:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
