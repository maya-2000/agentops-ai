"""The analytics layer's single execution boundary.

``QueryRunner`` wraps the Phase 1 ``Database`` protocol (it never imports a driver). It:

- binds exactly the named parameters (``$name``) that a statement uses,
- records every executed statement with its parameters as a ``QueryTrace``,
- converts driver exceptions into ``AnalyticsDatabaseError``, and
- builds the ``Provenance`` attached to results.

One runner is created per analytics operation, so a result's provenance lists precisely the
queries that produced it.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from app.analytics.errors import AnalyticsDatabaseError, AnalyticsError, QueryTimeoutAnalyticsError
from app.analytics.models import Provenance, QueryTrace
from app.database.base import Database, QueryResult
from app.database.deadline import QueryTimeoutError
from app.database.lineage import new_tool_run_id

_PARAM_RE = re.compile(r"\$([a-z_][a-z0-9_]*)")


def used_parameters(sql: str) -> set[str]:
    """Named parameters (``$name``) referenced by a SQL statement."""
    return set(_PARAM_RE.findall(sql))


def _jsonable(value: Any) -> Any:
    return value.isoformat() if isinstance(value, date) else value


class QueryRunner:
    def __init__(self, db: Database, operation: str):
        self.db = db
        self.operation = operation
        self.operation_id = new_tool_run_id()
        self.traces: list[QueryTrace] = []

    def run(self, sql: str, params: dict[str, Any] | None = None, *, calculation: str | None = None) -> QueryResult:
        """Execute a read query, binding only the parameters it references."""
        needed = used_parameters(sql)
        supplied = params or {}
        missing = needed - supplied.keys()
        if missing:
            raise AnalyticsError(f"Internal error: SQL references unbound parameters {sorted(missing)}")
        bound = {k: supplied[k] for k in sorted(needed)}
        try:
            result = self.db.query(sql, bound, tool_run_id=self.operation_id, calculation=calculation)
        except AnalyticsError:
            raise
        except QueryTimeoutError as exc:
            raise QueryTimeoutAnalyticsError(f"{self.operation}: {exc}") from exc
        except Exception as exc:  # driver-specific errors are normalised at this boundary
            raise AnalyticsDatabaseError(f"{self.operation}: query failed: {exc}") from exc
        self.traces.append(
            QueryTrace(
                query_id=result.query_id,
                sql=sql,
                parameters={k: _jsonable(v) for k, v in bound.items()},
                row_count=result.row_count,
                execution_time_ms=result.execution_time_ms,
                lineage=result.lineage,
            )
        )
        return result

    def records(
        self, sql: str, params: dict[str, Any] | None = None, *, calculation: str | None = None
    ) -> list[dict[str, Any]]:
        return self.run(sql, params, calculation=calculation).to_records()

    def provenance(self, calculation: str) -> Provenance:
        return Provenance(
            operation=self.operation,
            operation_id=self.operation_id,
            calculation=calculation,
            dataset_version=self.db.dataset_version,
            queries=list(self.traces),
        )

    def absorb(self, provenance: Provenance) -> None:
        """Include the queries of a nested operation (e.g. a KPI used by a module function)."""
        self.traces.extend(provenance.queries)
