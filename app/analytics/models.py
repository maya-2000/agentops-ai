"""Evidence-ready result contract shared by every analytics operation.

Every public analytics function returns a typed result carrying its own provenance: which
SQL ran with which bound parameters, which tables were read, which calculation turned the
rows into the reported numbers, and when. The Phase 4 evidence layer can then answer
"where did this number come from?" without re-deriving anything.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field, computed_field

from app.analytics.errors import ResultStatus
from app.analytics.periods import Period
from app.database.lineage import LineageRecord

Scalar = bool | int | float | str | None


class QueryTrace(BaseModel):
    """One executed query: the exact SQL, its bound parameters and its lineage record."""

    query_id: str
    sql: str
    parameters: dict[str, Any]
    row_count: int
    execution_time_ms: float
    lineage: LineageRecord


class Provenance(BaseModel):
    """How a result was produced (reuses the Phase 1 ``LineageRecord`` per query)."""

    operation: str
    operation_id: str
    calculation: str
    dataset_version: str
    queries: list[QueryTrace] = Field(default_factory=list)
    execution_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def source_tables(self) -> list[str]:
        return sorted({t for q in self.queries for t in q.lineage.source_tables})

    @computed_field  # type: ignore[prop-decorator]
    @property
    def query_ids(self) -> list[str]:
        return [q.query_id for q in self.queries]


RowT = TypeVar("RowT", bound=BaseModel)


class AnalyticsResult(BaseModel, Generic[RowT]):
    """Generic result: typed rows plus headline figures, context and provenance."""

    operation: str
    status: ResultStatus
    period: Period | None = None
    comparison_period: Period | None = None
    filters: dict[str, str] = Field(default_factory=dict)
    dimensions: list[str] = Field(default_factory=list)
    data: list[RowT] = Field(default_factory=list)
    summary: dict[str, Scalar] = Field(default_factory=dict)
    message: str | None = None
    limitations: list[str] = Field(default_factory=list)
    provenance: Provenance

    @computed_field  # type: ignore[prop-decorator]
    @property
    def source_tables(self) -> list[str]:
        return self.provenance.source_tables


def to_number(value: Any) -> float | int | None:
    """Convert DB numerics (Decimal, numpy) to plain Python numbers; keep ints as ints."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return float(value)


def safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    """``numerator / denominator``, or ``None`` when either is missing or the denominator is 0."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator) / float(denominator)
