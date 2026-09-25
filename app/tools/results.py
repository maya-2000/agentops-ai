"""Result types that tools add on top of the Phase 2/3 results (which are returned unchanged)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.analytics.kpis import KPIResult
from app.analytics.models import Scalar


class KPIComparison(BaseModel):
    """A KPI in two periods. The change is computed by the tool layer with Phase 2's ``pct_change``."""

    key: str
    name: str
    unit: str
    current: KPIResult
    comparison: KPIResult
    absolute_change: float | None
    percentage_change: float | None  # current / comparison - 1; None when the comparison value is 0 or missing
    direction: Literal["increase", "decline", "unchanged"] | None


class SQLResult(BaseModel):
    columns: list[str]
    rows: list[list[Scalar]]
    row_count: int
    truncated: bool  # more rows existed than max_rows; the rows shown are NOT complete
    max_rows: int
    sql: str  # the executed, validated statement
    parameters: dict[str, Scalar] = Field(default_factory=dict)
    query_id: str
    source_tables: list[str]
    execution_timestamp: datetime
    description: str
