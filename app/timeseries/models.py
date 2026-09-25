"""Typed monthly time series with provenance."""

from __future__ import annotations

from datetime import date
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, computed_field

from app.analytics.models import Provenance
from app.timeseries.errors import SeriesStatus

# ``observed``: the value comes from rows in the database.
# ``true_zero``: no rows exist and zero is the semantically correct value (see metrics.py).
# ``missing``: the value is unknown. It is never replaced by zero or by an imputed value.
Observation = Literal["observed", "true_zero", "missing"]


class TimeSeriesPoint(BaseModel):
    """One calendar month of a series."""

    period: str  # YYYY-MM
    start: date
    end: date
    value: float | None
    observation: Observation


class TimeSeries(BaseModel):
    """A continuous, chronological monthly series of one metric for one (optionally filtered) group."""

    metric: str
    metric_name: str
    unit: str
    grain: Literal["month"] = "month"
    status: SeriesStatus
    start: date | None
    end: date | None
    points: list[TimeSeriesPoint] = Field(default_factory=list)
    filters: dict[str, str] = Field(default_factory=dict)
    calculation: str
    message: str | None = None
    limitations: list[str] = Field(default_factory=list)
    provenance: Provenance

    @computed_field  # type: ignore[prop-decorator]
    @property
    def source_tables(self) -> list[str]:
        return self.provenance.source_tables

    @computed_field  # type: ignore[prop-decorator]
    @property
    def query_ids(self) -> list[str]:
        return self.provenance.query_ids

    @computed_field  # type: ignore[prop-decorator]
    @property
    def query_id(self) -> str | None:
        """The query that returned the monthly values (the last query of the operation)."""
        return self.provenance.queries[-1].query_id if self.provenance.queries else None

    def values(self) -> np.ndarray:
        """Values as floats; missing months are ``nan``."""
        return np.array([np.nan if p.value is None else p.value for p in self.points], dtype=float)

    def labels(self) -> list[str]:
        return [p.period for p in self.points]

    @property
    def missing_count(self) -> int:
        return sum(1 for p in self.points if p.observation == "missing")

    def contiguous_tail_start(self) -> int:
        """Index of the first point of the last run of non-missing months (``len(points)`` if the last is missing)."""
        start = len(self.points)
        for index in range(len(self.points) - 1, -1, -1):
            if self.points[index].value is None:
                break
            start = index
        return start
