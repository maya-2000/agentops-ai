"""Synthetic monthly series for Phase 3 tests (no database, fully deterministic)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

import numpy as np

from app.analytics.models import Provenance
from app.timeseries.calendar import add_months, month_end, month_label
from app.timeseries.metrics import get_series_metric
from app.timeseries.models import TimeSeries, TimeSeriesPoint

START = date(2024, 9, 1)


def synthetic_series(
    values: Sequence[float | None],
    *,
    metric: str = "revenue",
    start: date = START,
    filters: dict[str, str] | None = None,
) -> TimeSeries:
    """A ``TimeSeries`` with one point per value; ``None`` / ``nan`` become missing months."""
    spec = get_series_metric(metric)
    points = []
    for i, value in enumerate(values):
        first = add_months(start, i)
        missing = value is None or (isinstance(value, float) and math.isnan(value))
        points.append(
            TimeSeriesPoint(
                period=month_label(first),
                start=first,
                end=month_end(first),
                value=None if missing else float(value),  # type: ignore[arg-type]
                observation="missing" if missing else "observed",
            )
        )
    observed = any(p.value is not None for p in points)
    return TimeSeries(
        metric=spec.key,
        metric_name=spec.name,
        unit=spec.unit,
        status="ok" if observed else "no_data",
        start=points[0].start if points else None,
        end=points[-1].end if points else None,
        points=points,
        filters=filters or {},
        calculation="synthetic test series",
        provenance=Provenance(
            operation="synthetic",
            operation_id="T-synthetic",
            calculation="synthetic test series",
            dataset_version="test",
        ),
    )


def constant(n: int = 24, value: float = 100.0) -> list[float]:
    return [value] * n


def linear(n: int = 24, start: float = 100.0, slope: float = 10.0) -> list[float]:
    return [start + slope * i for i in range(n)]


def seasonal(n: int = 36, base: float = 100.0, amplitude: float = 20.0, period: int = 12) -> list[float]:
    return [base + amplitude * math.sin(2 * math.pi * i / period) for i in range(n)]


def noisy(n: int = 24, level: float = 1000.0, slope: float = 5.0, sd: float = 20.0, seed: int = 7) -> list[float]:
    rng = np.random.default_rng(seed)
    return [level + slope * i + float(e) for i, e in enumerate(rng.normal(0.0, sd, n))]


def with_change(values: Sequence[float], index: int, delta: float) -> list[float]:
    out = list(values)
    out[index] += delta
    return out
