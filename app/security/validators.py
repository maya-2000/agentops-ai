"""Central value validators: the one place that decides whether a value is acceptable.

The input guard (question and model understanding), the plan validator and the tool
authorization policy all call these functions, so the rules for metrics, dimensions, filters,
dates, periods, horizons, detectors, enums and text sizes are defined once. Each validator
returns a list of ``Violation`` objects and never raises. An empty list means the value is
accepted. Unknown values are rejected: the validators never guess.

The vocabularies come from the Phase 2/3 registries: KPI registry, dimension allow-list,
time-series metrics, forecast horizons and detector names.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any

from pydantic import BaseModel

from app.analytics.dimensions import DIMENSIONS
from app.analytics.kpis import KPI_REGISTRY
from app.anomalies.config import DETECTOR_NAMES
from app.forecasting.config import SUPPORTED_HORIZONS
from app.timeseries.metrics import SERIES_METRIC_KEYS

EARLIEST_DATE = date(2000, 1, 1)  # plausibility bounds for any date argument
LATEST_DATE = date(2100, 12, 31)
MAX_RANGE_DAYS = 3700  # about ten years
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PERIOD_SPEC = re.compile(
    r"^(last_month|latest_month|previous_month|last_quarter|previous_quarter|last_year|ytd|"
    r"trailing_\d{1,3}_months|\d{4}-(0[1-9]|1[0-2])|\d{4}-[Qq][1-4]|\d{4})$"
)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class Violation(BaseModel):
    code: str
    field: str
    message: str


def _v(code: str, field: str, message: str) -> list[Violation]:
    return [Violation(code=code, field=field, message=message)]


def check_text(value: Any, field: str, max_chars: int) -> list[Violation]:
    if not isinstance(value, str):
        return _v("invalid_arguments", field, f"{field} must be text")
    if len(value) > max_chars:
        return _v("oversized_input", field, f"{field} is longer than {max_chars} characters")
    if _CONTROL.search(value):
        return _v("invalid_arguments", field, f"{field} contains control characters")
    return []


def check_identifier(value: Any, field: str) -> list[Violation]:
    if not isinstance(value, str) or not _IDENTIFIER.match(value):
        return _v("invalid_arguments", field, f"{field} must be a lower-case identifier")
    return []


def check_kpi(value: Any, field: str = "kpi") -> list[Violation]:
    if not isinstance(value, str) or value not in KPI_REGISTRY:
        return _v("unsupported_kpi", field, f"Unknown KPI {str(value)[:40]!r}")
    return []


def check_series_metric(value: Any, field: str = "metric") -> list[Violation]:
    if not isinstance(value, str) or value not in SERIES_METRIC_KEYS:
        return _v("unsupported_metric", field, f"{str(value)[:40]!r} is not a forecastable metric")
    return []


def check_dimension(value: Any, field: str = "dimension") -> list[Violation]:
    if not isinstance(value, str) or value not in DIMENSIONS:
        return _v("unsupported_dimension", field, f"Unknown dimension {str(value)[:40]!r}")
    return []


def check_filters(filters: Any, *, max_filters: int, max_value_chars: int = 100) -> list[Violation]:
    if not isinstance(filters, Mapping):
        return _v("invalid_arguments", "filters", "filters must be an object")
    if len(filters) > max_filters:
        return _v("oversized_input", "filters", f"At most {max_filters} filters are allowed")
    problems: list[Violation] = []
    for key, value in filters.items():
        spec = DIMENSIONS.get(str(key))
        if spec is None or not spec.filterable:
            problems += _v("unsupported_dimension", f"filters.{str(key)[:40]}", f"Unknown filter {str(key)[:40]!r}")
            continue
        text = check_text(value, f"filters.{key}", max_value_chars)
        if text:
            problems += text
            continue
        if spec.allowed_values is not None and value not in spec.allowed_values:
            problems += _v("invalid_filter_value", f"filters.{key}", f"{str(value)[:40]!r} is not a valid {key}")
    return problems


def check_date(value: Any, field: str) -> list[Violation]:
    if isinstance(value, str):
        try:
            value = date.fromisoformat(value)
        except ValueError:
            return _v("invalid_arguments", field, f"{field} is not an ISO date (YYYY-MM-DD)")
    if not isinstance(value, date):
        return _v("invalid_arguments", field, f"{field} must be a date")
    if not EARLIEST_DATE <= value <= LATEST_DATE:
        return _v("invalid_date_range", field, f"{field} is outside the plausible range")
    return []


def check_date_range(start: Any, end: Any, field: str = "period") -> list[Violation]:
    problems = check_date(start, f"{field}.start") + check_date(end, f"{field}.end")
    if problems:
        return problems
    first = date.fromisoformat(start) if isinstance(start, str) else start
    last = date.fromisoformat(end) if isinstance(end, str) else end
    if first > last:
        return _v("invalid_date_range", field, f"{field} starts after it ends")
    if (last - first).days > MAX_RANGE_DAYS:
        return _v("invalid_date_range", field, f"{field} spans more than ten years")
    return []


def check_period_spec(value: Any, field: str = "period") -> list[Violation]:
    if not isinstance(value, str) or not _PERIOD_SPEC.match(value):
        return _v("invalid_period", field, f"{str(value)[:40]!r} is not a supported period")
    return []


def check_horizon(value: Any, field: str = "horizon") -> list[Violation]:
    if isinstance(value, bool) or not isinstance(value, int) or value not in SUPPORTED_HORIZONS:
        return _v(
            "invalid_horizon",
            field,
            f"Forecast horizons are {SUPPORTED_HORIZONS.start}-{SUPPORTED_HORIZONS.stop - 1} months",
        )
    return []


def check_detector(value: Any, field: str = "detector") -> list[Violation]:
    if value not in DETECTOR_NAMES:
        return _v("unsupported_method", field, f"Unknown detector {str(value)[:40]!r}")
    return []


def check_enum(value: Any, allowed: Iterable[str], field: str) -> list[Violation]:
    options = tuple(allowed)
    if value not in options:
        return _v("invalid_arguments", field, f"{field} must be one of {', '.join(options)}")
    return []


def check_finite(value: Any, field: str) -> list[Violation]:
    """Every float nested in ``value`` must be finite (no NaN or infinity)."""
    if isinstance(value, float):
        return [] if math.isfinite(value) else _v("invalid_tool_output", field, f"{field} is not a finite number")
    if isinstance(value, Mapping):
        return [p for k, v in value.items() for p in check_finite(v, f"{field}.{k}")]
    if isinstance(value, (list, tuple)):
        return [p for i, v in enumerate(value) for p in check_finite(v, f"{field}[{i}]")]
    return []
