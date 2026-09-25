"""Typed errors and result statuses for the analytics layer.

Two kinds of "no answer" are deliberately kept apart:

- **Errors** (exceptions below) mean the *request* is wrong or cannot be served: unknown KPI,
  a dimension the KPI does not support, an invalid filter value, a malformed period, or a
  database/calculation failure.
- **Result statuses** (``ResultStatus``) describe a valid request whose *data* does not support
  a value: nothing observed (``no_data``) or a denominator/evidence problem
  (``insufficient_data``). The value is then ``None``; the layer never fabricates a zero.
"""

from __future__ import annotations

from typing import Literal

ResultStatus = Literal["ok", "no_data", "insufficient_data"]


class AnalyticsError(Exception):
    """Base class for all analytics-layer errors."""

    code = "analytics_error"


class InvalidRequestError(AnalyticsError):
    """The request is malformed or internally inconsistent."""

    code = "invalid_request"


class UnsupportedKPIError(InvalidRequestError):
    """The KPI key is not in the registry."""

    code = "unsupported_kpi"


class UnsupportedDimensionError(InvalidRequestError):
    """The dimension or filter is unknown or not applicable to the requested calculation."""

    code = "unsupported_dimension"


class InvalidFilterValueError(InvalidRequestError):
    """A filter value is not one of the values that exist for that dimension."""

    code = "invalid_filter_value"


class InvalidPeriodError(InvalidRequestError):
    """A period specification could not be parsed or is inconsistent."""

    code = "invalid_period"


class CalculationError(AnalyticsError):
    """A calculation could not be completed (e.g. an unexpected result shape)."""

    code = "calculation_error"


class AnalyticsDatabaseError(AnalyticsError):
    """The database raised an error while executing an analytics query."""

    code = "database_error"
