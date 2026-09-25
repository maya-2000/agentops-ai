"""Typed errors and statuses shared by the Phase 3 time-series, forecasting and anomaly layers.

The errors extend the Phase 2 hierarchy, so a caller that already handles
``InvalidRequestError`` / ``AnalyticsError`` handles these too.
"""

from __future__ import annotations

from typing import Literal

from app.analytics.errors import InvalidRequestError

# ``ok``: a result was produced. ``no_data``: nothing is observed for the metric and filters.
# ``insufficient_history``: too few months to fit, backtest or score defensibly.
# ``insufficient_data``: enough months, but the series is unsuitable (e.g. mostly zeros).
SeriesStatus = Literal["ok", "no_data", "insufficient_history", "insufficient_data"]


class UnsupportedMetricError(InvalidRequestError):
    """The metric is not one of the registered time-series metrics."""

    code = "unsupported_metric"


class InvalidHorizonError(InvalidRequestError):
    """The forecast horizon is outside the supported range."""

    code = "invalid_horizon"


class UnsupportedMethodError(InvalidRequestError):
    """The forecasting model or anomaly detector is not in the allow-list."""

    code = "unsupported_method"
