"""Display names for metric and dimension identifiers (from the registries; never from the data)."""

from __future__ import annotations

from app.analytics.dimensions import DIMENSIONS
from app.analytics.kpis import KPI_REGISTRY
from app.timeseries import SERIES_METRICS


def metric_label(metric: str) -> str:
    """The registered display name of a KPI or series metric, else the identifier in words."""
    if metric in KPI_REGISTRY:
        return KPI_REGISTRY[metric].name
    if metric in SERIES_METRICS:
        return SERIES_METRICS[metric].name
    words = metric.replace("_", " ").strip()
    return words[:1].upper() + words[1:]


def dimension_label(dimension: str) -> str:
    spec = DIMENSIONS.get(dimension)
    return spec.display_name if spec else metric_label(dimension)
