"""Read access to the KPI registry (what the Phase 4 agent's ``get_kpi_definition`` will call)."""

from __future__ import annotations

from app.analytics.errors import UnsupportedKPIError
from app.analytics.kpis.definitions import KPI_KEYS, KPI_REGISTRY
from app.analytics.kpis.models import KPIDefinition


def get_kpi_definition(key: str) -> KPIDefinition:
    """Return the registered definition for ``key`` (case-insensitive)."""
    definition = KPI_REGISTRY.get(key.strip().lower())
    if definition is None:
        raise UnsupportedKPIError(f"Unknown KPI {key!r}. Registered KPIs: {', '.join(KPI_KEYS)}")
    return definition


def list_kpi_definitions() -> list[KPIDefinition]:
    return [KPI_REGISTRY[key] for key in KPI_KEYS]
