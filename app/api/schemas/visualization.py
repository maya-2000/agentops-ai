"""A small, renderer-neutral chart specification.

Every spec is built from the run's evidence or from the typed forecast/anomaly results the evidence
was built from. The numbers in ``rows`` are copied, never recomputed; each spec names the evidence
items it depicts. A renderer (the Streamlit UI, or any client) only draws what is here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.analytics.models import Scalar

ChartKind = Literal["kpi_card", "time_series", "bar", "comparison", "forecast", "anomaly", "table"]
ChartSource = Literal["evidence", "forecast_result", "anomaly_report"]


class ChartField(BaseModel):
    """One column of ``rows``: its key, a display label, and how its values are typed."""

    key: str
    label: str
    role: Literal["x", "y", "category", "lower", "upper", "flag", "series", "label", "value", "detail"] = "detail"
    unit: str | None = None


class VisualizationSpec(BaseModel):
    chart_id: str  # V1, V2, ... (stable within one response)
    kind: ChartKind
    title: str
    subtitle: str | None = None
    metric: str | None = None
    unit: str | None = None
    fields: list[ChartField] = Field(default_factory=list)
    rows: list[dict[str, Scalar]] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    source: ChartSource = "evidence"
    notes: list[str] = Field(default_factory=list)  # labelling that must travel with the chart (e.g. "forecast")

    def field(self, role: str) -> ChartField | None:
        return next((f for f in self.fields if f.role == role), None)
