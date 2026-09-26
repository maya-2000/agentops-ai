"""Output contracts of the three LLM tasks, as Pydantic models plus strict JSON schemas.

The JSON schemas are written by hand in the subset that structured-output APIs accept: every
object lists all properties as required and sets ``additionalProperties: false``, and optional
values are nullable. Free-form tool arguments are carried as a JSON string (``arguments_json``)
and validated against the tool's own input model by the agent.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Intent(StrEnum):
    KPI_LOOKUP = "kpi_lookup"
    PERIOD_COMPARISON = "period_comparison"
    DIMENSIONAL_COMPARISON = "dimensional_comparison"
    REVENUE_INVESTIGATION = "revenue_investigation"
    CUSTOMER_INVESTIGATION = "customer_investigation"
    SALES_ANALYSIS = "sales_analysis"
    MARKETING_ANALYSIS = "marketing_analysis"
    SUPPORT_ANALYSIS = "support_analysis"
    PRODUCT_ANALYSIS = "product_analysis"
    FORECAST = "forecast"
    ANOMALY_DETECTION = "anomaly_detection"
    MIXED_INVESTIGATION = "mixed_investigation"
    UNSUPPORTED = "unsupported"


AnalysisType = Literal[
    "value",
    "change",
    "contribution",
    "highest",
    "lowest",
    "largest_decrease",
    "largest_increase",
    "largest_pct_decrease",
    "largest_pct_increase",
    "trend",
    "causal",
    "risk",
    "cohort",
]
ANALYSIS_TYPES: tuple[str, ...] = (
    "value",
    "change",
    "contribution",
    "highest",
    "lowest",
    "largest_decrease",
    "largest_increase",
    "largest_pct_decrease",
    "largest_pct_increase",
    "trend",
    "causal",
    "risk",
    "cohort",
)
# Rankings of members by their *change* between two periods ("which region had the largest decline"),
# as opposed to "highest"/"lowest", which rank levels ("which region had the most revenue").
CHANGE_RANKINGS: tuple[str, ...] = (
    "largest_decrease",
    "largest_increase",
    "largest_pct_decrease",
    "largest_pct_increase",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FilterItem(_Strict):
    dimension: str
    value: str


class UnderstandingOutput(_Strict):
    intent: Intent
    metric: str | None = None
    period: str | None = None
    comparison_period: str | None = None
    dimensions: list[str] = Field(default_factory=list)
    filters: list[FilterItem] = Field(default_factory=list)
    horizon: int | None = None
    analysis_type: AnalysisType | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    ambiguities: list[str] = Field(default_factory=list)
    material_ambiguity: bool = False
    unsupported_reason: str | None = None


class PlanStepOutput(_Strict):
    tool_name: str
    arguments_json: str
    purpose: str


class PlanOutput(_Strict):
    steps: list[PlanStepOutput] = Field(default_factory=list)
    rationale: str = ""
    sufficient: bool = False  # follow-up planning: the collected evidence already answers the question


class DraftItemOutput(_Strict):
    text: str
    claim_ids: list[str] = Field(default_factory=list)


class ResponseDraftOutput(_Strict):
    answer: str
    answer_claim_ids: list[str] = Field(default_factory=list)
    key_findings: list[DraftItemOutput] = Field(default_factory=list)
    interpretation: list[DraftItemOutput] = Field(default_factory=list)
    recommendations: list[DraftItemOutput] = Field(default_factory=list)


def _nullable(kind: str) -> dict[str, Any]:
    return {"type": [kind, "null"]}


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


_STRINGS = {"type": "array", "items": {"type": "string"}}
_DRAFT_ITEM = _object({"text": {"type": "string"}, "claim_ids": _STRINGS})

UNDERSTANDING_SCHEMA: dict[str, Any] = _object(
    {
        "intent": {"type": "string", "enum": [i.value for i in Intent]},
        "metric": _nullable("string"),
        "period": _nullable("string"),
        "comparison_period": _nullable("string"),
        "dimensions": _STRINGS,
        "filters": {"type": "array", "items": _object({"dimension": {"type": "string"}, "value": {"type": "string"}})},
        "horizon": _nullable("integer"),
        "analysis_type": {"type": ["string", "null"], "enum": [*ANALYSIS_TYPES, None]},
        "confidence": {"type": "number"},
        "ambiguities": _STRINGS,
        "material_ambiguity": {"type": "boolean"},
        "unsupported_reason": _nullable("string"),
    }
)

PLAN_SCHEMA: dict[str, Any] = _object(
    {
        "steps": {
            "type": "array",
            "items": _object(
                {"tool_name": {"type": "string"}, "arguments_json": {"type": "string"}, "purpose": {"type": "string"}}
            ),
        },
        "rationale": {"type": "string"},
        "sufficient": {"type": "boolean"},
    }
)

RESPONSE_SCHEMA: dict[str, Any] = _object(
    {
        "answer": {"type": "string"},
        "answer_claim_ids": _STRINGS,
        "key_findings": {"type": "array", "items": _DRAFT_ITEM},
        "interpretation": {"type": "array", "items": _DRAFT_ITEM},
        "recommendations": {"type": "array", "items": _DRAFT_ITEM},
    }
)
