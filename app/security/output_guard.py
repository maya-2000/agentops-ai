"""Tool-output validation: a tool result is untrusted until it passes these checks.

The evidence layer only receives validated results. A result is rejected (fail closed) when:

- **Type.** It is not the typed result expected for the tool.
- **Provenance.** It has no provenance: query IDs, source tables and a calculation are all
  required, and every source table must be an approved business relation.
- **Values.**
  - A number is not finite (NaN or infinity).
  - A date is implausible.
  - A KPI or metric identifier is unknown.
- **Semantics.**
  - A forecast lacks its model, has points on or before its cutoff, or has an unsupported
    horizon.
  - An anomaly report names an unknown detector or has results without thresholds.
  - An SQL result is inconsistent: the row count does not match the rows, rows exceed the
    limit, or a truncation flag contradicts the row count.
- **Exposure.** It exposes a field it should not:
  - a hidden-state or credential-like key;
  - a PII field outside the operation declared for it;
  - more customer-level rows than allowed;
  - an SQL column that is withheld.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

from app.analytics.kpis import KPI_REGISTRY, KPIResult
from app.analytics.models import AnalyticsResult
from app.anomalies import AnomalyReport
from app.anomalies.config import DETECTOR_NAMES
from app.forecasting import ForecastResult
from app.forecasting.config import SUPPORTED_HORIZONS
from app.llm.schemas import ResponseDraftOutput
from app.security.data_policy import (
    CUSTOMER_LEVEL_OPERATIONS,
    PII_ALLOWED_OPERATIONS,
    DataExposurePolicy,
    names_hidden_state,
)
from app.security.limits import SecurityLimits
from app.security.validators import EARLIEST_DATE, LATEST_DATE, Violation
from app.timeseries.metrics import SERIES_METRIC_KEYS
from app.tools.base import ToolResult
from app.tools.results import KPIComparison, SQLResult

EXPECTED_RESULT_TYPES: dict[str, tuple[type, ...]] = {
    "get_kpi": (KPIResult, KPIComparison),
    "analyze_revenue": (AnalyticsResult,),
    "analyze_customers": (AnalyticsResult,),
    "analyze_sales": (AnalyticsResult,),
    "analyze_marketing": (AnalyticsResult,),
    "analyze_support": (AnalyticsResult,),
    "analyze_product": (AnalyticsResult,),
    "get_cohort_analysis": (AnalyticsResult,),
    "get_customer_risk": (AnalyticsResult,),
    "forecast_metric": (ForecastResult,),
    "detect_anomalies": (AnomalyReport,),
    "run_safe_sql": (SQLResult,),
}


def _bad(field: str, message: str, code: str = "invalid_tool_output") -> Violation:
    return Violation(code=code, field=field, message=message)


class ToolOutputValidator:
    def __init__(self, limits: SecurityLimits, exposure: DataExposurePolicy):
        self.limits = limits
        self.exposure = exposure

    def validate(self, result: ToolResult) -> list[Violation]:
        if not result.success:
            return []  # a failed call carries no output; it produces no evidence
        payload = result.result
        expected = EXPECTED_RESULT_TYPES.get(result.tool_name)
        if expected is None:
            return [_bad("tool_name", "No output contract for this tool")]
        if payload is None or not isinstance(payload, expected):
            return [_bad("result", f"Unexpected result type {type(payload).__name__}")]
        problems = self._provenance(result)
        operation = f"{result.tool_name}.{result.arguments.get('operation')}" if "operation" in result.arguments else ""
        dumped = payload.model_dump(mode="python")
        problems += self._values(dumped, "result")
        problems += self._keys(dumped, operation, "result")
        if isinstance(payload, KPIResult):
            problems += self._kpi(payload.key)
        elif isinstance(payload, KPIComparison):
            problems += self._kpi(payload.key) + self._kpi(payload.current.key) + self._kpi(payload.comparison.key)
        elif isinstance(payload, ForecastResult):
            problems += self._forecast(payload)
        elif isinstance(payload, AnomalyReport):
            problems += self._anomalies(payload)
        elif isinstance(payload, SQLResult):
            problems += self._sql(payload)
        elif (
            isinstance(payload, AnalyticsResult)
            and result.tool_name in CUSTOMER_LEVEL_OPERATIONS
            and len(payload.data) > self.limits.max_customer_rows
        ):
            problems.append(_bad("data", "Too many customer-level rows", "data_policy"))
        return problems

    # ------------------------------------------------------------------ checks
    def _provenance(self, result: ToolResult) -> list[Violation]:
        problems: list[Violation] = []
        if not result.query_ids:
            problems.append(_bad("query_ids", "Missing query IDs"))
        if not result.source_tables:
            problems.append(_bad("source_tables", "Missing source tables"))
        unknown = set(result.source_tables) - self.exposure.approved_relations
        if unknown:
            problems.append(_bad("source_tables", "Reads a relation outside the approved business data", "data_policy"))
        if not result.calculation:
            problems.append(_bad("calculation", "Missing calculation description"))
        return problems

    def _values(self, value: Any, field: str) -> list[Violation]:
        if isinstance(value, float):
            return [] if math.isfinite(value) else [_bad(field, "Non-finite number")]
        if isinstance(value, datetime):
            return [] if EARLIEST_DATE <= value.date() <= LATEST_DATE else [_bad(field, "Implausible timestamp")]
        if isinstance(value, date):
            return [] if EARLIEST_DATE <= value <= LATEST_DATE else [_bad(field, "Implausible date")]
        if isinstance(value, dict):
            return [p for k, v in value.items() for p in self._values(v, f"{field}.{k}")]
        if isinstance(value, (list, tuple)):
            return [p for i, v in enumerate(value) for p in self._values(v, f"{field}[{i}]")]
        return []

    def _keys(self, value: Any, operation: str, field: str) -> list[Violation]:
        allowed_pii = PII_ALLOWED_OPERATIONS.get(operation, frozenset())
        problems: list[Violation] = []
        stack: list[Any] = [value]
        pii = self.exposure.all_pii_columns
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for key, child in item.items():
                    name = str(key)
                    if names_hidden_state(name):
                        problems.append(_bad(f"{field}.{name}", "Hidden-state or credential field", "data_policy"))
                    elif name in pii and name not in allowed_pii:
                        problems.append(_bad(f"{field}.{name}", "PII field not allowed here", "data_policy"))
                    stack.append(child)
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
        return problems

    @staticmethod
    def _kpi(key: str) -> list[Violation]:
        return [] if key in KPI_REGISTRY else [_bad("key", "Unknown KPI identifier")]

    @staticmethod
    def _forecast(r: ForecastResult) -> list[Violation]:
        problems: list[Violation] = []
        if r.metric not in SERIES_METRIC_KEYS:
            problems.append(_bad("metric", "Unknown forecast metric"))
        if r.horizon not in SUPPORTED_HORIZONS:
            problems.append(_bad("horizon", "Unsupported horizon"))
        if r.status == "ok":
            if not r.model:
                problems.append(_bad("model", "A forecast must name its model"))
            if any(p.start <= r.cutoff_date for p in r.forecast_points):
                problems.append(_bad("forecast_points", "Forecast points must lie after the cutoff"))
            if len(r.forecast_points) != r.horizon:
                problems.append(_bad("forecast_points", "Point count does not match the horizon"))
        return problems

    @staticmethod
    def _anomalies(r: AnomalyReport) -> list[Violation]:
        problems: list[Violation] = []
        if r.metric not in SERIES_METRIC_KEYS:
            problems.append(_bad("metric", "Unknown anomaly metric"))
        if r.detector not in DETECTOR_NAMES:
            problems.append(_bad("detector", "Unknown detector"))
        if any(res.threshold is None or res.period_start > r.evaluation_end for res in r.results):
            problems.append(_bad("results", "Anomaly results need a threshold and must lie in the evaluation window"))
        return problems

    def _sql(self, r: SQLResult) -> list[Violation]:
        problems: list[Violation] = []
        if r.row_count != len(r.rows) or r.row_count > r.max_rows or r.max_rows > self.limits.sql_row_limit:
            problems.append(_bad("rows", "Row count is inconsistent or exceeds the row limit"))
        if r.truncated and r.row_count < r.max_rows:
            problems.append(_bad("truncated", "Truncation flag contradicts the row count"))
        if any(len(row) != len(r.columns) for row in r.rows):
            problems.append(_bad("rows", "Row width does not match the columns"))
        withheld = self.exposure.all_withheld_columns
        for column in r.columns:
            if column.lower() in withheld or names_hidden_state(column):
                problems.append(_bad(f"columns.{column[:40]}", "Withheld column in SQL output", "data_policy"))
        return problems


# ---------------------------------------------------------------------------------------- response length


def draft_length(draft: ResponseDraftOutput) -> int:
    items = [*draft.key_findings, *draft.interpretation, *draft.recommendations]
    return len(draft.answer) + sum(len(i.text) for i in items)


def shrink_draft(draft: ResponseDraftOutput, max_chars: int) -> ResponseDraftOutput | None:
    """Fit a draft within ``max_chars`` by dropping whole items, never cutting text mid-sentence.

    Items are dropped from the lowest priority up: recommendations, then interpretation, then
    trailing key findings. The answer is never cut. If it alone is too long, ``None`` is returned
    and the caller fails closed.
    """
    if len(draft.answer) > max_chars:
        return None
    findings = list(draft.key_findings)
    interpretation = list(draft.interpretation)
    recommendations = list(draft.recommendations)
    shrunk = draft.model_copy(deep=True)
    for bucket in (recommendations, interpretation, findings):
        while bucket and draft_length(shrunk) > max_chars:
            bucket.pop()
            shrunk = draft.model_copy(
                update={"key_findings": findings, "interpretation": interpretation, "recommendations": recommendations}
            )
    return shrunk if draft_length(shrunk) <= max_chars else None
