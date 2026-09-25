"""Deterministic validation of the understood question against the real catalogue and data coverage.

Validation turns the model's ``UnderstandingOutput`` into a ``ValidatedRequest`` or a typed outcome:

- ``unsupported``: outside the dataset or the supported analytics (unknown metric or dimension).
- ``clarify``: an ambiguity that would materially change the answer, or an uninterpretable period.
- ``insufficient``: a valid question the data cannot answer (a future period, before the data
  starts, a forecast horizon beyond 6 months, a filter value that does not exist).

Documented defaults are applied and recorded as assumptions. A missing period means the latest
complete month (``last_month`` = 2026-08 for the business as-of date 2026-08-31), and a change
question without a comparison compares with the previous period of the same length.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from app.analytics.dimensions import DIMENSIONS, Filters
from app.analytics.errors import InvalidFilterValueError, InvalidPeriodError, UnsupportedDimensionError
from app.analytics.kpis import KPI_KEYS, KPI_REGISTRY
from app.analytics.periods import Period, previous_period, resolve_period
from app.forecasting.config import SUPPORTED_HORIZONS
from app.llm.schemas import Intent, UnderstandingOutput
from app.timeseries.metrics import SERIES_METRIC_KEYS

Outcome = Literal["valid", "unsupported", "clarify", "insufficient"]

_SERIES_INTENTS = (Intent.FORECAST, Intent.ANOMALY_DETECTION)
_CHANGE_INTENTS = (
    Intent.PERIOD_COMPARISON,
    Intent.REVENUE_INVESTIGATION,
    Intent.CUSTOMER_INVESTIGATION,
    Intent.SUPPORT_ANALYSIS,
    Intent.MIXED_INVESTIGATION,
)
_METRIC_REQUIRED = (Intent.KPI_LOOKUP, Intent.PERIOD_COMPARISON, Intent.FORECAST)


class ValidatedRequest(BaseModel):
    intent: Intent
    metric: str | None = None
    metric_name: str | None = None
    period: Period | None = None
    comparison_period: Period | None = None
    dimensions: list[str] = Field(default_factory=list)
    filters: dict[str, str] = Field(default_factory=dict)
    horizon: int | None = None
    analysis_type: str | None = None
    causal_question: bool = False
    assumptions: list[str] = Field(default_factory=list)


class RequestValidation(BaseModel):
    outcome: Outcome
    request: ValidatedRequest | None = None
    message: str | None = None


def validate_understanding(u: UnderstandingOutput, *, as_of: date, coverage: tuple[date, date]) -> RequestValidation:
    first, last = coverage
    if u.intent == Intent.UNSUPPORTED:
        return RequestValidation(outcome="unsupported", message=u.unsupported_reason or "The request is out of scope.")
    if u.material_ambiguity:
        detail = "; ".join(u.ambiguities) or "the question can be read in more than one materially different way"
        return RequestValidation(outcome="clarify", message=f"Please clarify: {detail}.")

    assumptions: list[str] = []
    metric = u.metric.strip().lower() if u.metric else None
    if metric is not None and metric not in KPI_REGISTRY:
        return RequestValidation(
            outcome="unsupported",
            message=f"The metric {u.metric!r} is not available. Supported KPIs: {', '.join(KPI_KEYS)}.",
        )
    if u.intent in _SERIES_INTENTS and metric is not None and metric not in SERIES_METRIC_KEYS:
        return RequestValidation(
            outcome="insufficient",
            message=(
                f"{metric} has no monthly series for forecasting or anomaly detection. "
                f"Supported metrics: {', '.join(SERIES_METRIC_KEYS)}."
            ),
        )
    if metric is None and u.intent in _METRIC_REQUIRED:
        return RequestValidation(
            outcome="clarify", message="Please say which metric you mean (for example revenue or MRR)."
        )

    for dimension in u.dimensions:
        if dimension not in DIMENSIONS:
            return RequestValidation(
                outcome="unsupported",
                message=f"The dimension {dimension!r} is not supported. Supported: {', '.join(DIMENSIONS)}.",
            )
    try:
        filters = Filters(**{f.dimension: f.value for f in u.filters}).active()
    except InvalidFilterValueError as exc:
        return RequestValidation(outcome="insufficient", message=f"No data exists for that filter: {exc}")
    except (UnsupportedDimensionError, TypeError, ValueError) as exc:
        return RequestValidation(outcome="unsupported", message=f"Unsupported filter: {exc}")

    horizon: int | None = None
    period: Period | None = None
    comparison: Period | None = None
    if u.intent == Intent.FORECAST:
        horizon = u.horizon if u.horizon is not None else 1
        if u.horizon is None:
            assumptions.append("No horizon given; forecasting the next month.")
        if horizon not in SUPPORTED_HORIZONS:
            return RequestValidation(
                outcome="insufficient",
                message=(
                    f"A {horizon}-month horizon is not supported: forecasts cover 1 to "
                    f"{SUPPORTED_HORIZONS.stop - 1} months after the latest data ({last.isoformat()})."
                ),
            )
    else:
        spec = u.period
        if spec is None:
            spec = "last_month"
            assumptions.append(_default_period_note(as_of))
        try:
            period = resolve_period(spec, as_of=as_of)
        except InvalidPeriodError as exc:
            return RequestValidation(outcome="clarify", message=f"The period could not be interpreted: {exc}")
        issue = _coverage_issue(period, first, last, as_of)
        if issue:
            return RequestValidation(outcome="insufficient", message=issue)
        wants_change = u.intent in _CHANGE_INTENTS or u.analysis_type in ("change", "contribution")
        if u.comparison_period is not None:
            try:
                comparison = resolve_period(u.comparison_period, as_of=as_of)
            except InvalidPeriodError as exc:
                return RequestValidation(
                    outcome="clarify", message=f"The comparison period could not be interpreted: {exc}"
                )
        elif wants_change:
            comparison = previous_period(period)
            assumptions.append(f"Comparing {_label(period)} with the previous period, {_label(comparison)}.")
        if comparison is not None:
            issue = _coverage_issue(comparison, first, last, as_of)
            if issue:
                return RequestValidation(outcome="insufficient", message=f"Comparison period: {issue}")

    definition = KPI_REGISTRY.get(metric) if metric else None
    return RequestValidation(
        outcome="valid",
        request=ValidatedRequest(
            intent=u.intent,
            metric=metric,
            metric_name=definition.name if definition else None,
            period=period,
            comparison_period=comparison,
            dimensions=list(dict.fromkeys(u.dimensions)),
            filters=filters,
            horizon=horizon,
            analysis_type=u.analysis_type,
            causal_question=u.analysis_type == "causal",
            assumptions=assumptions,
        ),
    )


def _label(period: Period) -> str:
    return period.label if " to " not in period.label else f"{period.start.isoformat()} to {period.end.isoformat()}"


def _default_period_note(as_of: date) -> str:
    latest = resolve_period("last_month", as_of=as_of)
    return (
        f"No period given; using the latest complete month, {_label(latest)} (business as-of date {as_of.isoformat()})."
    )


def _coverage_issue(period: Period, first: date, last: date, as_of: date) -> str | None:
    if period.start > min(last, as_of):
        return (
            f"{_label(period)} is after the latest available data ({last.isoformat()}); future periods cannot be "
            "observed. A forecast can estimate up to 6 months ahead."
        )
    if period.end < first:
        return f"{_label(period)} is before the data begins ({first.isoformat()})."
    return None
