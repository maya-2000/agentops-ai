"""The forecasting service: validate, prepare the series, backtest, select, fit, forecast, attach provenance.

``ForecastService`` knows nothing about agents, LLMs, prompts, MCP, HTTP or UIs. The pure core,
``forecast_series``, works on a prepared ``TimeSeries``, so it can be tested without a database.

Point-in-time rules:

- The series ends at the last complete month on or before the cutoff. The cutoff defaults to
  the business as-of date (``Settings.as_of_date``), never the machine clock, and later cutoffs are rejected.
- Backtests, selection and the final fit use only that series. The final model is refitted on
  all of it, and the forecast covers the ``horizon`` months after the series ends.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np

from app.analytics.dimensions import Filters
from app.analytics.errors import InvalidRequestError
from app.analytics.kpis import KPIService
from app.analytics.models import Provenance
from app.database.base import Database
from app.database.lineage import new_tool_run_id
from app.forecasting.backtest import BacktestResult
from app.forecasting.base import MethodOutput, MethodUnavailableError
from app.forecasting.config import BASELINE_MODEL, ForecastConfig
from app.forecasting.methods import build_method
from app.forecasting.models import (
    AUTO_MODEL,
    CandidateSummary,
    ForecastPoint,
    ForecastRequest,
    ForecastResult,
    ModelSpecification,
    PredictionInterval,
    SeasonalityAssessment,
)
from app.forecasting.selection import (
    Selection,
    SelectionEvaluation,
    backtest_candidates,
    evaluate_selection,
    select_model,
)
from app.timeseries.calendar import following_months
from app.timeseries.errors import SeriesStatus
from app.timeseries.metrics import get_series_metric
from app.timeseries.models import TimeSeries
from app.timeseries.preparation import prepare_monthly_series

INTERVAL_LIMITATION = (
    "Bounds are prediction intervals for individual future months, not confidence intervals for a mean. They "
    "assume normally distributed, uncorrelated errors and ignore parameter-estimation uncertainty, so they can "
    "be too narrow; the backtest interval coverage shows how they performed historically."
)
SELECTION_LIMITATION = (
    "The selected model's backtest metrics were also used to select it, so they are optimistic. The naive "
    "baseline's metrics are not affected by selection."
)
EXTRAPOLATION_LIMITATION = (
    "Forecasts extrapolate the observed monthly history. They cannot anticipate events that have not yet "
    "affected the data, and they do not explain why the metric moves."
)


class ForecastService:
    def __init__(self, db: Database, *, config: ForecastConfig | None = None, as_of: date | None = None):
        self.db = db
        self.config = config or ForecastConfig()
        self.kpi = KPIService(db, as_of=as_of)
        self.as_of = self.kpi.as_of

    def prepare(self, request: ForecastRequest) -> TimeSeries:
        return prepare_monthly_series(
            self.db,
            request.metric,
            filters=request.filters,
            end_date=request.cutoff_date or self.as_of,
            kpi_service=self.kpi,
        )

    def forecast(self, request: ForecastRequest | dict[str, Any] | None = None, **kwargs: Any) -> ForecastResult:
        req = _request(request, kwargs)
        get_series_metric(req.metric)
        cutoff = req.cutoff_date or self.as_of
        return forecast_series(self.prepare(req), req, self.config, cutoff=cutoff)

    def evaluate_selection(
        self,
        metric: str,
        *,
        horizon: int = 3,
        cutoff_date: date | None = None,
        filters: Filters | dict[str, str] | None = None,
    ) -> SelectionEvaluation:
        """Out-of-sample evaluation of the backtest-based selection rule (nested rolling origins)."""
        req = _request({"metric": metric, "horizon": horizon, "cutoff_date": cutoff_date, "filters": filters or {}}, {})
        series = self.prepare(req)
        points = series.points[series.contiguous_tail_start() :]
        values = [float(p.value) for p in points if p.value is not None]
        return evaluate_selection(values, [p.period for p in points], self.config, horizon)


def forecast_metric(
    db: Database,
    metric: str,
    horizon: int = 3,
    *,
    cutoff_date: date | None = None,
    filters: Filters | dict[str, str] | None = None,
    model: str = AUTO_MODEL,
    confidence_level: float | None = None,
    config: ForecastConfig | None = None,
    as_of: date | None = None,
) -> ForecastResult:
    """Forecast a supported metric (convenience wrapper around ``ForecastService``)."""
    return ForecastService(db, config=config, as_of=as_of).forecast(
        metric=metric,
        horizon=horizon,
        cutoff_date=cutoff_date,
        filters=filters or {},
        model=model,
        confidence_level=confidence_level,
    )


def forecast_series(
    series: TimeSeries, request: ForecastRequest, config: ForecastConfig, *, cutoff: date
) -> ForecastResult:
    """Backtest, select and forecast from an already-prepared series (no database access)."""
    if series.end is not None and series.end > cutoff:
        raise InvalidRequestError(
            f"The series ends {series.end.isoformat()}, after the cutoff {cutoff.isoformat()}: refusing to "
            "forecast with information from after the cutoff."
        )
    spec = get_series_metric(series.metric)
    level = request.confidence_level or config.confidence_level
    horizon = request.horizon
    limitations = [EXTRAPOLATION_LIMITATION, *series.limitations]

    def stop(status: SeriesStatus, message: str) -> ForecastResult:
        return _result(series, request, config, cutoff, level, status=status, message=message, limitations=limitations)

    if series.status != "ok":
        return stop(series.status, series.message or "No observations for this metric and these filters.")

    tail = series.contiguous_tail_start()
    points = series.points[tail:]
    if 0 < tail < len(series.points):
        limitations.append(
            f"{tail} month(s) up to {series.points[tail - 1].period} are excluded: {series.points[tail - 1].period} "
            "is missing and the models need a gap-free series, so only the months after it are used."
        )
    required = config.required_history(horizon)
    if len(points) < required:
        window = f" ({points[0].period} to {points[-1].period})" if points else ""
        return stop(
            "insufficient_history",
            f"{len(points)} usable month(s){window}. A {horizon}-month forecast needs at least {required}: "
            f"{config.minimum_history} months for the first backtest training window, then "
            f"{config.min_backtest_folds} rolling-origin fold(s) with {horizon} validation month(s).",
        )

    values = np.array([p.value for p in points], dtype=float)
    labels = [p.period for p in points]
    zero_share = float(np.mean(values == 0))
    if zero_share > config.max_zero_share:
        return stop(
            "insufficient_data",
            f"{zero_share:.0%} of the {len(points)} months are zero (limit {config.max_zero_share:.0%}). The series is "
            "intermittent; the implemented models assume a continuous level, so no forecast is produced.",
        )

    candidates = list(config.candidates) if request.model == AUTO_MODEL else _unique([BASELINE_MODEL, request.model])
    backtests = backtest_candidates(values, labels, config, horizon, candidates)
    if request.model == AUTO_MODEL:
        selection = select_model(backtests, config)
    else:
        selection = _explicit_selection(request.model, backtests, config)

    chosen = selection.model
    try:
        output = build_method(chosen, config).forecast(values, horizon, level)
    except MethodUnavailableError as exc:
        if request.model != AUTO_MODEL:
            return stop("insufficient_history", f"The requested model {chosen} cannot be fitted: {exc}")
        limitations.append(f"{chosen} could not be refitted on the full history ({exc}); using the naive baseline.")
        chosen = BASELINE_MODEL
        selection = Selection(
            model=BASELINE_MODEL,
            reason=f"{selection.reason} Refit failed, so the naive baseline is used.",
            baseline_model=BASELINE_MODEL,
            improvement_over_baseline=0.0,
        )
        output = build_method(chosen, config).forecast(values, horizon, level)

    forecast_points, clipped = _points(output, points[-1].end, horizon, spec.lower_limit, spec.upper_limit)
    if clipped:
        possible = _range(spec.lower_limit, spec.upper_limit)
        limitations.append(
            f"Forecast values or bounds outside the metric's possible range ({possible}) were limited to that range."
        )
    interval_available = output.lower is not None
    if interval_available:
        limitations.append(INTERVAL_LIMITATION)
        coverage = backtests[chosen].metrics
        if coverage.interval_coverage is not None:
            limitations.append(
                f"In the historical backtest, {chosen}'s {level:.0%} intervals contained "
                f"{coverage.interval_coverage:.0%} of {coverage.interval_sample_count} actual month(s)."
            )
    else:
        limitations.append(f"No prediction interval: {output.interval_note or 'not available for this model'}")
    if request.model == AUTO_MODEL:
        limitations.append(SELECTION_LIMITATION)
    limitations.extend(f"Model fit note: {note}" for note in output.notes)

    method = build_method(chosen, config)
    seasonality = _seasonality(len(points), config.season_length)
    specification = ModelSpecification(
        model=chosen,
        display_name=method.display_name,
        description=method.description,
        parameters=output.parameters,
        configuration=config,
        training_start=points[0].start,
        training_end=points[-1].end,
        training_observations=len(points),
        cutoff_date=cutoff,
        random_seed=config.random_seed,
        notes=list(output.notes),
    )
    calculation = (
        f"{method.display_name} fitted on {len(points)} months ({labels[0]} to {labels[-1]}) of: {series.calculation} "
        f"Forecast for the {horizon} month(s) after {labels[-1]}. "
        + (
            f"{level:.0%} prediction interval, {method.interval_method}."
            if interval_available
            else "No prediction interval."
        )
    )
    return _result(
        series,
        request,
        config,
        cutoff,
        level,
        status="ok",
        message=None,
        limitations=limitations,
        model=chosen,
        points=forecast_points,
        interval=PredictionInterval(
            confidence_level=level,
            available=interval_available,
            method=method.interval_method if interval_available else None,
            note=output.interval_note,
        ),
        history_points=len(points),
        historical=(points[0].start, points[-1].end),
        seasonality=seasonality,
        selection=selection,
        backtests=[backtests[name] for name in candidates],
        specification=specification,
        calculation=calculation,
    )


# ---------------------------------------------------------------------------------------------------


def _request(request: ForecastRequest | dict[str, Any] | None, overrides: dict[str, Any]) -> ForecastRequest:
    if isinstance(request, ForecastRequest) and not overrides:
        return request
    base = request.model_dump() if isinstance(request, ForecastRequest) else dict(request or {})
    merged = {**base, **overrides}
    if merged.get("filters") is None:
        merged["filters"] = {}
    return ForecastRequest.model_validate(merged)


def _unique(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))


def _explicit_selection(model: str, backtests: dict[str, BacktestResult], config: ForecastConfig) -> Selection:
    chosen, baseline = backtests[model].metrics, backtests[BASELINE_MODEL].metrics
    metric = config.selection_metric
    chosen_value = getattr(chosen, metric)
    baseline_value = getattr(baseline, metric)
    if model == BASELINE_MODEL:
        comparison = "it is the baseline."
    elif chosen_value is None or baseline_value is None or not backtests[model].eligible:
        comparison = "it could not be backtested on every fold, so no comparison with the naive baseline is available."
    else:
        relation = "lower" if chosen_value < baseline_value else "not lower"
        comparison = (
            f"its historical backtest {metric.upper()} ({chosen_value:,.4g}) was {relation} than the naive "
            f"baseline's ({baseline_value:,.4g})."
        )
    improvement = None
    if chosen_value is not None and baseline_value:
        improvement = (baseline_value - chosen_value) / baseline_value
    return Selection(
        model=model,
        reason=f"{model} was requested explicitly (no automatic selection); {comparison}",
        baseline_model=BASELINE_MODEL,
        improvement_over_baseline=improvement,
    )


def _points(
    output: MethodOutput, last_end: date, horizon: int, lower_limit: float | None, upper_limit: float | None
) -> tuple[list[ForecastPoint], bool]:
    clipped = False

    def limit(value: float | None) -> float | None:
        nonlocal clipped
        if value is None:
            return None
        limited = value
        if lower_limit is not None and limited < lower_limit:
            limited = lower_limit
        if upper_limit is not None and limited > upper_limit:
            limited = upper_limit
        clipped = clipped or limited != value
        return limited

    months = following_months(last_end, horizon)
    points = []
    for step, month in enumerate(months):
        predicted = limit(output.mean[step])
        assert predicted is not None
        points.append(
            ForecastPoint(
                period=month.label,
                start=month.start,
                end=month.end,
                predicted_value=predicted,
                lower_bound=limit(output.lower[step]) if output.lower is not None else None,
                upper_bound=limit(output.upper[step]) if output.upper is not None else None,
            )
        )
    return points, clipped


def _range(lower: float | None, upper: float | None) -> str:
    low = "-inf" if lower is None else f"{lower:g}"
    high = "+inf" if upper is None else f"{upper:g}"
    return f"{low} to {high}"


def _seasonality(observations: int, season_length: int) -> SeasonalityAssessment:
    cycles = observations // season_length
    estimable = cycles >= 2
    return SeasonalityAssessment(
        season_length=season_length,
        history_observations=observations,
        full_cycles=cycles,
        estimable=estimable,
        seasonal_naive_available=observations >= season_length,
        note=(
            f"{cycles} full {season_length}-month cycle(s) of history. "
            + (
                "A seasonal pattern is observed at least twice, but no seasonal statistical model is fitted: "
                "backtest folds have fewer than two cycles, so it could not be validated. "
                if estimable
                else "Seasonality cannot be estimated from fewer than two cycles. "
            )
            + "The seasonal-naive baseline repeats the value from one season earlier."
        ),
    )


def _result(
    series: TimeSeries,
    request: ForecastRequest,
    config: ForecastConfig,
    cutoff: date,
    level: float,
    *,
    status: SeriesStatus,
    message: str | None,
    limitations: list[str],
    model: str | None = None,
    points: list[ForecastPoint] | None = None,
    interval: PredictionInterval | None = None,
    history_points: int = 0,
    historical: tuple[date, date] | None = None,
    seasonality: SeasonalityAssessment | None = None,
    selection: Selection | None = None,
    backtests: list[BacktestResult] | None = None,
    specification: ModelSpecification | None = None,
    calculation: str | None = None,
) -> ForecastResult:
    calc = calculation or f"No forecast ({status}). Series: {series.calculation}"
    provenance = Provenance(
        operation=f"forecast:{series.metric}",
        operation_id=new_tool_run_id(),
        calculation=calc,
        dataset_version=series.provenance.dataset_version,
        queries=list(series.provenance.queries),
    )
    backtest_list = backtests or []
    return ForecastResult(
        metric=series.metric,
        metric_name=series.metric_name,
        unit=series.unit,
        status=status,
        message=message,
        model=model,
        cutoff_date=cutoff,
        horizon=request.horizon,
        confidence_level=level,
        forecast_points=points or [],
        interval=interval,
        historical_start=historical[0] if historical else series.start,
        historical_end=historical[1] if historical else series.end,
        history_observations=history_points,
        sufficient_history=status == "ok",
        seasonality=seasonality,
        selected_model_reason=selection.reason if selection else None,
        improvement_over_baseline=selection.improvement_over_baseline if selection else None,
        backtests=backtest_list,
        candidates=[
            CandidateSummary(model=b.model, eligible=b.eligible, metrics=b.metrics, failures=b.failures)
            for b in backtest_list
        ],
        method=specification,
        filters=series.filters,
        calculation=calc,
        limitations=limitations,
        history=series,
        provenance=provenance,
    )
