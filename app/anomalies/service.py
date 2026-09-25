"""The anomaly service: validate, prepare the series, run one detector, attach explanations and provenance.

``AnomalyService`` knows nothing about agents, LLMs, prompts, MCP, HTTP or UIs. The pure core,
``detect_in_series``, works on a prepared ``TimeSeries``.

Scope is always explicit: a caller names a metric, a detector and, optionally, one set of group
filters (for example ``segment=SMB`` or ``region=EMEA``). Dimension combinations are
never scanned automatically. ``scan`` runs one detector over an explicit list of metrics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from app.analytics.dimensions import Filters
from app.analytics.errors import InvalidPeriodError, InvalidRequestError
from app.analytics.kpis import KPIService
from app.analytics.models import Provenance
from app.anomalies.config import DETECTOR_NAMES, AnomalyConfig, DetectorName
from app.anomalies.detectors import Assessment, DetectorOutput, forecast_residual, iqr_detector, rolling_zscore
from app.anomalies.models import (
    AnomalyReport,
    AnomalyResult,
    AnomalyScan,
    DetectorSpecification,
    Direction,
    ForecastResidualDetails,
    HistoricalWindow,
    IQRDetails,
    RollingZScoreDetails,
    SkippedPeriod,
)
from app.anomalies.thresholds import flag_threshold, is_flagged, severity_policy_text
from app.database.base import Database
from app.database.lineage import new_tool_run_id
from app.timeseries.calendar import add_months, last_complete_month_end, month_start
from app.timeseries.errors import SeriesStatus, UnsupportedMethodError
from app.timeseries.metrics import SERIES_METRICS, Transform, get_series_metric
from app.timeseries.models import TimeSeries
from app.timeseries.preparation import prepare_monthly_series

DEFAULT_EVALUATION_MONTHS = 12
DIRECTION_LIMITATION = (
    "Direction is statistical: 'positive' means above the expected value and 'negative' below it. It does not "
    "mean favourable or unfavourable (for example, higher ticket volume is 'positive')."
)
CAUSALITY_LIMITATION = (
    "An anomaly is a movement that is unusual relative to the preceding months. It does not establish a cause, "
    "and it is not by itself evidence of a business failure."
)
CHANGE_LIMITATION = (
    "Scores compare month-over-month changes, so the month after an unusual month can also score as unusual "
    "when the metric moves back towards its previous level."
)


class AnomalyService:
    def __init__(self, db: Database, *, config: AnomalyConfig | None = None, as_of: date | None = None):
        self.db = db
        self.config = config or AnomalyConfig()
        self.kpi = KPIService(db, as_of=as_of)
        self.as_of = self.kpi.as_of

    def detect(
        self,
        metric: str,
        start_date: date | None = None,
        end_date: date | None = None,
        detector: str | None = None,
        filters: Filters | Mapping[str, str] | None = None,
        window: int | None = None,
        *,
        transform: Transform | None = None,
        config: AnomalyConfig | None = None,
    ) -> AnomalyReport:
        cfg = _configure(config or self.config, detector, window, transform)
        get_series_metric(metric)
        evaluation_start, evaluation_end = self._evaluation_range(start_date, end_date)
        series = prepare_monthly_series(self.db, metric, filters=filters, end_date=evaluation_end, kpi_service=self.kpi)
        return detect_in_series(series, cfg, evaluation_start, evaluation_end)

    def scan(
        self,
        metrics: Sequence[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        detector: str | None = None,
        filters: Filters | Mapping[str, str] | None = None,
        window: int | None = None,
    ) -> AnomalyScan:
        """Run one detector over an explicit list of metrics. Default: every metric without a required filter."""
        chosen = (
            list(metrics) if metrics is not None else [k for k, m in SERIES_METRICS.items() if not m.required_filters]
        )
        if not chosen:
            raise InvalidRequestError("scan needs at least one metric")
        reports = [self.detect(m, start_date, end_date, detector, filters, window) for m in chosen]
        return AnomalyScan(
            detector=reports[0].detector,
            evaluation_start=reports[0].evaluation_start,
            evaluation_end=reports[0].evaluation_end,
            filters=reports[0].filters,
            reports=reports,
        )

    def _evaluation_range(self, start_date: date | None, end_date: date | None) -> tuple[date, date]:
        end = end_date or self.as_of
        if end > self.as_of:
            raise InvalidPeriodError(
                f"end_date {end.isoformat()} is after the business as-of date {self.as_of.isoformat()}: "
                "observations after the as-of date do not exist."
            )
        evaluation_end = last_complete_month_end(end)
        evaluation_start = (
            month_start(start_date)
            if start_date is not None
            else add_months(month_start(evaluation_end), -(DEFAULT_EVALUATION_MONTHS - 1))
        )
        if evaluation_start > evaluation_end:
            raise InvalidPeriodError(
                f"The evaluation range {evaluation_start.isoformat()} to {evaluation_end.isoformat()} contains no "
                "complete month."
            )
        return evaluation_start, evaluation_end


def detect_anomalies(
    db: Database,
    metric: str,
    start_date: date | None = None,
    end_date: date | None = None,
    detector: str = "rolling_zscore",
    filters: Filters | Mapping[str, str] | None = None,
    window: int = 12,
    *,
    transform: Transform | None = None,
    config: AnomalyConfig | None = None,
    as_of: date | None = None,
) -> AnomalyReport:
    """Detect anomalies in one metric's monthly series (convenience wrapper around ``AnomalyService``)."""
    return AnomalyService(db, config=config, as_of=as_of).detect(
        metric, start_date, end_date, detector, filters, window, transform=transform
    )


def detect_in_series(
    series: TimeSeries, config: AnomalyConfig, evaluation_start: date, evaluation_end: date
) -> AnomalyReport:
    """Score every month of the series inside ``[evaluation_start, evaluation_end]`` (no database access)."""
    if series.end is not None and series.end > evaluation_end:
        raise InvalidRequestError(
            f"The series ends {series.end.isoformat()}, after the evaluation end {evaluation_end.isoformat()}."
        )
    spec = get_series_metric(series.metric)
    detector = config.detector
    transform: Transform | None = (
        None if detector == "forecast_residual" else (config.transform or spec.anomaly_transform)
    )
    limitations = [DIRECTION_LIMITATION, CAUSALITY_LIMITATION]
    if transform in ("difference", "pct_change"):
        limitations.append(CHANGE_LIMITATION)
    limitations.extend(series.limitations)
    method = _specification(detector, config, transform)
    calculation = (
        f"{spec.name}: {method.description} Severity: {method.severity_policy} "
        f"Evaluated {evaluation_start.isoformat()} to {evaluation_end.isoformat()}. Series: {series.calculation}"
    )
    provenance = Provenance(
        operation=f"detect_anomalies:{series.metric}",
        operation_id=new_tool_run_id(),
        calculation=calculation,
        dataset_version=series.provenance.dataset_version,
        queries=list(series.provenance.queries),
    )

    def report(
        status: SeriesStatus,
        message: str | None,
        results: list[AnomalyResult] | None = None,
        skipped: list[SkippedPeriod] | None = None,
    ) -> AnomalyReport:
        return AnomalyReport(
            metric=series.metric,
            metric_name=series.metric_name,
            unit=series.unit,
            detector=detector,
            status=status,
            message=message,
            filters=series.filters,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            method=method,
            results=results or [],
            skipped=skipped or [],
            calculation=calculation,
            limitations=limitations,
            series=series,
            provenance=provenance,
        )

    if series.status != "ok":
        return report(series.status, series.message or "No observations for this metric and these filters.")
    indices = [i for i, p in enumerate(series.points) if p.start >= evaluation_start and p.end <= evaluation_end]
    if not indices:
        return report("no_data", "No month of the series falls inside the evaluation range.")

    values, labels = series.values(), series.labels()
    output = _run(detector, values, indices, config, transform, labels)
    skipped = [SkippedPeriod(period=labels[s.index], reason=s.reason) for s in output.skipped]
    if not output.assessments:
        return report(
            "insufficient_history",
            f"No month in the evaluation range could be scored: at least {config.min_history} prior observations "
            f"within a {config.window}-month window are required.",
            skipped=skipped,
        )
    results = [_result(series, a, config, provenance, labels, limitations) for a in output.assessments]
    return report("ok", None, _with_runs(results), skipped)


# ---------------------------------------------------------------------------------------------------


def _configure(
    base: AnomalyConfig, detector: str | None, window: int | None, transform: Transform | None
) -> AnomalyConfig:
    if detector is not None and detector not in DETECTOR_NAMES:
        raise UnsupportedMethodError(f"Unknown anomaly detector {detector!r}. Supported: {', '.join(DETECTOR_NAMES)}")
    updates: dict[str, Any] = {}
    if detector is not None:
        updates["detector"] = detector
    if window is not None:
        updates["window"] = window
        updates["min_history"] = min(base.min_history, window)
    if transform is not None:
        updates["transform"] = transform
    return AnomalyConfig.model_validate({**base.model_dump(), **updates}) if updates else base


def _run(
    detector: DetectorName,
    values: Any,
    indices: list[int],
    config: AnomalyConfig,
    transform: Transform | None,
    labels: list[str],
) -> DetectorOutput:
    if detector == "rolling_zscore":
        assert transform is not None
        return rolling_zscore(values, indices, config, transform)
    if detector == "iqr":
        assert transform is not None
        return iqr_detector(values, indices, config, transform)
    return forecast_residual(values, indices, config, labels)


_TRANSFORM_TEXT = {
    "level": "value",
    "difference": "month-over-month change",
    "pct_change": "month-over-month percentage change",
}


def _specification(detector: DetectorName, config: AnomalyConfig, transform: Transform | None) -> DetectorSpecification:
    w = config.window
    if detector == "rolling_zscore":
        description = (
            f"Rolling z-score: the month's {_TRANSFORM_TEXT[transform or 'level']} minus the mean of the previous "
            f"{w} months, divided by their sample standard deviation (ddof=1). The month itself is excluded."
        )
    elif detector == "iqr":
        description = (
            f"Robust IQR: the month's {_TRANSFORM_TEXT[transform or 'level']} against Tukey fences from the "
            f"quartiles of the previous {w} months (linear interpolation). The month itself is excluded."
        )
    else:
        description = (
            f"Forecast residual: a one-step {config.expectation_model} forecast fitted on the previous {w} months; the "
            "residual is standardised by the mean and standard deviation of the prior one-step residuals in the "
            "window. The month itself is excluded from the fit and from the residual statistics."
        )
    return DetectorSpecification(
        detector=detector,
        description=description,
        transform=transform,
        window=w,
        min_history=config.min_history,
        threshold=flag_threshold(detector, config),
        severity_policy=severity_policy_text(detector, config),
        expectation_model=config.expectation_model if detector == "forecast_residual" else None,
        configuration=config,
    )


def _result(
    series: TimeSeries,
    a: Assessment,
    config: AnomalyConfig,
    provenance: Provenance,
    labels: list[str],
    limitations: list[str],
) -> AnomalyResult:
    point = series.points[a.index]
    direction: Direction = "positive" if a.deviation > 0 else "negative" if a.deviation < 0 else "none"
    window = HistoricalWindow(
        start=labels[a.window_start],
        end=labels[a.window_end],
        observations=a.window_observations,
        window_months=config.window,
    )
    notes = list(limitations)
    if a.window_observations < config.window:
        notes.append(
            f"The baseline uses {a.window_observations} observations, fewer than the {config.window}-month window."
        )
    percentage = a.deviation / abs(a.expected) if a.expected != 0 else None
    flagged = is_flagged(a.severity, config.flag_severity)
    return AnomalyResult(
        metric=series.metric,
        metric_name=series.metric_name,
        unit=series.unit,
        period=point.period,
        period_start=point.start,
        period_end=point.end,
        observed_value=a.observed,
        expected_value=a.expected,
        deviation=a.deviation,
        deviation_percentage=percentage,
        score=a.score,
        threshold=a.threshold,
        lower_bound=a.lower_bound,
        upper_bound=a.upper_bound,
        detector=config.detector,
        severity=a.severity,
        direction=direction,
        is_anomaly=flagged,
        historical_window=window,
        explanation=_explain(point.period, a, direction, window, flagged),
        filters=series.filters,
        source_tables=provenance.source_tables,
        calculation=provenance.calculation,
        query_id=series.query_id,
        operation_id=provenance.operation_id,
        execution_timestamp=provenance.execution_timestamp,
        limitations=notes,
        details=a.details,
    )


def _with_runs(results: list[AnomalyResult]) -> list[AnomalyResult]:
    """Set ``anomaly_start``: the first month of each consecutive, same-direction run of flagged months."""
    out: list[AnomalyResult] = []
    run_start: str | None = None
    previous: AnomalyResult | None = None
    for result in results:
        if result.is_anomaly:
            continues = (
                previous is not None
                and previous.is_anomaly
                and previous.direction == result.direction
                and _consecutive(previous.period_end, result.period_start)
            )
            run_start = run_start if continues else result.period
            out.append(result.model_copy(update={"anomaly_start": run_start}))
        else:
            run_start = None
            out.append(result)
        previous = result
    return out


def _consecutive(previous_end: date, start: date) -> bool:
    return (start - previous_end).days == 1


def _fmt(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:,.2f}"
    return f"{value:.4f}"


def _signed(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}{_fmt(abs(value))}"


def _fmt_transformed(value: float, transform: Transform, *, signed: bool = True) -> str:
    if transform == "pct_change":
        return f"{value:+.2%}" if signed else f"{value:.2%}"
    return _signed(value) if signed else _fmt(value)


def _explain(period: str, a: Assessment, direction: Direction, window: HistoricalWindow, flagged: bool) -> str:
    percentage = f", {a.deviation / abs(a.expected):+.1%}" if a.expected else ""
    head = f"{period}: observed {_fmt(a.observed)} vs expected {_fmt(a.expected)} ({_signed(a.deviation)}{percentage})."
    span = f"{window.observations} prior months ({window.start} to {window.end})"
    d = a.details
    if isinstance(d, RollingZScoreDetails):
        basis = (
            f" This month's {_TRANSFORM_TEXT[d.transform]} {_fmt_transformed(d.transformed_value, d.transform)} vs a "
            f"mean of {_fmt_transformed(d.baseline_mean, d.transform)} "
            f"(std {_fmt_transformed(d.baseline_std, d.transform, signed=False)}) "
            f"over {span}."
        )
    elif isinstance(d, IQRDetails):
        basis = (
            f" This month's {_TRANSFORM_TEXT[d.transform]} {_fmt_transformed(d.transformed_value, d.transform)}; "
            f"over {span} Q1 {_fmt_transformed(d.q1, d.transform)}, Q3 {_fmt_transformed(d.q3, d.transform)}, "
            f"IQR {_fmt_transformed(d.iqr, d.transform, signed=False)}."
        )
    else:
        assert isinstance(d, ForecastResidualDetails)
        basis = (
            f" One-step {d.expectation_model} forecast {_fmt(d.one_step_forecast)} (fitted on {d.training_start} to "
            f"{d.training_end}) plus the mean prior residual {_signed(d.residual_mean)}; residual std "
            f"{_fmt(d.residual_std)} over {span}."
        )
    score = "undefined (the prior window has no variation)" if a.score is None else f"{a.score:+.2f}"
    verdict = "flagged" if flagged else "not flagged"
    bounds = (
        f" Values outside {_fmt(a.lower_bound)} to {_fmt(a.upper_bound)} are flagged."
        if a.lower_bound is not None and a.upper_bound is not None
        else ""
    )
    return (
        f"{head}{basis} Score {score}; flag threshold {a.threshold:g}: {a.severity}, {verdict}, "
        f"direction {direction}.{bounds}"
    )
