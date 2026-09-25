"""Centralised preparation of monthly business time series.

This is the only Phase 3 code that reads the database, and it does so exclusively through
Phase 2: ``KPIService.calculate_kpi`` (a KPI broken down by month) or ``revenue.mrr_series``
(month-end MRR and active customers). Aggregation happens in DuckDB; Python receives at most
one row per month.

Point-in-time guarantee: the queries are bounded by ``end_date`` (the cutoff). The series
ends at the last complete month on or before the cutoff, and no row after the cutoff is read.
A cutoff after the business as-of date is rejected: those observations do not exist yet.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from app.analytics import revenue
from app.analytics.dimensions import Filters
from app.analytics.errors import InvalidPeriodError
from app.analytics.executor import QueryRunner
from app.analytics.kpis import KPIService, get_kpi_definition
from app.analytics.models import Provenance
from app.analytics.periods import explicit_period
from app.database.base import Database
from app.timeseries.calendar import first_complete_month_start, last_complete_month_end, months_between
from app.timeseries.metrics import SeriesMetric, get_series_metric, validate_series_filters
from app.timeseries.models import TimeSeries, TimeSeriesPoint


def prepare_monthly_series(
    db: Database,
    metric: str,
    *,
    filters: Filters | Mapping[str, str] | None = None,
    end_date: date | None = None,
    start_date: date | None = None,
    as_of: date | None = None,
    kpi_service: KPIService | None = None,
) -> TimeSeries:
    """Build the monthly series of ``metric`` for complete months in ``[start_date, end_date]``.

    ``end_date`` is the information cutoff (default: the business as-of date). ``start_date``
    defaults to the start of the data coverage.
    """
    spec = get_series_metric(metric)
    active = validate_series_filters(spec, filters)
    service = kpi_service or KPIService(db, as_of=as_of)
    business_date = service.as_of
    cutoff = end_date or business_date
    if cutoff > business_date:
        raise InvalidPeriodError(
            f"The cutoff {cutoff.isoformat()} is after the business as-of date {business_date.isoformat()}: "
            "observations after the as-of date do not exist."
        )
    if start_date is not None and start_date > cutoff:
        raise InvalidPeriodError(f"start_date {start_date.isoformat()} is after the cutoff {cutoff.isoformat()}")

    first, last = service.coverage()
    window_start = first_complete_month_start(max(start_date or first, first))
    window_end = last_complete_month_end(min(cutoff, last))
    if window_end < window_start:
        runner = QueryRunner(db, f"timeseries:{spec.key}")
        return _series(
            spec,
            active,
            [],
            runner.provenance("no complete calendar month in the requested window"),
            calculation=_calculation(spec, active, None, None),
            status="no_data",
            message=(
                f"No complete calendar month lies between {window_start.isoformat()} and "
                f"{window_end.isoformat()} within the data coverage ({first.isoformat()} to {last.isoformat()})."
            ),
        )

    values, provenance, first_observed = _fetch(db, service, spec, active, window_start, window_end)
    notes: list[str] = []
    months = months_between(window_start, window_end)
    if not spec.empty_month_is_zero:
        observed = [m for m in months if m.label in values]
        if observed and observed[0].label != months[0].label:
            notes.append(
                f"The series starts in {observed[0].label}, the first month with observations; earlier months "
                "have no observations and are not treated as zero."
            )
            months = [m for m in months if m.start >= observed[0].start]
    if first_observed is not None and months and first_observed > months[0].start:
        notes.append(
            f"The first month ({months[0].label}) is partial: the first observation is {first_observed.isoformat()}."
        )

    points: list[TimeSeriesPoint] = []
    for month in months:
        raw = values.get(month.label)
        if raw is not None:
            points.append(
                TimeSeriesPoint(period=month.label, start=month.start, end=month.end, value=raw, observation="observed")
            )
        elif spec.empty_month_is_zero:
            points.append(
                TimeSeriesPoint(
                    period=month.label, start=month.start, end=month.end, value=0.0, observation="true_zero"
                )
            )
        else:
            points.append(
                TimeSeriesPoint(period=month.label, start=month.start, end=month.end, value=None, observation="missing")
            )

    calculation = _calculation(spec, active, window_start, window_end)
    if not any(p.observation == "observed" for p in points):
        return _series(
            spec,
            active,
            [],
            provenance,
            calculation=calculation,
            status="no_data",
            message="No observations exist for this metric and these filters in the data coverage.",
            notes=notes,
        )
    missing = sum(1 for p in points if p.observation == "missing")
    if missing:
        notes.append(f"{missing} month(s) have no observation and are reported as missing (not zero).")
    return _series(spec, active, points, provenance, calculation=calculation, status="ok", notes=notes)


def _fetch(
    db: Database, service: KPIService, spec: SeriesMetric, filters: dict[str, str], start: date, end: date
) -> tuple[dict[str, float], Provenance, date | None]:
    """Monthly values keyed by ``YYYY-MM`` (months without rows are absent), provenance and first observation date."""
    if spec.source == "kpi_monthly":
        kpi = service.calculate_kpi(spec.kpi_key, start_date=start, end_date=end, dimension="month", filters=filters)
        values: dict[str, float] = {}
        first_observed: date | None = None
        for row in kpi.breakdown:
            raw = row.components.get(spec.value_field)
            if raw is not None and not isinstance(raw, str):
                values[row.dimension_value] = float(raw)
            first_text = row.components.get("first_observed_date")
            if isinstance(first_text, str):
                observed_on = date.fromisoformat(first_text)
                first_observed = observed_on if first_observed is None else min(first_observed, observed_on)
        return values, kpi.provenance, first_observed
    result = revenue.mrr_series(db, explicit_period(start, end), filters=filters, as_of=service.as_of)
    return {row.month: float(getattr(row, spec.value_field)) for row in result.data}, result.provenance, None


def _calculation(spec: SeriesMetric, filters: dict[str, str], start: date | None, end: date | None) -> str:
    formula = get_kpi_definition(spec.kpi_key).formula
    window = f"{start.isoformat()} to {end.isoformat()}" if start and end else "no complete month"
    filter_text = ", ".join(f"{k}={v}" for k, v in filters.items()) or "none"
    if spec.source == "mrr_series":
        how = f"{spec.name} at each month end from the Phase 2 month-end MRR series ({spec.kpi_key} KPI: {formula})"
    else:
        how = f"{spec.name} per calendar month from the {spec.kpi_key} KPI broken down by month ({formula})"
    empty = "true zero" if spec.empty_month_is_zero else "missing"
    return f"{how}. Window: {window}. Filters: {filter_text}. Months without rows: {empty}."


def _series(
    spec: SeriesMetric,
    filters: dict[str, str],
    points: list[TimeSeriesPoint],
    provenance: Provenance,
    *,
    calculation: str,
    status: str,
    message: str | None = None,
    notes: list[str] | None = None,
) -> TimeSeries:
    return TimeSeries.model_validate(
        {
            "metric": spec.key,
            "metric_name": spec.name,
            "unit": spec.unit,
            "status": status,
            "start": points[0].start if points else None,
            "end": points[-1].end if points else None,
            "points": points,
            "filters": filters,
            "calculation": calculation,
            "message": message,
            "limitations": list(notes or []),
            "provenance": provenance,
        }
    )
