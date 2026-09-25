"""The KPI calculation engine.

``calculate_kpi(db, key, params)`` is the one way to compute a registered KPI:

1. look up the definition (``UnsupportedKPIError`` if unknown),
2. validate typed parameters, dimension and filters against the allow-list and the KPI,
3. resolve the period (and comparison period) against the business as-of date,
4. check the period against the data coverage,
5. render the registered SQL template, bind parameters and execute it through the
   ``Database`` abstraction (a total query, plus a breakdown query when a dimension is given),
6. derive the value from the returned components with the KPI's value rule, and
7. return a ``KPIResult`` carrying the SQL, parameters, source tables and lineage.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from pydantic import ValidationError

from app.analytics.dimensions import get_dimension, normalise_value
from app.analytics.errors import (
    AnalyticsError,
    InvalidFilterValueError,
    InvalidRequestError,
    ResultStatus,
    UnsupportedDimensionError,
)
from app.analytics.executor import QueryRunner
from app.analytics.kpis.models import KPIBreakdownRow, KPIDefinition, KPIParameters, KPIResult
from app.analytics.kpis.registry import get_kpi_definition
from app.analytics.kpis.sql import TEMPLATES, SQLTemplate
from app.analytics.models import Scalar, to_number
from app.analytics.periods import Period, explicit_period, previous_period, resolve_period
from app.config import get_settings
from app.database.base import Database

_NO_OBSERVATIONS = "No observations available for the requested period and filters."


def coerce_parameters(params: KPIParameters | dict[str, Any] | None, overrides: dict[str, Any]) -> KPIParameters:
    """Build ``KPIParameters`` from a model, a dict and/or keyword overrides."""
    if isinstance(params, KPIParameters) and not overrides:
        return params
    base = params.model_dump(exclude_none=True) if isinstance(params, KPIParameters) else dict(params or {})
    if isinstance(base.get("filters"), dict):
        base["filters"] = {k: v for k, v in base["filters"].items() if v is not None}
    try:
        return KPIParameters.model_validate({**base, **overrides})
    except ValidationError as exc:
        unknown = [str(e["loc"][-1]) for e in exc.errors() if e["type"] == "extra_forbidden"]
        if unknown:
            raise UnsupportedDimensionError(f"Unknown parameter(s) or filter(s): {', '.join(unknown)}") from None
        raise InvalidRequestError(f"Invalid KPI parameters: {exc}") from None


class KPIService:
    """Calculates registered KPIs against a ``Database``."""

    def __init__(self, db: Database, *, as_of: date | None = None):
        self.db = db
        self.as_of = as_of or get_settings().as_of_date
        self._coverage: tuple[date, date] | None = None
        self._known_values: set[tuple[str, str]] = set()

    # ---------------------------------------------------------------- public API
    def get_kpi_definition(self, key: str) -> KPIDefinition:
        return get_kpi_definition(key)

    def coverage(self) -> tuple[date, date]:
        """First and last day with observed revenue (the dataset's reporting window)."""
        if self._coverage is None:
            rows = QueryRunner(self.db, "coverage").run("SELECT MIN(date), MAX(date) FROM daily_revenue").rows
            start, end = rows[0]
            if start is None:
                raise AnalyticsError("The database contains no revenue observations")
            self._coverage = (start, end)
        return self._coverage

    def resolve(self, spec: str | None, start: date | None, end: date | None) -> Period:
        if start is not None and end is not None:
            return explicit_period(start, end)
        return resolve_period(spec, as_of=self.as_of)

    def calculate_kpi(self, key: str, params: KPIParameters | dict[str, Any] | None = None, **kwargs: Any) -> KPIResult:
        definition = get_kpi_definition(key)
        p = coerce_parameters(params, kwargs)
        period = self.resolve(p.period, p.start_date, p.end_date)
        comparison = self._comparison_period(definition, p, period)

        filters = p.filters.active()
        breakdown: str | None = None
        if p.dimension is not None:
            get_dimension(p.dimension)
            if p.dimension_value is not None:
                filters = {**filters, p.dimension: normalise_value(p.dimension, p.dimension_value)}
            else:
                breakdown = p.dimension
        template = TEMPLATES[definition.template]
        runner = QueryRunner(self.db, f"kpi:{definition.key}")

        def result(status: ResultStatus, message: str, limitations: list[str] | None = None) -> KPIResult:
            return self._result(
                definition,
                runner,
                status,
                None,
                {},
                [],
                period,
                comparison,
                filters,
                p.dimension,
                message,
                limitations or [],
            )

        # Grains the data cannot support defensibly: an explicit insufficient-evidence result.
        for grain in [*filters, *([breakdown] if breakdown else [])]:
            if grain in definition.insufficient_grains:
                return result("insufficient_data", definition.insufficient_grains[grain])
        for grain, filter_value in filters.items():
            reason = definition.insufficient_filter_values.get(grain, {}).get(filter_value)
            if reason:
                return result("insufficient_data", reason)

        self._check_applicable(definition, template, filters, breakdown)
        self._validate_lookup_values(filters)

        coverage_issue, coverage_notes = self._coverage_check(definition, template, period, comparison)
        if coverage_issue is not None:
            return result(coverage_issue[0], coverage_issue[1], coverage_notes)

        bind = self._bind(period, comparison, filters)
        total_rows = self._run(runner, definition, template, None, filters, bind)
        value, status, message, components = self._evaluate(definition, total_rows[0] if total_rows else None)
        rows: list[KPIBreakdownRow] = []
        if breakdown is not None:
            for row in self._run(runner, definition, template, breakdown, filters, bind):
                row_value, row_status, row_message, row_components = self._evaluate(definition, row)
                rows.append(
                    KPIBreakdownRow(
                        dimension_value=str(row["dimension_value"]),
                        status=row_status,
                        value=row_value,
                        components=row_components,
                        message=row_message,
                    )
                )
            rows.sort(key=lambda r: r.dimension_value)
        return self._result(
            definition,
            runner,
            status,
            value,
            components,
            rows,
            period,
            comparison,
            filters,
            p.dimension,
            message,
            coverage_notes,
        )

    # ---------------------------------------------------------------- validation
    def _comparison_period(self, definition: KPIDefinition, p: KPIParameters, period: Period) -> Period | None:
        given = p.comparison_period is not None or p.comparison_start_date is not None
        if not definition.requires_comparison:
            if given:
                raise InvalidRequestError(f"{definition.key} does not use a comparison period")
            return None
        if p.comparison_start_date is not None and p.comparison_end_date is not None:
            return explicit_period(p.comparison_start_date, p.comparison_end_date)
        if p.comparison_period is not None:
            return resolve_period(p.comparison_period, as_of=self.as_of)
        return previous_period(period)

    @staticmethod
    def _check_applicable(
        definition: KPIDefinition, template: SQLTemplate, filters: dict[str, str], breakdown: str | None
    ) -> None:
        for grain in filters:
            if grain not in template.filter_columns:
                raise UnsupportedDimensionError(
                    f"{definition.key} cannot be filtered by {grain!r}. Supported filters: "
                    f"{', '.join(template.filter_columns) or 'none'}"
                )
        if breakdown is not None and breakdown not in template.dimension_columns:
            raise UnsupportedDimensionError(
                f"{definition.key} cannot be broken down by {breakdown!r}. Supported dimensions: "
                f"{', '.join(template.dimension_columns) or 'none'}"
            )
        if definition.required_any_of and not any(g in filters or g == breakdown for g in definition.required_any_of):
            raise InvalidRequestError(
                f"{definition.key} needs one of {', '.join(definition.required_any_of)} as a filter or dimension"
            )

    def _validate_lookup_values(self, filters: dict[str, str]) -> None:
        """Open-ended filter values must exist in the data (checked with a bound parameter)."""
        for key, value in filters.items():
            spec = get_dimension(key)
            if spec.lookup_sql is None or (key, value) in self._known_values:
                continue
            count = QueryRunner(self.db, "validate_filter").run(spec.lookup_sql, {"value": value}).rows[0][0]
            if not count:
                raise InvalidFilterValueError(f"No {spec.display_name.lower()} {value!r} exists in the data")
            self._known_values.add((key, value))

    def _coverage_check(
        self, definition: KPIDefinition, template: SQLTemplate, period: Period, comparison: Period | None
    ) -> tuple[tuple[ResultStatus, str] | None, list[str]]:
        first, last = self.coverage()
        window = f"{first.isoformat()} to {last.isoformat()}"
        notes: list[str] = []
        if template.temporal == "point_in_time":
            if not first - timedelta(days=1) <= period.end <= last:
                return ("no_data", f"No state is observed at {period.end.isoformat()}; data covers {window}."), notes
        elif template.temporal == "cohort_flow":
            if period.opening_date < first - timedelta(days=1) or period.end > last:
                return (
                    (
                        "insufficient_data",
                        f"The opening state and the whole period must lie within the data ({window}).",
                    ),
                    notes,
                )
        else:
            if period.end < first or period.start > last:
                return ("no_data", f"The period is outside the data coverage ({window})."), notes
            if comparison is not None:
                for label, span in (("period", period), ("comparison period", comparison)):
                    if span.start < first or span.end > last:
                        return (
                            ("insufficient_data", f"The {label} {span.label} is not fully within the data ({window})."),
                            notes,
                        )
            elif period.start < first or period.end > last:
                notes.append(
                    f"The period extends beyond the data coverage ({window}); only observed days are included."
                )
        return None, notes

    # ---------------------------------------------------------------- execution
    @staticmethod
    def _bind(period: Period, comparison: Period | None, filters: dict[str, str]) -> dict[str, Any]:
        bind: dict[str, Any] = {
            "start_date": period.start,
            "end_date": period.end,
            "opening_date": period.opening_date,
            "period_months": period.months,
        }
        if comparison is not None:
            bind["comparison_start_date"] = comparison.start
            bind["comparison_end_date"] = comparison.end
        bind.update({f"f_{k}": v for k, v in filters.items()})
        return bind

    @staticmethod
    def _run(
        runner: QueryRunner,
        definition: KPIDefinition,
        template: SQLTemplate,
        dimension: str | None,
        filters: dict[str, str],
        bind: dict[str, Any],
    ) -> list[dict[str, Any]]:
        sql = template.render(dimension, filters)
        label = f"{definition.key} by {dimension}" if dimension else definition.key
        return runner.records(sql, bind, calculation=f"{label}: {definition.formula}")

    @staticmethod
    def _evaluate(
        definition: KPIDefinition, row: dict[str, Any] | None
    ) -> tuple[float | None, ResultStatus, str | None, dict[str, Scalar]]:
        if row is None:
            if not definition.zero_when_empty:
                return None, "no_data", _NO_OBSERVATIONS, {}
            components: dict[str, Scalar] = dict.fromkeys(definition.components, 0)
        else:
            components = {c: (row[c] if isinstance(row[c], str) else to_number(row[c])) for c in definition.components}
        observed = definition.observation_component
        if observed is not None and not components.get(observed):
            return None, "no_data", _NO_OBSERVATIONS, components
        value, status, message = definition.value_rule.apply(components)
        return value, status, message, components

    @staticmethod
    def _result(
        definition: KPIDefinition,
        runner: QueryRunner,
        status: ResultStatus,
        value: float | None,
        components: dict[str, Scalar],
        rows: list[KPIBreakdownRow],
        period: Period,
        comparison: Period | None,
        filters: dict[str, str],
        dimension: str | None,
        message: str | None,
        notes: list[str],
    ) -> KPIResult:
        scope = f"{period.label} ({period.start.isoformat()} to {period.end.isoformat()})"
        if comparison is not None:
            scope += f" vs {comparison.label} ({comparison.start.isoformat()} to {comparison.end.isoformat()})"
        filter_text = ", ".join(f"{k}={v}" for k, v in filters.items()) or "none"
        calculation = f"{definition.name} = {definition.formula}. Period: {scope}. Filters: {filter_text}."
        return KPIResult(
            key=definition.key,
            name=definition.name,
            status=status,
            value=value if status == "ok" else None,
            unit=definition.unit,
            period=period,
            comparison_period=comparison,
            filters=filters,
            dimension=dimension,
            breakdown=rows,
            components=components,
            formula=definition.formula,
            interpretation=definition.interpretation,
            limitations=[*definition.limitations, *notes],
            message=message,
            provenance=runner.provenance(calculation),
        )


def calculate_kpi(
    db: Database,
    key: str,
    params: KPIParameters | dict[str, Any] | None = None,
    *,
    as_of: date | None = None,
    **kwargs: Any,
) -> KPIResult:
    """Calculate a registered KPI (convenience wrapper around ``KPIService``)."""
    return KPIService(db, as_of=as_of).calculate_kpi(key, params, **kwargs)
