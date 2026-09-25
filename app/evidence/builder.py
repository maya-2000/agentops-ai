"""Deterministic conversion of executed tool results into evidence items.

Every number in an evidence item is copied from a typed Phase 1-3 result; nothing is recomputed
here. Each item keeps the tool's provenance: call ID, query IDs, source tables, calculation,
execution timestamp and limitations. Forecast evidence additionally keeps the model, cutoff,
horizon and backtest metrics; anomaly evidence keeps the detector, threshold, score and expected
value. A failed tool call produces no evidence: the failure stays in the tool trace.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.analytics.kpis import KPIResult, get_kpi_definition
from app.analytics.models import AnalyticsResult, Scalar
from app.analytics.periods import Period
from app.anomalies import AnomalyReport, AnomalyResult
from app.evidence.formatting import format_money_change, format_percent, format_value
from app.evidence.models import Evidence, EvidenceGraph, EvidenceStatus, EvidenceType
from app.forecasting import ForecastResult
from app.tools.base import ToolResult
from app.tools.results import KPIComparison, SQLResult

MAX_ROWS_PER_RESULT = 8  # breakdown / decomposition members kept as evidence (largest first)
MAX_SQL_ROWS_AS_EVIDENCE = 10


def period_label(period: Period | None) -> str | None:
    """Readable label: YYYY-MM, YYYY-Qn or YYYY for calendar units, else the explicit date range."""
    if period is None:
        return None
    if period.is_calendar_months:
        months = int(period.months)
        if months == 1:
            return period.start.strftime("%Y-%m")
        if months == 3 and period.start.month in (1, 4, 7, 10):
            return f"{period.start.year}-Q{(period.start.month - 1) // 3 + 1}"
        if months == 12 and period.start.month == 1:
            return str(period.start.year)
    return period.label if " to " not in period.label else f"{period.start.isoformat()} to {period.end.isoformat()}"


def _for(period: Period | None) -> str:
    label = period_label(period)
    return f" for {label}" if label else ""


def _filters_text(filters: dict[str, str]) -> str:
    return f" [{', '.join(f'{k}={v}' for k, v in filters.items())}]" if filters else ""


def _status(status: str) -> EvidenceStatus:
    return status if status in ("ok", "no_data", "insufficient_data", "insufficient_history") else "no_data"  # type: ignore[return-value]


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


class _Collector:
    """Creates evidence items for one tool result with shared provenance."""

    def __init__(self, graph: EvidenceGraph, result: ToolResult, operation: str):
        self.graph = graph
        self.result = result
        self.operation = operation
        self.items: list[Evidence] = []

    def add(
        self,
        evidence_type: EvidenceType,
        statement: str,
        *,
        period: Period | None = None,
        comparison: Period | None = None,
        value: Scalar = None,
        unit: str | None = None,
        metric: str | None = None,
        filters: dict[str, str] | None = None,
        dimension: str | None = None,
        dimension_value: str | None = None,
        attributes: dict[str, Scalar] | None = None,
        details: dict[str, Scalar] | None = None,
        status: str = "ok",
        truncated: bool = False,
        limitations: list[str] | None = None,
    ) -> Evidence:
        evidence = Evidence(
            evidence_id=self.graph.next_evidence_id(),
            evidence_type=evidence_type,
            statement=statement,
            metric=metric,
            value=value,
            unit=unit,
            display_value=format_value(_num(value), unit) if _num(value) is not None else None,
            period_label=period_label(period),
            period_start=period.start if period else None,
            period_end=period.end if period else None,
            comparison_label=period_label(comparison),
            comparison_start=comparison.start if comparison else None,
            comparison_end=comparison.end if comparison else None,
            filters=dict(filters or {}),
            dimension=dimension,
            dimension_value=dimension_value,
            attributes=dict(attributes or {}),
            details=dict(details or {}),
            status=_status(status),
            truncated=truncated,
            tool_name=self.result.tool_name,
            tool_call_id=self.result.call_id,
            operation=self.operation,
            query_ids=list(self.result.query_ids),
            source_tables=list(self.result.source_tables),
            calculation=self.result.calculation,
            execution_timestamp=self.result.finished_at,
            limitations=list(limitations if limitations is not None else self.result.limitations),
            confidence="high" if status == "ok" else "low",
        )
        self.graph.add_evidence(evidence)
        self.items.append(evidence)
        return evidence


def build_evidence(result: ToolResult, graph: EvidenceGraph) -> list[Evidence]:
    """Add evidence for one tool result to ``graph`` and return the new items."""
    if not result.success or result.result is None:
        return []
    payload = result.result
    operation = str(result.arguments.get("operation", result.tool_name))
    collector = _Collector(
        graph, result, f"{result.tool_name}.{operation}" if operation != result.tool_name else operation
    )
    if isinstance(payload, KPIComparison):
        _kpi_comparison(collector, payload)
    elif isinstance(payload, KPIResult):
        _kpi(collector, payload)
    elif isinstance(payload, ForecastResult):
        _forecast(collector, payload)
    elif isinstance(payload, AnomalyReport):
        _anomalies(collector, payload)
    elif isinstance(payload, SQLResult):
        _sql(collector, payload)
    elif isinstance(payload, AnalyticsResult):
        _ANALYTICS.get(operation, _generic)(collector, payload)
    else:  # pragma: no cover - every registered tool returns one of the types above
        collector.add("derived", f"{result.tool_name} returned a {type(payload).__name__}.", status=result.status)
    return collector.items


# ------------------------------------------------------------------------------------------------ KPI


def _kpi_type(key: str) -> EvidenceType:
    return "observed" if get_kpi_definition(key).value_rule.kind == "component" else "calculated"


def _kpi(c: _Collector, r: KPIResult) -> None:
    label = period_label(r.period)
    if r.status != "ok":
        c.add(
            "derived",
            f"{r.name} for {label}{_filters_text(r.filters)}: no value ({r.status}). {r.message or ''}".strip(),
            period=r.period,
            metric=r.key,
            unit=r.unit,
            filters=r.filters,
            status=r.status,
        )
        return
    c.add(
        _kpi_type(r.key),
        f"{r.name} for {label}{_filters_text(r.filters)}: {format_value(r.value, r.unit)}.",
        period=r.period,
        comparison=r.comparison_period,
        value=r.value,
        unit=r.unit,
        metric=r.key,
        filters=r.filters,
        attributes={k: v for k, v in r.components.items() if _num(v) is not None},
    )
    rows = sorted((row for row in r.breakdown if row.value is not None), key=lambda row: -(row.value or 0.0))
    for rank, row in enumerate(rows[:MAX_ROWS_PER_RESULT], start=1):
        c.add(
            _kpi_type(r.key),
            f"{r.name} for {label}{_filters_text(r.filters)}, {r.dimension} {row.dimension_value}: "
            f"{format_value(row.value, r.unit)} (rank {rank} of {len(rows)} by value).",
            period=r.period,
            value=row.value,
            unit=r.unit,
            metric=r.key,
            filters=r.filters,
            dimension=r.dimension,
            dimension_value=row.dimension_value,
            attributes={"rank": rank, "members": len(rows)},
        )


def _kpi_comparison(c: _Collector, r: KPIComparison) -> None:
    _kpi(c, r.current)
    _kpi(c, r.comparison)
    if r.absolute_change is None:
        return
    pct = f" ({format_percent(r.percentage_change, signed=True)})" if r.percentage_change is not None else ""
    c.add(
        "calculated",
        f"{r.name} changed from {format_value(r.comparison.value, r.unit)} in {period_label(r.comparison.period)} to "
        f"{format_value(r.current.value, r.unit)} in {period_label(r.current.period)}: "
        f"{'+' if r.absolute_change >= 0 else '-'}{format_value(abs(r.absolute_change), r.unit)}{pct}.",
        period=r.current.period,
        comparison=r.comparison.period,
        value=r.absolute_change,
        unit=r.unit,
        metric=r.key,
        filters=r.current.filters,
        attributes={
            "absolute_change": r.absolute_change,
            "percentage_change": r.percentage_change,
            "current_value": r.current.value,
            "comparison_value": r.comparison.value,
        },
        details={"direction": r.direction},
    )


# ------------------------------------------------------------------------------------------------ analytics

_Extractor = Callable[[_Collector, AnalyticsResult[Any]], None]


def _no_result(c: _Collector, r: AnalyticsResult[Any]) -> bool:
    if r.status == "ok":
        return False
    c.add(
        "derived",
        f"{_humanize(c.operation)}{_for(r.period)}: no result ({r.status}). {r.message or ''}".strip(),
        period=r.period,
        comparison=r.comparison_period,
        filters=r.filters,
        status=r.status,
    )
    return True


def _revenue_change(c: _Collector, r: AnalyticsResult[Any]) -> None:
    if _no_result(c, r):
        return
    s = r.summary
    cur, prev, change, pct = (
        _num(s.get("current_revenue")),
        _num(s.get("comparison_revenue")),
        _num(s.get("absolute_change")),
        _num(s.get("percentage_change")),
    )
    period, comparison, f = r.period, r.comparison_period, r.filters
    c.add(
        "observed",
        f"Revenue for {period_label(period)}{_filters_text(f)}: {format_value(cur, 'SGD')}.",
        period=period,
        value=cur,
        unit="SGD",
        metric="revenue",
        filters=f,
    )
    c.add(
        "observed",
        f"Revenue for {period_label(comparison)}{_filters_text(f)}: {format_value(prev, 'SGD')}.",
        period=comparison,
        value=prev,
        unit="SGD",
        metric="revenue",
        filters=f,
    )
    if change is not None:
        c.add(
            "calculated",
            f"Revenue changed by {'+' if change >= 0 else '-'}{format_money_change(change)}"
            f"{f' ({format_percent(pct, signed=True)})' if pct is not None else ''} from {period_label(comparison)} "
            f"to {period_label(period)}{_filters_text(f)}.",
            period=period,
            comparison=comparison,
            value=change,
            unit="SGD",
            metric="revenue",
            filters=f,
            attributes={
                "absolute_change": change,
                "percentage_change": pct,
                "current_value": cur,
                "comparison_value": prev,
            },
            details={"direction": s.get("direction")},
        )


def _decomposition(c: _Collector, r: AnalyticsResult[Any]) -> None:
    if _no_result(c, r):
        return
    s = r.summary
    total = _num(s.get("total_change")) or 0.0
    dimension = r.dimensions[0] if r.dimensions else "dimension"
    c.add(
        "calculated",
        f"Revenue change {period_label(r.comparison_period)} to {period_label(r.period)}{_filters_text(r.filters)} "
        f"decomposed by {dimension}: total {'+' if total >= 0 else '-'}{format_money_change(total)}; largest decline: "
        f"{s.get('largest_decline') or 'none'}; largest increase: {s.get('largest_increase') or 'none'}.",
        period=r.period,
        comparison=r.comparison_period,
        value=total,
        unit="SGD",
        metric="revenue",
        filters=r.filters,
        dimension=dimension,
        attributes={
            "total_change": total,
            "total_percentage_change": _num(s.get("total_percentage_change")),
            "gross_decline": _num(s.get("gross_decline")),
            "gross_increase": _num(s.get("gross_increase")),
        },
        details={
            "largest_decline": s.get("largest_decline"),
            "largest_increase": s.get("largest_increase"),
            "reconciled": s.get("reconciled"),
        },
    )
    rows = sorted(r.data, key=lambda row: row.absolute_change if total < 0 else -row.absolute_change)
    for rank, row in enumerate(rows[:MAX_ROWS_PER_RESULT], start=1):
        share_key = "share_of_gross_decline" if row.absolute_change < 0 else "share_of_gross_increase"
        share = getattr(row, share_key)
        share_text = (
            f"; {format_percent(share)} of the gross {'decline' if row.absolute_change < 0 else 'increase'}"
            if share is not None
            else ""
        )
        pct = f" ({format_percent(row.percentage_change, signed=True)})" if row.percentage_change is not None else ""
        c.add(
            "calculated",
            f"{dimension} {row.dimension_value}{_filters_text(r.filters)}: revenue changed by "
            f"{'+' if row.absolute_change >= 0 else '-'}{format_money_change(row.absolute_change)}{pct} from "
            f"{period_label(r.comparison_period)} to {period_label(r.period)}{share_text}.",
            period=r.period,
            comparison=r.comparison_period,
            value=row.absolute_change,
            unit="SGD",
            metric="revenue",
            filters=r.filters,
            dimension=dimension,
            dimension_value=row.dimension_value,
            attributes={
                "absolute_change": row.absolute_change,
                "percentage_change": row.percentage_change,
                "current_value": row.current_value,
                "previous_value": row.previous_value,
                "share_of_total_change": row.share_of_total_change,
                "share_of_gross_decline": row.share_of_gross_decline,
                "share_of_gross_increase": row.share_of_gross_increase,
                "rank": rank,
            },
        )


def _bridge(c: _Collector, r: AnalyticsResult[Any]) -> None:
    if _no_result(c, r):
        return
    for row in r.data:
        if row.mrr is None:
            continue
        observed = row.component in ("opening_mrr", "closing_mrr")
        name = row.component.replace("_mrr", " MRR").replace("_", " ")
        customers = f" ({row.customers:,} customers)" if row.customers is not None else ""
        c.add(
            "observed" if observed else "calculated",
            f"MRR bridge {period_label(r.period)}{_filters_text(r.filters)}: {name} "
            f"{'' if observed else ('+' if row.mrr >= 0 else '-')}{format_money_change(row.mrr)}{customers}.",
            period=r.period,
            value=row.mrr,
            unit="SGD per month",
            metric="mrr",
            filters=r.filters,
            dimension="mrr_component",
            dimension_value=row.component,
            attributes={"customers": row.customers},
        )
    net = _num(r.summary.get("net_change"))
    if net is not None:
        c.add(
            "calculated",
            f"Net MRR change in {period_label(r.period)}{_filters_text(r.filters)}: "
            f"{'+' if net >= 0 else '-'}{format_money_change(net)}.",
            period=r.period,
            value=net,
            unit="SGD per month",
            metric="mrr",
            filters=r.filters,
            dimension="mrr_component",
            dimension_value="net_change",
        )


def _churn_summary(c: _Collector, r: AnalyticsResult[Any]) -> None:
    if _no_result(c, r):
        return
    for row in r.data:
        if row.value is None:
            continue
        c.add(
            "calculated",
            f"{row.name} for {period_label(r.period)}{_filters_text(r.filters)}: {format_value(row.value, row.unit)}.",
            period=r.period,
            value=row.value,
            unit=row.unit,
            metric=row.key,
            filters=r.filters,
        )
    for key in ("opening_customers", "churned_customers"):
        value = _num(r.summary.get(key))
        if value is not None:
            c.add(
                "observed",
                f"{key.replace('_', ' ').capitalize()} in {period_label(r.period)}{_filters_text(r.filters)}: "
                f"{format_value(value, 'customers')}.",
                period=r.period,
                value=value,
                unit="customers",
                metric=key,
                filters=r.filters,
            )


def _churn_by_dimension(c: _Collector, r: AnalyticsResult[Any]) -> None:
    if _no_result(c, r):
        return
    rows = sorted(r.data, key=lambda row: -(row.logo_churn_rate or 0.0))
    for rank, row in enumerate(rows[:MAX_ROWS_PER_RESULT], start=1):
        if row.logo_churn_rate is None:
            continue
        interval = (
            f"; 95% interval {format_percent(row.logo_churn_ci_low)} to {format_percent(row.logo_churn_ci_high)}"
            if row.logo_churn_ci_low is not None and row.logo_churn_ci_high is not None
            else ""
        )
        sample = "" if row.sufficient_sample else "; sample too small for a reliable comparison"
        c.add(
            "calculated",
            f"Logo churn for {row.dimension} {row.dimension_value} in {period_label(r.period)}"
            f"{_filters_text(r.filters)}: {format_percent(row.logo_churn_rate)} ({row.churned_customers:,} of "
            f"{row.opening_customers:,} opening customers{interval}{sample}); rank {rank} of {len(rows)}.",
            period=r.period,
            value=row.logo_churn_rate,
            unit="ratio",
            metric="logo_churn_rate",
            filters=r.filters,
            dimension=row.dimension,
            dimension_value=row.dimension_value,
            attributes={
                "churned_customers": row.churned_customers,
                "opening_customers": row.opening_customers,
                "ci_low": row.logo_churn_ci_low,
                "ci_high": row.logo_churn_ci_high,
                "revenue_churn_rate": row.revenue_churn_rate,
                "churned_mrr": row.churned_mrr,
                "rank": rank,
                "sufficient_sample": row.sufficient_sample,
            },
            status="ok",
        )


def _usage_churn(c: _Collector, r: AnalyticsResult[Any]) -> None:
    if _no_result(c, r):
        return
    for row in r.data:
        if row.segment != "All":
            continue
        c.add(
            "calculated",
            f"Customers who {row.outcome.replace('churned in period', 'churned')} in {period_label(r.period)}: "
            f"{row.customers:,} customers, median usage ratio {row.median_usage_ratio:.2f}, "
            f"{format_percent(row.share_with_usage_decline)} with a usage decline, "
            f"{row.tickets_per_customer_recent_90d:.2f} tickets per customer in the prior 90 days.",
            period=r.period,
            value=row.tickets_per_customer_recent_90d,
            unit="tickets per customer",
            metric="tickets_per_customer_recent_90d",
            filters=r.filters,
            dimension="outcome",
            dimension_value=row.outcome,
            attributes={
                "customers": row.customers,
                "median_usage_ratio": row.median_usage_ratio,
                "share_with_usage_decline": row.share_with_usage_decline,
                "tickets_per_customer_recent_90d": row.tickets_per_customer_recent_90d,
                "tickets_per_customer_prior_90d": row.tickets_per_customer_prior_90d,
            },
            details={"association_only": True, "recent_window_days": 90},
        )


def _support_change(c: _Collector, r: AnalyticsResult[Any]) -> None:
    if _no_result(c, r):
        return
    units = {
        "tickets": "tickets",
        "tickets_per_active_customer": "tickets per customer",
        "average_resolution_hours": "hours",
    }
    for row in r.data:
        if row.change is None:
            continue
        unit = units.get(row.metric, "count")
        c.add(
            "calculated",
            f"{row.metric.replace('_', ' ').capitalize()}{_filters_text(r.filters)}: "
            f"{format_value(row.current, unit)} in {period_label(r.period)} vs {format_value(row.comparison, unit)} "
            f"in {period_label(r.comparison_period)} ({format_percent(row.change, signed=True)}).",
            period=r.period,
            comparison=r.comparison_period,
            value=row.change,
            unit="ratio",
            metric=row.metric,
            filters=r.filters,
            attributes={
                "current_value": row.current,
                "comparison_value": row.comparison,
                "percentage_change": row.change,
            },
        )


def _grouped_rows(c: _Collector, r: AnalyticsResult[Any]) -> None:
    """Rows keyed by ``dimension``/``dimension_value`` with numeric fields (support, sales, ...)."""
    if _no_result(c, r):
        return
    _generic_summary(c, r)
    for row in r.data[:MAX_ROWS_PER_RESULT]:
        data = row.model_dump()
        numbers = {k: v for k, v in data.items() if _num(v) is not None}
        label = data.get("dimension_value") or data.get("feature") or data.get("channel") or data.get("campaign_id")
        fields = ", ".join(f"{k.replace('_', ' ')} {_fmt_field(k, v)}" for k, v in list(numbers.items())[:5])
        c.add(
            "calculated",
            f"{_humanize(c.operation)} {data.get('dimension') or ''} {label or ''}{_for(r.period)}"
            f"{_filters_text(r.filters)}: {fields}.".replace("  ", " "),
            period=r.period,
            filters=r.filters,
            dimension=data.get("dimension"),
            dimension_value=str(label) if label is not None else None,
            attributes=numbers,
        )


_OPERATION_NAMES = {"get_customer_risk": "Customer risk", "get_cohort_analysis": "Cohort retention"}


def _humanize(operation: str) -> str:
    return _OPERATION_NAMES.get(operation, operation.split(".")[-1].replace("_", " ").capitalize())


def _fmt_field(key: str, value: Any) -> str:
    if any(t in key for t in ("rate", "share", "ratio", "roas")) and isinstance(value, float) and abs(value) <= 5:
        return format_percent(value)
    if any(t in key for t in ("revenue", "spend", "value", "mrr", "cac")):
        return format_value(value, "SGD")
    return format_value(value, "count") if isinstance(value, int) else f"{value:,.4g}"


def _generic_summary(c: _Collector, r: AnalyticsResult[Any]) -> None:
    for key, value in list(r.summary.items())[:MAX_ROWS_PER_RESULT]:
        if _num(value) is None:
            continue
        c.add(
            "calculated",
            f"{_humanize(c.operation)}{_for(r.period)}{_filters_text(r.filters)}: {key.replace('_', ' ')} "
            f"{_fmt_field(key, value)}.",
            period=r.period,
            comparison=r.comparison_period,
            value=value,
            metric=key,
            filters=r.filters,
        )


def _generic(c: _Collector, r: AnalyticsResult[Any]) -> None:
    if _no_result(c, r):
        return
    _generic_summary(c, r)
    if not c.items:
        c.add("derived", f"{_humanize(c.operation)} returned {len(r.data)} rows{_for(r.period)}.", period=r.period)


def _risk(c: _Collector, r: AnalyticsResult[Any]) -> None:
    if _no_result(c, r):
        return
    _generic_summary(c, r)
    for rank, row in enumerate(r.data[:5], start=1):
        signals = ", ".join(s.signal.replace("_", " ") for s in row.signals) or "none"
        c.add(
            "calculated",
            f"Customer {row.customer_id} ({row.segment}, {row.region}): risk score {row.risk_score} ({row.risk_band} "
            f"band), MRR {format_value(row.mrr, 'SGD')}; signals: {signals}.",
            value=row.risk_score,
            unit="points",
            filters=r.filters,
            dimension="customer_id",
            dimension_value=row.customer_id,
            attributes={"risk_score": row.risk_score, "mrr": row.mrr, "rank": rank},
            details={"risk_band": row.risk_band},
        )


_ANALYTICS: dict[str, _Extractor] = {
    "revenue_change": _revenue_change,
    "decompose_revenue_change": _decomposition,
    "revenue_bridge": _bridge,
    "churn_summary": _churn_summary,
    "churn_by_dimension": _churn_by_dimension,
    "usage_churn_relationship": _usage_churn,
    "support_volume_change": _support_change,
    "support_by_dimension": _grouped_rows,
    "sales_performance": _grouped_rows,
    "channel_performance": _grouped_rows,
    "feature_adoption": _grouped_rows,
    "get_customer_risk": _risk,
}


# ------------------------------------------------------------------------------------------------ Phase 3


def _forecast(c: _Collector, r: ForecastResult) -> None:
    details: dict[str, Scalar] = {
        "model": r.model,
        "cutoff_date": r.cutoff_date.isoformat(),
        "horizon": r.horizon,
        "history_end": r.historical_end.isoformat() if r.historical_end else None,
        "confidence_level": r.confidence_level,
    }
    if r.status != "ok":
        c.add(
            "derived",
            f"No forecast for {r.metric_name}{_filters_text(r.filters)} ({r.status}): {r.message}",
            metric=r.metric,
            unit=r.unit,
            filters=r.filters,
            status=r.status,
            details=details,
        )
        return
    backtest, baseline = r.backtest_metrics, r.baseline_metrics
    details |= {
        "backtest_mae": backtest.mae if backtest else None,
        "backtest_wape": backtest.wape if backtest else None,
        "backtest_folds": backtest.fold_count if backtest else None,
        "interval_coverage": backtest.interval_coverage if backtest else None,
        "baseline_mae": baseline.mae if baseline else None,
        "interval_method": r.interval.method if r.interval else None,
    }
    level = f"{r.confidence_level:.0%}"
    for step, point in enumerate(r.forecast_points, start=1):
        interval = (
            f" ({level} prediction interval {format_value(point.lower_bound, r.unit)} to "
            f"{format_value(point.upper_bound, r.unit)})"
            if point.lower_bound is not None and point.upper_bound is not None
            else " (no prediction interval)"
        )
        c.add(
            "forecast",
            f"Forecast {r.metric_name}{_filters_text(r.filters)} for {point.period}: "
            f"{format_value(point.predicted_value, r.unit)}{interval}; model {r.model}, data to {r.historical_end}.",
            period=Period(start=point.start, end=point.end, label=point.period),
            value=point.predicted_value,
            unit=r.unit,
            metric=r.metric,
            filters=r.filters,
            attributes={"lower_bound": point.lower_bound, "upper_bound": point.upper_bound, "step": step},
            details=details,
        )


def _anomaly_statement(a: AnomalyResult) -> str:
    score = "unbounded" if a.score is None else f"{a.score:+.2f}"
    pct = f", {format_percent(a.deviation_percentage, signed=True)}" if a.deviation_percentage is not None else ""
    verdict = "flagged as statistically unusual" if a.is_anomaly else "not flagged"
    return (
        f"{a.metric_name}{_filters_text(a.filters)} in {a.period} was {verdict} by the {a.detector} detector "
        f"({a.severity}, direction {a.direction}): observed {format_value(a.observed_value, a.unit)} vs expected "
        f"{format_value(a.expected_value, a.unit)}{pct}; score {score} vs threshold {a.threshold:g}."
    )


def _anomalies(c: _Collector, r: AnomalyReport) -> None:
    base_details: dict[str, Scalar] = {
        "detector": r.detector,
        "window": r.method.window,
        "transform": r.method.transform,
        "threshold": r.method.threshold,
    }
    if r.status != "ok":
        c.add(
            "derived",
            f"No anomaly assessment for {r.metric_name}{_filters_text(r.filters)} ({r.status}): {r.message}",
            metric=r.metric,
            filters=r.filters,
            status=r.status,
            details=base_details,
        )
        return
    evaluated = Period(
        start=r.evaluation_start, end=r.evaluation_end, label=f"{r.evaluation_start} to {r.evaluation_end}"
    )
    flagged = r.anomalies
    listing = ", ".join(f"{a.period} ({a.direction})" for a in flagged) or "none"
    c.add(
        "derived",
        f"{r.detector} check of {r.metric_name}{_filters_text(r.filters)} for {len(r.results)} months "
        f"({r.results[0].period} to {r.results[-1].period}): {len(flagged)} flagged as statistically unusual: "
        f"{listing}.",
        period=evaluated,
        value=len(flagged),
        unit="count",
        metric=r.metric,
        filters=r.filters,
        attributes={"months_scored": len(r.results), "months_flagged": len(flagged)},
        details=base_details,
    )
    shown = list(flagged[:5])
    latest = r.results[-1]
    if latest not in shown:
        shown.append(latest)
    for a in shown:
        c.add(
            "anomaly",
            _anomaly_statement(a),
            period=Period(start=a.period_start, end=a.period_end, label=a.period),
            value=a.observed_value,
            unit=a.unit,
            metric=a.metric,
            filters=a.filters,
            attributes={
                "expected_value": a.expected_value,
                "deviation": a.deviation,
                "deviation_percentage": a.deviation_percentage,
                "score": a.score,
                "threshold": a.threshold,
            },
            details=base_details
            | {
                "severity": a.severity,
                "direction": a.direction,
                "is_anomaly": a.is_anomaly,
                "anomaly_start": a.anomaly_start,
                "window_start": a.historical_window.start,
                "window_end": a.historical_window.end,
            },
            limitations=list(a.limitations[:3]),
        )


# ------------------------------------------------------------------------------------------------ SQL


def _sql(c: _Collector, r: SQLResult) -> None:
    truncated = " TRUNCATED: more rows exist than were returned; the rows are not complete." if r.truncated else ""
    c.add(
        "observed",
        f"Query '{r.description}' returned {r.row_count} row(s) with columns {', '.join(r.columns)}.{truncated}",
        value=r.row_count,
        unit="rows",
        attributes={"row_count": r.row_count, "max_rows": r.max_rows},
        details={"sql": r.sql, "truncated": r.truncated},
        truncated=r.truncated,
        status="ok" if r.row_count else "no_data",
    )
    for index, row in enumerate(r.rows[:MAX_SQL_ROWS_AS_EVIDENCE], start=1):
        cells = dict(zip(r.columns, row, strict=True))
        c.add(
            "observed",
            f"Query '{r.description}' row {index}: " + ", ".join(f"{k}={v}" for k, v in cells.items()) + ".",
            attributes={k: v for k, v in cells.items() if _num(v) is not None},
            details={k: v for k, v in cells.items() if _num(v) is None},
            truncated=r.truncated,
        )


def evidence_summary(evidence: Evidence) -> dict[str, Any]:
    """The compact form sent to an LLM (no raw rows, no SQL text)."""
    return {
        "evidence_id": evidence.evidence_id,
        "type": evidence.evidence_type,
        "statement": evidence.statement,
        "status": evidence.status,
        "tool": evidence.tool_name,
        "period": evidence.period_label,
        "dimension": evidence.dimension,
        "dimension_value": evidence.dimension_value,
        "filters": evidence.filters,
        "truncated": evidence.truncated,
    }
