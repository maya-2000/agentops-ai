"""Chart specs from one agent run: KPI cards, period comparisons, breakdown bars and tables, time
series, forecasts with prediction intervals, and anomaly markers.

Rules:

- Numbers are copied from evidence items (``value``/``attributes``) or from the typed forecast and
  anomaly results those evidence items were built from. Nothing is recomputed, rescaled or
  aggregated; a chart that would need a calculation is not produced.
- Every spec lists the evidence it depicts. Forecast and anomaly specs carry their notices, so the
  labelling ("an estimate, not observed data"; "unusual is not necessarily bad") cannot be lost.
- Customer-level results are shown by customer ID only, as the evidence states them; withheld
  fields never reach a raw result that is charted (only forecast and anomaly results are).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence

from app.analytics.models import Scalar
from app.anomalies import AnomalyReport
from app.api.labels import dimension_label, metric_label
from app.api.schemas.responses import ANOMALY_NOTICE, FORECAST_NOTICE, AnomalySection, ForecastSection, KPIValue
from app.api.schemas.visualization import ChartField, VisualizationSpec
from app.evidence.models import Claim, Evidence
from app.forecasting import ForecastResult

MAX_COMPARISONS = 2
MAX_BREAKDOWNS = 3
_MONTH = re.compile(r"^\d{4}-\d{2}$")
_PAIRS = (("current_value", "comparison_value"), ("current_value", "previous_value"))


def _number(value: Scalar) -> float | int | None:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else None


def _words(key: str) -> str:
    text = key.replace("_", " ")
    return text[:1].upper() + text[1:]


# ------------------------------------------------------------------------------------------ KPI cards


def kpi_cards(kpis: Sequence[KPIValue]) -> list[VisualizationSpec]:
    if not kpis:
        return []
    rows: list[dict[str, Scalar]] = [
        {
            "label": k.label,
            "display_value": k.display_value,
            "value": k.value,
            "unit": k.unit,
            "period": k.period,
            "comparison_period": k.comparison_period,
            "percentage_change": k.percentage_change,
            "evidence_type": k.evidence_type,
            "evidence_id": k.evidence_id,
        }
        for k in kpis
    ]
    return [
        VisualizationSpec(
            chart_id="",
            kind="kpi_card",
            title="Key figures",
            fields=[
                ChartField(key="label", label="Metric", role="label"),
                ChartField(key="display_value", label="Value", role="value"),
                ChartField(key="percentage_change", label="Change", role="detail"),
            ],
            rows=rows,
            evidence_ids=[k.evidence_id for k in kpis],
        )
    ]


# ------------------------------------------------------------------------------------------ comparisons


def _pair(e: Evidence) -> tuple[float | int, float | int] | None:
    for current_key, previous_key in _PAIRS:
        current, previous = _number(e.attributes.get(current_key)), _number(e.attributes.get(previous_key))
        if current is not None and previous is not None:
            return current, previous
    return None


def comparisons(evidence: Sequence[Evidence], claims: Sequence[Claim]) -> list[VisualizationSpec]:
    """Current vs comparison period of one metric, from a change evidence item's own attributes."""
    primary = {e for c in claims if c.primary for e in c.evidence_ids}
    candidates = [
        e
        for e in evidence
        if e.status == "ok" and e.comparison_label and e.period_label and e.dimension_value is None and _pair(e)
    ]
    chosen = [e for e in candidates if e.evidence_id in primary] or candidates[:1]
    specs = []
    for e in chosen[:MAX_COMPARISONS]:
        pair = _pair(e)
        assert pair is not None and e.comparison_label and e.period_label
        current, previous = pair
        # A change stated as a ratio (its value is the percentage change) says nothing about the unit
        # of the two levels, so the levels are shown without one.
        unit = None if e.value == e.attributes.get("percentage_change") else e.unit
        name = metric_label(e.metric) if e.metric else "Value"
        specs.append(
            VisualizationSpec(
                chart_id="",
                kind="comparison",
                title=f"{name}: {e.period_label} vs {e.comparison_label}",
                subtitle=e.statement,
                metric=e.metric,
                unit=unit,
                fields=[
                    ChartField(key="period", label="Period", role="x"),
                    ChartField(key="value", label=name, role="y", unit=unit),
                    ChartField(key="role", label="Role", role="category"),
                ],
                rows=[
                    {"period": e.comparison_label, "value": previous, "role": "comparison"},
                    {"period": e.period_label, "value": current, "role": "current"},
                ],
                evidence_ids=[e.evidence_id],
            )
        )
    return specs


# ------------------------------------------------------------------------------------------ breakdowns


def _breakdown_groups(evidence: Sequence[Evidence]) -> list[list[Evidence]]:
    groups: dict[tuple[str, str, str | None], list[Evidence]] = defaultdict(list)
    for e in evidence:
        if e.status == "ok" and e.dimension and e.dimension_value is not None:
            groups[(e.tool_call_id, e.dimension, e.metric)].append(e)
    return [members for members in groups.values() if len(members) >= 2]


def _measure(members: Sequence[Evidence]) -> str | None:
    """``value`` when every member states one; else the first numeric attribute every member has."""
    if all(_number(e.value) is not None for e in members):
        return "value"
    shared = [k for k in members[0].attributes if all(_number(e.attributes.get(k)) is not None for e in members)]
    return shared[0] if shared else None


def breakdowns(evidence: Sequence[Evidence]) -> list[VisualizationSpec]:
    """One bar chart and one table per dimension breakdown (members in the tool's own order)."""
    specs: list[VisualizationSpec] = []
    for members in _breakdown_groups(evidence)[:MAX_BREAKDOWNS]:
        first = members[0]
        assert first.dimension is not None
        measure = _measure(members)
        dimension = dimension_label(first.dimension)
        change = " change" if first.comparison_label else ""
        name = metric_label(first.metric) if first.metric and measure == "value" else _words(measure or "value")
        when = f"{first.period_label} vs {first.comparison_label}" if first.comparison_label else first.period_label
        title = f"{name}{change} by {dimension}" + (f" ({when})" if when else "")
        ids = [e.evidence_id for e in members]
        if measure is not None:
            unit = first.unit if measure == "value" else None
            specs.append(
                VisualizationSpec(
                    chart_id="",
                    kind="bar",
                    title=title,
                    metric=first.metric,
                    unit=unit,
                    fields=[
                        ChartField(key="member", label=dimension, role="category"),
                        ChartField(key="value", label=f"{name}{change}", role="y", unit=unit),
                        ChartField(key="display_value", label="Display", role="label"),
                    ],
                    rows=[
                        {
                            "member": e.dimension_value,
                            "value": e.value if measure == "value" else e.attributes.get(measure),
                            "display_value": e.display_value if measure == "value" else None,
                            "evidence_id": e.evidence_id,
                        }
                        for e in members
                    ],
                    evidence_ids=ids,
                )
            )
        columns = list(dict.fromkeys(k for e in members for k in e.attributes))
        specs.append(
            VisualizationSpec(
                chart_id="",
                kind="table",
                title=title,
                metric=first.metric,
                unit=first.unit,
                fields=[
                    ChartField(key="member", label=dimension, role="category"),
                    ChartField(key="display_value", label="Value", role="value", unit=first.unit),
                    *(ChartField(key=k, label=_words(k)) for k in columns),
                    ChartField(key="evidence_id", label="Evidence"),
                ],
                rows=[
                    {
                        "member": e.dimension_value,
                        "display_value": e.display_value,
                        **{k: e.attributes.get(k) for k in columns},
                        "evidence_id": e.evidence_id,
                    }
                    for e in members
                ],
                evidence_ids=ids,
            )
        )
    return specs


# ------------------------------------------------------------------------------------------ time series


def time_series(evidence: Sequence[Evidence]) -> list[VisualizationSpec]:
    """Three or more monthly values of one metric stated as separate evidence items."""
    series: dict[tuple[str, str | None], dict[str, Evidence]] = defaultdict(dict)
    for e in evidence:
        if (
            e.evidence_type in ("observed", "calculated")
            and e.status == "ok"
            and e.metric
            and e.dimension_value is None
            and not e.comparison_label
            and not e.filters
            and e.period_label
            and _MONTH.match(e.period_label)
            and _number(e.value) is not None
        ):
            series[(e.metric, e.unit)].setdefault(e.period_label, e)
    specs = []
    for (metric, unit), by_period in series.items():
        if len(by_period) < 3:
            continue
        points = [by_period[p] for p in sorted(by_period)]
        name = metric_label(metric)
        specs.append(
            VisualizationSpec(
                chart_id="",
                kind="time_series",
                title=f"{name} by month",
                metric=metric,
                unit=unit,
                fields=[
                    ChartField(key="period", label="Month", role="x"),
                    ChartField(key="value", label=name, role="y", unit=unit),
                ],
                rows=[{"period": e.period_label, "value": e.value, "evidence_id": e.evidence_id} for e in points],
                evidence_ids=[e.evidence_id for e in points],
            )
        )
    return specs


# ------------------------------------------------------------------------------------------ forecasts


def forecast_chart(section: ForecastSection, result: ForecastResult) -> VisualizationSpec:
    rows: list[dict[str, Scalar]] = [
        {"period": p.period, "value": p.value, "series": "actual", "lower": None, "upper": None}
        for p in result.history.points
    ]
    rows += [
        {
            "period": p.period,
            "value": p.predicted_value,
            "series": "forecast",
            "lower": p.lower_bound,
            "upper": p.upper_bound,
        }
        for p in result.forecast_points
    ]
    view = section.forecast
    interval = f"{view.confidence_level:.0%} prediction interval" if view.interval_available else "no interval"
    return VisualizationSpec(
        chart_id="",
        kind="forecast",
        title=f"{view.metric_name} forecast: next {view.horizon} month{'s' if view.horizon != 1 else ''}",
        subtitle=f"Model {view.model or 'n/a'}, data to {view.cutoff_date:%Y-%m-%d}, {interval}.",
        metric=view.metric,
        unit=view.unit,
        fields=[
            ChartField(key="period", label="Month", role="x"),
            ChartField(key="value", label=view.metric_name, role="y", unit=view.unit),
            ChartField(key="series", label="Series", role="series"),
            ChartField(key="lower", label="Lower bound", role="lower", unit=view.unit),
            ChartField(key="upper", label="Upper bound", role="upper", unit=view.unit),
        ],
        rows=rows,
        evidence_ids=section.evidence_ids,
        source="forecast_result",
        notes=[FORECAST_NOTICE, *section.limitations[:2]],
    )


# ------------------------------------------------------------------------------------------ anomalies


def anomaly_chart(section: AnomalySection, report: AnomalyReport) -> VisualizationSpec:
    scored = {r.period: r for r in report.results}
    rows: list[dict[str, Scalar]] = []
    for p in report.series.points:
        r = scored.get(p.period)
        rows.append(
            {
                "period": p.period,
                "value": p.value,
                "expected": r.expected_value if r else None,
                "lower": r.lower_bound if r else None,
                "upper": r.upper_bound if r else None,
                "flagged": r.is_anomaly if r else False,
                "severity": r.severity if r else None,
                "direction": r.direction if r else None,
                "score": r.score if r else None,
            }
        )
    view = section.report
    flagged = [a.period for a in view.flagged]
    return VisualizationSpec(
        chart_id="",
        kind="anomaly",
        title=f"{view.metric_name}: {view.detector.replace('_', ' ')} check",
        subtitle=(
            f"{len(flagged)} of {view.scored_periods} months flagged"
            + (f" ({', '.join(flagged)})" if flagged else "")
            + f"; window {view.window} months, threshold {view.threshold:g}."
        ),
        metric=view.metric,
        unit=view.unit,
        fields=[
            ChartField(key="period", label="Month", role="x"),
            ChartField(key="value", label=view.metric_name, role="y", unit=view.unit),
            ChartField(key="expected", label="Expected", role="detail", unit=view.unit),
            ChartField(key="lower", label="Lower bound", role="lower", unit=view.unit),
            ChartField(key="upper", label="Upper bound", role="upper", unit=view.unit),
            ChartField(key="flagged", label="Flagged", role="flag"),
        ],
        rows=rows,
        evidence_ids=section.evidence_ids,
        source="anomaly_report",
        notes=[ANOMALY_NOTICE],
    )


# ------------------------------------------------------------------------------------------ all


def build_visualizations(
    evidence: Sequence[Evidence],
    claims: Sequence[Claim],
    kpis: Sequence[KPIValue],
    *,
    forecasts: Sequence[tuple[ForecastSection, ForecastResult]] = (),
    anomalies: Sequence[tuple[AnomalySection, AnomalyReport]] = (),
) -> list[VisualizationSpec]:
    specs = [
        *kpi_cards(kpis),
        *[forecast_chart(s, r) for s, r in forecasts],
        *[anomaly_chart(s, r) for s, r in anomalies],
        *comparisons(evidence, claims),
        *breakdowns(evidence),
        *time_series(evidence),
    ]
    return [s.model_copy(update={"chart_id": f"V{i}"}) for i, s in enumerate(specs, start=1)]
