"""Pure transformations from API responses (JSON dictionaries) to what the UI shows.

Nothing here talks to Streamlit, the network or the data. Numbers are only formatted, never
calculated: every value shown comes from the API response (evidence, claims, chart specs, forecast
and anomaly sections). Kept separate from ``render`` so it can be unit-tested without a browser.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from app.ui.client import APIFailure

Json = Mapping[str, Any]

# ------------------------------------------------------------------------------------------ vocabulary


BadgeColor = Literal["red", "orange", "yellow", "blue", "green", "violet", "gray", "grey", "primary"]


@dataclass(frozen=True)
class ClaimStyle:
    label: str
    color: BadgeColor  # a Streamlit badge colour
    icon: str
    note: str  # how to read a claim of this type


CLAIM_STYLES: dict[str, ClaimStyle] = {
    "observed_fact": ClaimStyle("Observed", "blue", ":material/visibility:", "Read directly from recorded data."),
    "calculated_result": ClaimStyle(
        "Calculated", "green", ":material/calculate:", "Computed from recorded data by a registered calculation."
    ),
    "inference": ClaimStyle(
        "Inferred", "orange", ":material/lightbulb:", "A reading of the evidence, not an observed fact."
    ),
    "recommendation": ClaimStyle(
        "Recommended", "violet", ":material/assistant_direction:", "A suggested next step, not a finding."
    ),
}
CLAIM_ORDER = tuple(CLAIM_STYLES)
SUPPORT_LABELS = {
    "supported": "Supported by evidence",
    "partially_supported": "Partially supported",
    "unsupported": "Not supported",
}

TOOL_LABELS = {
    "get_kpi": "KPI lookup",
    "analyze_revenue": "Revenue analysis",
    "analyze_customers": "Customer analysis",
    "analyze_sales": "Sales analysis",
    "analyze_marketing": "Marketing analysis",
    "analyze_support": "Support analysis",
    "analyze_product": "Product analysis",
    "get_cohort_analysis": "Cohort analysis",
    "get_customer_risk": "Customer risk scoring",
    "forecast_metric": "Forecast",
    "detect_anomalies": "Anomaly detection",
    "run_safe_sql": "Read-only query",
}


@dataclass(frozen=True)
class Banner:
    level: str  # "success", "info", "warning" or "error"
    title: str
    guidance: str | None = None


OUTCOME_BANNERS: dict[str, Banner] = {
    "answered": Banner("success", "Answer"),
    "partial": Banner(
        "warning",
        "Partial answer",
        "The explanation could not be fully validated against the evidence, so only validated findings are shown.",
    ),
    "refused": Banner(
        "error",
        "I can't answer that safely from the available data.",
        "Ask a business question about the dataset, for example about revenue, churn, CAC or support tickets.",
    ),
    "unsupported": Banner(
        "info",
        "I can't answer that from the available data.",
        "AgentOps answers questions about the Northwind Cloud business data: KPIs, period comparisons, breakdowns "
        "by region, segment or channel, forecasts and anomaly checks.",
    ),
    "insufficient_evidence": Banner(
        "warning",
        "There is not enough evidence for a reliable answer.",
        "Name the metric and the period (for example 'revenue last month' or 'churn in Q2'), "
        "or ask a narrower question.",
    ),
    "failed": Banner(
        "error",
        "The analysis could not be completed.",
        "Try again, or rephrase the question. The request ID below identifies the run in the service logs.",
    ),
}
REFUSAL_GUIDANCE = {
    "policy": "The request asked for something the agent is not permitted to do, so nothing was analysed.",
    "out_of_scope": "The question is outside the business data AgentOps can analyse.",
    "invalid_input": "The question could not be accepted as written. Rephrase it as a plain business question.",
}

# ------------------------------------------------------------------------------------------ formatting

_MONEY = ("SGD",)
_COUNTS = {"count", "customers", "tickets", "deals", "accounts", "users", "seats", "subscriptions"}


def format_value(value: Any, unit: str | None = None) -> str:
    """Display formatting only (thousands separators, currency, percentages). Never a calculation."""
    if value is None:
        return "—"
    if isinstance(value, bool) or not isinstance(value, int | float):
        return str(value)
    v = float(value)
    unit = unit or ""
    if unit.startswith(_MONEY):
        text = f"SGD {v:,.0f}" if abs(v) >= 100 else f"SGD {v:,.2f}"
        suffix = unit[len("SGD") :].strip()
        return f"{text} {suffix}".strip()
    if unit in ("ratio", "rate", "share"):
        return format_percent(v)
    if unit in _COUNTS:
        return f"{v:,.0f}" if v.is_integer() else f"{v:,.1f}"
    if unit in ("hours", "days", "months"):
        return f"{v:,.1f} {unit}"
    if v.is_integer() and abs(v) < 1e15:
        return f"{v:,.0f}" + (f" {unit}" if unit else "")
    return f"{v:,.4g}" + (f" {unit}" if unit else "")


def format_percent(fraction: Any, *, signed: bool = False) -> str:
    if not isinstance(fraction, int | float) or isinstance(fraction, bool):
        return "—"
    pct = float(fraction) * 100
    decimals = 2 if abs(pct) < 10 else 1
    return f"{pct:+.{decimals}f}%" if signed else f"{pct:.{decimals}f}%"


_MARKDOWN_SPECIAL = set("\\`*_{}[]()#+-.!|<>~$:")


def escape_markdown(text: Any) -> str:
    """Show text literally in Streamlit markdown: no links, images, emphasis, LaTeX ($) or :directives:."""
    return "".join(f"\\{ch}" if ch in _MARKDOWN_SPECIAL else ch for ch in str(text))


def format_ms(value: Any) -> str:
    if not isinstance(value, int | float):
        return ""
    return f"{value:,.0f} ms" if value >= 10 else f"{value:.1f} ms"


def _period_text(period: Json | None, comparison: Json | None) -> str | None:
    if not period:
        return None
    text = str(period.get("label", ""))
    if comparison:
        text += f" vs {comparison.get('label', '')}"
    return text


# ------------------------------------------------------------------------------------------ answer


@dataclass(frozen=True)
class AnswerView:
    outcome: str
    banner: Banner
    answer: str
    request_id: str
    period: str | None
    caveats: list[str]
    assumptions: list[str]
    refusal_note: str | None
    show_analysis: bool  # evidence, claims and charts are worth showing


def answer_view(response: Json) -> AnswerView:
    outcome = str(response.get("outcome", "failed"))
    banner = OUTCOME_BANNERS.get(outcome, OUTCOME_BANNERS["failed"])
    body = response.get("response") or {}
    refusal = response.get("refusal") or None
    return AnswerView(
        outcome=outcome,
        banner=banner,
        answer=str(response.get("answer", "")),
        request_id=str(response.get("request_id", "")),
        period=_period_text(response.get("period"), response.get("comparison_period")),
        caveats=[str(c) for c in body.get("caveats", [])],
        assumptions=[str(a) for a in body.get("assumptions", [])],
        refusal_note=REFUSAL_GUIDANCE.get(str(refusal.get("kind"))) if refusal else None,
        show_analysis=bool(response.get("evidence")) and outcome not in ("refused", "unsupported"),
    )


# ------------------------------------------------------------------------------------------ claims


@dataclass(frozen=True)
class ClaimItem:
    claim_id: str
    style: ClaimStyle
    text: str
    primary: bool
    support: str
    evidence_ids: list[str]
    marker: ClaimStyle | None = None  # a second label for forecast and anomaly claims


# Claim kinds whose type alone could overstate certainty or imply a judgement get a second label.
KIND_MARKERS: dict[str, ClaimStyle] = {
    "forecast": ClaimStyle(
        "Forecast", "yellow", ":material/query_stats:", "An estimate from a model, not observed data."
    ),
    "forecast_quality": ClaimStyle(
        "Forecast quality", "yellow", ":material/query_stats:", "Past accuracy of the model, not a guarantee."
    ),
    "anomaly": ClaimStyle(
        "Anomaly check", "gray", ":material/insights:", "Statistically unusual is not necessarily good or bad."
    ),
    "anomaly_summary": ClaimStyle(
        "Anomaly check", "gray", ":material/insights:", "Statistically unusual is not necessarily good or bad."
    ),
}


def claim_items(response: Json) -> list[ClaimItem]:
    """Claims that answer the question first, then by type (observed, calculated, inferred, recommended)."""
    claims = [c for c in response.get("claims", []) if isinstance(c, Mapping)]

    def order(item: tuple[int, Json]) -> tuple[int, int, int]:
        index, c = item
        kind = str(c.get("claim_type"))
        return (0 if c.get("primary") else 1, CLAIM_ORDER.index(kind) if kind in CLAIM_ORDER else 9, index)

    items = []
    for _, c in sorted(enumerate(claims), key=order):
        kind = str(c.get("claim_type"))
        style = CLAIM_STYLES.get(kind, ClaimStyle(kind.replace("_", " ").title(), "gray", ":material/info:", ""))
        items.append(
            ClaimItem(
                claim_id=str(c.get("claim_id", "")),
                style=style,
                text=str(c.get("text", "")),
                primary=bool(c.get("primary")),
                support=SUPPORT_LABELS.get(str(c.get("support_status")), str(c.get("support_status", ""))),
                evidence_ids=[str(e) for e in c.get("evidence_ids", [])],
                marker=KIND_MARKERS.get(str(c.get("kind"))),
            )
        )
    return items


# ------------------------------------------------------------------------------------------ evidence


def evidence_rows(response: Json) -> list[dict[str, str]]:
    """The provenance table: what each evidence item says and where it comes from.

    Shown: statement, metric, period, comparison period, dimension, filters, source tables,
    calculation, query IDs and tool. Not shown: raw tool arguments and method internals.
    """
    rows = []
    for e in response.get("evidence", []):
        if not isinstance(e, Mapping):
            continue
        dimension = e.get("dimension")
        member = e.get("dimension_value")
        filters = e.get("filters") or {}
        rows.append(
            {
                "Evidence": str(e.get("evidence_id", "")),
                "Type": str(e.get("evidence_type", "")).replace("_", " ").capitalize(),
                "Statement": str(e.get("statement", "")),
                "Metric": str(e.get("metric") or ""),
                "Value": str(e.get("display_value") or format_value(e.get("value"), e.get("unit"))),
                "Period": str(e.get("period_label") or ""),
                "Comparison": str(e.get("comparison_label") or ""),
                "Dimension": f"{dimension} = {member}" if dimension and member else str(dimension or ""),
                "Filters": ", ".join(f"{k} = {v}" for k, v in filters.items()),
                "Source tables": ", ".join(str(t) for t in e.get("source_tables", [])),
                "Calculation": str(e.get("calculation") or ""),
                "Query IDs": ", ".join(str(q) for q in e.get("query_ids", [])),
                "Tool": TOOL_LABELS.get(str(e.get("tool_name")), str(e.get("tool_name", ""))),
            }
        )
    return rows


# ------------------------------------------------------------------------------------------ trace


@dataclass(frozen=True)
class TraceRow:
    level: int  # 0: an agent stage; 1: a tool call inside "Running analysis tools"
    mark: str  # "✓", "✗" or "■" (the run stopped here)
    label: str
    duration: str
    detail: str


def trace_rows(response: Json) -> list[TraceRow]:
    """Stages of the run with their durations, and each tool call with its purpose (no prompts or reasoning)."""
    run = response.get("run") or {}
    tools = [
        TraceRow(
            level=1,
            mark="✓" if t.get("success") else "✗",
            label=TOOL_LABELS.get(str(t.get("tool_name")), str(t.get("tool_name", ""))),
            duration=format_ms(t.get("execution_time_ms")),
            detail="; ".join(
                part
                for part in (
                    str(t.get("purpose") or ""),
                    str(t.get("error") or "") if not t.get("success") else "",
                    f"{int(t.get('attempts', 1))} attempts" if int(t.get("attempts", 1)) > 1 else "",
                )
                if part
            ),
        )
        for t in response.get("trace", [])
        if isinstance(t, Mapping)
    ]
    stages = [s for s in run.get("stages", []) if isinstance(s, Mapping)]
    if not stages:
        return tools
    rows: list[TraceRow] = []
    inserted = False
    for s in stages:
        stage = str(s.get("stage", ""))
        if stage == "execute_tools" and inserted:
            continue  # one stage row per tool loop; its calls are listed once
        rows.append(
            TraceRow(
                level=0,
                mark="✓" if s.get("ok", True) else "■",
                label=str(s.get("label", stage)),
                duration=format_ms(s.get("duration_ms")),
                detail="",
            )
        )
        if stage == "execute_tools":
            rows.extend(tools)
            inserted = True
    if not inserted:
        rows.extend(tools)
    return rows


# ------------------------------------------------------------------------------------------ KPI cards


@dataclass(frozen=True)
class KPICard:
    label: str
    value: str
    delta: str | None
    caption: str
    help: str


def kpi_cards(spec: Json) -> list[KPICard]:
    cards = []
    for row in spec.get("rows", []):
        period = row.get("period")
        comparison = row.get("comparison_period")
        when = f"{period} vs {comparison}" if period and comparison else period or ""
        change = row.get("percentage_change")
        cards.append(
            KPICard(
                label=str(row.get("label", "")),
                value=str(row.get("display_value") or format_value(row.get("value"), row.get("unit"))),
                delta=format_percent(change, signed=True) if isinstance(change, int | float) else None,
                caption=str(when),
                help=f"Evidence {row.get('evidence_id', '')} ({row.get('evidence_type', '')})",
            )
        )
    return cards


# ------------------------------------------------------------------------------------------ charts

VEGA_SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"
_BLUE, _ORANGE, _GREY, _AMBER = "#2f6fdf", "#e08a3c", "#9aa5b1", "#d98e04"


def _axis_format(unit: str | None) -> str:
    if unit in ("ratio", "rate", "share"):
        return ".1%"
    return ",.0f" if unit and (unit.startswith("SGD") or unit in _COUNTS) else ",.4~g"


def _field(spec: Json, role: str) -> Json | None:
    return next((f for f in spec.get("fields", []) if f.get("role") == role), None)


def _y_title(spec: Json) -> str:
    y = _field(spec, "y") or {}
    unit = y.get("unit") or spec.get("unit")
    return f"{y.get('label', 'Value')} ({unit})" if unit else str(y.get("label", "Value"))


def vega_lite(spec: Json) -> dict[str, Any] | None:
    """A Vega-Lite chart for a chart spec (``None`` for KPI cards and tables, which are not charts)."""
    kind = spec.get("kind")
    rows = [dict(r) for r in spec.get("rows", [])]
    if not rows or kind in ("kpi_card", "table"):
        return None
    unit = spec.get("unit")
    y_axis = {"title": _y_title(spec), "format": _axis_format(unit)}
    base: dict[str, Any] = {"$schema": VEGA_SCHEMA, "data": {"values": rows}}
    period_x = {"field": "period", "type": "ordinal", "title": None, "axis": {"labelAngle": -45}}
    # Line charts show levels over time, so their axis need not start at zero (bars always do).
    line_y = {"field": "value", "type": "quantitative", "axis": y_axis, "scale": {"zero": False}}
    if kind == "comparison":
        return base | {
            "mark": {"type": "bar", "cornerRadiusEnd": 3},
            "encoding": {
                "x": {"field": "period", "type": "ordinal", "title": None, "sort": None, "axis": {"labelAngle": 0}},
                "y": {"field": "value", "type": "quantitative", "axis": y_axis},
                "color": {
                    "field": "role",
                    "type": "nominal",
                    "scale": {"domain": ["comparison", "current"], "range": [_GREY, _BLUE]},
                    "legend": None,
                },
                "tooltip": [
                    {"field": "period", "title": "Period"},
                    {"field": "value", "type": "quantitative", "title": "Value", "format": _axis_format(unit)},
                ],
            },
        }
    if kind == "bar":
        category = _field(spec, "category") or {"label": "Member"}
        return base | {
            "height": {"step": 28},
            "mark": {"type": "bar", "cornerRadiusEnd": 3},
            "encoding": {
                "y": {"field": "member", "type": "nominal", "sort": None, "title": category.get("label")},
                "x": {"field": "value", "type": "quantitative", "axis": y_axis},
                # Direction only (two neutral hues): a lower value is not labelled good or bad.
                "color": {"condition": {"test": "datum.value < 0", "value": _ORANGE}, "value": _BLUE},
                "tooltip": [
                    {"field": "member", "title": category.get("label")},
                    {"field": "display_value", "title": "Value"},
                ],
            },
        }
    if kind == "time_series":
        return base | {
            "mark": {"type": "line", "point": True},
            "encoding": {"x": period_x, "y": line_y},
        }
    if kind == "forecast":
        return base | {
            "layer": [
                {
                    "transform": [{"filter": "datum.series === 'forecast' && datum.lower != null"}],
                    "mark": {"type": "area", "opacity": 0.2, "color": _ORANGE},
                    "encoding": {
                        "x": period_x,
                        "y": {"field": "lower", "type": "quantitative", "axis": y_axis, "scale": {"zero": False}},
                        "y2": {"field": "upper"},
                    },
                },
                {
                    "transform": [{"filter": "datum.series === 'actual'"}],
                    "mark": {"type": "line", "color": _BLUE, "point": {"size": 20}},
                    "encoding": {"x": period_x, "y": {"field": "value", "type": "quantitative"}},
                },
                {
                    "transform": [{"filter": "datum.series === 'forecast'"}],
                    "mark": {"type": "line", "color": _ORANGE, "strokeDash": [6, 4], "point": {"size": 40}},
                    "encoding": {
                        "x": period_x,
                        "y": {"field": "value", "type": "quantitative"},
                        "tooltip": [
                            {"field": "period", "title": "Month"},
                            {"field": "value", "title": "Forecast", "format": _axis_format(unit)},
                            {"field": "lower", "title": "Lower bound", "format": _axis_format(unit)},
                            {"field": "upper", "title": "Upper bound", "format": _axis_format(unit)},
                        ],
                    },
                },
            ]
        }
    if kind == "anomaly":
        return base | {
            "layer": [
                {
                    "transform": [{"filter": "datum.lower != null && datum.upper != null"}],
                    "mark": {"type": "area", "opacity": 0.15, "color": _GREY},
                    "encoding": {
                        "x": period_x,
                        "y": {"field": "lower", "type": "quantitative", "axis": y_axis, "scale": {"zero": False}},
                        "y2": {"field": "upper"},
                    },
                },
                {
                    "transform": [{"filter": "datum.expected != null"}],
                    "mark": {"type": "line", "color": _GREY, "strokeDash": [4, 3]},
                    "encoding": {"x": period_x, "y": {"field": "expected", "type": "quantitative"}},
                },
                {
                    "mark": {"type": "line", "color": _BLUE, "point": {"size": 20}},
                    "encoding": {"x": period_x, "y": {"field": "value", "type": "quantitative"}},
                },
                {
                    "transform": [{"filter": "datum.flagged === true"}],
                    "mark": {"type": "point", "shape": "diamond", "size": 160, "filled": True, "color": _AMBER},
                    "encoding": {
                        "x": period_x,
                        "y": {"field": "value", "type": "quantitative"},
                        "tooltip": [
                            {"field": "period", "title": "Month"},
                            {"field": "value", "title": "Observed", "format": _axis_format(unit)},
                            {"field": "expected", "title": "Expected", "format": _axis_format(unit)},
                            {"field": "severity", "title": "Severity"},
                            {"field": "direction", "title": "Direction"},
                            {"field": "score", "title": "Score", "format": ".2f"},
                        ],
                    },
                },
            ]
        }
    return None


def table_rows(spec: Json) -> list[dict[str, str]]:
    """A table spec's rows with its column labels and formatted values."""
    fields = [f for f in spec.get("fields", []) if isinstance(f, Mapping)]
    rows = []
    for row in spec.get("rows", []):
        rows.append(
            {
                str(f.get("label", f.get("key"))): (
                    str(row.get(f["key"]))
                    if isinstance(row.get(f["key"]), str)
                    else format_value(row.get(f["key"]), f.get("unit"))
                )
                for f in fields
            }
        )
    return rows


# ------------------------------------------------------------------------------------------ forecasts


@dataclass(frozen=True)
class ForecastPanel:
    title: str
    notice: str
    facts: list[tuple[str, str]]  # horizon, model, data cut-off, interval
    points: list[dict[str, str]]
    backtest: list[tuple[str, str]]
    limitations: list[str] = field(default_factory=list)


def forecast_panel(section: Json) -> ForecastPanel:
    f = section.get("forecast") or {}
    unit = f.get("unit")
    level = f.get("confidence_level")
    interval = f"{level:.0%} prediction interval" if f.get("interval_available") and level else "No interval"
    points = [
        {
            "Month": str(p.get("period", "")),
            "Forecast": format_value(p.get("predicted_value"), unit),
            "Lower bound": format_value(p.get("lower_bound"), unit),
            "Upper bound": format_value(p.get("upper_bound"), unit),
        }
        for p in f.get("points", [])
    ]
    backtest: list[tuple[str, str]] = []
    selected, baseline = f.get("backtest") or {}, f.get("baseline") or {}
    if selected.get("mae") is not None:
        folds = f" over {selected.get('fold_count')} folds" if selected.get("fold_count") else ""
        backtest.append(("Backtest mean absolute error", format_value(selected["mae"], unit) + folds))
    if selected.get("wape") is not None:
        backtest.append(("Backtest weighted absolute % error", format_percent(selected["wape"])))
    if selected.get("interval_coverage") is not None:
        backtest.append(("Actuals inside the backtest intervals", format_percent(selected["interval_coverage"])))
    if baseline.get("mae") is not None:
        backtest.append(("Naive baseline mean absolute error", format_value(baseline["mae"], unit)))
    horizon = f.get("horizon")
    return ForecastPanel(
        title=f"Forecast: {f.get('metric_name', f.get('metric', ''))}",
        notice=str(section.get("notice", "")),
        facts=[
            ("Horizon", f"{horizon} month{'s' if horizon != 1 else ''}" if horizon else "—"),
            ("Model", str(f.get("model") or "—")),
            ("Data up to", str(f.get("cutoff_date") or "—")),
            ("Interval", interval),
        ],
        points=points,
        backtest=backtest,
        limitations=[str(x) for x in section.get("limitations", [])],
    )


# ------------------------------------------------------------------------------------------ anomalies

DIRECTION_LABELS = {"positive": "Above expected", "negative": "Below expected", "none": "—"}


@dataclass(frozen=True)
class AnomalyPanel:
    title: str
    notice: str
    summary: str
    flagged: list[dict[str, str]]
    limitations: list[str] = field(default_factory=list)


def anomaly_panel(section: Json) -> AnomalyPanel:
    r = section.get("report") or {}
    unit = r.get("unit")
    flagged = [
        {
            "Metric": str(r.get("metric_name", r.get("metric", ""))),
            "Month": str(a.get("period", "")),
            "Observed": format_value(a.get("observed_value"), unit),
            "Expected": format_value(a.get("expected_value"), unit),
            "Deviation": format_percent(a.get("deviation_percentage"), signed=True),
            "Score": f"{a['score']:+.2f}" if isinstance(a.get("score"), int | float) else "unbounded",
            "Severity": str(a.get("severity", "")),
            "Direction": DIRECTION_LABELS.get(str(a.get("direction")), str(a.get("direction", ""))),
        }
        for a in r.get("flagged", [])
    ]
    detector = str(r.get("detector", "")).replace("_", " ")
    summary = (
        f"{len(flagged)} of {r.get('scored_periods', 0)} months flagged by the {detector} detector "
        f"(window {r.get('window')} months, threshold {r.get('threshold')})."
    )
    return AnomalyPanel(
        title=f"Anomaly check: {r.get('metric_name', r.get('metric', ''))}",
        notice=str(section.get("notice", "")),
        summary=summary,
        flagged=flagged,
        limitations=[str(x) for x in section.get("limitations", [])],
    )


# ------------------------------------------------------------------------------------------ errors and history


@dataclass(frozen=True)
class ErrorView:
    title: str
    message: str
    hint: str | None
    request_id: str | None


def error_view(failure: APIFailure) -> ErrorView:
    if failure.kind == "connection":
        return ErrorView(
            "The analysis service is not reachable.",
            failure.message,
            "Start the API with `python -m app.api` (or check UI_API_URL), then try again.",
            None,
        )
    if failure.kind == "timeout":
        return ErrorView("The analysis took too long.", failure.message, "Try a narrower question.", None)
    hints = {
        "empty_question": "Type a question first.",
        "question_too_long": "Shorten the question.",
        "busy": "Another analysis is running. Try again in a few seconds.",
        "timeout": "Try a narrower question.",
        "agent_unavailable": "Check that the database has been generated and restart the API.",
    }
    hint = hints.get(failure.code or "")
    return ErrorView("The request could not be completed.", failure.message, hint, failure.request_id)


@dataclass(frozen=True)
class HistoryEntry:
    question: str
    outcome: str
    answer: str
    request_id: str | None
    asked_at: str


def history_entry(question: str, *, response: Json | None = None, failure: APIFailure | None = None) -> HistoryEntry:
    now = datetime.now().strftime("%H:%M:%S")
    if response is not None:
        return HistoryEntry(
            question=question,
            outcome=str(response.get("outcome", "")),
            answer=str(response.get("answer", "")),
            request_id=str(response.get("request_id", "")) or None,
            asked_at=now,
        )
    message = failure.message if failure else "No response."
    return HistoryEntry(question, "error", message, failure.request_id if failure else None, now)


def visualization_groups(response: Json) -> dict[str, list[Json]]:
    """Chart specs by kind, in response order (KPI cards, charts, tables)."""
    groups: dict[str, list[Json]] = {"kpi_card": [], "chart": [], "table": []}
    for spec in response.get("visualizations", []):
        if not isinstance(spec, Mapping):
            continue
        kind = spec.get("kind")
        groups["kpi_card" if kind == "kpi_card" else "table" if kind == "table" else "chart"].append(spec)
    return groups


def examples(capabilities: Json | None, fallback: Sequence[str]) -> list[str]:
    """Example questions from the API's capabilities, else the UI's own list."""
    listed = capabilities.get("example_questions") if capabilities else None
    return [str(q) for q in listed] if isinstance(listed, list) and listed else list(fallback)
