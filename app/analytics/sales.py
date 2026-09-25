"""Sales analytics: pipeline, closed business, conversion, rep and segment performance, funnel stages.

Closed-deal metrics come from the registered KPIs (``win_rate`` / ``average_order_value`` /
``sales_cycle`` share one template, so a single breakdown query yields all three consistently).

Rep comparisons use neutral, descriptive language only (e.g. "lower observed conversion rate
than the team median"). They carry 95% Wilson intervals and are not made for reps below a
minimum number of closed opportunities.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from app.analytics.common import (
    PeriodSpec,
    business_as_of,
    filter_clause,
    median,
    to_filters,
    to_period,
    wilson_interval,
)
from app.analytics.dimensions import Filters
from app.analytics.errors import InvalidRequestError
from app.analytics.executor import QueryRunner
from app.analytics.kpis.service import KPIService
from app.analytics.kpis.sql import CLOSED_OPPORTUNITIES
from app.analytics.models import AnalyticsResult, safe_ratio, to_number
from app.database.base import Database
from app.database.metadata import FUNNEL_STAGES, OPEN_OPPORTUNITY_STAGES

MIN_CLOSED_FOR_REP_COMPARISON = 30
_SALES_COLUMNS = dict(CLOSED_OPPORTUNITIES.filter_columns)


class PipelineStageRow(BaseModel):
    stage: str
    open_opportunities: int
    pipeline_value: float
    weighted_value: float | None
    probability: float | None


class ClosedPerformanceRow(BaseModel):
    dimension_value: str
    closed_opportunities: int
    won_opportunities: int
    lost_opportunities: int
    win_rate: float | None
    closed_won_value: float
    closed_lost_value: float
    average_order_value: float | None
    average_sales_cycle_days: float | None


class RepPerformanceRow(BaseModel):
    sales_rep: str
    region: str
    opportunities_created: int
    closed_opportunities: int
    wins: int
    losses: int
    win_rate: float | None
    win_rate_ci_low: float | None
    win_rate_ci_high: float | None
    team_median_win_rate: float | None
    difference_from_team_median: float | None
    sufficient_sample: bool
    rank: int | None  # 1 = highest observed win rate among reps with a sufficient sample
    observation: str


class FunnelStageRow(BaseModel):
    stage: str
    reached: int
    advanced: int | None
    lost_at_stage: int | None
    stage_conversion_rate: float | None
    drop_off_rate: float | None


class ConversionRow(BaseModel):
    dimension_value: str
    opportunities_created: int
    won: int
    lost: int
    still_open: int
    created_to_won_rate: float | None
    resolved_win_rate: float | None


def pipeline_summary(
    db: Database, *, filters: Filters | dict[str, str] | None = None, as_of: date | None = None
) -> AnalyticsResult[PipelineStageRow]:
    """Open pipeline at the close of the as-of date, by current stage (weighted by CRM probability).

    Weighted values use each opportunity's current stage probability, which is only known for
    the current pipeline. For an earlier ``as_of`` the stage at that time is not recorded, so
    only the unweighted total (the ``pipeline_value`` KPI) is reported.
    """
    service = KPIService(db, as_of=as_of)
    as_of_date = business_as_of(as_of)
    active = to_filters(filters)
    kpi = service.calculate_kpi("pipeline_value", start_date=as_of_date, end_date=as_of_date, filters=active)
    runner = QueryRunner(db, "pipeline_summary")
    runner.absorb(kpi.provenance)
    _, last = service.coverage()
    rows: list[PipelineStageRow] = []
    limitations = list(kpi.limitations)
    if kpi.status == "ok" and as_of_date >= last:
        clause, bind = filter_clause(active, _SALES_COLUMNS, "pipeline_summary")
        records = runner.records(
            f"""
SELECT o.stage, COUNT(*) AS open_opportunities, SUM(o.deal_value) AS pipeline_value,
       SUM(o.deal_value * o.probability) AS weighted_value, MAX(o.probability) AS probability
FROM sales_opportunities AS o
WHERE o.close_date IS NULL{clause}
GROUP BY 1
""",
            bind,
            calculation="current open pipeline by stage; weighted value = deal_value x stage probability",
        )
        by_stage = {r["stage"]: r for r in records}
        for stage in OPEN_OPPORTUNITY_STAGES:
            r = by_stage.get(stage)
            rows.append(
                PipelineStageRow(
                    stage=stage,
                    open_opportunities=int(r["open_opportunities"]) if r else 0,
                    pipeline_value=_num(r["pipeline_value"]) if r else 0.0,
                    weighted_value=_num(r["weighted_value"]) if r else 0.0,
                    probability=_num(r["probability"]) if r else None,
                )
            )
    else:
        limitations.append("Stage breakdown and weighted pipeline are only available for the current pipeline.")
    return AnalyticsResult[PipelineStageRow](
        operation="pipeline_summary",
        status=kpi.status,
        filters=active,
        dimensions=["stage"] if rows else [],
        data=rows,
        summary={
            "as_of": as_of_date.isoformat(),
            "pipeline_value": kpi.value,
            "open_opportunities": kpi.components.get("open_opportunities"),
            "weighted_pipeline_value": sum(r.weighted_value or 0 for r in rows) if rows else None,
        },
        message=kpi.message,
        limitations=limitations,
        provenance=runner.provenance("pipeline_value KPI plus current stage breakdown"),
    )


def sales_performance(
    db: Database,
    period: PeriodSpec = None,
    *,
    dimension: str | None = None,
    filters: Filters | dict[str, str] | None = None,
    as_of: date | None = None,
) -> AnalyticsResult[ClosedPerformanceRow]:
    """Closed-won/lost value, win rate, average order value and sales cycle (optionally by a dimension)."""
    current = to_period(period, as_of)
    kpi = KPIService(db, as_of=as_of).calculate_kpi(
        "win_rate", start_date=current.start, end_date=current.end, dimension=dimension, filters=to_filters(filters)
    )
    groups = kpi.breakdown if dimension else []
    rows = [_closed_row(r.dimension_value, r.components) for r in groups]
    total = _closed_row("All", kpi.components) if kpi.components else None
    runner = QueryRunner(db, "sales_performance")
    runner.absorb(kpi.provenance)
    clause, bind = filter_clause(to_filters(filters), _SALES_COLUMNS, "sales_performance")
    median_won = runner.records(
        f"""
SELECT MEDIAN(datediff('day', o.created_date, o.close_date)) AS median_days
FROM sales_opportunities AS o
WHERE o.stage = 'Won' AND o.close_date BETWEEN $start_date AND $end_date{clause}
""",
        {"start_date": current.start, "end_date": current.end, **bind},
        calculation="median days from creation to close for won opportunities",
    )[0]["median_days"]
    summary = total.model_dump(exclude={"dimension_value"}) if total else {}
    summary["median_won_sales_cycle_days"] = to_number(median_won)
    return AnalyticsResult[ClosedPerformanceRow](
        operation="sales_performance",
        status=kpi.status,
        period=current,
        filters=kpi.filters,
        dimensions=[dimension] if dimension else [],
        data=rows,
        summary=summary,
        message=kpi.message,
        limitations=kpi.limitations,
        provenance=runner.provenance("win_rate / average_order_value / sales_cycle from one closed-opportunity query"),
    )


def segment_performance(
    db: Database, period: PeriodSpec = None, *, as_of: date | None = None
) -> AnalyticsResult[ClosedPerformanceRow]:
    """Closed-deal performance by segment."""
    result = sales_performance(db, period, dimension="segment", as_of=as_of)
    return result.model_copy(update={"operation": "segment_performance"})


def rep_performance(
    db: Database,
    period: PeriodSpec = "trailing_12_months",
    *,
    filters: Filters | dict[str, str] | None = None,
    min_closed: int = MIN_CLOSED_FOR_REP_COMPARISON,
    as_of: date | None = None,
) -> AnalyticsResult[RepPerformanceRow]:
    """Per-rep win rates vs the team median, with 95% Wilson intervals and minimum-sample flags."""
    current = to_period(period, as_of)
    active = to_filters(filters)
    kpi = KPIService(db, as_of=as_of).calculate_kpi(
        "win_rate", start_date=current.start, end_date=current.end, dimension="sales_rep", filters=active
    )
    runner = QueryRunner(db, "rep_performance")
    runner.absorb(kpi.provenance)
    clause, bind = filter_clause(active, _SALES_COLUMNS, "rep_performance")
    created = {
        r["sales_rep"]: r
        for r in runner.records(
            f"""
SELECT o.sales_rep, MIN(o.region) AS region,
       COUNT(*) FILTER (WHERE o.created_date BETWEEN $start_date AND $end_date) AS opportunities_created
FROM sales_opportunities AS o
WHERE TRUE{clause}
GROUP BY 1
""",
            {"start_date": current.start, "end_date": current.end, **bind},
            calculation="opportunities created in the period per rep (and the rep's territory)",
        )
    }
    stats = []
    for r in kpi.breakdown:
        wins, losses = int(r.components["won_opportunities"] or 0), int(r.components["lost_opportunities"] or 0)
        stats.append((r.dimension_value, wins, losses, r.value))
    eligible = [rate for _, w, lo, rate in stats if w + lo >= min_closed and rate is not None]
    team_median = median(eligible)
    ranked = sorted((rate, rep) for rep, w, lo, rate in stats if w + lo >= min_closed and rate is not None)
    rank_of = {rep: i for i, (_, rep) in enumerate(reversed(ranked), start=1)}
    rows = []
    for rep, wins, losses, rate in stats:
        closed = wins + losses
        interval = wilson_interval(wins, closed)
        sufficient = closed >= min_closed
        info = created.get(rep, {})
        rows.append(
            RepPerformanceRow(
                sales_rep=rep,
                region=str(info.get("region", "")),
                opportunities_created=int(info.get("opportunities_created", 0)),
                closed_opportunities=closed,
                wins=wins,
                losses=losses,
                win_rate=rate,
                win_rate_ci_low=interval[0] if interval else None,
                win_rate_ci_high=interval[1] if interval else None,
                team_median_win_rate=team_median,
                difference_from_team_median=(rate - team_median)
                if rate is not None and team_median is not None
                else None,
                sufficient_sample=sufficient,
                rank=rank_of.get(rep),
                observation=_rep_observation(sufficient, interval, team_median, min_closed),
            )
        )
    rows.sort(key=lambda r: (r.rank is None, r.rank or 0, r.sales_rep))
    below = [r.sales_rep for r in rows if r.observation.startswith("Lower observed")]
    return AnalyticsResult[RepPerformanceRow](
        operation="rep_performance",
        status=kpi.status,
        period=current,
        filters=active,
        dimensions=["sales_rep"],
        data=rows,
        summary={
            "team_median_win_rate": team_median,
            "reps_compared": len(eligible),
            "reps_below_median_with_confidence": ", ".join(below) or None,
            "min_closed": min_closed,
        },
        message=kpi.message,
        limitations=[
            *kpi.limitations,
            "Win rates depend on the mix of segments and deal types each rep handles; differences are observations, "
            "not assessments of individual performance.",
            f"Reps with fewer than {min_closed} closed opportunities are not compared or ranked.",
        ],
        provenance=runner.provenance(
            "win rate per rep = won / closed; 95% Wilson interval; team median over reps with a sufficient sample"
        ),
    )


def opportunity_conversion(
    db: Database,
    period: PeriodSpec = "trailing_12_months",
    *,
    dimension: str | None = None,
    filters: Filters | dict[str, str] | None = None,
    as_of: date | None = None,
) -> AnalyticsResult[ConversionRow]:
    """Outcome, as of the as-of date, of opportunities *created* in the period (a creation cohort)."""
    current = to_period(period, as_of)
    active = to_filters(filters)
    dims = {k: v for k, v in _SALES_COLUMNS.items() if k != "customer_id"}
    if dimension is not None and dimension not in dims:
        raise InvalidRequestError(f"opportunity_conversion supports dimensions {', '.join(dims)}")
    clause, bind = filter_clause(active, _SALES_COLUMNS, "opportunity_conversion")
    group = f"CAST({dims[dimension]} AS VARCHAR)" if dimension else "'All'"
    runner = QueryRunner(db, "opportunity_conversion")
    records = runner.records(
        f"""
SELECT {group} AS dimension_value, COUNT(*) AS created,
       COUNT(*) FILTER (WHERE o.stage = 'Won') AS won, COUNT(*) FILTER (WHERE o.stage = 'Lost') AS lost
FROM sales_opportunities AS o
WHERE o.created_date BETWEEN $start_date AND $end_date{clause}
GROUP BY 1
ORDER BY 1
""",
        {"start_date": current.start, "end_date": current.end, **bind},
        calculation="outcome of opportunities created in the period",
    )
    rows = [
        ConversionRow(
            dimension_value=str(r["dimension_value"]),
            opportunities_created=int(r["created"]),
            won=int(r["won"]),
            lost=int(r["lost"]),
            still_open=int(r["created"]) - int(r["won"]) - int(r["lost"]),
            created_to_won_rate=safe_ratio(r["won"], r["created"]),
            resolved_win_rate=safe_ratio(r["won"], int(r["won"]) + int(r["lost"])),
        )
        for r in records
    ]
    return AnalyticsResult[ConversionRow](
        operation="opportunity_conversion",
        status="ok" if rows else "no_data",
        period=current,
        filters=active,
        dimensions=[dimension] if dimension else [],
        data=rows,
        message=None if rows else "No opportunities were created in the period.",
        limitations=[
            "Recent creation cohorts still contain open opportunities, so created_to_won_rate rises as deals close.",
            "Conversion here is opportunity-to-win; marketing lead-to-customer conversion is the conversion_rate KPI.",
        ],
        provenance=runner.provenance("created_to_won_rate = won / created; resolved_win_rate = won / (won + lost)"),
    )


def funnel_stage_distribution(
    db: Database,
    period: PeriodSpec = "trailing_12_months",
    *,
    filters: Filters | dict[str, str] | None = None,
    as_of: date | None = None,
) -> AnalyticsResult[FunnelStageRow]:
    """Stage reach, stage-to-stage conversion and drop-off for opportunities closed in the period.

    ``furthest_stage`` records where each lost deal stopped; a deal that reached stage N also
    passed every earlier stage.
    """
    current = to_period(period, as_of)
    active = to_filters(filters)
    clause, bind = filter_clause(active, _SALES_COLUMNS, "funnel_stage_distribution")
    runner = QueryRunner(db, "funnel_stage_distribution")
    counts = {
        r["furthest_stage"]: int(r["n"])
        for r in runner.records(
            f"""
SELECT o.furthest_stage, COUNT(*) AS n
FROM sales_opportunities AS o
WHERE o.stage IN ('Won', 'Lost') AND o.close_date BETWEEN $start_date AND $end_date{clause}
GROUP BY 1
""",
            {"start_date": current.start, "end_date": current.end, **bind},
            calculation="closed opportunities by furthest stage reached",
        )
    }
    reached = [sum(counts.get(s, 0) for s in FUNNEL_STAGES[i:]) for i in range(len(FUNNEL_STAGES))]
    rows = []
    for i, stage in enumerate(FUNNEL_STAGES):
        is_last = i == len(FUNNEL_STAGES) - 1
        advanced = None if is_last else reached[i + 1]
        rows.append(
            FunnelStageRow(
                stage=stage,
                reached=reached[i],
                advanced=advanced,
                lost_at_stage=None if is_last else counts.get(stage, 0),
                stage_conversion_rate=None if is_last else safe_ratio(advanced, reached[i]),
                drop_off_rate=None if is_last else safe_ratio(counts.get(stage, 0), reached[i]),
            )
        )
    drops = [r for r in rows if r.drop_off_rate is not None]
    largest = max(drops, key=lambda r: r.drop_off_rate or 0.0) if drops and reached[0] else None
    return AnalyticsResult[FunnelStageRow](
        operation="funnel_stage_distribution",
        status="ok" if reached[0] else "no_data",
        period=current,
        filters=active,
        dimensions=["stage"],
        data=rows,
        summary={
            "closed_opportunities": reached[0],
            "largest_drop_off_stage": largest.stage if largest else None,
            "largest_drop_off_rate": largest.drop_off_rate if largest else None,
        },
        message=None if reached[0] else "No opportunities closed in the period.",
        limitations=["Only closed opportunities; open deals have not finished moving through the funnel."],
        provenance=runner.provenance(
            "reached(stage) = closed deals whose furthest stage is at or beyond it; conversion = reached(next) / "
            "reached(stage); drop-off = lost at stage / reached(stage)"
        ),
    )


# ---------------------------------------------------------------------------------------------------


def _num(value: object) -> float:
    number = to_number(value)
    return float(number) if number is not None else 0.0


def _closed_row(label: str, c: dict[str, float | int | str | None]) -> ClosedPerformanceRow:
    closed, won = int(c.get("closed_opportunities") or 0), int(c.get("won_opportunities") or 0)
    return ClosedPerformanceRow(
        dimension_value=label,
        closed_opportunities=closed,
        won_opportunities=won,
        lost_opportunities=int(c.get("lost_opportunities") or 0),
        win_rate=safe_ratio(won, closed),
        closed_won_value=_num(c.get("won_value")),
        closed_lost_value=_num(c.get("lost_value")),
        average_order_value=safe_ratio(_num(c.get("won_value")), won),
        average_sales_cycle_days=safe_ratio(_num(c.get("total_cycle_days")), closed),
    )


def _rep_observation(
    sufficient: bool, interval: tuple[float, float] | None, team_median: float | None, min_closed: int
) -> str:
    if not sufficient or interval is None or team_median is None:
        return f"Fewer than {min_closed} closed opportunities; not compared with the team."
    low, high = interval
    if high < team_median:
        return "Lower observed conversion rate than the team median (95% interval below the median)."
    if low > team_median:
        return "Higher observed conversion rate than the team median (95% interval above the median)."
    return "Observed conversion rate is not clearly different from the team median."
