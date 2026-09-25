"""Analytics modules on the full dataset: reconciliation, cross-checks against the independent
reference, denominators, wording rules and the risk backtest."""

from __future__ import annotations

import re
from datetime import date, timedelta

import pandas as pd
import pytest

from app.analytics import cohorts, customers, marketing, product, revenue, risk, sales, support
from app.analytics.common import median, wilson_interval
from app.analytics.kpis import calculate_kpi
from app.analytics.periods import explicit_period, month_period
from app.database.base import Database
from tests.integration.reference_kpis import Reference, ts

pytestmark = pytest.mark.slow

EVALUATIVE_WORDS = re.compile(r"\b(bad|weak|poor|underperform\w*|failing|lazy)\b", re.IGNORECASE)
CAUSAL_WORDS = re.compile(r"\b(caused|causes|because of|due to|will churn|led to)\b", re.IGNORECASE)


# ---- revenue ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dimension", revenue.DECOMPOSITION_DIMENSIONS)
def test_decomposition_reconciles_and_matches_reference(
    full_db: Database, reference: Reference, dimension: str
) -> None:
    result = revenue.decompose_revenue_change(full_db, dimension, "last_month")
    s = result.summary
    assert s["reconciled"] is True and abs(s["reconciliation_difference"]) <= 0.01
    rows = result.data
    assert sum(r.absolute_change for r in rows) == pytest.approx(s["total_change"], abs=0.01)
    assert sum(r.share_of_total_change or 0 for r in rows) == pytest.approx(1.0)
    assert sum(r.contribution_to_total_change or 0 for r in rows) == pytest.approx(s["total_percentage_change"])
    declines = [r for r in rows if r.absolute_change < 0]
    if declines:
        assert sum(r.share_of_gross_decline or 0 for r in declines) == pytest.approx(1.0)
    assert [r.absolute_change for r in rows] == sorted(r.absolute_change for r in rows)
    assert s["direction"] == ("decline" if s["total_change"] < 0 else "increase")
    for row in rows[:3]:  # spot-check members against the reference
        filters = {dimension: row.dimension_value}
        assert row.current_value == pytest.approx(
            reference.revenue(date(2026, 8, 1), date(2026, 8, 31), filters)["revenue"], abs=0.01
        )
        assert row.previous_value == pytest.approx(
            reference.revenue(date(2026, 7, 1), date(2026, 7, 31), filters)["revenue"], abs=0.01
        )


@pytest.mark.parametrize(
    "period",
    [month_period(2026, 8), month_period(2025, 2), explicit_period(date(2025, 3, 17), date(2025, 11, 2))],
)
def test_mrr_bridge_reconciles(full_db: Database, reference: Reference, period) -> None:  # type: ignore[no-untyped-def]
    result = revenue.revenue_bridge(full_db, period)
    assert result.summary["reconciled"] is True
    rows = {r.component: r for r in result.data}
    opening = reference.recurring_state(period.opening_date, {})["mrr"]
    closing = reference.recurring_state(period.end, {})["mrr"]
    assert rows["opening_mrr"].mrr == pytest.approx(opening, abs=0.01)
    assert rows["closing_mrr"].mrr == pytest.approx(closing, abs=0.01)
    subs = reference.t["subscriptions"]
    new = subs[
        (subs["change_type"] == "new")
        & (subs["start_date"] >= ts(period.start))
        & (subs["start_date"] <= ts(period.end))
    ]
    assert rows["new"].mrr == pytest.approx(new["monthly_recurring_revenue"].sum(), abs=0.01)
    assert rows["contraction"].mrr <= 0 <= rows["expansion"].mrr and rows["churn"].mrr <= 0
    assert rows["reactivation"].mrr is None  # not observed in the data model: never reported as zero


def test_mrr_series_matches_phase1_view_and_kpi(full_db: Database) -> None:
    series = revenue.mrr_series(full_db, "trailing_12_months")
    view = dict(full_db.query("SELECT strftime(month, '%Y-%m'), SUM(mrr) FROM v_monthly_mrr GROUP BY 1").rows)
    assert len(series.data) == 12
    for row in series.data:
        assert row.mrr == pytest.approx(float(view[row.month]), abs=0.01)
        assert row.arr == pytest.approx(row.mrr * 12)
    last = series.data[-1]
    assert last.mrr == pytest.approx(calculate_kpi(full_db, "mrr", period="last_month").value)


def test_revenue_by_period_sums_to_kpi(full_db: Database) -> None:
    result = revenue.revenue_by_period(full_db, "2025", grain="quarter")
    assert [r.period for r in result.data] == ["2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4"]
    assert sum(r.revenue for r in result.data) == pytest.approx(result.summary["total_revenue"], abs=0.02)
    for row in result.data:
        assert row.subscription_revenue + row.usage_revenue == pytest.approx(row.revenue, abs=0.01)


def test_revenue_concentration(full_db: Database, reference: Reference) -> None:
    result = revenue.revenue_concentration(full_db, "last_month", top_n=5)
    shares = [r.cumulative_share for r in result.data]
    assert shares == sorted(shares) and shares[-1] <= 1
    r = reference.t["daily_revenue"]
    month = r[(r["date"] >= ts(date(2026, 8, 1))) & (r["date"] <= ts(date(2026, 8, 31)))]
    top = month.groupby("customer_id")["revenue"].sum().sort_values(ascending=False)
    assert result.data[0].customer_id == top.index[0]
    assert result.data[0].share_of_revenue == pytest.approx(top.iloc[0] / top.sum())


# ---- customers, churn, cohorts ------------------------------------------------------------------------------


@pytest.mark.parametrize("period", ["last_month", "2026-Q2", "2025"])
def test_customer_movements_identity(full_db: Database, period: str) -> None:
    result = customers.customer_movements(full_db, period)
    s = result.summary
    assert s["closing_customers"] == s["opening_customers"] + s["new_customers"] - s["churned_customers"]
    assert s["closing_customers"] == calculate_kpi(full_db, "customer_count", period=period).value
    churn = calculate_kpi(full_db, "logo_churn_rate", period=period)
    assert s["opening_customers"] == churn.components["opening_customers"]
    by_segment = customers.customer_movements(full_db, period, dimension="segment")
    assert sum(r.closing_customers for r in by_segment.data) == s["closing_customers"]


def test_churn_by_dimension(full_db: Database, reference: Reference) -> None:
    result = customers.churn_by_dimension(full_db, "segment", "trailing_12_months")
    period = result.period
    assert period is not None
    for row in result.data:
        expected = reference.logo_churn(period.start, period.end, {"segment": row.dimension_value})
        assert row.logo_churn_rate == pytest.approx(expected)
        assert row.logo_churn_ci_low <= row.logo_churn_rate <= row.logo_churn_ci_high  # type: ignore[operator]
    ranked = [r.logo_churn_rate for r in result.data if r.sufficient_sample]
    assert ranked == sorted(ranked, reverse=True)
    assert 0 <= result.summary["overall_logo_churn_rate"] <= 1


def test_monthly_churn_series_matches_monthly_kpi(full_db: Database) -> None:
    result = customers.monthly_churn_series(full_db, "2026-Q2")
    assert [r.month for r in result.data] == ["2026-04", "2026-05", "2026-06"]
    for row in result.data:
        kpi = calculate_kpi(full_db, "logo_churn_rate", period=row.month)
        assert row.logo_churn_rate == kpi.value
        nrr = calculate_kpi(full_db, "nrr", period=row.month)
        assert row.net_revenue_retention == pytest.approx(nrr.value)


def test_churn_summary_is_one_consistent_base(full_db: Database) -> None:
    s = customers.churn_summary(full_db, "last_quarter").summary
    assert s["retention_rate"] == pytest.approx(1 - s["logo_churn_rate"])
    assert 0 <= s["logo_churn_rate"] <= 1 and s["nrr"] > 0


def test_usage_churn_relationship_is_segmented_and_associative(full_db: Database) -> None:
    result = customers.usage_churn_relationship(full_db, "2026-Q2")
    segments = {r.segment for r in result.data}
    assert {"SMB", "Mid-Market", "Enterprise", "All"} <= segments
    for outcome in ("churned in period", "retained"):
        by_segment = sum(r.customers for r in result.data if r.outcome == outcome and r.segment != "All")
        total = next(r.customers for r in result.data if r.outcome == outcome and r.segment == "All")
        assert by_segment == total
    assert not any(CAUSAL_WORDS.search(text) for text in result.limitations)


def test_cohort_denominators_and_bounds(full_db: Database, reference: Reference) -> None:
    result = cohorts.cohort_retention(full_db)
    c = reference.t["customers"]
    in_window = c[c["signup_date"] >= ts(date(2024, 9, 1))]
    expected_sizes = in_window.groupby(in_window["signup_date"].dt.strftime("%Y-%m")).size()
    sizes = {cell.cohort_month: cell.cohort_size for cell in result.data}
    assert sizes == expected_sizes.to_dict()
    assert min(sizes) == "2024-09"  # the window start does not create an earlier artificial cohort
    assert result.summary["pre_window_customers_excluded"] == int((c["signup_date"] < ts(date(2024, 9, 1))).sum())
    for cell in result.data:
        assert 0 <= cell.active_customers <= cell.cohort_size
        assert 0 <= cell.logo_retention <= 1
    matrix = cohorts.cohort_matrix(result)
    assert max(max(row) for row in matrix.values()) <= 23 and len(matrix["2026-08"]) == 1
    # Independent check of one cell: cohort 2025-01, 6 months after signup (month end 2025-07-31).
    members = set(in_window[in_window["signup_date"].dt.strftime("%Y-%m") == "2025-01"]["customer_id"])
    active = reference.in_force(date(2025, 7, 31))
    cell = next(x for x in result.data if x.cohort_month == "2025-01" and x.months_since_signup == 6)
    assert cell.active_customers == active[active["customer_id"].isin(members)]["customer_id"].nunique()
    base = reference.in_force(date(2025, 1, 31))
    assert cell.cohort_initial_mrr == pytest.approx(
        base[base["customer_id"].isin(members)]["monthly_recurring_revenue"].sum(), abs=0.01
    )


# ---- risk ---------------------------------------------------------------------------------------------------


def test_risk_scores_are_transparent_and_observable(full_db: Database) -> None:
    result = risk.score_customer_risk(full_db)
    assert set(result.source_tables) <= {"customers", "subscriptions", "usage_events", "support_tickets"}
    assert result.summary["customers_scored"] == calculate_kpi(full_db, "customer_count").value
    scores = [c.risk_score for c in result.data]
    assert scores == sorted(scores, reverse=True)
    rules = risk.RULES
    for c in result.data:
        assert c.risk_score == min(100, sum(s.points for s in c.signals))
        expected_band = (
            "high" if c.risk_score >= rules.high_band else "medium" if c.risk_score >= rules.medium_band else "low"
        )
        assert c.risk_band == expected_band
        for signal in c.signals:
            assert not CAUSAL_WORDS.search(signal.description + signal.observed)


def test_risk_bands_rank_subsequent_churn(full_db: Database, reference: Reference) -> None:
    """Time-based validation: score at a past date with data up to that date only, then compare churn
    observed over the following three months by band."""
    churn_dates = reference.churn_dates()
    pooled = {"high": [0, 0], "medium": [0, 0], "low": [0, 0]}
    for as_of in (date(2025, 8, 31), date(2025, 11, 30), date(2026, 2, 28), date(2026, 5, 31)):
        until = as_of + timedelta(days=92)
        churned = set(churn_dates[(churn_dates > ts(as_of)) & (churn_dates <= ts(until))].index)
        scored = risk.score_customer_risk(full_db, as_of=as_of).data
        rate = {}
        for band in pooled:
            members = [c for c in scored if c.risk_band == band]
            hits = sum(c.customer_id in churned for c in members)
            pooled[band][0] += hits
            pooled[band][1] += len(members)
            rate[band] = hits / len(members)
        assert rate["high"] > rate["low"] and rate["medium"] > rate["low"], (as_of, rate)
    pooled_rate = {band: hits / n for band, (hits, n) in pooled.items()}
    assert pooled_rate["high"] > pooled_rate["medium"] > pooled_rate["low"]


# ---- sales ----------------------------------------------------------------------------------------------------


def test_rep_performance(full_db: Database, reference: Reference) -> None:
    result = sales.rep_performance(full_db, "trailing_12_months")
    period = result.period
    assert period is not None
    closed = reference.closed(period.start, period.end, {})
    eligible = []
    for row in result.data:
        mine = closed[closed["sales_rep"] == row.sales_rep]
        assert row.wins == int((mine["stage"] == "Won").sum()) and row.losses == int((mine["stage"] == "Lost").sum())
        if row.sufficient_sample:
            eligible.append(row.win_rate)
            assert row.win_rate_ci_low <= row.win_rate <= row.win_rate_ci_high  # type: ignore[operator]
        else:
            assert row.rank is None and "not compared" in row.observation
        assert not EVALUATIVE_WORDS.search(row.observation)
    assert result.summary["team_median_win_rate"] == pytest.approx(median(eligible))
    assert all(not EVALUATIVE_WORDS.search(text) for text in result.limitations)


def test_rep_performance_small_sample_is_not_ranked(full_db: Database) -> None:
    result = sales.rep_performance(full_db, "last_month", min_closed=1000)
    assert all(r.rank is None and not r.sufficient_sample for r in result.data)


def test_sales_performance_and_funnel(full_db: Database, reference: Reference) -> None:
    perf = sales.sales_performance(full_db, "2025")
    closed = reference.closed(date(2025, 1, 1), date(2025, 12, 31), {})
    won = closed[closed["stage"] == "Won"]
    assert perf.summary["closed_won_value"] == pytest.approx(won["deal_value"].sum(), abs=0.01)
    assert perf.summary["median_won_sales_cycle_days"] == pytest.approx(
        (won["close_date"] - won["created_date"]).dt.days.median()
    )
    funnel = sales.funnel_stage_distribution(full_db, "2025")
    reached = [r.reached for r in funnel.data]
    assert reached[0] == len(closed) and reached == sorted(reached, reverse=True)
    for row in funnel.data[:-1]:
        assert row.advanced + row.lost_at_stage == row.reached  # type: ignore[operator]
    assert funnel.data[-1].reached == len(won)


def test_opportunity_conversion_and_pipeline(full_db: Database, reference: Reference) -> None:
    conv = sales.opportunity_conversion(full_db, "2025", dimension="segment")
    o = reference.t["sales_opportunities"]
    created = o[(o["created_date"] >= ts(date(2025, 1, 1))) & (o["created_date"] <= ts(date(2025, 12, 31)))]
    assert sum(r.opportunities_created for r in conv.data) == len(created)
    for row in conv.data:
        assert row.won + row.lost + row.still_open == row.opportunities_created
    pipeline = sales.pipeline_summary(full_db)
    assert sum(r.pipeline_value for r in pipeline.data) == pytest.approx(pipeline.summary["pipeline_value"], abs=0.01)
    assert pipeline.summary["weighted_pipeline_value"] <= pipeline.summary["pipeline_value"]
    historical = sales.pipeline_summary(full_db, as_of=date(2025, 12, 31))
    assert historical.data == [] and historical.summary["weighted_pipeline_value"] is None
    assert historical.summary["pipeline_value"] == pytest.approx(reference.pipeline(date(2025, 12, 31), {}), abs=0.01)


# ---- marketing ---------------------------------------------------------------------------------------------


def test_channel_and_campaign_totals_reconcile(full_db: Database, reference: Reference) -> None:
    channels = marketing.channel_performance(full_db, "2025")
    campaigns = marketing.campaign_performance(full_db, "2025")
    expected = reference.marketing(date(2025, 1, 1), date(2025, 12, 31), {})
    assert sum(r.spend for r in channels.data) == pytest.approx(expected["spend"], abs=0.01)
    assert sum(r.spend for r in campaigns.data) == pytest.approx(expected["spend"], abs=0.01)
    assert sum(r.conversions for r in campaigns.data) == expected["conversions"]
    assert {r.channel for r in channels.data} <= {"Paid Search", "Paid Social", "Email", "Organic", "Partner", "Events"}
    for row in campaigns.data:
        if row.cac is not None:
            assert row.cac == pytest.approx(row.spend / row.conversions)
        if not row.sufficient_sample:
            assert row.cac_vs_channel_median is None and not row.high_cac_relative_to_channel
        elif row.cac_vs_channel_median is not None:
            assert row.high_cac_relative_to_channel == (row.cac_vs_channel_median >= marketing.HIGH_CAC_MULTIPLE)


def test_channel_roas(full_db: Database, reference: Reference) -> None:
    recent = marketing.channel_roas(full_db, "last_month")
    assert recent.status == "insufficient_data" and not recent.data
    result = marketing.channel_roas(full_db, "2026-Q1", window_days=90)
    c = reference.t["customers"]
    acquired = c[
        (c["signup_date"] >= ts(date(2026, 1, 1)))
        & (c["signup_date"] <= ts(date(2026, 3, 31)))
        & (c["acquisition_channel"] == "Paid Search")
    ]
    r = reference.t["daily_revenue"].merge(acquired[["customer_id", "signup_date"]], on="customer_id")
    window = r[(r["date"] >= r["signup_date"]) & (r["date"] <= r["signup_date"] + pd.Timedelta(days=89))]
    row = next(x for x in result.data if x.channel == "Paid Search")
    assert row.customers_acquired == len(acquired)
    assert row.attributed_revenue == pytest.approx(window["revenue"].sum(), abs=0.01)
    assert row.roas == pytest.approx(row.attributed_revenue / row.spend)


# ---- support ------------------------------------------------------------------------------------------------


def test_support_summary_and_breakdowns(full_db: Database, reference: Reference) -> None:
    summary = support.support_summary(full_db, "last_month").summary
    tickets = reference.tickets(date(2026, 8, 1), date(2026, 8, 31), {})
    assert summary["tickets"] == len(tickets)
    assert summary["resolved_tickets"] + summary["unresolved_tickets"] == summary["tickets"]
    resolved = tickets[tickets["status"] == "Resolved"]["resolution_time"]
    assert summary["median_resolution_hours"] == pytest.approx(resolved.median())
    lifetimes = reference.t["customers"].assign(
        churn=reference.t["customers"]["customer_id"].map(reference.churn_dates())
    )
    active = lifetimes[
        (lifetimes["signup_date"] <= ts(date(2026, 8, 31)))
        & (lifetimes["churn"].isna() | (lifetimes["churn"] >= ts(date(2026, 8, 1))))
    ]
    assert summary["active_customers_in_period"] == len(active)
    by_category = support.support_by_dimension(full_db, "ticket_category", "last_month")
    assert sum(r.tickets for r in by_category.data) == len(tickets)
    assert sum(r.share_of_tickets or 0 for r in by_category.data) == pytest.approx(1.0)
    change = support.support_volume_change(full_db, "2026-06", "2026-05")
    may = len(reference.tickets(date(2026, 5, 1), date(2026, 5, 31), {}))
    june = len(reference.tickets(date(2026, 6, 1), date(2026, 6, 30), {}))
    assert change.summary["ticket_volume_change"] == pytest.approx(june / may - 1)


# ---- product -------------------------------------------------------------------------------------------------


def test_product_adoption_views(full_db: Database, reference: Reference) -> None:
    features = product.feature_adoption(full_db, "last_month")
    assert len(features.data) == reference.t["product_features"]["feature_name"].nunique()
    for row in features.data:
        assert row.average_daily_adoption_rate == pytest.approx(
            reference.adoption(date(2026, 8, 1), date(2026, 8, 31), row.feature)
        )
    trend = product.adoption_trend(full_db, "Reports", "2026-Q2")
    assert [r.period for r in trend.data] == ["2026-04", "2026-05", "2026-06"]
    assert trend.data[0].average_daily_adoption_rate == pytest.approx(
        reference.adoption(date(2026, 4, 1), date(2026, 4, 30), "Reports")
    )
    p = reference.t["product_features"]
    first = p[p["feature_name"] == "AI Insights"]["date"].min().date()
    launch = product.feature_launch_summary(full_db, "AI Insights")
    assert launch.summary["first_observed_date"] == first.isoformat()
    breadth = product.adoption_breadth_by(full_db, "segment", "last_month")
    u = reference.t["usage_events"].merge(reference.t["customers"][["customer_id", "segment"]], on="customer_id")
    u = u[(u["event_date"] >= ts(date(2026, 8, 1))) & (u["event_date"] <= ts(date(2026, 8, 31)))]
    expected = u.groupby("segment")["feature_usage"].mean()
    for row in breadth.data:
        assert row.average_features_used == pytest.approx(expected[row.dimension_value])
    distribution = product.feature_usage_distribution(full_db, "last_month")
    assert sum(r.share for r in distribution.data) == pytest.approx(1.0)


# ---- smaller window ---------------------------------------------------------------------------------------------


def test_modules_run_on_the_small_fixture_window(small_db: Database) -> None:
    """The layer adapts to a different data window (Mar-Aug 2026, 100 customers)."""
    assert revenue.decompose_revenue_change(small_db, "segment", "last_month").summary["reconciled"] is True
    assert revenue.revenue_bridge(small_db, "last_quarter").summary["reconciled"] is True
    cohort = cohorts.cohort_retention(small_db)
    assert min(c.cohort_month for c in cohort.data) == "2026-03"
    assert calculate_kpi(small_db, "revenue", period="2025").status == "no_data"
    assert calculate_kpi(small_db, "logo_churn_rate", period="2026-03").status == "ok"
    assert calculate_kpi(small_db, "logo_churn_rate", period="2026-02").status == "insufficient_data"


def test_wilson_interval_reference_values() -> None:
    low, high = wilson_interval(20, 100)  # type: ignore[misc]
    assert low == pytest.approx(0.1333, abs=1e-4) and high == pytest.approx(0.2888, abs=1e-4)
    assert wilson_interval(0, 0) is None
