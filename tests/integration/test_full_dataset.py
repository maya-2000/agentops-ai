"""Full-scale dataset (the real configuration) built from scratch into a temp directory.

Covers the Phase 1 acceptance criteria that need realistic scale: counts and dimensions,
injected-event validation, realism ranges and hidden-data exposure. Realism assertions use
wide ranges so they test plausibility, not exact values.
"""

from __future__ import annotations

import json
import re
from typing import Any

import duckdb
import pytest

from app.database.metadata import COUNTRIES, PLANS, REGIONS, SEGMENTS, TABLES, VIEWS
from data.generator.generate import GenerationResult

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def con(full_dataset: GenerationResult) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(str(full_dataset.config.db_path), read_only=True)
    yield connection
    connection.close()


def scalar(con: duckdb.DuckDBPyConnection, sql: str) -> Any:
    return con.execute(sql).fetchone()[0]


# ---- scale and dimensions ------------------------------------------------------------------------------


def test_business_dimensions(full_dataset: GenerationResult, con: duckdb.DuckDBPyConnection) -> None:
    m = full_dataset.manifest
    assert m["customer_count"] == 5000 == scalar(con, "SELECT COUNT(*) FROM customers")
    assert (m["period_start"], m["period_end"]) == ("2024-09-01", "2026-08-31")
    assert (m["region_count"], m["country_count"], m["segment_count"], m["plan_count"]) == (4, 13, 3, 4)
    assert 18 <= m["sales_rep_count"] <= 22
    assert {r[0] for r in con.execute("SELECT DISTINCT region FROM customers").fetchall()} == set(REGIONS)
    assert {r[0] for r in con.execute("SELECT DISTINCT country FROM customers").fetchall()} == set(COUNTRIES)
    assert {r[0] for r in con.execute("SELECT DISTINCT segment FROM customers").fetchall()} == set(SEGMENTS)
    assert {r[0] for r in con.execute("SELECT DISTINCT plan FROM subscriptions").fetchall()} == set(PLANS)


def test_history_covers_the_full_window(con: duckdb.DuckDBPyConnection) -> None:
    lo, hi = con.execute("SELECT MIN(date), MAX(date) FROM daily_revenue").fetchone()
    assert (lo.isoformat(), hi.isoformat()) == ("2024-09-01", "2026-08-31")
    assert scalar(con, "SELECT COUNT(DISTINCT month) FROM v_monthly_mrr") == 24
    assert scalar(con, "SELECT COUNT(DISTINCT event_date) FROM usage_events") == 104


def test_quality_checks_all_pass(full_dataset: GenerationResult) -> None:
    failures = [f"{c.name}: {c.details}" for c in full_dataset.quality_checks if not c.passed]
    assert not failures, failures
    assert all(c.severity == "error" for c in full_dataset.quality_checks), "all checks are blocking at full scale"


def test_manifest_row_counts_match_database(full_dataset: GenerationResult, con: duckdb.DuckDBPyConnection) -> None:
    for table, entry in full_dataset.manifest["tables"].items():
        assert entry["row_count"] == scalar(con, f'SELECT COUNT(*) FROM "{table}"') > 0


# ---- injected events ----------------------------------------------------------------------------------


def test_all_seven_events_injected(full_dataset: GenerationResult) -> None:
    events = {e["event_id"]: e for e in full_dataset.ground_truth["events"]}
    assert sorted(events) == [f"E{i}" for i in range(1, 8)]
    assert all(e["injected"] for e in events.values())


def test_generator_side_event_validation_passes(full_dataset: GenerationResult) -> None:
    by_event: dict[str, list[str]] = {}
    for check in full_dataset.event_checks:
        by_event.setdefault(check.event_id, []).append(check.status)
    assert sorted(by_event) == [f"E{i}" for i in range(1, 8)]
    failures = [f"{c.event_id} {c.check}: {c.details}" for c in full_dataset.event_checks if c.status != "passed"]
    assert not failures, failures


def test_ground_truth_file_matches_result(full_dataset: GenerationResult) -> None:
    on_disk = json.loads(full_dataset.config.ground_truth_path.read_text())
    assert on_disk == full_dataset.ground_truth


# ---- hidden mechanism and ground truth are not exposed --------------------------------------------------


def test_database_contains_only_documented_objects(con: duckdb.DuckDBPyConnection) -> None:
    objects = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    assert objects == {t.name for t in TABLES} | {v.name for v in VIEWS}


def test_no_hidden_health_or_event_columns(con: duckdb.DuckDBPyConnection) -> None:
    columns = [r[0] for r in con.execute("SELECT column_name FROM information_schema.columns").fetchall()]
    pattern = re.compile(r"health|score|engagement|propensity|risk|injected|truth|cohort|skill|intensity", re.I)
    assert not [c for c in columns if pattern.search(c)]


def test_event_details_not_leaked_into_values(full_dataset: GenerationResult, con: duckdb.DuckDBPyConnection) -> None:
    names = [r[0] for r in con.execute("SELECT DISTINCT campaign_name FROM marketing_campaigns").fetchall()]
    assert not [n for n in names if re.search(r"ineffic|\bE[1-7]\b|injected|ground.?truth", n, re.I)]
    rep = next(e for e in full_dataset.ground_truth["events"] if e["event_id"] == "E4")["details"]["sales_rep"]
    assert "low" not in rep.lower()


# ---- realism (wide plausibility ranges) -------------------------------------------------------------------


def test_segment_economics(con: duckdb.DuckDBPyConnection) -> None:
    med = dict(
        con.execute("""SELECT c.segment, MEDIAN(s.monthly_recurring_revenue) FROM subscriptions s
                              JOIN customers c USING (customer_id) WHERE s.status = 'active' GROUP BY 1""").fetchall()
    )
    assert 100 < med["SMB"] < 600
    assert 600 < med["Mid-Market"] < 3000
    assert 2000 < med["Enterprise"] < 12000


def test_monthly_logo_churn_rates_plausible(con: duckdb.DuckDBPyConnection) -> None:
    rates = dict(
        con.execute("""
        WITH base AS (SELECT month, segment, COUNT(*) n FROM v_monthly_mrr GROUP BY 1, 2),
             churned AS (SELECT CAST(date_trunc('month', s.end_date) AS DATE) AS month, c.segment, COUNT(*) AS k
                         FROM subscriptions s JOIN customers c USING (customer_id)
                         WHERE s.status = 'churned' GROUP BY 1, 2)
        SELECT b.segment, AVG(COALESCE(k, 0)::DOUBLE / n) FROM base b
        LEFT JOIN churned ch ON ch.segment = b.segment AND ch.month = b.month + INTERVAL 1 MONTH
        WHERE b.month < '2026-08-01' GROUP BY 1""").fetchall()
    )
    assert 0.015 < rates["SMB"] < 0.05
    assert 0.005 < rates["Mid-Market"] < 0.025
    assert 0.002 < rates["Enterprise"] < 0.015


def test_mrr_grows_then_declines_in_august(con: duckdb.DuckDBPyConnection) -> None:
    mrr = [
        float(r[1]) for r in con.execute("SELECT month, SUM(mrr) FROM v_monthly_mrr GROUP BY 1 ORDER BY 1").fetchall()
    ]
    assert mrr[-2] > mrr[0] * 1.2, "the business should have grown over the window"
    revenue = dict(con.execute("SELECT month, SUM(revenue) FROM v_monthly_revenue GROUP BY 1").fetchall())
    months = sorted(revenue)
    change = float(revenue[months[-1]]) / float(revenue[months[-2]]) - 1
    assert -0.08 < change < -0.005, f"August revenue change {change:.2%}"


def test_marketing_funnel_plausible(con: duckdb.DuckDBPyConnection) -> None:
    ctr, lead_rate, conv_rate, cac = con.execute("""SELECT SUM(clicks)::DOUBLE / SUM(impressions),
        SUM(leads)::DOUBLE / SUM(clicks), SUM(conversions)::DOUBLE / SUM(leads), SUM(spend) / SUM(conversions)
        FROM marketing_campaigns""").fetchone()
    assert 0.002 < ctr < 0.1
    assert 0.001 < lead_rate < 0.5
    assert 0.02 < conv_rate < 0.2
    assert 500 < float(cac) < 6000


def test_diminishing_returns_to_spend(con: duckdb.DuckDBPyConnection) -> None:
    """Within a channel, weeks with higher spend yield fewer leads per SGD."""
    low, high = con.execute("""
        WITH w AS (SELECT channel, spend, leads, NTILE(4) OVER (PARTITION BY channel ORDER BY spend) q
                   FROM marketing_campaigns WHERE channel = 'Paid Search')
        SELECT SUM(leads) FILTER (WHERE q = 1) / SUM(spend) FILTER (WHERE q = 1),
               SUM(leads) FILTER (WHERE q = 4) / SUM(spend) FILTER (WHERE q = 4) FROM w""").fetchone()
    assert high < low


def test_sales_funnel_plausible(con: duckdb.DuckDBPyConnection) -> None:
    win_rate = scalar(
        con,
        "SELECT AVG(CASE WHEN stage = 'Won' THEN 1.0 ELSE 0 END) FROM sales_opportunities "
        "WHERE stage IN ('Won', 'Lost')",
    )
    assert 0.1 < win_rate < 0.35
    reached = dict(
        con.execute("""SELECT furthest_stage, COUNT(*) FROM sales_opportunities
                                  WHERE stage IN ('Won', 'Lost') GROUP BY 1""").fetchall()
    )
    # each later stage is reached by fewer deals
    order = ["Lead", "Qualified", "Proposal", "Negotiation", "Won"]
    at_least = [sum(reached.get(s, 0) for s in order[i:]) for i in range(len(order))]
    assert at_least == sorted(at_least, reverse=True)
    cycle = dict(
        con.execute("""SELECT segment, MEDIAN(datediff('day', created_date, close_date)) FROM sales_opportunities
                                WHERE stage = 'Won' AND opportunity_type = 'New Business' GROUP BY 1""").fetchall()
    )
    assert cycle["Enterprise"] > cycle["Mid-Market"]


def test_won_deals_match_new_mrr(con: duckdb.DuckDBPyConnection) -> None:
    mismatches = scalar(
        con,
        """
        SELECT COUNT(*) FROM sales_opportunities o JOIN subscriptions s
          ON s.customer_id = o.customer_id AND s.change_type = 'new'
        WHERE o.stage = 'Won' AND o.opportunity_type = 'New Business'
          AND (ABS(o.deal_value - 12 * s.monthly_recurring_revenue) > 0.05 OR o.close_date <> s.start_date)""",
    )
    assert mismatches == 0


def test_support_and_usage_relationships(con: duckdb.DuckDBPyConnection) -> None:
    """Within each segment, churned customers raised more tickets per month, with more negative
    sentiment, than retained ones. (Pooled across segments the ticket comparison reverses because
    churners are mostly small SMB accounts: an emergent Simpson's-paradox trap.)"""
    rows = con.execute("""
        WITH exposure AS (SELECT customer_id, COUNT(DISTINCT date_trunc('month', date)) AS m
                          FROM daily_revenue GROUP BY 1),
             t AS (SELECT customer_id, COUNT(*) AS n, SUM(CASE WHEN sentiment = 'Negative' THEN 1 ELSE 0 END) AS neg
                   FROM support_tickets GROUP BY 1)
        SELECT c.segment, c.status, SUM(COALESCE(t.n, 0))::DOUBLE / SUM(e.m), SUM(t.neg)::DOUBLE / SUM(t.n)
        FROM customers c JOIN exposure e USING (customer_id) LEFT JOIN t USING (customer_id)
        GROUP BY 1, 2""").fetchall()
    stats = {(seg, status): (rate, neg) for seg, status, rate, neg in rows}
    for segment in ("SMB", "Mid-Market", "Enterprise"):
        churned, active = stats[(segment, "churned")], stats[(segment, "active")]
        assert churned[0] > active[0], segment
        assert churned[1] > active[1], segment


def test_feature_adoption_plausible(con: duckdb.DuckDBPyConnection) -> None:
    rates = dict(
        con.execute("""SELECT feature_name, AVG(adoption_rate) FROM product_features
                                WHERE date >= '2026-08-01' GROUP BY 1""").fetchall()
    )
    assert len(rates) == 12 and "AI Insights" in rates
    assert max(rates, key=rates.get) == "Dashboards"
    assert all(0.0 < r < 1.0 for r in rates.values())


def test_parquet_export(full_dataset: GenerationResult) -> None:
    parquet_dir = full_dataset.config.parquet_dir
    assert parquet_dir is not None
    assert sorted(p.stem for p in parquet_dir.glob("*.parquet")) == sorted(t.name for t in TABLES)
