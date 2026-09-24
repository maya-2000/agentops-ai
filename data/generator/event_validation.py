"""Generator-side validation that the seven injected events are present in the data.

INTERNAL ONLY. These checks read the ground-truth file and use it to confirm the generator
did its job. They must never be exposed to the agent as tools or analytical shortcuts; the
agent has to discover the events from the business tables alone.

Each check measures the event through observable business data (the same data the agent
sees), using thresholds wide enough to tolerate realistic noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import duckdb

from data.generator.config import GeneratorConfig
from data.generator.events import EVENTS, load_ground_truth

Status = Literal["passed", "failed", "skipped"]


@dataclass(frozen=True)
class EventCheck:
    event_id: str
    check: str
    status: Status
    details: str
    metrics: dict[str, Any] = field(default_factory=dict)


def _month(d: date, offset: int = 0) -> date:
    idx = d.year * 12 + d.month - 1 + offset
    return date(idx // 12, idx % 12 + 1, 1)


class _EventValidator:
    def __init__(self, con: duckdb.DuckDBPyConnection, truth: dict[str, dict[str, Any]], config: GeneratorConfig):
        self.con = con
        self.truth = truth
        self.config = config
        self.results: list[EventCheck] = []

    def rows(self, sql: str, params: list[Any] | None = None) -> list[tuple[Any, ...]]:
        return self.con.execute(sql, params or []).fetchall()

    def scalar(self, sql: str, params: list[Any] | None = None) -> Any:
        row = self.con.execute(sql, params or []).fetchone()
        return row[0] if row else None

    def add(self, event_id: str, check: str, passed: bool, details: str, **metrics: Any) -> None:
        self.results.append(EventCheck(event_id, check, "passed" if passed else "failed", details, metrics))

    def skip(self, event_id: str, reason: str) -> None:
        self.results.append(EventCheck(event_id, "event_present_in_window", "skipped", reason))

    def injected(self, event_id: str) -> bool:
        event = self.truth.get(event_id)
        if not event or not event.get("injected"):
            reason = (event or {}).get("details", {}).get("reason", "not injected in this window")
            self.skip(event_id, reason)
            return False
        return True

    def window_has(self, *months: date) -> bool:
        return all(self.config.start_date <= m <= self.config.end_date for m in months)

    # ---- E1 ---------------------------------------------------------------------------------------
    def e1(self) -> None:
        if not self.injected("E1"):
            return
        details = self.truth["E1"]["details"]
        aug = EVENTS.e1_sg_enterprise_churn.month
        jul = _month(aug, -1)
        if not self.window_has(jul):
            self.skip("E1", "previous month not in window")
            return

        churn_by_month = dict(
            self.rows("""
            WITH base AS (SELECT month, COUNT(*) AS n FROM v_monthly_mrr
                          WHERE country = 'Singapore' AND segment = 'Enterprise' GROUP BY 1),
                 churned AS (SELECT CAST(date_trunc('month', s.end_date) AS DATE) AS month, COUNT(*) AS k
                             FROM subscriptions s JOIN customers c USING (customer_id)
                             WHERE s.status = 'churned' AND c.country = 'Singapore'
                               AND c.segment = 'Enterprise' GROUP BY 1)
            SELECT CAST(b.month + INTERVAL 1 MONTH AS DATE), COALESCE(k, 0)::DOUBLE / n FROM base b
            LEFT JOIN churned ch ON ch.month = b.month + INTERVAL 1 MONTH""")
        )
        event_rate = churn_by_month.get(aug, 0.0)
        prior = [v for m, v in churn_by_month.items() if _month(aug, -12) <= m < aug]
        baseline = sum(prior) / len(prior) if prior else 0.0
        self.add(
            "E1",
            "sg_enterprise_churn_rate_spike",
            event_rate >= max(4 * baseline, 0.08),
            f"Aug churn {event_rate:.1%} vs trailing average {baseline:.2%}",
            event_rate=event_rate,
            baseline=baseline,
        )

        def mrr_delta(dimension: str, where: str = "") -> list[tuple[str, float]]:
            return [
                (r[0], float(r[1]))
                for r in self.rows(
                    f"""
                SELECT {dimension}, SUM(CASE WHEN month = ? THEN mrr ELSE -mrr END) AS delta
                FROM v_monthly_mrr WHERE month IN (?, ?) {where} GROUP BY 1 ORDER BY 2""",
                    [aug, jul, aug],
                )
            ]

        countries = mrr_delta("country")
        self.add(
            "E1",
            "singapore_largest_country_mrr_decline",
            countries[0][0] == "Singapore",
            f"most negative country: {countries[0][0]} ({countries[0][1]:,.0f}); next: "
            f"{countries[1][0]} ({countries[1][1]:,.0f})",
            ranking=countries[:3],
        )
        segments = mrr_delta("segment")
        sg_segments = mrr_delta("segment", "AND country = 'Singapore'")
        self.add(
            "E1",
            "enterprise_largest_segment_decline",
            segments[0][0] == "Enterprise" and sg_segments[0][0] == "Enterprise",
            f"overall: {segments[0][0]} ({segments[0][1]:,.0f}); within Singapore: {sg_segments[0][0]}",
            overall=segments,
            singapore=sg_segments,
        )

        revenue = dict(
            self.rows(
                """SELECT month, SUM(revenue) FROM v_monthly_revenue
                                    WHERE month IN (?, ?) GROUP BY 1""",
                [jul, aug],
            )
        )
        change = float(revenue[aug]) / float(revenue[jul]) - 1
        self.add(
            "E1",
            "total_revenue_declined_month_over_month",
            change < 0,
            f"Aug vs Jul revenue {change:+.2%}",
            change=change,
        )
        # Realism guard: the event should not be visible as a collapse of the whole business.
        self.add(
            "E1",
            "decline_is_not_trivially_extreme",
            change > -0.10,
            f"Aug vs Jul revenue {change:+.2%} (guard: > -10%)",
        )

        ids = details["churned_customer_ids"]
        w_lo, w_hi = EVENTS.e1_sg_enterprise_churn.last_service_day_window
        matched = self.scalar(
            f"""SELECT COUNT(*) FROM subscriptions WHERE status = 'churned'
            AND end_date BETWEEN ? AND ? AND customer_id IN ({",".join("?" * len(ids))})""",
            [w_lo, w_hi, *ids],
        )
        self.add(
            "E1",
            "ground_truth_churners_observable",
            matched == len(ids),
            f"{matched}/{len(ids)} ground-truth churners have a churned subscription in the wave window",
        )

        billing = dict(
            self.rows(
                """
            SELECT CASE WHEN created_at >= ? THEN 'event' ELSE 'baseline' END,
                   COUNT(*)::DOUBLE / COUNT(DISTINCT CAST(date_trunc('month', created_at) AS DATE))
            FROM support_tickets t JOIN customers c USING (customer_id)
            WHERE c.country = 'Singapore' AND c.segment = 'Enterprise' AND t.category = 'Billing'
              AND created_at >= ? AND created_at < ? GROUP BY 1""",
                [jul, _month(aug, -5), _month(aug, 1)],
            )
        )
        ratio = billing.get("event", 0.0) / max(billing.get("baseline", 0.0), 1e-9)
        self.add(
            "E1",
            "sg_enterprise_billing_complaints_rise",
            ratio >= 2.0,
            f"SG Enterprise Billing tickets per month: Jul-Aug {billing.get('event', 0):.1f} vs "
            f"Mar-Jun {billing.get('baseline', 0):.1f} ({ratio:.1f}x)",
        )

    # ---- E2 ---------------------------------------------------------------------------------------
    def e2(self) -> None:
        if not self.injected("E2"):
            return
        spike = EVENTS.e2_support_spike
        event_months = (_month(spike.start), _month(spike.end))
        base_months = (_month(spike.start, -3), _month(spike.start, -1))
        if not self.window_has(base_months[0]):
            self.skip("E2", "baseline months not in window")
            return
        stats = {
            r[0]: r[1:]
            for r in self.rows(
                """
            WITH t AS (
                SELECT CASE WHEN created_at >= ? AND created_at < ? THEN 'event'
                            WHEN created_at >= ? AND created_at < ? THEN 'baseline' END AS period, *
                FROM support_tickets),
            active AS (
                SELECT CASE WHEN month BETWEEN ? AND ? THEN 'event'
                            WHEN month BETWEEN ? AND ? THEN 'baseline' END AS period,
                       COUNT(*) AS customer_months FROM v_monthly_mrr GROUP BY 1)
            SELECT t.period, COUNT(*)::DOUBLE / MAX(a.customer_months),
                   AVG(CASE WHEN category IN ('Bug', 'Integration') THEN 1.0 ELSE 0 END),
                   MEDIAN(resolution_time)
            FROM t JOIN active a USING (period) WHERE t.period IS NOT NULL GROUP BY 1""",
                [
                    event_months[0],
                    _month(event_months[1], 1),
                    base_months[0],
                    _month(base_months[1], 1),
                    *event_months,
                    *base_months,
                ],
            )
        }
        if not {"event", "baseline"} <= stats.keys():
            self.skip("E2", "no tickets in the event or baseline period at this scale")
            return
        ev, base = stats["event"], stats["baseline"]
        self.add(
            "E2",
            "tickets_per_active_customer_spike",
            ev[0] >= 1.2 * base[0],
            f"{ev[0]:.2f} vs {base[0]:.2f} tickets per active customer-month ({ev[0] / base[0]:.2f}x)",
        )
        self.add(
            "E2",
            "bug_integration_share_rises",
            ev[1] >= base[1] + 0.05,
            f"Bug+Integration share {ev[1]:.1%} vs {base[1]:.1%}",
        )
        self.add(
            "E2",
            "resolution_time_lengthens",
            ev[2] >= 1.3 * base[2],
            f"median resolution {ev[2]:.1f}h vs {base[2]:.1f}h",
        )

    # ---- E3 ---------------------------------------------------------------------------------------
    def e3(self) -> None:
        if not self.injected("E3"):
            return
        target = self.truth["E3"]["details"]["campaign_id"]
        # CAC from a handful of conversions is noise, so rank campaigns with >= 10 conversions.
        cac = self.rows("""SELECT campaign_id, channel, spend / conversions AS cac, conversions FROM v_campaign_summary
                           WHERE conversions >= 10 ORDER BY cac DESC""")
        target_row = next((r for r in cac if r[0] == target), None)
        if target_row is None:
            self.skip("E3", "event campaign has < 10 conversions; CAC not measurable at this scale")
            return
        top_id = cac[0][0]
        peers = sorted(float(r[2]) for r in cac if r[1] == EVENTS.e3_paid_social.channel and r[0] != target)
        median = peers[len(peers) // 2] if peers else float("nan")
        target_cac = float(target_row[2]) if target_row else float("inf")
        self.add(
            "E3",
            "highest_cac_campaign",
            top_id == target,
            f"highest CAC campaign (>= 10 conversions) {top_id} (ground truth {target})",
        )
        self.add(
            "E3",
            "cac_well_above_channel_median",
            target_cac >= 2 * median,
            f"CAC {target_cac:,.0f} vs Paid Social median {median:,.0f} ({target_cac / median:.1f}x)",
        )

    # ---- E4 ---------------------------------------------------------------------------------------
    def e4(self) -> None:
        if not self.injected("E4"):
            return
        target = self.truth["E4"]["details"]["sales_rep"]
        rates = self.rows("""SELECT sales_rep, COUNT(*) AS closed,
                                    AVG(CASE WHEN stage = 'Won' THEN 1.0 ELSE 0 END) AS win_rate
                             FROM sales_opportunities WHERE stage IN ('Won', 'Lost') GROUP BY 1 ORDER BY win_rate""")
        target_row = next((r for r in rates if r[0] == target), None)
        if target_row is None or len(rates) < 3:
            self.skip("E4", "too few closed opportunities to compare reps at this scale")
            return
        ordered = sorted(float(r[2]) for r in rates)
        median = ordered[len(ordered) // 2]
        self.add(
            "E4",
            "lowest_win_rate_rep",
            rates[0][0] == target,
            f"lowest win rate: {rates[0][0]} ({float(rates[0][2]):.1%}); ground truth {target}",
        )
        self.add(
            "E4",
            "win_rate_clearly_below_median",
            float(target_row[2]) <= median - 0.05 and int(target_row[1]) >= 30,
            f"{float(target_row[2]):.1%} vs team median {median:.1%} on {target_row[1]} closed deals",
        )

    # ---- E5 ---------------------------------------------------------------------------------------
    def e5(self) -> None:
        if not self.injected("E5"):
            return
        launch = EVENTS.e5_ai_insights.launch_date
        name = EVENTS.e5_ai_insights.feature_name
        first, last = self.rows("SELECT MIN(date), MAX(date) FROM product_features WHERE feature_name = ?", [name])[0]
        if first is None:
            self.add("E5", "rows_start_at_launch", False, "no AI Insights rows found")
            return
        self.add(
            "E5",
            "rows_start_at_launch",
            first is not None and first == launch,
            f"first AI Insights row {first} (launch {launch})",
        )
        early = self.scalar(
            "SELECT AVG(adoption_rate) FROM product_features WHERE feature_name = ? AND date < ?",
            [name, launch + timedelta(days=28)],
        )
        late = self.scalar(
            "SELECT AVG(adoption_rate) FROM product_features WHERE feature_name = ? AND date > ?",
            [name, last - timedelta(days=28)],
        )
        self.add(
            "E5",
            "adoption_ramps_up",
            late is not None and late >= max(3 * early, 0.05),
            f"adoption first 4 weeks {early:.1%} -> last 4 weeks {late:.1%}",
        )

    # ---- E6 ---------------------------------------------------------------------------------------
    def e6(self) -> None:
        rates = dict(
            self.rows(
                """
            WITH base AS (SELECT month, segment, COUNT(*) AS n FROM v_monthly_mrr GROUP BY 1, 2),
                 churned AS (SELECT CAST(date_trunc('month', s.end_date) AS DATE) AS month, c.segment, COUNT(*) AS k
                             FROM subscriptions s JOIN customers c USING (customer_id)
                             WHERE s.status = 'churned' GROUP BY 1, 2)
            SELECT b.segment, AVG(COALESCE(k, 0)::DOUBLE / n) FROM base b
            LEFT JOIN churned ch ON ch.segment = b.segment AND ch.month = b.month + INTERVAL 1 MONTH
            WHERE b.month + INTERVAL 1 MONTH <= ? GROUP BY 1""",
                [self.config.end_date],
            )
        )
        if set(rates) != {"SMB", "Mid-Market", "Enterprise"}:
            self.skip("E6", "not all segments present at this scale")
            return
        self.add(
            "E6",
            "smb_highest_monthly_churn",
            rates["SMB"] >= 1.5 * rates["Mid-Market"] and rates["SMB"] > rates["Enterprise"],
            "avg monthly logo churn " + ", ".join(f"{k} {v:.2%}" for k, v in sorted(rates.items())),
        )

    # ---- E7 ---------------------------------------------------------------------------------------
    def e7(self) -> None:
        if not self.injected("E7"):
            return
        details = self.truth["E7"]["details"]
        churned, cohort = details["churned_customer_ids"], details["cohort_customer_ids"]
        if len(churned) < 5:
            self.skip("E7", f"only {len(churned)} cohort churners; too few to measure")
            return

        def wau_ratio(ids_clause: str, params: list[Any]) -> float | None:
            return self.scalar(
                f"""
                WITH ch AS (SELECT customer_id, end_date FROM subscriptions WHERE status = 'churned' {ids_clause}),
                u AS (SELECT datediff('day', u.event_date, ch.end_date) AS d, u.active_users
                      FROM usage_events u JOIN ch USING (customer_id))
                SELECT AVG(CASE WHEN d BETWEEN 0 AND 28 THEN active_users END)
                     / AVG(CASE WHEN d BETWEEN 84 AND 140 THEN active_users END) FROM u""",
                params,
            )

        placeholders = ",".join("?" * len(churned))
        cohort_ratio = wau_ratio(f"AND customer_id IN ({placeholders})", churned)
        drift_start = EVENTS.e7_pre_churn_decline.drift_start
        others = wau_ratio(f"AND end_date >= ? AND customer_id NOT IN ({placeholders})", [drift_start, *churned])
        self.add(
            "E7",
            "usage_declines_before_churn",
            cohort_ratio is not None and cohort_ratio <= 0.75 and (others is None or cohort_ratio < others),
            f"WAU in last 4 weeks vs weeks 12-20 before churn: cohort {cohort_ratio:.2f}, other churners "
            f"{(others or float('nan')):.2f}",
        )

        # Tickets per active customer-month (exposure = customer-months with any revenue).
        cohort_ph = ",".join("?" * len(cohort))
        since = _month(drift_start, -3)
        tickets = dict(
            self.rows(
                f"""
            WITH exposure AS (
                SELECT CASE WHEN date >= ? THEN 'after' ELSE 'before' END AS period,
                       COUNT(DISTINCT (customer_id, date_trunc('month', date))) AS customer_months
                FROM daily_revenue WHERE customer_id IN ({cohort_ph}) AND date >= ? GROUP BY 1),
            t AS (
                SELECT CASE WHEN created_at >= ? THEN 'after' ELSE 'before' END AS period, COUNT(*) AS n
                FROM support_tickets WHERE customer_id IN ({cohort_ph}) AND created_at >= ? GROUP BY 1)
            SELECT period, n::DOUBLE / customer_months FROM t JOIN exposure USING (period)""",
                [drift_start, *cohort, since, drift_start, *cohort, since],
            )
        )
        before, after = tickets.get("before", 0.0), tickets.get("after", 0.0)
        self.add(
            "E7",
            "cohort_support_tickets_rise",
            after > before,
            f"cohort tickets per active customer-month: before {before:.2f}, after {after:.2f}",
        )


def validate_events(db_path: Path, ground_truth_path: Path, config: GeneratorConfig) -> list[EventCheck]:
    truth = load_ground_truth(ground_truth_path)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        validator = _EventValidator(con, truth, config)
        for step in (validator.e1, validator.e2, validator.e3, validator.e4, validator.e5, validator.e6, validator.e7):
            step()
        return validator.results
    finally:
        con.close()
