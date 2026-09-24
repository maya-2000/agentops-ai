"""Data-quality validation of the loaded business database.

Checks are grouped by the Phase 1 acceptance categories (A-N) plus an exposure check (O)
confirming that no hidden generator variable or ground truth reached the database. Checks
tolerate realistic statistical variation: sanity checks compare aggregates, not individual rows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import duckdb

from app.database.metadata import COUNTRY_REGION, TABLES, VIEWS
from data.generator.config import COMPANY_FOUNDED, PLAN_PARAMS, GeneratorConfig

Severity = Literal["error", "warning"]

# Expected null-rate bounds for nullable columns (min, max).
NULL_RATE_BOUNDS: dict[tuple[str, str], tuple[float, float]] = {
    ("subscriptions", "end_date"): (0.2, 0.9),
    ("sales_opportunities", "customer_id"): (0.3, 0.95),
    ("sales_opportunities", "close_date"): (0.0, 0.4),  # open deals; higher share in short windows
    ("support_tickets", "resolved_at"): (0.0, 0.05),
    ("support_tickets", "resolution_time"): (0.0, 0.05),
}

MIN_CUSTOMERS_FOR_STATISTICS = 1000

# Column names that would reveal the hidden mechanism or the injected-event ground truth.
FORBIDDEN_NAME_PATTERN = re.compile(
    r"health|engagement|propensity|latent|hidden|injected|ground_truth|business_event|event_flag|cohort|skill",
    re.IGNORECASE,
)


def _count_where(table: str, predicate: str) -> str:
    return f'SELECT COUNT(*) FROM "{table}" WHERE {predicate}'


@dataclass(frozen=True)
class CheckResult:
    name: str
    category: str
    passed: bool
    details: str
    severity: Severity = "error"


class _Checker:
    def __init__(self, con: duckdb.DuckDBPyConnection, config: GeneratorConfig):
        self.con = con
        self.config = config
        self.start = config.start_date.isoformat()
        self.end = config.end_date.isoformat()
        self.results: list[CheckResult] = []
        # Aggregate comparisons are statistically meaningless on tiny datasets (e.g. a 100-customer
        # test fixture); there they are reported as warnings instead of failing the build.
        self.statistical: Severity = "error" if config.customer_count >= MIN_CUSTOMERS_FOR_STATISTICS else "warning"

    def scalar(self, sql: str) -> Any:
        row = self.con.execute(sql).fetchone()
        return row[0] if row else None

    def check(self, category: str, name: str, passed: bool, details: str, severity: Severity = "error") -> None:
        self.results.append(CheckResult(name, category, bool(passed), details, severity))

    def zero(self, category: str, name: str, sql: str, what: str, severity: Severity = "error") -> None:
        """Pass when ``sql`` (a COUNT query) returns 0."""
        count = int(self.scalar(sql) or 0)
        self.check(category, name, count == 0, f"{count} {what}", severity)

    # ---- A. row counts ------------------------------------------------------------------------
    def row_counts(self) -> None:
        for table in TABLES:
            n = int(self.scalar(f'SELECT COUNT(*) FROM "{table.name}"'))
            self.check("A", f"rows:{table.name}", n > 0, f"{n} rows")
        n_customers = int(self.scalar("SELECT COUNT(*) FROM customers"))
        self.check(
            "A",
            "rows:customer_count_matches_config",
            n_customers == self.config.customer_count,
            f"{n_customers} customers (configured {self.config.customer_count})",
        )

    # ---- B. date ranges -----------------------------------------------------------------------
    def date_ranges(self) -> None:
        s, e = self.start, self.end
        last_week = (self.config.end_date - timedelta(days=6)).isoformat()
        rules = {  # name: (table, predicate that must match no rows)
            "daily_revenue.date": ("daily_revenue", f"date < '{s}' OR date > '{e}'"),
            "usage_events.event_date": ("usage_events", f"event_date < '{s}' OR event_date > '{last_week}'"),
            "marketing_campaigns.date": ("marketing_campaigns", f"date < '{s}' OR date > '{last_week}'"),
            "product_features.date": ("product_features", f"date < '{s}' OR date > '{e}'"),
            "support_tickets.created_at": ("support_tickets", f"created_at < '{s}' OR created_at >= DATE '{e}' + 1"),
            "customers.signup_date": ("customers", f"signup_date < '{COMPANY_FOUNDED}' OR signup_date > '{e}'"),
            "subscriptions.dates": ("subscriptions", f"start_date > '{e}' OR end_date > '{e}'"),
            "sales_opportunities.close_date": ("sales_opportunities", f"close_date < '{s}' OR close_date > '{e}'"),
            "sales_opportunities.created_date": ("sales_opportunities", f"created_date > '{e}'"),
        }
        for name, (table, predicate) in rules.items():
            self.zero("B", f"date_range:{name}", _count_where(table, predicate), "rows outside the expected range")
        n_days = (self.config.end_date - self.config.start_date).days + 1
        covered = int(self.scalar("SELECT COUNT(DISTINCT date) FROM daily_revenue"))
        self.check("B", "date_range:daily_revenue_has_every_day", covered == n_days, f"{covered}/{n_days} days")

    # ---- C/D/G. keys, required fields, categorical domains ------------------------------------------
    def keys_nulls_domains(self) -> None:
        for table in TABLES:
            pk = ", ".join(f'"{c}"' for c in table.primary_key)
            dupes = int(
                self.scalar(f'SELECT COUNT(*) FROM (SELECT {pk} FROM "{table.name}" GROUP BY {pk} HAVING COUNT(*) > 1)')
            )
            self.check("C", f"primary_key:{table.name}", dupes == 0, f"{dupes} duplicated keys")
            for col in table.columns:
                if not col.nullable:
                    self.zero(
                        "D",
                        f"required:{table.name}.{col.name}",
                        f'SELECT COUNT(*) FROM "{table.name}" WHERE "{col.name}" IS NULL',
                        "NULL values",
                    )
                if col.allowed_values:
                    values = ", ".join("'" + v.replace("'", "''") + "'" for v in col.allowed_values)
                    self.zero(
                        "G",
                        f"domain:{table.name}.{col.name}",
                        f'SELECT COUNT(*) FROM "{table.name}" WHERE "{col.name}" NOT IN ({values})',
                        "values outside the allowed set",
                    )

    # ---- E. null rates ------------------------------------------------------------------------
    def null_rates(self) -> None:
        for (table, column), (lo, hi) in NULL_RATE_BOUNDS.items():
            rate = float(self.scalar(f'SELECT AVG(CASE WHEN "{column}" IS NULL THEN 1.0 ELSE 0 END) FROM "{table}"'))
            self.check(
                "E", f"null_rate:{table}.{column}", lo <= rate <= hi, f"null rate {rate:.3f} (expected {lo}-{hi})"
            )
        self.zero(
            "E",
            "null_rate:ticket_resolution_fields_agree",
            "SELECT COUNT(*) FROM support_tickets WHERE (resolved_at IS NULL) <> (resolution_time IS NULL) "
            "OR (resolved_at IS NULL) <> (status = 'Open')",
            "tickets with inconsistent open/resolved fields",
        )

    # ---- F. referential integrity ------------------------------------------------------------------
    def referential_integrity(self) -> None:
        for table in TABLES:
            for col in table.columns:
                if col.foreign_key is None:
                    continue
                fk = col.foreign_key
                self.zero(
                    "F",
                    f"fk:{table.name}.{col.name}",
                    f'SELECT COUNT(*) FROM "{table.name}" t LEFT JOIN "{fk.table}" p '
                    f'ON t."{col.name}" = p."{fk.column}" WHERE t."{col.name}" IS NOT NULL AND p."{fk.column}" IS NULL',
                    "orphaned references",
                )
        self.zero(
            "F",
            "fk:daily_revenue_attributes_match_customer",
            "SELECT COUNT(*) FROM daily_revenue d JOIN customers c USING (customer_id) "
            "WHERE d.region <> c.region OR d.segment <> c.segment",
            "rows whose region/segment differ from the customer",
        )
        self.zero(
            "F",
            "fk:won_opportunities_have_customer",
            "SELECT COUNT(*) FROM sales_opportunities WHERE stage = 'Won' AND customer_id IS NULL",
            "won opportunities without a customer",
        )

    # ---- H. non-negative / bounded values -------------------------------------------------------------
    def value_bounds(self) -> None:
        rules = {  # name: (table, predicate that must match no rows)
            "revenue>=0": ("daily_revenue", "revenue < 0"),
            "mrr>=0": ("subscriptions", "monthly_recurring_revenue < 0 OR previous_mrr < 0 OR current_mrr < 0"),
            "seats>0": ("subscriptions", "seats <= 0"),
            "spend>=0": (
                "marketing_campaigns",
                "spend < 0 OR impressions < 0 OR clicks < 0 OR leads < 0 OR conversions < 0",
            ),
            "deal_value>0": ("sales_opportunities", "deal_value <= 0"),
            "probability_0_1": ("sales_opportunities", "probability < 0 OR probability > 1"),
            "usage>=0": (
                "usage_events",
                "active_users < 0 OR sessions < 0 OR api_calls < 0 OR feature_usage < 0 OR feature_usage > 12",
            ),
            "resolution_time>0": ("support_tickets", "resolution_time <= 0"),
            "adoption_rate_0_1": ("product_features", "adoption_rate < 0 OR adoption_rate > 1 OR active_users < 0"),
        }
        for name, (table, predicate) in rules.items():
            self.zero("H", f"bounds:{name}", _count_where(table, predicate), "violating rows")

    # ---- I. logical date relationships --------------------------------------------------------------
    def date_logic(self) -> None:
        rules = {
            "close_date>=created_date": "SELECT COUNT(*) FROM sales_opportunities WHERE close_date < created_date",
            "resolved_at>=created_at": "SELECT COUNT(*) FROM support_tickets WHERE resolved_at < created_at",
            "end_date>=start_date": "SELECT COUNT(*) FROM subscriptions WHERE end_date < start_date",
            "signup=first_subscription": (
                "SELECT COUNT(*) FROM customers c JOIN (SELECT customer_id, MIN(start_date) first_start "
                "FROM subscriptions GROUP BY 1) s USING (customer_id) WHERE c.signup_date <> s.first_start"
            ),
            "closed_opportunities_have_close_date": (
                "SELECT COUNT(*) FROM sales_opportunities WHERE (stage IN ('Won','Lost')) <> (close_date IS NOT NULL)"
            ),
        }
        for name, sql in rules.items():
            self.zero("I", f"date_logic:{name}", sql, "violating rows")

    # ---- J. revenue consistency ---------------------------------------------------------------------------
    def revenue_consistency(self) -> None:
        self.zero(
            "J",
            "revenue:subscription_revenue_only_while_in_force",
            """
            SELECT COUNT(*) FROM daily_revenue d
            WHERE d.revenue_type = 'subscription' AND NOT EXISTS (
                SELECT 1 FROM subscriptions s WHERE s.customer_id = d.customer_id AND s.start_date <= d.date
                  AND (s.end_date IS NULL OR s.end_date >= d.date) AND s.plan = d.plan)""",
            "revenue rows without a matching subscription/plan in force",
        )
        self.zero(
            "J",
            "revenue:every_in_force_day_has_revenue",
            f"""
            WITH spans AS (
                SELECT subscription_id, customer_id,
                       GREATEST(start_date, DATE '{self.start}') AS s,
                       LEAST(COALESCE(end_date, DATE '{self.end}'), DATE '{self.end}') AS e
                FROM subscriptions WHERE COALESCE(end_date, DATE '{self.end}') >= DATE '{self.start}'),
            expected AS (SELECT customer_id, SUM(datediff('day', s, e) + 1) AS days FROM spans GROUP BY 1),
            actual AS (SELECT customer_id, COUNT(*) AS days FROM daily_revenue
                       WHERE revenue_type = 'subscription' GROUP BY 1)
            SELECT COUNT(*) FROM expected e LEFT JOIN actual a USING (customer_id)
            WHERE COALESCE(a.days, 0) <> e.days""",
            "customers whose revenue days differ from subscription days",
        )
        self.zero(
            "J",
            "revenue:full_month_revenue_equals_mrr",
            """
            WITH m AS (
                SELECT customer_id, CAST(date_trunc('month', date) AS DATE) AS month, plan,
                       COUNT(*) AS n_days, SUM(revenue) AS revenue
                FROM daily_revenue WHERE revenue_type = 'subscription' GROUP BY ALL)
            SELECT COUNT(*) FROM m JOIN subscriptions s
              ON s.customer_id = m.customer_id AND s.start_date <= m.month
             AND (s.end_date IS NULL OR s.end_date >= last_day(m.month))
            WHERE m.n_days = datediff('day', m.month, last_day(m.month)) + 1
              AND ABS(m.revenue - s.monthly_recurring_revenue) > 0.16""",
            "full customer-months where recognised revenue differs from MRR by > SGD 0.16 (cent rounding)",
        )
        self.zero(
            "J",
            "revenue:no_usage_revenue_on_starter_plan",
            "SELECT COUNT(*) FROM daily_revenue WHERE revenue_type = 'usage' AND plan = 'Starter'",
            "usage revenue rows on the Starter plan",
        )

    # ---- K. subscription consistency ------------------------------------------------------------------------
    def subscription_consistency(self) -> None:
        ordered = """WITH o AS (SELECT *, LAG(end_date) OVER w AS prev_end,
                     LAG(monthly_recurring_revenue) OVER w AS prev_mrr,
                     LEAD(start_date) OVER w AS next_start, ROW_NUMBER() OVER w AS rn
                     FROM subscriptions WINDOW w AS (PARTITION BY customer_id ORDER BY start_date))"""
        rules = {
            "records_contiguous_no_overlap": (
                f"{ordered} SELECT COUNT(*) FROM o WHERE rn > 1 AND start_date <> prev_end + 1"
            ),
            "previous_mrr_matches_prior_record": (
                f"{ordered} SELECT COUNT(*) FROM o WHERE previous_mrr <> COALESCE(prev_mrr, 0)"
            ),
            "change_type_matches_mrr_movement": f"""{ordered} SELECT COUNT(*) FROM o WHERE
                (change_type = 'new') <> (rn = 1)
                OR (change_type = 'expansion' AND monthly_recurring_revenue <= previous_mrr)
                OR (change_type = 'contraction' AND monthly_recurring_revenue >= previous_mrr)""",
            "status_matches_end_date": f"""{ordered} SELECT COUNT(*) FROM o WHERE
                (status = 'active') <> (end_date IS NULL)
                OR (status = 'superseded') <> (next_start IS NOT NULL)""",
            "current_mrr_rule": """SELECT COUNT(*) FROM subscriptions WHERE current_mrr <>
                CASE WHEN status = 'active' THEN monthly_recurring_revenue ELSE 0 END""",
        }
        for name, sql in rules.items():
            self.zero("K", f"subscription:{name}", sql, "violating records")
        price_case = " ".join(f"WHEN '{p}' THEN {v.price_per_seat}" for p, v in PLAN_PARAMS.items())
        self.zero(
            "K",
            "subscription:mrr_consistent_with_seats_and_price",
            f"""
            SELECT COUNT(*) FROM subscriptions WHERE
              monthly_recurring_revenue > seats * (CASE plan {price_case} END) + 0.01
              OR monthly_recurring_revenue < seats * (CASE plan {price_case} END) * 0.70""",
            "records priced outside list price minus a 0-30% discount",
        )

    # ---- L. customer consistency ----------------------------------------------------------------------------
    def customer_consistency(self) -> None:
        rules = {
            "every_customer_has_one_new_subscription": """SELECT COUNT(*) FROM customers c LEFT JOIN
                (SELECT customer_id, COUNT(*) FILTER (WHERE change_type = 'new') AS n FROM subscriptions GROUP BY 1) s
                USING (customer_id) WHERE COALESCE(s.n, 0) <> 1""",
            "status_matches_subscriptions": """SELECT COUNT(*) FROM customers c JOIN
                (SELECT customer_id, COUNT(*) FILTER (WHERE status = 'active') AS n_active,
                        COUNT(*) FILTER (WHERE status = 'churned') AS n_churned FROM subscriptions GROUP BY 1) s
                USING (customer_id) WHERE NOT (
                    (c.status = 'active' AND s.n_active = 1 AND s.n_churned = 0) OR
                    (c.status = 'churned' AND s.n_active = 0 AND s.n_churned = 1))""",
            "region_matches_country": "SELECT COUNT(*) FROM customers WHERE region <> CASE country "
            + " ".join(f"WHEN '{c}' THEN '{r}'" for c, r in COUNTRY_REGION.items())
            + " END",
            "segment_matches_company_size": """SELECT COUNT(*) FROM customers WHERE segment <> CASE
                WHEN company_size IN ('1-50', '51-200') THEN 'SMB' WHEN company_size = '201-1000' THEN 'Mid-Market'
                ELSE 'Enterprise' END""",
            "unique_company_names": "SELECT COUNT(*) - COUNT(DISTINCT company_name) FROM customers",
        }
        for name, sql in rules.items():
            self.zero("L", f"customer:{name}", sql, "violating customers")

        churned = "(SELECT customer_id, end_date FROM subscriptions WHERE status = 'churned')"
        signup = "customers"
        in_force_mid_week = """usage_events u JOIN subscriptions s ON s.customer_id = u.customer_id
                AND s.start_date <= u.event_date + 3 AND (s.end_date IS NULL OR s.end_date >= u.event_date + 3)"""
        activity = {
            "no_revenue_after_churn": f"""SELECT COUNT(*) FROM daily_revenue d
                JOIN {churned} c USING (customer_id) WHERE d.date > c.end_date""",
            "no_usage_after_churn": f"""SELECT COUNT(*) FROM usage_events u
                JOIN {churned} c USING (customer_id) WHERE u.event_date > c.end_date""",
            "no_tickets_after_churn": f"""SELECT COUNT(*) FROM support_tickets t
                JOIN {churned} c USING (customer_id) WHERE CAST(t.created_at AS DATE) > c.end_date""",
            "no_activity_before_signup": f"""SELECT
                (SELECT COUNT(*) FROM usage_events u JOIN {signup} c USING (customer_id)
                 WHERE u.event_date + 6 < c.signup_date)
              + (SELECT COUNT(*) FROM support_tickets t JOIN {signup} c USING (customer_id)
                 WHERE CAST(t.created_at AS DATE) < c.signup_date)
              + (SELECT COUNT(*) FROM daily_revenue d JOIN {signup} c USING (customer_id)
                 WHERE d.date < c.signup_date)""",
            "active_users_within_seats": f"SELECT COUNT(*) FROM {in_force_mid_week} WHERE u.active_users > s.seats",
            "no_api_calls_on_starter": (
                f"SELECT COUNT(*) FROM {in_force_mid_week} WHERE s.plan = 'Starter' AND u.api_calls > 0"
            ),
        }
        for name, sql in activity.items():
            self.zero("L", f"customer:{name}", sql, "violating rows")

    # ---- M. marketing consistency ------------------------------------------------------------------------------
    def marketing_consistency(self) -> None:
        self.zero(
            "M",
            "marketing:funnel_monotonic",
            _count_where("marketing_campaigns", "conversions > leads OR leads > clicks OR clicks > impressions"),
            "rows where conversions <= leads <= clicks <= impressions is violated",
        )
        self.zero(
            "M",
            "marketing:conversions_reconcile_with_customers",
            f"""
            WITH conv AS (SELECT channel, SUM(conversions) AS n FROM marketing_campaigns GROUP BY 1),
                 cust AS (SELECT acquisition_channel AS channel, COUNT(*) AS n FROM customers
                          WHERE signup_date >= '{self.start}' GROUP BY 1)
            SELECT COUNT(*) FROM conv LEFT JOIN cust USING (channel) WHERE conv.n <> COALESCE(cust.n, 0)""",
            "channels where campaign conversions differ from customers acquired",
        )
        self.zero(
            "M",
            "marketing:one_channel_per_campaign",
            """SELECT COUNT(*) FROM (SELECT campaign_id FROM marketing_campaigns GROUP BY 1
               HAVING COUNT(DISTINCT channel) > 1 OR COUNT(DISTINCT campaign_name) > 1)""",
            "campaigns with inconsistent channel/name",
        )
        self.zero(
            "M",
            "marketing:spend_positive",
            "SELECT COUNT(*) FROM marketing_campaigns WHERE spend <= 0",
            "weeks without spend",
        )

    # ---- N. business sanity -------------------------------------------------------------------------------------
    def business_sanity(self) -> None:
        med = dict(
            self.con.execute("""SELECT c.segment, MEDIAN(s.monthly_recurring_revenue) FROM subscriptions s
            JOIN customers c USING (customer_id) WHERE s.change_type = 'new' GROUP BY 1""").fetchall()
        )
        self.check(
            "N",
            "sanity:mrr_increases_with_segment",
            med["Enterprise"] > med["Mid-Market"] > med["SMB"],
            "median initial MRR " + ", ".join(f"{k} {float(v):,.0f}" for k, v in med.items()),
            severity=self.statistical,
        )
        deal = dict(
            self.con.execute("""SELECT segment, AVG(deal_value) FROM sales_opportunities
                WHERE stage = 'Won' AND opportunity_type = 'New Business' GROUP BY 1""").fetchall()
        )
        self.check(
            "N",
            "sanity:enterprise_deals_largest",
            deal.get("Enterprise", 0) > deal.get("Mid-Market", 0) > deal.get("SMB", 0),
            "avg won deal " + ", ".join(f"{k} {float(v):,.0f}" for k, v in deal.items()),
            severity=self.statistical,
        )
        monthly = [
            float(r[1])
            for r in self.con.execute("SELECT month, SUM(mrr) FROM v_monthly_mrr GROUP BY 1 ORDER BY 1").fetchall()
        ]
        swings = [abs(b / a - 1) for a, b in pairwise(monthly)]
        self.check(
            "N",
            "sanity:mrr_month_to_month_change_below_10pct",
            max(swings, default=0) < 0.10,
            f"largest MoM MRR change {max(swings, default=0):.1%}",
            severity=self.statistical,
        )
        churn = dict(
            self.con.execute("""
            SELECT c.segment, COUNT(*) FILTER (WHERE s.status = 'churned')::DOUBLE / COUNT(*) FROM customers c
            JOIN subscriptions s USING (customer_id) WHERE s.status IN ('active','churned') GROUP BY 1""").fetchall()
        )
        self.check(
            "N",
            "sanity:smb_churns_most",
            churn["SMB"] > churn["Mid-Market"] > churn["Enterprise"],
            "share of customers churned " + ", ".join(f"{k} {v:.1%}" for k, v in churn.items()),
            severity=self.statistical,
        )
        per_cust = dict(
            self.con.execute("""SELECT c.segment, COUNT(t.ticket_id)::DOUBLE / COUNT(DISTINCT c.customer_id)
            FROM customers c LEFT JOIN support_tickets t USING (customer_id) GROUP BY 1""").fetchall()
        )
        self.check(
            "N",
            "sanity:larger_customers_raise_more_tickets",
            per_cust["Enterprise"] > per_cust["SMB"],
            "tickets per customer " + ", ".join(f"{k} {v:.1f}" for k, v in per_cust.items()),
            severity=self.statistical,
        )
        wau_by_status = dict(
            self.con.execute("""
            SELECT c.status, AVG(u.active_users::DOUBLE / s.seats) FROM usage_events u
            JOIN customers c USING (customer_id)
            JOIN subscriptions s ON s.customer_id = u.customer_id AND s.start_date <= u.event_date + 3
                 AND (s.end_date IS NULL OR s.end_date >= u.event_date + 3) GROUP BY 1""").fetchall()
        )
        self.check(
            "N",
            "sanity:churned_customers_used_product_less",
            wau_by_status.get("churned", 0) < wau_by_status.get("active", 1),
            "avg seat utilisation " + ", ".join(f"{k} {v:.2f}" for k, v in wau_by_status.items()),
            severity=self.statistical,
        )

    # ---- O. exposure of hidden data -----------------------------------------------------------------------------
    def no_hidden_exposure(self) -> None:
        objects = {
            r[0]
            for r in self.con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema NOT IN ('information_schema', 'pg_catalog')"
            ).fetchall()
        }
        expected = {t.name for t in TABLES} | {v.name for v in VIEWS}
        self.check(
            "O",
            "exposure:only_documented_tables_and_views",
            objects == expected,
            f"unexpected={sorted(objects - expected)}, missing={sorted(expected - objects)}",
        )
        columns = self.con.execute(
            "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'main'"
        ).fetchall()
        leaked = [
            f"{t}.{c}" for t, c in columns if FORBIDDEN_NAME_PATTERN.search(c) or FORBIDDEN_NAME_PATTERN.search(t)
        ]
        self.check("O", "exposure:no_hidden_or_ground_truth_columns", not leaked, f"suspicious columns: {leaked}")
        for table in TABLES:
            actual = [
                r[0]
                for r in self.con.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position",
                    [table.name],
                ).fetchall()
            ]
            self.check(
                "O",
                f"exposure:columns_match_metadata:{table.name}",
                actual == list(table.column_names),
                f"columns {actual}",
            )


def validate_database(db_path: Path, config: GeneratorConfig) -> list[CheckResult]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        checker = _Checker(con, config)
        for step in (
            checker.row_counts,
            checker.date_ranges,
            checker.keys_nulls_domains,
            checker.null_rates,
            checker.referential_integrity,
            checker.value_bounds,
            checker.date_logic,
            checker.revenue_consistency,
            checker.subscription_consistency,
            checker.customer_consistency,
            checker.marketing_consistency,
            checker.business_sanity,
            checker.no_hidden_exposure,
        ):
            step()
        return checker.results
    finally:
        con.close()
