"""Transparent, rule-based customer risk scoring from observable data only.

The score adds fixed points for observable warning signals measured as of a date:

| Signal | Observable measure | Points |
|---|---|---|
| low_seat_utilisation | mean weekly active users in the last 4 weeks / licensed seats | 30 if < 0.35, 15 if < 0.45 |
| usage_decline | mean weekly active users, last 4 weeks vs weeks 9-16 before | 30 if down >= 40%, 15 if down >= 25% |
| low_feature_breadth | mean distinct features used per week, last 4 weeks | 10 if < 4 |
| negative_sentiment | share of Negative tickets in the last 90 days (>= 2 tickets) | 10 if >= 50% |
| ticket_increase | tickets in the last 60 days minus the 60 days before | 10 if up by >= 2 |
| recent_contraction | a contraction subscription record in the last 90 days | 5 |

The maximum is 95. Bands: high >= 45, medium >= 25, otherwise low.

Calibration (observable data only): the signals and weights were chosen by scoring customers at
four past dates (2025-08-31, 2025-11-30, 2026-02-28, 2026-05-31) and comparing the churn observed
in the following three months by band. With these rules the bands were monotonic at every date:
high-band churn was 8.8-16.5% vs 4.4-6.2% in the low band (the high/medium gap was narrow at
2025-11-30). Slow ticket resolution was also evaluated but showed no consistent lift (0.9-1.2x),
so it is not scored. The analytics test suite re-runs this backtest.

This is a heuristic prioritisation aid, not a churn probability. It uses no hidden generator
variable. Its only inputs are usage_events, support_tickets, subscriptions and customers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from pydantic import BaseModel

from app.analytics.common import CUSTOMER_FILTER_COLUMNS, business_as_of, filter_clause, to_filters
from app.analytics.dimensions import Filters
from app.analytics.errors import InvalidRequestError
from app.analytics.executor import QueryRunner
from app.analytics.kpis.sql import in_force
from app.analytics.models import AnalyticsResult, to_number
from app.database.base import Database

RiskBand = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class RiskRules:
    """Thresholds and points (documented and justified in the module docstring)."""

    utilisation_low: float = 0.35
    utilisation_moderate: float = 0.45
    utilisation_points_low: int = 30
    utilisation_points_moderate: int = 15
    usage_decline_major: float = 0.40
    usage_decline_minor: float = 0.25
    usage_points_major: int = 30
    usage_points_minor: int = 15
    breadth_min_features: float = 4.0
    breadth_points: int = 10
    negative_share: float = 0.5
    negative_min_tickets: int = 2
    negative_points: int = 10
    tickets_increase: int = 2
    tickets_points: int = 10
    contraction_points: int = 5
    high_band: int = 45
    medium_band: int = 25


RULES = RiskRules()


class RiskSignal(BaseModel):
    signal: str
    observed: str
    points: int
    description: str


class CustomerRisk(BaseModel):
    customer_id: str
    company_name: str
    segment: str
    region: str
    mrr: float
    risk_score: int
    risk_band: RiskBand
    signals: list[RiskSignal]


def score_customer_risk(
    db: Database,
    *,
    as_of: date | None = None,
    filters: Filters | dict[str, str] | None = None,
    min_band: RiskBand = "low",
    limit: int | None = None,
    rules: RiskRules = RULES,
) -> AnalyticsResult[CustomerRisk]:
    """Score every customer active at the close of ``as_of`` (default: the business as-of date)."""
    if limit is not None and limit < 1:
        raise InvalidRequestError("limit must be positive")
    as_of_date = business_as_of(as_of)
    active = to_filters(filters)
    clause, bind = filter_clause(active, CUSTOMER_FILTER_COLUMNS, "score_customer_risk")
    runner = QueryRunner(db, "score_customer_risk")
    sql = f"""
WITH scored AS (
    SELECT c.customer_id, c.company_name, c.segment, c.region, s.seats, s.monthly_recurring_revenue AS mrr
    FROM subscriptions AS s
    JOIN customers AS c ON c.customer_id = s.customer_id
    WHERE {in_force("s", "$as_of")}{clause}
),
usage AS (
    SELECT
        u.customer_id,
        AVG(u.active_users) FILTER (WHERE u.event_date > $as_of - INTERVAL 28 DAY) AS recent_wau,
        AVG(u.active_users) FILTER (WHERE u.event_date <= $as_of - INTERVAL 56 DAY) AS baseline_wau,
        AVG(u.feature_usage) FILTER (WHERE u.event_date > $as_of - INTERVAL 28 DAY) AS recent_features
    FROM usage_events AS u
    WHERE u.event_date > $as_of - INTERVAL 112 DAY AND u.event_date <= $as_of
    GROUP BY 1
),
tickets AS (
    SELECT
        t.customer_id,
        COUNT(*) FILTER (WHERE t.created_at > $as_of - INTERVAL 60 DAY) AS tickets_recent_60d,
        COUNT(*) FILTER (WHERE t.created_at <= $as_of - INTERVAL 60 DAY) AS tickets_prior_60d,
        COUNT(*) FILTER (WHERE t.created_at > $as_of - INTERVAL 90 DAY) AS tickets_90d,
        COUNT(*) FILTER (WHERE t.created_at > $as_of - INTERVAL 90 DAY AND t.sentiment = 'Negative') AS negative_90d
    FROM support_tickets AS t
    WHERE t.created_at > $as_of - INTERVAL 120 DAY AND t.created_at < $as_of + INTERVAL 1 DAY
    GROUP BY 1
),
contractions AS (
    SELECT customer_id, COUNT(*) AS contractions_90d
    FROM subscriptions
    WHERE change_type = 'contraction' AND start_date > $as_of - INTERVAL 90 DAY AND start_date <= $as_of
    GROUP BY 1
)
SELECT
    sc.*, u.recent_wau, u.baseline_wau, u.recent_features,
    COALESCE(t.tickets_recent_60d, 0) AS tickets_recent_60d, COALESCE(t.tickets_prior_60d, 0) AS tickets_prior_60d,
    COALESCE(t.tickets_90d, 0) AS tickets_90d, COALESCE(t.negative_90d, 0) AS negative_90d,
    COALESCE(k.contractions_90d, 0) AS contractions_90d
FROM scored AS sc
LEFT JOIN usage AS u ON u.customer_id = sc.customer_id
LEFT JOIN tickets AS t ON t.customer_id = sc.customer_id
LEFT JOIN contractions AS k ON k.customer_id = sc.customer_id
ORDER BY sc.customer_id
"""
    records = runner.records(
        sql, {"as_of": as_of_date, **bind}, calculation="observable risk signals per active customer"
    )
    scored = [_score(r, rules) for r in records]
    band_rank = {"low": 0, "medium": 1, "high": 2}
    selected = [c for c in scored if band_rank[c.risk_band] >= band_rank[min_band]]
    selected.sort(key=lambda c: (-c.risk_score, -c.mrr, c.customer_id))
    if limit is not None:
        selected = selected[:limit]
    counts = {band: sum(c.risk_band == band for c in scored) for band in ("high", "medium", "low")}
    return AnalyticsResult[CustomerRisk](
        operation="score_customer_risk",
        status="ok" if scored else "no_data",
        filters=active,
        data=selected,
        summary={
            "as_of": as_of_date.isoformat(),
            "customers_scored": len(scored),
            **{f"{band}_risk_customers": n for band, n in counts.items()},
        },
        limitations=[
            "Rule-based heuristic for prioritising account reviews; the score is not a calibrated churn probability.",
            "Signals indicate observed patterns (for example, 'usage decline is present'); they do not establish why "
            "a customer may churn.",
            "Customers without usage 9-16 weeks before the as-of date have no baseline for the usage-decline signal.",
        ],
        provenance=runner.provenance(
            "risk_score = sum of points for observable signals (low seat utilisation, usage decline, low feature "
            "breadth, negative sentiment, ticket increase, recent contraction); bands high >= "
            f"{rules.high_band}, medium >= {rules.medium_band}"
        ),
    )


def _decline(recent: object, baseline: object) -> float | None:
    r, b = to_number(recent), to_number(baseline)
    if r is None or b is None or b <= 0:
        return None
    return 1.0 - float(r) / float(b)


def _score(r: dict[str, object], rules: RiskRules) -> CustomerRisk:
    signals: list[RiskSignal] = []
    recent_wau, seats = to_number(r["recent_wau"]), to_number(r["seats"])
    if recent_wau is not None and seats:
        utilisation = float(recent_wau) / float(seats)
        if utilisation < rules.utilisation_moderate:
            low = utilisation < rules.utilisation_low
            signals.append(
                RiskSignal(
                    signal="low_seat_utilisation",
                    observed=f"{utilisation:.0%} of seats active in the last 4 weeks",
                    points=rules.utilisation_points_low if low else rules.utilisation_points_moderate,
                    description="Low seat utilisation is present.",
                )
            )
    decline = _decline(r["recent_wau"], r["baseline_wau"])
    if decline is not None and decline >= rules.usage_decline_minor:
        major = decline >= rules.usage_decline_major
        signals.append(
            RiskSignal(
                signal="usage_decline",
                observed=f"weekly active users down {decline:.0%} vs 9-16 weeks earlier",
                points=rules.usage_points_major if major else rules.usage_points_minor,
                description="Usage decline is present.",
            )
        )
    features = to_number(r["recent_features"])
    if features is not None and recent_wau and float(features) < rules.breadth_min_features:
        signals.append(
            RiskSignal(
                signal="low_feature_breadth",
                observed=f"{float(features):.1f} distinct features used per week (last 4 weeks)",
                points=rules.breadth_points,
                description="Narrow feature usage is present.",
            )
        )
    total, negative = int(r["tickets_90d"]), int(r["negative_90d"])  # type: ignore[call-overload]
    if total >= rules.negative_min_tickets and negative / total >= rules.negative_share:
        signals.append(
            RiskSignal(
                signal="negative_sentiment",
                observed=f"{negative} of {total} tickets negative (90 days)",
                points=rules.negative_points,
                description="Negative ticket sentiment is present.",
            )
        )
    recent, prior = int(r["tickets_recent_60d"]), int(r["tickets_prior_60d"])  # type: ignore[call-overload]
    if recent - prior >= rules.tickets_increase:
        signals.append(
            RiskSignal(
                signal="ticket_increase",
                observed=f"{recent} tickets in 60 days vs {prior} before",
                points=rules.tickets_points,
                description="Ticket volume increased.",
            )
        )
    if int(r["contractions_90d"]) > 0:  # type: ignore[call-overload]
        signals.append(
            RiskSignal(
                signal="recent_contraction",
                observed="contraction in the last 90 days",
                points=rules.contraction_points,
                description="A recent contraction is present.",
            )
        )
    score = min(100, sum(s.points for s in signals))
    band: RiskBand = "high" if score >= rules.high_band else "medium" if score >= rules.medium_band else "low"
    return CustomerRisk(
        customer_id=str(r["customer_id"]),
        company_name=str(r["company_name"]),
        segment=str(r["segment"]),
        region=str(r["region"]),
        mrr=float(to_number(r["mrr"]) or 0.0),
        risk_score=score,
        risk_band=band,
        signals=signals,
    )
