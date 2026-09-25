"""Monthly series from the generated dataset, cross-checked against the independent pandas reference and Phase 2."""

from __future__ import annotations

from datetime import date

import pytest

from app.analytics.errors import InvalidFilterValueError, InvalidPeriodError
from app.analytics.kpis import KPIService
from app.database.base import Database
from app.timeseries import prepare_monthly_series
from tests.integration.reference_kpis import Reference

pytestmark = pytest.mark.slow

LABELS = [f"{2024 + (8 + i) // 12}-{(8 + i) % 12 + 1:02d}" for i in range(24)]  # 2024-09 .. 2026-08


@pytest.mark.parametrize(
    "filters", [{}, {"segment": "Enterprise"}, {"country": "Singapore", "segment": "Enterprise"}, {"plan": "Growth"}]
)
def test_revenue_series_matches_reference(full_db: Database, reference: Reference, filters: dict[str, str]) -> None:
    series = prepare_monthly_series(full_db, "revenue", filters=filters)
    assert series.status == "ok" and series.labels() == LABELS
    assert series.start == date(2024, 9, 1) and series.end == date(2026, 8, 31)
    for point in series.points:
        assert point.value == pytest.approx(reference.revenue(point.start, point.end, filters)["revenue"], rel=1e-9)
    assert set(series.source_tables) == {"customers", "daily_revenue"} and series.query_id in series.query_ids


@pytest.mark.parametrize("filters", [{}, {"region": "APAC"}, {"country": "Singapore", "segment": "Enterprise"}])
def test_state_series_match_reference(full_db: Database, reference: Reference, filters: dict[str, str]) -> None:
    mrr = prepare_monthly_series(full_db, "mrr", filters=filters)
    customers = prepare_monthly_series(full_db, "customer_count", filters=filters)
    assert mrr.labels() == customers.labels() == LABELS
    for m, c in zip(mrr.points, customers.points, strict=True):
        state = reference.recurring_state(m.end, filters)
        assert m.value == pytest.approx(state["mrr"], rel=1e-9)
        assert c.value == state["customers"]


def test_state_series_equal_phase2_kpis_at_month_ends(full_db: Database) -> None:
    service = KPIService(full_db)
    mrr = prepare_monthly_series(full_db, "mrr", filters={"segment": "SMB"})
    customers = prepare_monthly_series(full_db, "customer_count", filters={"segment": "SMB"})
    for index in (0, 11, 23):
        period = mrr.points[index].period
        assert mrr.points[index].value == pytest.approx(
            service.calculate_kpi("mrr", period=period, segment="SMB").value
        )
        kpi = service.calculate_kpi("customer_count", period=period, segment="SMB")
        assert customers.points[index].value == kpi.value


@pytest.mark.parametrize("filters", [{}, {"ticket_category": "Bug"}, {"region": "EMEA", "ticket_priority": "High"}])
def test_ticket_series_matches_reference(full_db: Database, reference: Reference, filters: dict[str, str]) -> None:
    series = prepare_monthly_series(full_db, "support_ticket_volume", filters=filters)
    for point in series.points:
        assert point.value == len(reference.tickets(point.start, point.end, filters))


def test_months_without_tickets_are_true_zeros(full_db: Database, reference: Reference) -> None:
    filters = {"country": "Mexico", "ticket_category": "How-To", "ticket_priority": "Urgent"}
    series = prepare_monthly_series(full_db, "support_ticket_volume", filters=filters)
    zeros = [p for p in series.points if p.observation == "true_zero"]
    assert zeros and len(series.points) == 24
    for point in zeros:
        assert point.value == 0.0 and len(reference.tickets(point.start, point.end, filters)) == 0
    assert all(p.observation != "missing" for p in series.points)


@pytest.mark.parametrize("feature", ["Dashboards", "AI Insights"])
def test_adoption_series_matches_reference(full_db: Database, reference: Reference, feature: str) -> None:
    series = prepare_monthly_series(full_db, "product_adoption", filters={"product_feature": feature})
    for point in series.points:
        assert point.value == pytest.approx(reference.adoption(point.start, point.end, feature), rel=1e-9)
    assert all(0 <= (p.value or 0) <= 1 for p in series.points)


def test_months_before_a_feature_exists_are_not_zeros(full_db: Database) -> None:
    series = prepare_monthly_series(full_db, "product_adoption", filters={"product_feature": "AI Insights"})
    assert series.points[0].period == "2026-03" and len(series.points) == 6  # not 24 months with leading zeros
    assert all(p.observation == "observed" for p in series.points)
    assert any("not treated as zero" in note for note in series.limitations)
    assert any("partial" in note for note in series.limitations)


def test_cutoff_bounds_the_series_and_every_query(full_db: Database) -> None:
    cutoff = date(2026, 5, 20)
    for metric in ("revenue", "mrr", "support_ticket_volume"):
        series = prepare_monthly_series(full_db, metric, end_date=cutoff)
        assert series.points[-1].period == "2026-04"  # May is incomplete at the cutoff
        for query in series.provenance.queries:
            for name, value in query.parameters.items():
                if name.endswith("date"):
                    assert date.fromisoformat(value) <= cutoff, (metric, name, value)


def test_invalid_requests(full_db: Database) -> None:
    with pytest.raises(InvalidPeriodError):
        prepare_monthly_series(full_db, "revenue", end_date=date(2026, 9, 30))  # after the as-of date
    with pytest.raises(InvalidPeriodError):
        prepare_monthly_series(full_db, "revenue", start_date=date(2026, 6, 1), end_date=date(2026, 5, 31))
    with pytest.raises(InvalidFilterValueError):
        prepare_monthly_series(full_db, "product_adoption", filters={"product_feature": "Teleportation"})
    before = prepare_monthly_series(full_db, "revenue", end_date=date(2024, 9, 15))
    assert before.status == "no_data" and not before.points


def test_start_date(full_db: Database) -> None:
    series = prepare_monthly_series(full_db, "revenue", start_date=date(2025, 6, 10))
    assert series.points[0].period == "2025-07"  # June is not complete from the start date
