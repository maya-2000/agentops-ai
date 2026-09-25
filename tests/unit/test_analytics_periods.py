"""Period resolution anchored to the business as-of date (never the system clock)."""

from __future__ import annotations

from datetime import date

import pytest

from app.analytics.errors import InvalidPeriodError
from app.analytics.periods import (
    Period,
    explicit_period,
    months_in,
    previous_period,
    resolve_period,
    same_period_last_year,
)

AS_OF = date(2026, 8, 31)


@pytest.mark.parametrize(
    ("spec", "start", "end", "label"),
    [
        ("last_month", date(2026, 8, 1), date(2026, 8, 31), "2026-08"),
        (None, date(2026, 8, 1), date(2026, 8, 31), "2026-08"),
        ("previous_month", date(2026, 7, 1), date(2026, 7, 31), "2026-07"),
        ("last_quarter", date(2026, 4, 1), date(2026, 6, 30), "2026-Q2"),
        ("previous_quarter", date(2026, 1, 1), date(2026, 3, 31), "2026-Q1"),
        ("last_year", date(2025, 1, 1), date(2025, 12, 31), "2025"),
        ("ytd", date(2026, 1, 1), date(2026, 8, 31), "2026 YTD"),
        ("trailing_12_months", date(2025, 9, 1), date(2026, 8, 31), "trailing 12 months to 2026-08"),
        ("2025-02", date(2025, 2, 1), date(2025, 2, 28), "2025-02"),
        ("month:2024-02", date(2024, 2, 1), date(2024, 2, 29), "2024-02"),
        ("2026-Q1", date(2026, 1, 1), date(2026, 3, 31), "2026-Q1"),
        ("quarter:2025-q4", date(2025, 10, 1), date(2025, 12, 31), "2025-Q4"),
        ("2025", date(2025, 1, 1), date(2025, 12, 31), "2025"),
    ],
)
def test_resolve_period(spec: str | None, start: date, end: date, label: str) -> None:
    period = resolve_period(spec, as_of=AS_OF)
    assert (period.start, period.end, period.label) == (start, end, label)


def test_last_month_uses_the_business_as_of_date_by_default() -> None:
    assert resolve_period("last_month").label == "2026-08"  # settings.as_of_date, not today's date


def test_mid_month_as_of_means_previous_complete_month() -> None:
    assert resolve_period("last_month", as_of=date(2026, 8, 15)).label == "2026-07"
    assert resolve_period("last_quarter", as_of=date(2026, 6, 29)).label == "2026-Q1"


@pytest.mark.parametrize("spec", ["last_fortnight", "2026-13", "2026-Q5", "trailing_0_months", ""])
def test_invalid_specs(spec: str) -> None:
    with pytest.raises(InvalidPeriodError):
        resolve_period(spec or "   ", as_of=AS_OF)


def test_end_before_start_rejected() -> None:
    with pytest.raises(InvalidPeriodError):
        Period(start=date(2026, 8, 31), end=date(2026, 8, 1), label="bad")


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("2026-03", "2026-02"), ("2026-01", "2025-12"), ("2026-Q1", "2025-Q4"), ("2025", "2024")],
)
def test_previous_period_calendar_aware(spec: str, expected: str) -> None:
    assert previous_period(resolve_period(spec, as_of=AS_OF)).label == expected


def test_previous_period_for_custom_range_has_equal_length() -> None:
    period = explicit_period(date(2026, 3, 10), date(2026, 3, 19))
    previous = previous_period(period)
    assert previous.end == date(2026, 3, 9) and previous.days == period.days == 10


def test_same_period_last_year_and_leap_day() -> None:
    assert same_period_last_year(resolve_period("2026-Q2", as_of=AS_OF)).start == date(2025, 4, 1)
    feb = same_period_last_year(resolve_period("2024-02", as_of=AS_OF))
    assert (feb.start, feb.end) == (date(2023, 2, 1), date(2023, 2, 28))


def test_months_and_opening_date() -> None:
    quarter = resolve_period("2026-Q2", as_of=AS_OF)
    assert quarter.months == 3.0 and quarter.opening_date == date(2026, 3, 31)
    assert [m.label for m in months_in(quarter)] == ["2026-04", "2026-05", "2026-06"]
    clipped = months_in(explicit_period(date(2026, 1, 15), date(2026, 2, 10)))
    assert (clipped[0].start, clipped[-1].end) == (date(2026, 1, 15), date(2026, 2, 10))
    assert explicit_period(date(2026, 1, 1), date(2026, 1, 10)).months == pytest.approx(10 / (365.25 / 12))
