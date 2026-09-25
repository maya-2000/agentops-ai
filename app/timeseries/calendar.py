"""Calendar-month helpers for monthly series (complete months only)."""

from __future__ import annotations

from calendar import monthrange
from datetime import date, timedelta

from app.analytics.periods import Period, month_period


def month_start(d: date) -> date:
    return d.replace(day=1)


def month_end(d: date) -> date:
    return date(d.year, d.month, monthrange(d.year, d.month)[1])


def add_months(d: date, months: int) -> date:
    """First day of the month ``months`` after the month of ``d``."""
    index = d.year * 12 + d.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def last_complete_month_end(d: date) -> date:
    """The latest month end on or before ``d`` (``d`` itself when it is a month end)."""
    return d if d == month_end(d) else month_start(d) - timedelta(days=1)


def first_complete_month_start(d: date) -> date:
    """The earliest month start on or after ``d``."""
    return d if d.day == 1 else add_months(d, 1)


def month_label(d: date) -> str:
    return d.strftime("%Y-%m")


def months_between(start: date, end: date) -> list[Period]:
    """Calendar months from the month of ``start`` to the month of ``end`` (inclusive)."""
    months: list[Period] = []
    cursor = month_start(start)
    while cursor <= end:
        months.append(month_period(cursor.year, cursor.month))
        cursor = add_months(cursor, 1)
    return months


def following_months(last: date, count: int) -> list[Period]:
    """The ``count`` calendar months after the month of ``last``."""
    first = add_months(last, 1)
    return [month_period(m.year, m.month) for m in (add_months(first, i) for i in range(count))]
