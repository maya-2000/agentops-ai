"""Period resolution anchored to the synthetic business's as-of date.

Business "today" is ``Settings.as_of_date`` (2026-08-31), never the system clock. Relative
periods resolve to the latest *complete* calendar unit on or before that date:

- ``last_month``: the latest complete month. With as-of 2026-08-31 (a month end) this is
  August 2026; with a mid-month as-of date it would be the previous month.
- ``previous_month``: the month before ``last_month`` (July 2026).
- ``last_quarter`` / ``last_year``: the latest complete quarter / calendar year
  (2026-Q2 and 2025 for as-of 2026-08-31).
- ``trailing_N_months``: the N complete months ending with ``last_month``.
- ``ytd``: 1 January of the as-of year to the as-of date.

Explicit forms: ``YYYY-MM`` / ``month:YYYY-MM``, ``YYYY-Qn`` / ``quarter:YYYY-Qn``,
``YYYY`` / ``year:YYYY``, or an explicit ``(start, end)`` pair (inclusive).
"""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import date, timedelta

from pydantic import BaseModel, model_validator

from app.analytics.errors import InvalidPeriodError
from app.config import get_settings

_DAYS_PER_MONTH = 365.25 / 12


class Period(BaseModel):
    """An inclusive date range with a human-readable label."""

    start: date
    end: date
    label: str

    @model_validator(mode="after")
    def _ordered(self) -> Period:
        if self.end < self.start:
            raise InvalidPeriodError(f"Period end {self.end} is before start {self.start}")
        return self

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    @property
    def is_calendar_months(self) -> bool:
        """True when the period starts on a month start and ends on a month end."""
        return self.start.day == 1 and self.end == _month_end(self.end)

    @property
    def months(self) -> float:
        """Length in months: exact for whole calendar months, else days / 30.44."""
        if self.is_calendar_months:
            return float((self.end.year - self.start.year) * 12 + self.end.month - self.start.month + 1)
        return self.days / _DAYS_PER_MONTH

    @property
    def opening_date(self) -> date:
        """The day before the period starts (its close is the period's opening state)."""
        return self.start - timedelta(days=1)


def _month_end(d: date) -> date:
    return date(d.year, d.month, monthrange(d.year, d.month)[1])


def _add_months(d: date, months: int) -> date:
    idx = d.year * 12 + d.month - 1 + months
    return date(idx // 12, idx % 12 + 1, 1)


def month_period(year: int, month: int) -> Period:
    start = date(year, month, 1)
    return Period(start=start, end=_month_end(start), label=start.strftime("%Y-%m"))


def quarter_period(year: int, quarter: int) -> Period:
    if quarter not in (1, 2, 3, 4):
        raise InvalidPeriodError(f"Quarter must be 1-4, got {quarter}")
    start = date(year, 3 * (quarter - 1) + 1, 1)
    return Period(start=start, end=_month_end(_add_months(start, 2)), label=f"{year}-Q{quarter}")


def year_period(year: int) -> Period:
    return Period(start=date(year, 1, 1), end=date(year, 12, 31), label=str(year))


def last_complete_month(as_of: date) -> Period:
    anchor = as_of if as_of == _month_end(as_of) else _add_months(as_of.replace(day=1), -1)
    return month_period(anchor.year, anchor.month)


def last_complete_quarter(as_of: date) -> Period:
    q = (as_of.month - 1) // 3 + 1
    current = quarter_period(as_of.year, q)
    if current.end <= as_of:
        return current
    return previous_period(current)


def explicit_period(start: date, end: date, label: str | None = None) -> Period:
    return Period(start=start, end=end, label=label or f"{start.isoformat()} to {end.isoformat()}")


_MONTH_RE = re.compile(r"^(?:month:)?(\d{4})-(\d{2})$")
_QUARTER_RE = re.compile(r"^(?:quarter:)?(\d{4})-Q([1-4])$", re.IGNORECASE)
_YEAR_RE = re.compile(r"^(?:year:)?(\d{4})$")
_TRAILING_RE = re.compile(r"^trailing_(\d{1,2})_months$")


def resolve_period(spec: str | Period | None = None, *, as_of: date | None = None) -> Period:
    """Resolve a period specification. ``None`` means ``last_month``."""
    if isinstance(spec, Period):
        return spec
    as_of = as_of or get_settings().as_of_date
    text = (spec or "last_month").strip()
    key = text.lower()

    if key in ("last_month", "latest_month"):
        return last_complete_month(as_of)
    if key == "previous_month":
        return previous_period(last_complete_month(as_of))
    if key == "last_quarter":
        return last_complete_quarter(as_of)
    if key == "previous_quarter":
        return previous_period(last_complete_quarter(as_of))
    if key == "last_year":
        return year_period(as_of.year - (0 if as_of == date(as_of.year, 12, 31) else 1))
    if key == "ytd":
        return Period(start=date(as_of.year, 1, 1), end=as_of, label=f"{as_of.year} YTD")
    if match := _TRAILING_RE.match(key):
        n = int(match.group(1))
        if n < 1:
            raise InvalidPeriodError("trailing period must cover at least one month")
        last = last_complete_month(as_of)
        start = _add_months(last.start, -(n - 1))
        return Period(start=start, end=last.end, label=f"trailing {n} months to {last.label}")
    if match := _MONTH_RE.match(key):
        year, month = int(match.group(1)), int(match.group(2))
        if not 1 <= month <= 12:
            raise InvalidPeriodError(f"Invalid month in {text!r}")
        return month_period(year, month)
    if match := _QUARTER_RE.match(text):
        return quarter_period(int(match.group(1)), int(match.group(2)))
    if match := _YEAR_RE.match(key):
        return year_period(int(match.group(1)))
    raise InvalidPeriodError(
        f"Unrecognised period {text!r}. Use last_month, previous_month, last_quarter, previous_quarter, "
        "last_year, ytd, trailing_N_months, YYYY-MM, YYYY-Qn, YYYY or explicit start/end dates."
    )


def previous_period(period: Period) -> Period:
    """The immediately preceding period of the same shape (calendar-aware for months/quarters/years)."""
    if period.is_calendar_months:
        months = int(period.months)
        start = _add_months(period.start, -months)
        end = _month_end(_add_months(period.start, -1))
        if months == 1:
            return month_period(start.year, start.month)
        if months == 3 and start.month in (1, 4, 7, 10):
            return quarter_period(start.year, (start.month - 1) // 3 + 1)
        if months == 12 and start.month == 1:
            return year_period(start.year)
        return Period(start=start, end=end, label=f"{start.strftime('%Y-%m')} to {end.strftime('%Y-%m')}")
    end = period.start - timedelta(days=1)
    start = end - timedelta(days=period.days - 1)
    return explicit_period(start, end)


def same_period_last_year(period: Period) -> Period:
    """The same calendar span one year earlier (29 February maps to 28 February)."""

    def shift(d: date) -> date:
        try:
            return d.replace(year=d.year - 1)
        except ValueError:  # 29 February
            return d.replace(year=d.year - 1, day=28)

    start, end = shift(period.start), shift(period.end)
    if period.is_calendar_months:
        end = _month_end(end)
    return Period(start=start, end=end, label=f"{period.label} (prior year)")


def months_in(period: Period) -> list[Period]:
    """Calendar months overlapping the period, clipped to the period bounds."""
    months: list[Period] = []
    cursor = period.start.replace(day=1)
    while cursor <= period.end:
        month = month_period(cursor.year, cursor.month)
        start, end = max(month.start, period.start), min(month.end, period.end)
        months.append(Period(start=start, end=end, label=month.label))
        cursor = _add_months(cursor, 1)
    return months
