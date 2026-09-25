"""Period labels used by scenarios (``2026-08``, ``2026-Q2``, ``2026``), parsed independently of production."""

from __future__ import annotations

import calendar
import re
from datetime import date

from pydantic import BaseModel

_MONTH = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")
_QUARTER = re.compile(r"^(\d{4})-Q([1-4])$")
_YEAR = re.compile(r"^(\d{4})$")
_RANGE = re.compile(r"^(\d{4}-\d{2}):(\d{4}-\d{2})$")


class PeriodRange(BaseModel):
    label: str
    start: date
    end: date


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def parse_period(label: str) -> PeriodRange:
    """``YYYY-MM``, ``YYYY-Qn``, ``YYYY`` or a month range ``YYYY-MM:YYYY-MM`` (inclusive)."""
    if match := _RANGE.match(label):
        opening, closing = parse_period(match.group(1)), parse_period(match.group(2))
        return PeriodRange(label=label, start=opening.start, end=closing.end)
    if match := _MONTH.match(label):
        year, month = int(match.group(1)), int(match.group(2))
        return PeriodRange(label=label, start=date(year, month, 1), end=_month_end(year, month))
    if match := _QUARTER.match(label):
        year, quarter = int(match.group(1)), int(match.group(2))
        first = 3 * (quarter - 1) + 1
        return PeriodRange(label=label, start=date(year, first, 1), end=_month_end(year, first + 2))
    if match := _YEAR.match(label):
        year = int(match.group(1))
        return PeriodRange(label=label, start=date(year, 1, 1), end=date(year, 12, 31))
    raise ValueError(f"Unsupported period label {label!r} (use YYYY-MM, YYYY-Qn or YYYY)")


def month_label(d: date) -> str:
    return f"{d.year}-{d.month:02d}"


def month_ranges(start: date, end: date) -> list[PeriodRange]:
    """Every calendar month from the month of ``start`` to the month of ``end``, inclusive."""
    months: list[PeriodRange] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append(parse_period(f"{year}-{month:02d}"))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def shift_months(label: str, offset: int) -> str:
    """The month label ``offset`` months from a month label (negative: earlier)."""
    period = parse_period(label)
    index = period.start.year * 12 + period.start.month - 1 + offset
    return f"{index // 12}-{index % 12 + 1:02d}"
