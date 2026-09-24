"""Calendar helpers: month and week grids for the reporting window."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from data.generator.config import ANNUAL_SPEND_GROWTH, SPEND_REFERENCE_DATE

# Weeks containing these dates get reduced activity (all regions unless listed).
HOLIDAY_WEEK_FACTORS: tuple[tuple[date, float, tuple[str, ...] | None], ...] = (
    (date(2024, 12, 25), 0.55, None),
    (date(2025, 1, 1), 0.75, None),
    (date(2025, 12, 25), 0.55, None),
    (date(2026, 1, 1), 0.75, None),
    (date(2025, 1, 29), 0.80, ("APAC",)),  # Lunar New Year
    (date(2026, 2, 17), 0.80, ("APAC",)),
)


def to_day(d: date) -> np.datetime64:
    return np.datetime64(d, "D")


@dataclass(frozen=True)
class Timeline:
    start: date
    end: date

    @property
    def month_starts(self) -> list[date]:
        months, current = [], self.start
        while current <= self.end:
            months.append(current)
            current = date(current.year + (current.month == 12), current.month % 12 + 1, 1)
        return months

    @property
    def month_ends(self) -> list[date]:
        starts = self.month_starts
        return [s - timedelta(days=1) for s in starts[1:]] + [self.end]

    @property
    def n_months(self) -> int:
        return len(self.month_starts)

    def month_index(self, d: date) -> int:
        """Index of the calendar month of ``d`` within the window (-1 if before the window)."""
        if d < self.start:
            return -1
        return (d.year - self.start.year) * 12 + d.month - self.start.month

    @property
    def week_starts(self) -> list[date]:
        """Mondays starting weeks that lie entirely inside the window."""
        first = self.start + timedelta(days=(7 - self.start.weekday()) % 7)
        weeks, current = [], first
        while current + timedelta(days=6) <= self.end:
            weeks.append(current)
            current += timedelta(days=7)
        return weeks

    @property
    def days(self) -> np.ndarray:
        return np.arange(to_day(self.start), to_day(self.end) + 1, dtype="datetime64[D]")


def month_index_array(days: np.ndarray, start: date) -> np.ndarray:
    """Vectorised month index relative to ``start`` for datetime64[D] values."""
    months = days.astype("datetime64[M]").astype(int)
    return months - np.datetime64(start, "M").astype(int)


def spend_growth_factor(d: date) -> float:
    years = (d - SPEND_REFERENCE_DATE).days / 365.25
    return 1.0 + ANNUAL_SPEND_GROWTH * years


def holiday_factor(week_start: date, region: str) -> float:
    factor = 1.0
    for holiday, value, regions in HOLIDAY_WEEK_FACTORS:
        if week_start <= holiday <= week_start + timedelta(days=6) and (regions is None or region in regions):
            factor = min(factor, value)
    return factor
