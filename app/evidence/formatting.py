"""Deterministic number display, and number extraction for response validation.

Claims embed numbers rendered by ``format_value``. The response validator extracts every number
from generated text with ``extract_numbers`` and checks it against the numbers the evidence
actually contains (``number_is_supported``), allowing only for display rounding.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_MONEY_UNITS = ("SGD", "SGD per month")
_COUNT_UNITS = ("customers", "tickets", "count", "opportunities", "rows", "leads", "conversions")


def format_value(value: float | int | None, unit: str | None) -> str:
    if value is None:
        return "not available"
    v = float(value)
    if unit in _MONEY_UNITS or (unit or "").startswith("SGD"):
        return f"SGD {v:,.0f}" if abs(v) >= 100 else f"SGD {v:,.2f}"
    if unit in ("ratio", "rate"):
        return format_percent(v)
    if unit in _COUNT_UNITS:
        return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.1f}"
    if unit == "hours":
        return f"{v:,.1f} hours"
    if unit == "days":
        return f"{v:,.1f} days"
    return f"{v:,.4g}"


def format_percent(fraction: float, *, signed: bool = False) -> str:
    """A fraction as a percentage: 2 decimals below 10%, 1 decimal otherwise."""
    pct = fraction * 100
    decimals = 2 if abs(pct) < 10 else 1
    return f"{pct:+.{decimals}f}%" if signed else f"{pct:.{decimals}f}%"


def format_money_change(value: float) -> str:
    """An unsigned money amount for "increased/decreased by" wording."""
    return format_value(abs(value), "SGD")


# ------------------------------------------------------------------------------------------------ extraction

_DATE_PATTERNS = (
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{4}-\d{2}\b"),
    re.compile(r"\b\d{4}-Q[1-4]\b", re.IGNORECASE),
    re.compile(r"\bQ[1-4]\b", re.IGNORECASE),
    re.compile(r"\b(?:19|20|21)\d{2}\b(?!\s*%)(?![.,]\d)"),  # a bare year
    # Identifiers are names, not quantities: evidence / claim / call IDs (E12, C3, T1) and prefixed IDs whose
    # digits follow letters and a separator (CUST-002529, EV-00042, INV-00731, Q-76ade5f37395, query_1847,
    # run-abc123, R-eval-direct-1).
    re.compile(r"\b[ECT]\d+\b"),
    re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*[-_][A-Za-z]*\d[A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*\b"),
)
_NUMBER = re.compile(
    r"(?<![\w.])(?P<sign>[-+\u2212])?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s*(?P<suffix>%|percent\b|pp\b|k\b|K\b|thousand\b|m\b|M\b|million\b|bn\b|billion\b))?"
)
_SCALE = {"k": 1e3, "K": 1e3, "thousand": 1e3, "m": 1e6, "M": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}


@dataclass(frozen=True)
class ParsedNumber:
    raw: str
    value: float
    decimals: int
    percent: bool
    scale: float

    @property
    def tolerance(self) -> float:
        return 0.5 * 10 ** (-self.decimals) * self.scale + 1e-9


def extract_numbers(text: str) -> list[ParsedNumber]:
    """Numbers stated in text, ignoring dates, years, quarters and evidence/claim identifiers."""
    cleaned = text
    for pattern in _DATE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    numbers = []
    for match in _NUMBER.finditer(cleaned):
        digits = match.group("num")
        suffix = match.group("suffix")
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        magnitude = float(digits.replace(",", ""))
        scale = _SCALE.get(suffix or "", 1.0)
        sign = -1.0 if match.group("sign") in ("-", "\u2212") else 1.0
        numbers.append(
            ParsedNumber(
                raw=match.group(0).strip(),
                value=sign * magnitude * scale,
                decimals=decimals,
                percent=suffix in ("%", "percent", "pp"),
                scale=scale,
            )
        )
    return numbers


def number_is_supported(number: ParsedNumber, allowed: list[float]) -> bool:
    """True when the stated number equals an evidence number up to display rounding (sign-insensitive)."""
    target = abs(number.value)
    for value in allowed:
        if not math.isfinite(value):
            continue
        candidates = (abs(value) * 100, abs(value)) if number.percent else (abs(value),)
        for candidate in candidates:
            if abs(candidate - target) <= number.tolerance + 1e-9 * max(1.0, candidate):
                return True
    return False


def is_small_count(number: ParsedNumber) -> bool:
    """Small whole numbers without units (e.g. "3 months", "top 5") are structural, not business values."""
    return not number.percent and number.scale == 1.0 and number.decimals == 0 and abs(number.value) <= 24
