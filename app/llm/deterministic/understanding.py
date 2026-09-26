"""Rule-based question understanding for the deterministic (offline) model.

It maps common business phrasings to the structured ``UnderstandingOutput`` using only the
vocabulary supplied in the request context (KPI keys, dimension values, feature names). It parses
language; it never looks up or produces a business number. Coverage is intentionally modest: a
network model handles open-ended phrasing through the same interface and validation.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from app.llm.schemas import AnalysisType, FilterItem, Intent, UnderstandingOutput

_OUT_OF_SCOPE = re.compile(
    r"\b(stock|share price|stock price|shares of|weather|bitcoin|crypto|apple|google|microsoft|amazon|tesla|"
    r"competitor\w*|salar(y|ies)|employee\w*|election|president|recipe|football|lottery|horoscope)\b",
    re.IGNORECASE,
)
# "drop" is a write only as a command ("Drop the revenue table"), not as a noun ("the biggest drop in revenue").
_WRITE = re.compile(
    r"\b(delete|insert|truncate|modify|overwrite|alter)\b|^\s*drop\b|"
    r"\bdrop\b(?=(?:\s+\w+){0,3}\s+(?:tables?|databases?|db|schemas?|views?|columns?|index|data|rows?|records?)\b)",
    re.IGNORECASE,
)

_METRICS: tuple[tuple[str, str], ...] = (
    (r"\bnet revenue retention\b|\bnrr\b", "nrr"),
    (r"\brevenue churn\b", "revenue_churn_rate"),
    (r"\brevenue growth\b|\bgrowth rate\b", "revenue_growth"),
    (r"\bmrr\b|\bmonthly recurring revenue\b|\brecurring revenue\b", "mrr"),
    (r"\barr\b|\bannual recurring revenue\b", "arr"),
    (r"\bretention\b", "retention_rate"),
    (r"\bchurn\w*", "logo_churn_rate"),
    (r"\bcac\b|\bcustomer acquisition cost\b|\bacquisition cost\b", "cac"),
    (r"\bclv\b|\bltv\b|\blifetime value\b", "clv"),
    (r"\barpu\b|\baverage revenue per (?:user|customer|account)\b", "arpu"),
    (r"\baverage order value\b|\baov\b|\bdeal size\b", "average_order_value"),
    (r"\bconversion rate\b|\blead conversion\b|\bconversions?\b", "conversion_rate"),
    (r"\bpipeline\b", "pipeline_value"),
    (r"\bwin rate\b|\bwin-rate\b", "win_rate"),
    (r"\bsales cycle\b", "sales_cycle"),
    (r"\bresolution time\b|\btime to resolve\b", "average_resolution_time"),
    (r"\bsupport[- ]tickets?\b|\btickets?\b|\bticket volume\b|\bsupport volume\b", "support_ticket_volume"),
    (r"\badoption\b", "product_adoption"),
    (r"\bcustomer count\b|\bnumber of customers\b|\bactive customers\b|\bhow many customers\b", "customer_count"),
    (r"\brevenue\b|\bsales revenue\b", "revenue"),
)
_DOMAINS: tuple[tuple[str, Intent], ...] = (
    (r"\b(sales|win rate|pipeline|deals?|opportunit\w+|reps?|quota)\b", Intent.SALES_ANALYSIS),
    (r"\b(marketing|campaigns?|channels?|cac|leads?|spend|roas)\b", Intent.MARKETING_ANALYSIS),
    (r"\b(support|tickets?|resolution)\b", Intent.SUPPORT_ANALYSIS),
    (r"\b(features?|adoption|product usage)\b", Intent.PRODUCT_ANALYSIS),
    (r"\b(churn\w*|retention|customers?|cohorts?|at[- ]risk|risk)\b", Intent.CUSTOMER_INVESTIGATION),
    (r"\b(revenue|mrr|arr|recurring)\b", Intent.REVENUE_INVESTIGATION),
)
_DIMENSION_STEMS = (
    ("segment", "segment"),
    ("region", "region"),
    ("countr", "country"),
    ("plan", "plan"),
    ("industr", "industry"),
    ("channel", "acquisition_channel"),
    ("categor", "ticket_category"),
    ("priorit", "ticket_priority"),
    ("rep", "sales_rep"),
    ("campaign", "campaign"),
    ("feature", "product_feature"),
)
_DIMENSION_ASK = re.compile(
    r"\b(?:by|which|per|across|each|what|every)\s+(?:customer\s+|ticket\s+|sales\s+|product\s+|marketing\s+|"
    r"acquisition\s+)?"
    r"(segment|region|countr(?:y|ies)|plan|industr(?:y|ies)|channel|categor(?:y|ies)|priority|rep|campaign|feature)s?\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "nine": 9, "twelve": 12, "a": 1}
_MONTHS_AHEAD = re.compile(
    r"\b(\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")[\s-]months?\s+"
    r"(?:ahead\b|(?:[a-z]+\s+){0,2}(?:forecast|projection|outlook)s?\b)"
)
_MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_MONTH_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t|tember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b(?:\s+(\d{4}))?",
    re.IGNORECASE,
)

_FORECAST = re.compile(
    r"\b(forecast\w*|predict\w*|projection\w*|outlook|next (?:month|quarter|year|\w+ months)|will .{0,40}\bbe)\b",
    re.IGNORECASE,
)
_ANOMALY = re.compile(r"\b(unusual|anomal\w*|abnormal|spikes?|outliers?|irregular|unexpected)\b", re.IGNORECASE)
_CAUSAL = re.compile(r"\b(caus\w+|what drove|drivers?|reason|explain\w*)\b", re.IGNORECASE)
_WHY = re.compile(r"\bwhy\b", re.IGNORECASE)
_CHANGE = re.compile(
    r"\b(chang\w*|increas\w*|decreas\w*|declin\w*|drop\w*|fell|fall\w*|grow\w*|grew|rise|rose|compar\w*|vs\.?|"
    r"versus|how much did|trend\w*|went (?:up|down))\b",
    re.IGNORECASE,
)
_CONTRIBUTION = re.compile(r"\b(contribut\w*|accounted for|account for|responsible for)\b", re.IGNORECASE)
_HIGHEST = re.compile(r"\b(highest|most|largest|biggest|top|maximum|best)\b", re.IGNORECASE)
_LOWEST = re.compile(r"\b(lowest|least|smallest|minimum|worst)\b", re.IGNORECASE)
_DECREASE = re.compile(r"\b(declin\w*|decreas\w*|drop\w*|fell|fall\w*|shr[ai]nk\w*|contract\w*)\b", re.IGNORECASE)
_INCREASE = re.compile(r"\b(increas\w*|grow\w*|grew|growth|gain\w*|rise|rose|rising|jump\w*)\b", re.IGNORECASE)
_PERCENT = re.compile(r"%|\b(percent\w*|relative)\b", re.IGNORECASE)
_RISK = re.compile(r"\b(at[- ]risk|risk\w*)\b", re.IGNORECASE)
_COHORT = re.compile(r"\bcohorts?\b", re.IGNORECASE)
_COMPARISON_MARKER = re.compile(
    r"\b(?:compared (?:with|to)|vs\.?|versus|than|relative to|against)\s+(?:the\s+|that\s+of\s+(?:the\s+)?)?$"
)
_FROM_MARKER = re.compile(r"\bfrom\s+(?:the\s+)?$")
# "may" is a month (not the verb) after a preposition or comparison word: "in May", "compared with May",
# "from May to July", "July vs May", "July and May".
_MAY_MONTH_CONTEXT = re.compile(
    r"\b(?:in|of|for|since|until|till|to|from|and|or|vs\.?|versus|with|than|against|between|through|during|"
    r"before|after|over)\s+$"
)
_ISO_MONTH = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])\b(?!-\d)")
_ISO_DATE = r"20\d{2}-\d{2}-\d{2}"
_ISO_RANGE = re.compile(rf"\b({_ISO_DATE})(?:\s+(?:to|until|through|-|\u2013)\s+|\.\.)({_ISO_DATE})\b")
_ISO_DAY = re.compile(rf"\b{_ISO_DATE}\b")
# A day of a month in words: "3 March YYYY", "the 3rd of March", "March 3", "March 3rd, YYYY". A day
# number next to a month name names one day, never the month. "March YYYY" and "during March" do not
# match: a four-digit year is not a day, and "3 months" / "3 weeks" are durations.
_MONTH_NAME = (
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t|tember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_DAY_NUMBER = r"(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?"
_DAY_OF_MONTH = re.compile(
    rf"\b{_DAY_NUMBER}\s+(?:of\s+)?{_MONTH_NAME}\b"
    rf"|\b{_MONTH_NAME}\.?\s+{_DAY_NUMBER}\b(?!\s*(?:months?|weeks?|days?|years?|quarters?|%|\d))",
    re.IGNORECASE,
)
# 03/04/YYYY or 3.4.YY: a single day, and ambiguous (day-month or month-day order).
_NUMERIC_DATE = re.compile(r"\b\d{1,2}[/.]\d{1,2}[/.](?:20)?\d{2}\b")
_ISO_QUARTER = re.compile(r"\b(20\d{2})-q([1-4])\b")


def understand(context: dict[str, Any]) -> dict[str, Any]:
    question: str = context["question"]
    q = " ".join(question.split())
    as_of = date.fromisoformat(context["as_of"])

    if _OUT_OF_SCOPE.search(q) or _WRITE.search(q):
        reason = (
            "The request asks to change data; the system is read-only."
            if _WRITE.search(q) and not _OUT_OF_SCOPE.search(q)
            else "The question is not about the Northwind Cloud business dataset."
        )
        return UnderstandingOutput(intent=Intent.UNSUPPORTED, unsupported_reason=reason, confidence=0.9).model_dump(
            mode="json"
        )

    metrics = _metrics(q)
    metric = metrics[0] if metrics else None
    dimensions = list(dict.fromkeys(_dimension_key(m.group(1)) for m in _DIMENSION_ASK.finditer(q)))
    filters = _filters(q, context, dimensions)
    periods, comparisons, day_level = _periods(q, as_of)
    horizon = _horizon(q, as_of)
    ambiguities: list[str] = []

    intent, analysis = _intent(q, metric, dimensions, horizon)
    if intent == Intent.UNSUPPORTED:
        return UnderstandingOutput(
            intent=intent,
            unsupported_reason="The question does not mention a supported business metric or area.",
            confidence=0.6,
        ).model_dump(mode="json")
    if intent == Intent.KPI_LOOKUP and (comparisons or len(periods) > 1):
        intent, analysis = Intent.PERIOD_COMPARISON, "change"  # two periods: a comparison, not a lookup
    if intent == Intent.FORECAST and metric is None and "revenue" in q.lower():
        metric = "revenue"
    if intent == Intent.CUSTOMER_INVESTIGATION and metric is None:
        metric = "logo_churn_rate" if re.search(r"churn", q, re.IGNORECASE) else None
    if len({_family(m) for m in metrics}) > 1 and intent in (Intent.REVENUE_INVESTIGATION, Intent.PERIOD_COMPARISON):
        intent = Intent.MIXED_INVESTIGATION
    material = False
    if day_level:
        ambiguities.append(
            "A single day was asked for, but KPIs are reported for whole months, quarters or years, so a day's "
            'value is not available. Ask for the whole month (for example "revenue in March", or YYYY-MM) or a '
            "quarter (YYYY-Qn)."
        )
        material = True
    if analysis == "lowest" and dimensions and _change_ranking(q, metric):
        # "Smallest decline" has no analytics answer (only the largest decline/increase are named): ask, never
        # answer with a level ranking.
        ambiguities.append(
            "A ranking by the smallest change is not available; ask for the largest decline or the largest increase."
        )
        material = True
    vocab: dict[str, list[str]] = context.get("dimension_values", {})
    shared = set(vocab.get("segment", [])) & set(vocab.get("plan", []))
    for item in filters:
        if item.dimension == "segment" and item.value in shared:
            ambiguities.append(f"'{item.value}' read as the {item.value} segment (not the {item.value} plan).")

    return UnderstandingOutput(
        intent=intent,
        metric=metric,
        period=periods[0] if periods and intent != Intent.FORECAST else None,
        comparison_period=comparisons[0] if comparisons else periods[1] if len(periods) > 1 else None,
        dimensions=dimensions,
        filters=filters,
        horizon=horizon if intent == Intent.FORECAST else None,
        analysis_type=analysis,
        confidence=0.7,
        ambiguities=ambiguities,
        material_ambiguity=material,
    ).model_dump(mode="json")


def _metrics(q: str) -> list[str]:
    """Metrics in the order they are mentioned; a specific phrase ("revenue churn") hides its parts."""
    taken: list[tuple[int, int]] = []
    found: list[tuple[int, str]] = []
    for pattern, key in _METRICS:
        for match in re.finditer(pattern, q, re.IGNORECASE):
            if any(match.start() < end and start < match.end() for start, end in taken):
                continue
            taken.append(match.span())
            found.append((match.start(), key))
    return list(dict.fromkeys(key for _, key in sorted(found)))


def _dimension_key(word: str) -> str:
    lowered = word.lower()
    return next(key for stem, key in _DIMENSION_STEMS if lowered.startswith(stem))


def _family(metric: str) -> str:
    if metric in ("revenue", "mrr", "arr", "revenue_growth", "arpu"):
        return "revenue"
    if metric in ("logo_churn_rate", "revenue_churn_rate", "retention_rate", "nrr", "customer_count", "clv"):
        return "customers"
    if metric in ("support_ticket_volume", "average_resolution_time"):
        return "support"
    return metric


def _intent(
    q: str, metric: str | None, dimensions: list[str], horizon: int | None
) -> tuple[Intent, AnalysisType | None]:
    if _FORECAST.search(q) or horizon is not None:
        return Intent.FORECAST, None
    if _ANOMALY.search(q):
        return Intent.ANOMALY_DETECTION, None
    causal = bool(_CAUSAL.search(q))
    if _WHY.search(q) or causal:
        analysis: AnalysisType = "causal" if causal else "change"
        family = _family(metric) if metric else None
        if family == "revenue" or (metric is None and re.search(r"revenue|mrr", q, re.IGNORECASE)):
            return Intent.REVENUE_INVESTIGATION, analysis
        if family == "customers" or re.search(r"churn|customer", q, re.IGNORECASE):
            return Intent.CUSTOMER_INVESTIGATION, analysis
        if family == "support":
            return Intent.SUPPORT_ANALYSIS, analysis
        for pattern, intent in _DOMAINS:
            if re.search(pattern, q, re.IGNORECASE):
                return intent, analysis
        return Intent.UNSUPPORTED, None
    if _RISK.search(q):
        return Intent.CUSTOMER_INVESTIGATION, "risk"
    if _COHORT.search(q):
        return Intent.CUSTOMER_INVESTIGATION, "cohort"
    if dimensions and _CONTRIBUTION.search(q):
        return Intent.DIMENSIONAL_COMPARISON, "contribution"
    if dimensions and _HIGHEST.search(q) and (ranking := _change_ranking(q, metric)):
        return Intent.DIMENSIONAL_COMPARISON, ranking
    if dimensions and (_HIGHEST.search(q) or _LOWEST.search(q)):
        return Intent.DIMENSIONAL_COMPARISON, "lowest" if _LOWEST.search(q) else "highest"
    if _CHANGE.search(q) and metric is not None:
        return Intent.PERIOD_COMPARISON, "change"
    if metric is not None:
        return Intent.KPI_LOOKUP, "value"
    for pattern, intent in _DOMAINS:
        if re.search(pattern, q, re.IGNORECASE):
            return intent, None
    return Intent.UNSUPPORTED, None


def _change_ranking(q: str, metric: str | None) -> AnalysisType | None:
    """A ranking of members by their change ("largest decline"), not by their level ("highest revenue")."""
    if metric == "revenue_growth":
        return None  # already a change metric: ranking its values is a level ranking
    percent = bool(_PERCENT.search(q))
    if _DECREASE.search(q):
        return "largest_pct_decrease" if percent else "largest_decrease"
    if _INCREASE.search(q):
        return "largest_pct_increase" if percent else "largest_increase"
    return None


def _filters(q: str, context: dict[str, Any], asked: list[str]) -> list[FilterItem]:
    vocab: dict[str, list[str]] = context.get("dimension_values", {})
    guards = {
        "plan": r"\bplans?\b",
        "ticket_category": r"\b(tickets?|support|category|categories)\b",
        "ticket_priority": r"\bpriorit\w+\b",
        "acquisition_channel": r"\b(channels?|campaigns?|marketing|acquired)\b",
        "product_feature": r"\b(features?|adoption)\b",
        "industry": r"\bindustr\w+\b",
    }
    items: list[FilterItem] = []
    taken: set[str] = set()
    for dimension, values in vocab.items():
        if dimension in asked:
            continue  # a dimension being compared is a breakdown, not a filter
        guard = guards.get(dimension)
        if guard and not re.search(guard, q, re.IGNORECASE):
            continue
        for value in sorted(values, key=len, reverse=True):
            if value.lower() in taken:
                continue
            if re.search(rf"(?<![\w-]){re.escape(value)}(?![\w-])", q, re.IGNORECASE):
                items.append(FilterItem(dimension=dimension, value=value))
                taken.add(value.lower())
                break
    return items


def _month_spec(name: str, year: str | None, as_of: date) -> str:
    month = next(i for i, full in enumerate(_MONTHS, start=1) if full.startswith(name.lower()[:3]))
    default_year = as_of.year if month <= as_of.month else as_of.year - 1
    return f"{default_year if year is None else int(year)}-{month:02d}"


def _periods(q: str, as_of: date) -> tuple[list[str], list[str], bool]:
    """(periods asked about, comparison periods, whether a day-level date could not be used) in question order.

    A period introduced by "compared with", "vs", "than", "relative to" or "against" is a comparison
    period, and so is the first period of "from A to B". An explicit ISO date range that covers whole
    months, a quarter or a year becomes that period; any other day-level date is reported, never read as
    a year.
    """
    found: list[tuple[int, str]] = []
    lowered = q.lower()
    iso_spans: list[tuple[int, int]] = []
    ranges: set[int] = set()
    day_level = False
    for match in _ISO_RANGE.finditer(q):
        iso_spans.append(match.span())
        spec = _range_spec(date.fromisoformat(match.group(1)), date.fromisoformat(match.group(2)))
        if spec is None:
            day_level = True
        else:
            found.append((match.start(), spec))
            ranges.add(match.start())
    for match in _ISO_DAY.finditer(q):
        if not any(start <= match.start() < end for start, end in iso_spans):
            iso_spans.append(match.span())
            day_level = True
    if any(_names_a_day(q, m) for m in _DAY_OF_MONTH.finditer(q)) or _NUMERIC_DATE.search(q):
        day_level = True
    for phrase, spec in (
        ("last month", "last_month"),
        ("this month", "last_month"),
        ("previous month", "previous_month"),
        ("prior month", "previous_month"),
        ("last quarter", "last_quarter"),
        ("previous quarter", "previous_quarter"),
        ("last year", "last_year"),
        ("year to date", "ytd"),
        ("this year", "ytd"),
        ("ytd", "ytd"),
    ):
        for match in re.finditer(rf"\b{phrase}\b", lowered):
            found.append((match.start(), spec))
    for match in re.finditer(r"\b(?:last|past|previous|trailing)\s+(\d+|\w+)\s+months\b", lowered):
        count = int(match.group(1)) if match.group(1).isdigit() else _NUMBER_WORDS.get(match.group(1), 0)
        if count:
            found.append((match.start(), f"trailing_{count}_months"))
    for match in _MONTH_RE.finditer(q):
        if match.group(1).lower() == "may" and not _may_is_month(q, match):
            continue  # "may" as a verb
        found.append((match.start(), _month_spec(match.group(1), match.group(2), as_of)))
    # Explicit ISO labels ("2026-07", "2026-Q2") name one period each, not a bare year.
    for match in _ISO_MONTH.finditer(q):
        found.append((match.start(), f"{match.group(1)}-{match.group(2)}"))
        iso_spans.append(match.span())
    for match in _ISO_QUARTER.finditer(lowered):
        found.append((match.start(), f"{match.group(1)}-Q{match.group(2)}"))
        iso_spans.append(match.span())

    def in_iso(position: int) -> bool:
        return any(start <= position < end for start, end in iso_spans)

    for match in re.finditer(r"\bq([1-4])\s*(\d{4})?\b", lowered):
        if in_iso(match.start()):
            continue
        year = match.group(2) or str(as_of.year)
        found.append((match.start(), f"{year}-Q{match.group(1)}"))
    month_years = {m.group(2) for m in _MONTH_RE.finditer(q) if m.group(2)}
    for match in re.finditer(r"\b(20\d{2})\b", q):
        if in_iso(match.start()):
            continue
        if match.group(1) not in month_years and not re.search(rf"q[1-4]\s*{match.group(1)}", lowered):
            found.append((match.start(), match.group(1)))
    ordered = sorted(found)
    periods: list[str] = []
    comparisons: list[str] = []
    for index, (position, spec) in enumerate(ordered):
        before = lowered[:position]
        from_a_to_b = _FROM_MARKER.search(before) and index + 1 < len(ordered) and position not in ranges
        if _COMPARISON_MARKER.search(before) or from_a_to_b:
            comparisons.append(spec)
        else:
            periods.append(spec)
    return periods, comparisons, day_level


def _names_a_day(q: str, match: re.Match[str]) -> bool:
    """A day-of-month match, unless its month word is the verb "may" ("the top 3 may change")."""
    month = (match.group(1) or match.group(2) or "").lower()
    if month != "may":
        return True
    after = q[match.end() :]
    return bool(re.match(r"\s*(?:,?\s*20\d{2}\b|[?.!,;]|$)", after)) or match.group(0).lower().startswith("may")


def _range_spec(start: date, end: date) -> str | None:
    """The month, quarter or year an explicit date range covers exactly (None for any other range)."""
    if end < start or start.day != 1 or (end + timedelta(days=1)).day != 1:
        return None
    months = (end.year - start.year) * 12 + end.month - start.month + 1
    if months == 1:
        return f"{start.year}-{start.month:02d}"
    if months == 3 and start.month in (1, 4, 7, 10):
        return f"{start.year}-Q{(start.month - 1) // 3 + 1}"
    if months == 12 and start.month == 1:
        return str(start.year)
    return None


def _may_is_month(q: str, match: re.Match[str]) -> bool:
    """Whether a "may" token names the month: followed by a year, or after a preposition or comparison word."""
    if match.group(2):
        return True
    before = q[: match.start()].lower()
    if _MAY_MONTH_CONTEXT.search(before):
        return True
    # A capitalised "May" inside the sentence (not its first word) is the month.
    return match.group(1) == "May" and bool(before.strip())


def _horizon(q: str, as_of: date) -> int | None:
    lowered = q.lower()
    if match := re.search(r"\bnext\s+(\d+|\w+)\s+months\b", lowered):
        return int(match.group(1)) if match.group(1).isdigit() else _NUMBER_WORDS.get(match.group(1))
    if re.search(r"\bnext month\b", lowered):
        return 1
    if re.search(r"\bnext quarter\b", lowered):
        return 3
    if re.search(r"\bnext year\b", lowered):
        return 12
    # "a 3-month forecast", "six month outlook", "3 months ahead": a count of months naming the horizon.
    if match := _MONTHS_AHEAD.search(lowered):
        return int(match.group(1)) if match.group(1).isdigit() else _NUMBER_WORDS[match.group(1)]
    if _FORECAST.search(q) and (match := re.search(r"\b(20\d{2})\b", q)):
        year = int(match.group(1))
        if year > as_of.year:
            return (year - as_of.year) * 12 + (12 - as_of.month)
    return None
