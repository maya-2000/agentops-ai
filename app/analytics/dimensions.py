"""The single allow-list of analysis dimensions and typed filters.

Nothing a caller supplies is ever placed into SQL text. A caller can only *name* a dimension
from this registry. Each calculation then maps that name to a column expression it declares
itself. Filter values are validated (enumerated dimensions against the Phase 1 metadata
vocabularies; open-ended dimensions against the database) and always passed as bound
parameters.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from app.analytics.errors import InvalidFilterValueError, UnsupportedDimensionError
from app.database.metadata import (
    ACQUISITION_CHANNELS,
    COUNTRIES,
    INDUSTRIES,
    OPPORTUNITY_TYPES,
    PLANS,
    REGIONS,
    REVENUE_TYPES,
    SEGMENTS,
    TICKET_CATEGORIES,
    TICKET_PRIORITIES,
)

ValueKind = Literal["enumerated", "lookup", "time"]


class DimensionSpec(BaseModel):
    """A dimension that can be used to filter and/or break down analytics results."""

    model_config = ConfigDict(frozen=True)

    key: str
    display_name: str
    source_table: str
    source_column: str
    value_kind: ValueKind
    allowed_values: tuple[str, ...] | None = None
    filterable: bool = True
    filter_semantics: str

    @property
    def lookup_sql(self) -> str | None:
        """Existence check for open-ended (lookup) values; the value is always bound as ``$value``."""
        if self.value_kind != "lookup":
            return None
        return f'SELECT COUNT(*) FROM "{self.source_table}" WHERE "{self.source_column}" = $value'


def _enum(key: str, name: str, table: str, column: str, values: tuple[str, ...], semantics: str) -> DimensionSpec:
    return DimensionSpec(
        key=key,
        display_name=name,
        source_table=table,
        source_column=column,
        value_kind="enumerated",
        allowed_values=values,
        filter_semantics=semantics,
    )


def _lookup(key: str, name: str, table: str, column: str, semantics: str) -> DimensionSpec:
    return DimensionSpec(
        key=key,
        display_name=name,
        source_table=table,
        source_column=column,
        value_kind="lookup",
        filter_semantics=semantics,
    )


DIMENSIONS: dict[str, DimensionSpec] = {
    spec.key: spec
    for spec in (
        _enum(
            "region",
            "Region",
            "customers",
            "region",
            REGIONS,
            "Customer's sales region. For sales KPIs, the opportunity's region (rep territory).",
        ),
        _enum("country", "Country", "customers", "country", COUNTRIES, "Customer's billing country."),
        _enum(
            "segment",
            "Segment",
            "customers",
            "segment",
            SEGMENTS,
            "Customer segment. For sales KPIs, the opportunity's segment (prospects included).",
        ),
        _enum("industry", "Industry", "customers", "industry", INDUSTRIES, "Customer's primary industry."),
        _enum(
            "plan",
            "Plan",
            "subscriptions",
            "plan",
            PLANS,
            "Plan in force: per revenue day, at the measurement date for point-in-time KPIs, "
            "or at the period opening for churn/retention KPIs.",
        ),
        _enum(
            "acquisition_channel",
            "Acquisition channel",
            "customers",
            "acquisition_channel",
            ACQUISITION_CHANNELS,
            "Channel that acquired the customer. For marketing KPIs, the campaign channel.",
        ),
        _enum(
            "revenue_type",
            "Revenue type",
            "daily_revenue",
            "revenue_type",
            REVENUE_TYPES,
            "Subscription (recurring) or usage (API overage) revenue.",
        ),
        _enum(
            "opportunity_type",
            "Opportunity type",
            "sales_opportunities",
            "opportunity_type",
            OPPORTUNITY_TYPES,
            "New Business or Expansion opportunity.",
        ),
        _enum(
            "ticket_category",
            "Ticket category",
            "support_tickets",
            "category",
            TICKET_CATEGORIES,
            "Support ticket category.",
        ),
        _enum(
            "ticket_priority",
            "Ticket priority",
            "support_tickets",
            "priority",
            TICKET_PRIORITIES,
            "Support ticket priority.",
        ),
        _lookup("sales_rep", "Sales rep", "sales_opportunities", "sales_rep", "Opportunity owner."),
        _lookup("campaign", "Campaign", "marketing_campaigns", "campaign_id", "Marketing campaign identifier."),
        _lookup("product_feature", "Product feature", "product_features", "feature_name", "Product feature name."),
        _lookup("customer_id", "Customer", "customers", "customer_id", "A single customer account."),
        DimensionSpec(
            key="month",
            display_name="Month",
            source_table="(time)",
            source_column="(period date)",
            value_kind="time",
            filterable=False,
            filter_semantics="Breakdown only (YYYY-MM); restrict time with the period instead.",
        ),
        DimensionSpec(
            key="quarter",
            display_name="Quarter",
            source_table="(time)",
            source_column="(period date)",
            value_kind="time",
            filterable=False,
            filter_semantics="Breakdown only (YYYY-Qn); restrict time with the period instead.",
        ),
    )
}

FILTER_KEYS: tuple[str, ...] = tuple(k for k, spec in DIMENSIONS.items() if spec.filterable)


def get_dimension(key: str) -> DimensionSpec:
    try:
        return DIMENSIONS[key]
    except KeyError:
        raise UnsupportedDimensionError(
            f"Unknown dimension {key!r}. Allowed dimensions: {', '.join(DIMENSIONS)}"
        ) from None


def normalise_value(key: str, value: str) -> str:
    """Validate an enumerated filter value and return its canonical spelling (case-insensitive)."""
    spec = get_dimension(key)
    if not spec.filterable:
        raise UnsupportedDimensionError(f"{spec.display_name} cannot be used as a filter: {spec.filter_semantics}")
    if spec.value_kind != "enumerated":
        return value.strip()
    assert spec.allowed_values is not None
    by_lower = {v.lower(): v for v in spec.allowed_values}
    canonical = by_lower.get(value.strip().lower())
    if canonical is None:
        raise InvalidFilterValueError(
            f"{value!r} is not a valid {spec.display_name.lower()}. Allowed: {', '.join(spec.allowed_values)}"
        )
    return canonical


class Filters(BaseModel):
    """Typed, validated filters. Every field maps 1:1 onto a filterable dimension."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    region: str | None = None
    country: str | None = None
    segment: str | None = None
    industry: str | None = None
    plan: str | None = None
    acquisition_channel: str | None = None
    revenue_type: str | None = None
    opportunity_type: str | None = None
    ticket_category: str | None = None
    ticket_priority: str | None = None
    sales_rep: str | None = None
    campaign: str | None = None
    product_feature: str | None = None
    customer_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalise(cls, data: object) -> object:
        if isinstance(data, dict):
            return {
                k: (normalise_value(k, v) if isinstance(v, str) and k in DIMENSIONS else v) for k, v in data.items()
            }
        return data

    def active(self) -> dict[str, str]:
        """Filters that are set, in registry order."""
        return {k: v for k in FILTER_KEYS if (v := getattr(self, k)) is not None}

    def merged(self, **extra: str) -> Filters:
        return Filters(**{**self.active(), **extra})


assert set(Filters.model_fields) == set(FILTER_KEYS), "Filters must mirror the filterable dimensions"
