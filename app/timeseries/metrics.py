"""The allow-list of business metrics that can be prepared as monthly time series.

Each metric is an existing Phase 2 KPI, so a forecast or anomaly is always about the same
number that ``calculate_kpi`` reports. Nothing here re-defines a KPI. The registry only records
how the KPI becomes a monthly series:

- ``kpi_monthly``: the KPI broken down by ``month`` in one query (flows and averages).
- ``mrr_series``: the Phase 2 month-end MRR series (point-in-time states, one query).

Missing-data policy (see ``empty_month_is_zero``):

- Inside the data coverage, a month without rows is a **true zero** only when zero is the
  correct value for that metric: no revenue recognised, no subscriptions in force, no tickets
  created. These tables record every event, so an absent row means none happened.
- For averages (product adoption) an absent month is **missing**, not zero: the average of
  nothing is undefined. Months before a feature is first observed are not part of its series.
- Months outside the data coverage and incomplete months are never part of a series.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from app.analytics.dimensions import Filters
from app.analytics.errors import InvalidRequestError, UnsupportedDimensionError
from app.timeseries.errors import UnsupportedMetricError

Measure = Literal["flow", "state", "average_rate"]
SeriesSource = Literal["kpi_monthly", "mrr_series"]
Transform = Literal["level", "difference", "pct_change"]

# Group filters a Phase 3 series may use. ``customer_id`` and other single-entity filters are
# excluded on purpose: single-account series are too small for monthly forecasting or scoring.
SERIES_FILTER_KEYS: tuple[str, ...] = (
    "region",
    "country",
    "segment",
    "industry",
    "acquisition_channel",
    "plan",
    "revenue_type",
    "ticket_category",
    "ticket_priority",
    "product_feature",
)

_CUSTOMER_GROUPS = ("region", "country", "segment", "industry", "acquisition_channel")


class SeriesMetric(BaseModel):
    """How a registered KPI is turned into a continuous monthly series."""

    model_config = ConfigDict(frozen=True)

    key: str
    name: str
    unit: str
    kpi_key: str
    measure: Measure
    source: SeriesSource
    value_field: str
    supported_filters: tuple[str, ...]
    required_filters: tuple[str, ...] = ()
    empty_month_is_zero: bool
    lower_limit: float | None = 0.0
    upper_limit: float | None = None
    anomaly_transform: Transform
    description: str


SERIES_METRICS: dict[str, SeriesMetric] = {
    m.key: m
    for m in (
        SeriesMetric(
            key="revenue",
            name="Revenue",
            unit="SGD",
            kpi_key="revenue",
            measure="flow",
            source="kpi_monthly",
            value_field="revenue",
            supported_filters=(*_CUSTOMER_GROUPS, "plan", "revenue_type"),
            empty_month_is_zero=True,
            anomaly_transform="pct_change",
            description="Recognised revenue (subscription + usage) per calendar month: the revenue KPI by month.",
        ),
        SeriesMetric(
            key="mrr",
            name="Monthly recurring revenue",
            unit="SGD per month",
            kpi_key="mrr",
            measure="state",
            source="mrr_series",
            value_field="mrr",
            supported_filters=(*_CUSTOMER_GROUPS, "plan"),
            empty_month_is_zero=True,
            anomaly_transform="pct_change",
            description="MRR at each month end (subscriptions in force at the close of the month end).",
        ),
        SeriesMetric(
            key="customer_count",
            name="Active customer count",
            unit="customers",
            kpi_key="customer_count",
            measure="state",
            source="mrr_series",
            value_field="active_customers",
            supported_filters=(*_CUSTOMER_GROUPS, "plan"),
            empty_month_is_zero=True,
            anomaly_transform="pct_change",
            description="Customers with a subscription in force at the close of each month end.",
        ),
        SeriesMetric(
            key="support_ticket_volume",
            name="Support ticket volume",
            unit="tickets",
            kpi_key="support_ticket_volume",
            measure="flow",
            source="kpi_monthly",
            value_field="tickets",
            supported_filters=(*_CUSTOMER_GROUPS, "ticket_category", "ticket_priority"),
            empty_month_is_zero=True,
            anomaly_transform="pct_change",
            description="Support tickets created per calendar month.",
        ),
        SeriesMetric(
            key="product_adoption",
            name="Product adoption",
            unit="ratio",
            kpi_key="product_adoption",
            measure="average_rate",
            source="kpi_monthly",
            value_field="average_daily_adoption_rate",
            supported_filters=("product_feature",),
            required_filters=("product_feature",),
            empty_month_is_zero=False,
            upper_limit=1.0,
            anomaly_transform="difference",
            description="Average daily adoption rate (feature DAU / platform DAU) of one feature per calendar month.",
        ),
    )
}

SERIES_METRIC_KEYS: tuple[str, ...] = tuple(SERIES_METRICS)


def get_series_metric(key: str) -> SeriesMetric:
    metric = SERIES_METRICS.get(key.strip().lower())
    if metric is None:
        raise UnsupportedMetricError(
            f"{key!r} is not supported for forecasting or anomaly detection. "
            f"Supported metrics: {', '.join(SERIES_METRIC_KEYS)}"
        )
    return metric


def validate_series_filters(metric: SeriesMetric, filters: Filters | Mapping[str, str] | None) -> dict[str, str]:
    """Validate and canonicalise filters with the Phase 2 ``Filters`` model, then apply the metric's allow-list."""
    if filters is None:
        active: dict[str, str] = {}
    elif isinstance(filters, Filters):
        active = filters.active()
    else:
        try:
            active = Filters(**{k: v for k, v in dict(filters).items() if v is not None}).active()
        except ValidationError as exc:
            unknown = [str(e["loc"][-1]) for e in exc.errors() if e["type"] == "extra_forbidden"]
            if unknown:
                raise UnsupportedDimensionError(f"Unknown filter(s): {', '.join(unknown)}") from None
            raise InvalidRequestError(f"Invalid filters: {exc}") from None
    for key in active:
        if key not in metric.supported_filters:
            raise UnsupportedDimensionError(
                f"{metric.key} series cannot be filtered by {key!r}. Supported filters: "
                f"{', '.join(metric.supported_filters)}"
            )
    for key in metric.required_filters:
        if key not in active:
            raise InvalidRequestError(f"{metric.key} series requires the {key!r} filter")
    return active
