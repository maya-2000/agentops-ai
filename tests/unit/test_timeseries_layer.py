"""Static checks of the Phase 3 metric registry, filter validation, calendar helpers and the series model."""

from __future__ import annotations

import math
from datetime import date

import pytest

from app.analytics.common import CUSTOMER_FILTER_COLUMNS
from app.analytics.dimensions import Filters
from app.analytics.errors import InvalidFilterValueError, InvalidRequestError, UnsupportedDimensionError
from app.analytics.kpis import KPI_REGISTRY
from app.analytics.kpis.sql import TEMPLATES
from app.analytics.revenue import MRRSeriesRow
from app.timeseries import SERIES_METRIC_KEYS, SERIES_METRICS, UnsupportedMetricError, get_series_metric
from app.timeseries.calendar import (
    add_months,
    first_complete_month_start,
    following_months,
    last_complete_month_end,
    month_end,
    months_between,
)
from app.timeseries.metrics import SERIES_FILTER_KEYS, validate_series_filters
from tests.phase3_support import synthetic_series


def test_registered_metrics_are_the_phase3_scope() -> None:
    assert SERIES_METRIC_KEYS == ("revenue", "mrr", "customer_count", "support_ticket_volume", "product_adoption")


@pytest.mark.parametrize("key", SERIES_METRIC_KEYS)
def test_metric_is_an_existing_kpi_with_consistent_filters(key: str) -> None:
    spec = SERIES_METRICS[key]
    kpi = KPI_REGISTRY[spec.kpi_key]  # every series metric is a Phase 2 KPI
    assert spec.unit == kpi.unit
    assert set(spec.supported_filters) <= set(SERIES_FILTER_KEYS)
    assert "customer_id" not in spec.supported_filters
    assert set(spec.required_filters) <= set(spec.supported_filters)
    if spec.source == "kpi_monthly":
        template = TEMPLATES[kpi.template]
        assert set(spec.supported_filters) <= set(template.filter_columns)
        assert "month" in template.dimension_columns
        assert spec.value_field in kpi.components
    else:
        assert set(spec.supported_filters) <= set(CUSTOMER_FILTER_COLUMNS) | {"plan"}
        assert spec.value_field in MRRSeriesRow.model_fields


def test_missing_data_policy_per_metric() -> None:
    assert all(
        SERIES_METRICS[k].empty_month_is_zero for k in ("revenue", "mrr", "customer_count", "support_ticket_volume")
    )
    adoption = SERIES_METRICS["product_adoption"]
    assert not adoption.empty_month_is_zero and adoption.upper_limit == 1.0


@pytest.mark.parametrize("key", ["nrr", "gross_margin", "", "revenue; DROP TABLE customers"])
def test_unsupported_metric(key: str) -> None:
    with pytest.raises(UnsupportedMetricError):
        get_series_metric(key)


def test_metric_lookup_is_case_insensitive() -> None:
    assert get_series_metric(" MRR ").key == "mrr"


def test_filters_are_validated_and_canonicalised() -> None:
    revenue = SERIES_METRICS["revenue"]
    assert validate_series_filters(revenue, {"segment": "enterprise", "region": "apac"}) == {
        "region": "APAC",
        "segment": "Enterprise",
    }
    assert validate_series_filters(revenue, Filters(country="Singapore")) == {"country": "Singapore"}
    assert validate_series_filters(revenue, None) == {}


def test_filter_errors() -> None:
    revenue, adoption = SERIES_METRICS["revenue"], SERIES_METRICS["product_adoption"]
    with pytest.raises(UnsupportedDimensionError):
        validate_series_filters(revenue, {"ticket_category": "Bug"})  # valid dimension, not for revenue
    with pytest.raises(UnsupportedDimensionError):
        validate_series_filters(revenue, {"customer_id": "CUST-000001"})  # single-entity series excluded
    with pytest.raises(UnsupportedDimensionError):
        validate_series_filters(revenue, {"colour": "red"})
    with pytest.raises(InvalidFilterValueError):
        validate_series_filters(revenue, {"country": "Atlantis"})
    with pytest.raises(InvalidRequestError):
        validate_series_filters(adoption, {})  # product_feature is required


def test_calendar_helpers() -> None:
    assert last_complete_month_end(date(2026, 8, 31)) == date(2026, 8, 31)
    assert last_complete_month_end(date(2026, 8, 30)) == date(2026, 7, 31)
    assert first_complete_month_start(date(2024, 9, 1)) == date(2024, 9, 1)
    assert first_complete_month_start(date(2026, 3, 2)) == date(2026, 4, 1)
    assert add_months(date(2026, 11, 15), 3) == date(2027, 2, 1)
    assert add_months(date(2026, 1, 31), -1) == date(2025, 12, 1)
    assert month_end(date(2028, 2, 3)) == date(2028, 2, 29)
    assert [m.label for m in months_between(date(2025, 11, 20), date(2026, 2, 1))] == [
        "2025-11",
        "2025-12",
        "2026-01",
        "2026-02",
    ]
    assert [m.label for m in following_months(date(2026, 8, 31), 3)] == ["2026-09", "2026-10", "2026-11"]


def test_series_model_helpers() -> None:
    series = synthetic_series([1.0, None, 3.0, 4.0])
    values = series.values()
    assert values[0] == 1.0 and math.isnan(values[1])
    assert series.missing_count == 1
    assert series.contiguous_tail_start() == 2
    assert series.labels() == ["2024-09", "2024-10", "2024-11", "2024-12"]
    assert series.query_id is None and series.source_tables == []
    assert synthetic_series([1.0, 2.0, None]).contiguous_tail_start() == 3
    assert synthetic_series([1.0, 2.0]).contiguous_tail_start() == 0
