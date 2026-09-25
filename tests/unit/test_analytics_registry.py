"""Static checks of the KPI registry, dimension allow-list, templates and value rules."""

from __future__ import annotations

import re

import pytest
import sqlglot

from app.analytics.dimensions import DIMENSIONS, FILTER_KEYS, Filters, get_dimension, normalise_value
from app.analytics.errors import InvalidFilterValueError, UnsupportedDimensionError, UnsupportedKPIError
from app.analytics.kpis import (
    KPI_KEYS,
    KPI_REGISTRY,
    KPIParameters,
    ValueRule,
    get_kpi_definition,
    list_kpi_definitions,
)
from app.analytics.kpis.sql import TEMPLATES
from app.database import metadata

REQUIRED_KEYS = [
    "revenue",
    "mrr",
    "arr",
    "revenue_growth",
    "logo_churn_rate",
    "revenue_churn_rate",
    "retention_rate",
    "nrr",
    "cac",
    "clv",
    "arpu",
    "average_order_value",
    "conversion_rate",
    "pipeline_value",
    "win_rate",
    "sales_cycle",
    "support_ticket_volume",
    "average_resolution_time",
    "product_adoption",
    "customer_count",
]
KNOWN_OBJECTS = set(metadata.table_names()) | set(metadata.view_names())


def test_all_twenty_enumerated_kpis_registered_in_order() -> None:
    assert list(KPI_KEYS) == REQUIRED_KEYS
    assert [d.key for d in list_kpi_definitions()] == REQUIRED_KEYS


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_definition_is_complete(key: str) -> None:
    d = get_kpi_definition(key)
    for field in ("name", "definition", "formula", "sql", "unit", "time_grain", "interpretation"):
        assert getattr(d, field).strip(), f"{key}.{field} is empty"
    assert d.limitations and all(item.strip() for item in d.limitations)
    assert d.dependencies
    for dependency in d.dependencies:
        assert dependency in KNOWN_OBJECTS or dependency in KPI_REGISTRY, (key, dependency)
    assert d.components and d.sql == TEMPLATES[d.template].sql


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_templates_are_valid_parameterised_sql(name: str) -> None:
    template = TEMPLATES[name]
    assert "{dimension}" in template.sql and "{filters}" in template.sql
    for dimension in (None, *template.dimension_columns):
        sql = template.render(dimension, dict.fromkeys(template.filter_columns, "x"))
        sqlglot.parse_one(sql.replace("$", ":"), read="duckdb")  # parses as a single statement
        assert "'x'" not in sql  # filter values never appear in SQL text
        assert set(re.findall(r"\$f_(\w+)", sql)) == set(template.filter_columns)
    for table in template.source_tables:
        assert table in KNOWN_OBJECTS
    for key in (*template.filter_columns, *template.dimension_columns):
        assert key in DIMENSIONS, (name, key)


def test_render_rejects_unlisted_columns() -> None:
    template = TEMPLATES["revenue"]
    with pytest.raises(UnsupportedDimensionError):
        template.render("sales_rep", {})
    with pytest.raises(UnsupportedDimensionError):
        template.render(None, {"campaign": "CMP-001"})


def test_dimension_registry_uses_phase1_vocabularies() -> None:
    assert DIMENSIONS["segment"].allowed_values == metadata.SEGMENTS
    assert DIMENSIONS["region"].allowed_values == metadata.REGIONS
    assert DIMENSIONS["plan"].allowed_values == metadata.PLANS
    assert DIMENSIONS["country"].allowed_values == metadata.COUNTRIES
    for spec in DIMENSIONS.values():
        if spec.value_kind != "time":
            table_columns = metadata.TABLES_BY_NAME[spec.source_table].column_names
            assert spec.source_column in table_columns, spec.key
    assert set(FILTER_KEYS) == set(Filters.model_fields)
    assert not DIMENSIONS["month"].filterable


def test_filter_validation() -> None:
    assert normalise_value("segment", "mid-market") == "Mid-Market"
    with pytest.raises(InvalidFilterValueError):
        Filters(region="Antarctica")
    with pytest.raises(UnsupportedDimensionError):
        get_dimension("colour")
    with pytest.raises(UnsupportedDimensionError):
        normalise_value("month", "2026-08")
    assert get_dimension("sales_rep").lookup_sql is not None and "$value" in get_dimension("sales_rep").lookup_sql  # type: ignore[operator]


def test_parameters_accept_flat_filters() -> None:
    params = KPIParameters.model_validate({"period": "last_month", "segment": "smb", "region": "APAC"})
    assert params.filters.active() == {"region": "APAC", "segment": "SMB"}


def test_unknown_kpi() -> None:
    with pytest.raises(UnsupportedKPIError):
        get_kpi_definition("gross_margin")


@pytest.mark.parametrize(
    ("rule", "components", "expected", "status"),
    [
        (ValueRule(kind="component", component="x"), {"x": 5}, 5.0, "ok"),
        (ValueRule(kind="scaled", component="x", factor=12), {"x": 2}, 24.0, "ok"),
        (ValueRule(kind="ratio", numerator="a", denominator="b"), {"a": 1, "b": 4}, 0.25, "ok"),
        (ValueRule(kind="ratio", numerator="a", denominator="b"), {"a": 1, "b": 0}, None, "insufficient_data"),
        (ValueRule(kind="complement_ratio", numerator="a", denominator="b"), {"a": 1, "b": 4}, 0.75, "ok"),
        (ValueRule(kind="growth", numerator="a", denominator="b"), {"a": 90, "b": 100}, -0.1, "ok"),
        (ValueRule(kind="growth", numerator="a", denominator="b"), {"a": 90, "b": 0}, None, "insufficient_data"),
        (
            ValueRule(kind="net_retention"),
            {"opening_mrr": 100, "churned_mrr": 10, "contraction_mrr": 5, "expansion_mrr": 20},
            1.05,
            "ok",
        ),
        (ValueRule(kind="net_retention"), {"opening_mrr": 0}, None, "insufficient_data"),
        (
            ValueRule(kind="lifetime_value"),
            {
                "closing_mrr": 1000,
                "closing_customers": 10,
                "churned_customers": 2,
                "opening_customers": 100,
                "period_months": 1,
            },
            5000.0,
            "ok",
        ),
        (
            ValueRule(kind="lifetime_value"),
            {
                "closing_mrr": 1000,
                "closing_customers": 10,
                "churned_customers": 0,
                "opening_customers": 100,
                "period_months": 1,
            },
            None,
            "insufficient_data",
        ),
    ],
)
def test_value_rules(rule: ValueRule, components: dict[str, float], expected: float | None, status: str) -> None:
    value, got_status, _ = rule.apply(components)  # type: ignore[arg-type]
    assert got_status == status
    assert value == (pytest.approx(expected) if expected is not None else None)


def test_committed_kpi_catalog_is_up_to_date() -> None:
    from app.analytics.kpis.catalog import CATALOG_PATH, render_catalog

    assert CATALOG_PATH.read_text(encoding="utf-8") == render_catalog(), (
        "docs/kpi-catalog.md is stale: run `python -m app.analytics.kpis.catalog`"
    )
