"""Schema DDL, constraints and data-dictionary generation."""

from __future__ import annotations

import duckdb
import pytest

from app.database.data_dictionary import DICTIONARY_PATH, render_data_dictionary, render_er_diagram
from app.database.metadata import TABLES, VIEWS, table_names
from app.database.schema import create_table_sql, index_ddl


@pytest.fixture()
def empty_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    for table in TABLES:
        con.execute(create_table_sql(table))
    for stmt in index_ddl():
        con.execute(stmt)
    return con


def test_all_required_tables_defined() -> None:
    assert set(table_names()) == {
        "customers",
        "subscriptions",
        "usage_events",
        "sales_opportunities",
        "support_tickets",
        "marketing_campaigns",
        "daily_revenue",
        "product_features",
    }


def test_required_columns_present() -> None:
    required = {
        "customers": {
            "customer_id",
            "company_name",
            "country",
            "region",
            "industry",
            "company_size",
            "segment",
            "acquisition_channel",
            "signup_date",
        },
        "subscriptions": {
            "subscription_id",
            "customer_id",
            "plan",
            "monthly_recurring_revenue",
            "start_date",
            "end_date",
            "status",
            "previous_mrr",
            "current_mrr",
        },
        "usage_events": {
            "event_id",
            "customer_id",
            "event_date",
            "active_users",
            "sessions",
            "api_calls",
            "feature_usage",
        },
        "sales_opportunities": {
            "opportunity_id",
            "customer_id",
            "sales_rep",
            "created_date",
            "close_date",
            "stage",
            "deal_value",
            "probability",
        },
        "support_tickets": {
            "ticket_id",
            "customer_id",
            "created_at",
            "resolved_at",
            "priority",
            "category",
            "resolution_time",
            "sentiment",
        },
        "marketing_campaigns": {
            "campaign_id",
            "campaign_name",
            "channel",
            "date",
            "spend",
            "impressions",
            "clicks",
            "leads",
            "conversions",
        },
        "daily_revenue": {"date", "customer_id", "region", "segment", "plan", "revenue"},
        "product_features": {"feature_id", "feature_name", "date", "active_users", "adoption_rate"},
    }
    for table in TABLES:
        assert required[table.name] <= set(table.column_names), table.name


def test_every_column_documented() -> None:
    for table in TABLES:
        assert table.business_purpose and table.grain
        for col in table.columns:
            assert col.description, f"{table.name}.{col.name} lacks a description"


def test_ddl_creates_tables(empty_db: duckdb.DuckDBPyConnection) -> None:
    created = {r[0] for r in empty_db.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    assert created == set(table_names())


def test_check_constraint_rejects_unknown_category(empty_db: duckdb.DuckDBPyConnection) -> None:
    with pytest.raises(duckdb.ConstraintException):
        empty_db.execute(
            "INSERT INTO customers VALUES "
            "('CUST-1','X','Atlantis','APAC','Fintech','1-50','SMB','Referral','2025-01-01','active')"
        )


def test_foreign_key_rejects_orphans(empty_db: duckdb.DuckDBPyConnection) -> None:
    with pytest.raises(duckdb.ConstraintException):
        empty_db.execute(
            "INSERT INTO support_tickets VALUES ('T1','CUST-404','2025-01-01',NULL,'Low','Bug','Open',NULL,'Neutral')"
        )


def test_nullable_customer_on_prospect_opportunity(empty_db: duckdb.DuckDBPyConnection) -> None:
    empty_db.execute(
        "INSERT INTO sales_opportunities VALUES ('OPP-1',NULL,'Rep','New Business','SMB','APAC',"
        "'2025-01-01','2025-02-01','Lost','Lead',1000,0)"
    )


def test_primary_key_enforced(empty_db: duckdb.DuckDBPyConnection) -> None:
    row = "('F01','Dashboards','2025-01-01',10,0.5)"
    empty_db.execute(f"INSERT INTO product_features VALUES {row}")
    with pytest.raises(duckdb.ConstraintException):
        empty_db.execute(f"INSERT INTO product_features VALUES {row}")


def test_committed_data_dictionary_is_up_to_date() -> None:
    assert DICTIONARY_PATH.read_text(encoding="utf-8") == render_data_dictionary(), (
        "docs/data-dictionary.md is stale: run `python -m app.database.data_dictionary`"
    )


def test_er_diagram_covers_all_tables() -> None:
    diagram = render_er_diagram()
    assert diagram.startswith("erDiagram")
    for table in TABLES:
        assert f"{table.name.upper()} {{" in diagram
    assert "CUSTOMERS ||--o{ SUBSCRIPTIONS" in diagram


def test_dictionary_documents_views() -> None:
    text = render_data_dictionary()
    for view in VIEWS:
        assert f"`{view.name}`" in text
