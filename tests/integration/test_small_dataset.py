"""End-to-end generation on the small fixture (100 customers, 6 months)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb
import pytest

from app.database.metadata import TABLES, VIEWS
from data.generator.config import GeneratorConfig
from data.generator.generate import GenerationResult
from data.generator.manifest import CHECKSUMS_FILENAME, MANIFEST_FILENAME
from data.generator.validation import validate_database

MANIFEST_TEMPLATE_KEYS = [
    "dataset_name",
    "dataset_version",
    "random_seed",
    "generation_timestamp",
    "period_start",
    "period_end",
    "currency",
    "customer_count",
    "region_count",
    "country_count",
    "segment_count",
    "plan_count",
    "sales_rep_count",
    "tables",
    "schema_version",
]
MANIFEST_TABLES = [
    "customers",
    "subscriptions",
    "usage_events",
    "sales_opportunities",
    "support_tickets",
    "marketing_campaigns",
    "daily_revenue",
    "product_features",
]


def test_all_data_quality_checks_pass(small_dataset: GenerationResult) -> None:
    """Every structural check passes. Aggregate statistical sanity checks are only warnings at
    this scale (100 customers), so they are excluded here and asserted on the full dataset."""
    failures = [
        f"{c.name}: {c.details}" for c in small_dataset.quality_checks if not c.passed and c.severity == "error"
    ]
    assert not failures, failures
    assert all(c.severity == "warning" for c in small_dataset.quality_checks if c.category == "N")
    categories = {c.category for c in small_dataset.quality_checks}
    assert categories == set("ABCDEFGHIJKLMNO")


def test_all_tables_loaded(small_dataset: GenerationResult, small_config: GeneratorConfig) -> None:
    con = duckdb.connect(str(small_config.db_path), read_only=True)
    try:
        for table in TABLES:
            assert con.execute(f'SELECT COUNT(*) FROM "{table.name}"').fetchone()[0] > 0, table.name
        assert con.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 100
        span = con.execute("SELECT MIN(date), MAX(date) FROM daily_revenue").fetchone()
        assert (span[0].isoformat(), span[1].isoformat()) == ("2026-03-01", "2026-08-31")
    finally:
        con.close()


def test_manifest_follows_template_and_reflects_database(
    small_dataset: GenerationResult, small_config: GeneratorConfig
) -> None:
    path = small_config.metadata_dir / MANIFEST_FILENAME
    manifest = json.loads(path.read_text())
    assert list(manifest) == MANIFEST_TEMPLATE_KEYS
    assert list(manifest["tables"]) == MANIFEST_TABLES
    assert all(list(v) == ["row_count"] for v in manifest["tables"].values())
    assert manifest["dataset_name"] == "Northwind Cloud"
    assert manifest["currency"] == "SGD"
    assert manifest["random_seed"] == small_config.seed
    assert manifest["period_start"] == "2026-03-01" and manifest["period_end"] == "2026-08-31"
    assert manifest["generation_timestamp"] is not None
    assert manifest["customer_count"] == 100
    con = duckdb.connect(str(small_config.db_path), read_only=True)
    try:
        for table, entry in manifest["tables"].items():
            assert entry["row_count"] == con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        reps = con.execute("SELECT COUNT(DISTINCT sales_rep) FROM sales_opportunities").fetchone()[0]
    finally:
        con.close()
    assert manifest["sales_rep_count"] == reps
    assert manifest["tables"] == {k: {"row_count": v} for k, v in small_dataset.row_counts.items()}


def test_metadata_artifacts_written(small_dataset: GenerationResult, small_config: GeneratorConfig) -> None:
    checksums = json.loads((small_config.metadata_dir / CHECKSUMS_FILENAME).read_text())
    assert checksums["fingerprint_sha256"] == small_dataset.fingerprint
    assert set(checksums["tables"]) == set(MANIFEST_TABLES)
    dictionary = json.loads((small_config.metadata_dir / "data_dictionary.json").read_text())
    assert {t["name"] for t in dictionary["tables"]} == set(MANIFEST_TABLES)
    assert small_config.ground_truth_path.exists()


def test_ground_truth_is_outside_the_database(small_dataset: GenerationResult, small_config: GeneratorConfig) -> None:
    assert small_config.ground_truth_path.parent != small_config.db_path.parent
    con = duckdb.connect(str(small_config.db_path), read_only=True)
    try:
        names = [r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()]
    finally:
        con.close()
    assert set(names) == {t.name for t in TABLES} | {v.name for v in VIEWS}
    assert not any(word in n for n in names for word in ("truth", "health", "injected", "business_event"))


def test_marketing_conversions_reconcile_exactly(
    small_dataset: GenerationResult, small_config: GeneratorConfig
) -> None:
    con = duckdb.connect(str(small_config.db_path), read_only=True)
    try:
        rows = con.execute("""
            SELECT m.channel, m.conv, COALESCE(c.n, 0) FROM
              (SELECT channel, SUM(conversions) conv FROM marketing_campaigns GROUP BY 1) m
            LEFT JOIN (SELECT acquisition_channel channel, COUNT(*) n FROM customers
                       WHERE signup_date >= '2026-03-01' GROUP BY 1) c USING (channel)""").fetchall()
    finally:
        con.close()
    assert rows and all(conv == n for _, conv, n in rows)


# ---- the validator must actually catch problems (not pass vacuously) ----------------------------------


@pytest.fixture()
def corruptible_copy(small_dataset: GenerationResult, small_config: GeneratorConfig, tmp_path: Path) -> Path:
    copy = tmp_path / "copy.duckdb"
    shutil.copy(small_config.db_path, copy)
    return copy


@pytest.mark.parametrize(
    ("corruption", "expected_failure"),
    [
        ("UPDATE marketing_campaigns SET conversions = leads + 1 WHERE rowid = 0", "marketing:funnel_monotonic"),
        ("UPDATE daily_revenue SET revenue = -5 WHERE rowid = 0", "bounds:revenue>=0"),
        (
            "UPDATE support_tickets SET resolved_at = created_at - INTERVAL 1 HOUR "
            "WHERE ticket_id = (SELECT MIN(ticket_id) FROM support_tickets WHERE resolved_at IS NOT NULL)",
            "date_logic:resolved_at>=created_at",
        ),
        (
            "UPDATE subscriptions SET current_mrr = current_mrr + 10 WHERE status = 'active' "
            "AND subscription_id = (SELECT MIN(subscription_id) FROM subscriptions WHERE status = 'active')",
            "subscription:current_mrr_rule",
        ),
        ("CREATE TABLE business_events (event_id VARCHAR)", "exposure:only_documented_tables_and_views"),
        ("ALTER TABLE customers ADD COLUMN health_score DOUBLE", "exposure:no_hidden_or_ground_truth_columns"),
    ],
)
def test_validator_detects_corruption(
    corruptible_copy: Path, small_config: GeneratorConfig, corruption: str, expected_failure: str
) -> None:
    con = duckdb.connect(str(corruptible_copy))
    con.execute(corruption)
    con.close()
    failed = {c.name for c in validate_database(corruptible_copy, small_config) if not c.passed}
    assert expected_failure in failed
