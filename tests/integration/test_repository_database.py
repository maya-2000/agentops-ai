"""Checks against the repository's built database, when it exists.

These tests are skipped until `python -m data.generator.generate` has been run; the rest of
the suite never depends on a pre-built database.
"""

from __future__ import annotations

import json

import pytest

from app.config import PROJECT_ROOT
from app.database.factory import get_database
from data.generator.manifest import table_checksums

DB_PATH = PROJECT_ROOT / "database" / "northwind_cloud.duckdb"
METADATA = PROJECT_ROOT / "data" / "metadata"

pytestmark = [
    pytest.mark.full_data,
    pytest.mark.skipif(not DB_PATH.exists(), reason="repository database not built"),
]


def test_database_matches_committed_checksums() -> None:
    committed = json.loads((METADATA / "dataset_checksums.json").read_text())
    assert table_checksums(DB_PATH) == committed["tables"]


def test_manifest_matches_database() -> None:
    manifest = json.loads((METADATA / "dataset_manifest.json").read_text())
    db = get_database()
    try:
        for table, entry in manifest["tables"].items():
            assert db.query(f'SELECT COUNT(*) FROM "{table}"').rows[0][0] == entry["row_count"]
        assert db.dataset_version == manifest["dataset_version"]
    finally:
        db.close()
