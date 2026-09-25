"""Shared fixtures. Every dataset is generated from scratch into a temporary directory:
the suite needs no pre-built database, no network access, no LLM and no API key."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from data.generator.config import GeneratorConfig
from data.generator.generate import GenerationResult, generate_dataset


def make_config(root: Path, **overrides: object) -> GeneratorConfig:
    params: dict[str, object] = {
        "db_path": root / "northwind.duckdb",
        "parquet_dir": None,
        "metadata_dir": root / "metadata",
        "ground_truth_path": root / "ground_truth" / "injected_events.json",
    }
    params.update(overrides)
    return GeneratorConfig(**params)  # type: ignore[arg-type]


@pytest.fixture()
def config_factory(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Build a GeneratorConfig writing into this test's temp directory."""
    return lambda **overrides: make_config(tmp_path, **overrides)


@pytest.fixture(scope="session")
def small_config(tmp_path_factory: pytest.TempPathFactory) -> GeneratorConfig:
    """100 customers over 6 months: fast structural tests.

    Too small for statistical event detection, so event validation is lenient here.
    """
    return make_config(
        tmp_path_factory.mktemp("small"),
        customer_count=100,
        start_date=date(2026, 3, 1),
        end_date=date(2026, 8, 31),
        strict_event_validation=False,
    )


@pytest.fixture(scope="session")
def small_dataset(small_config: GeneratorConfig) -> GenerationResult:
    return generate_dataset(small_config)


@pytest.fixture(scope="session")
def full_dataset(tmp_path_factory: pytest.TempPathFactory) -> GenerationResult:
    """The real configuration (5,000 customers, Sep 2024 - Aug 2026), built into a temp dir.

    Generation raises if any data-quality or injected-event check fails (strict mode).
    """
    root = tmp_path_factory.mktemp("full")
    return generate_dataset(make_config(root, parquet_dir=root / "parquet"))


# ---- Phase 2 analytics fixtures ------------------------------------------------------------------------


@pytest.fixture(scope="session")
def full_db(full_dataset: GenerationResult):  # type: ignore[no-untyped-def]
    """Read-only ``Database`` (the Phase 1 abstraction) on the full generated dataset."""
    from app.database.duckdb_backend import DuckDBDatabase

    db = DuckDBDatabase(full_dataset.config.db_path, dataset_version=full_dataset.manifest["dataset_version"])
    yield db
    db.close()


@pytest.fixture(scope="session")
def small_db(small_dataset: GenerationResult):  # type: ignore[no-untyped-def]
    from app.database.duckdb_backend import DuckDBDatabase

    db = DuckDBDatabase(small_dataset.config.db_path, dataset_version=small_dataset.manifest["dataset_version"])
    yield db
    db.close()


@pytest.fixture(scope="session")
def reference(full_dataset: GenerationResult):  # type: ignore[no-untyped-def]
    """Independent pandas reference over raw table extracts of the full dataset."""
    from tests.integration.reference_kpis import Reference, load_raw_tables

    return Reference(load_raw_tables(full_dataset.config.db_path))
