"""The evaluation context: which data the production system runs on, and what the evaluator knows about it.

Two worlds are kept apart here:

- **Observable business data**: a read-only production ``Database`` handle. It is the only thing
  handed to the production system.
- **Evaluation-only knowledge**: the independent reference tables (raw extracts for the pandas
  reference), the hidden labels, and the values of withheld fields (so that leaks can be
  detected). None of it is passed to production code.

A data source is either the repository database (seed 42, ``database/northwind_cloud.duckdb`` with
``data/seeds/injected_events.json``) or a dataset generated for another seed into a cache
directory with its own ground truth (multi-seed runs). Generation uses the existing Phase 1
generator unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.config import PROJECT_ROOT, get_settings
from app.database.duckdb_backend import DuckDBDatabase
from app.database.factory import parse_duckdb_path
from app.security.data_policy import DataExposurePolicy, default_exposure_policy
from evals.reference.labels import HiddenLabels

if TYPE_CHECKING:
    from tests.integration.reference_kpis import Reference

REPOSITORY_GROUND_TRUTH = PROJECT_ROOT / "data" / "seeds" / "injected_events.json"
DEFAULT_CACHE = PROJECT_ROOT / ".eval_cache"


@dataclass(frozen=True)
class DataSource:
    db_path: Path
    ground_truth_path: Path
    seed: int
    dataset_version: str
    as_of: date
    origin: str  # "repository" or "generated"
    coverage_start: date = date(2024, 9, 1)


def repository_source() -> DataSource:
    """The configured database and the repository ground truth (both from the default seed-42 generation)."""
    settings = get_settings()
    db_path = parse_duckdb_path(settings.database_url, settings)
    if not db_path.exists():
        raise FileNotFoundError(
            "The business database has not been built. Run: python -m data.generator.generate "
            "(or pass --seed to generate a dataset for the evaluation)."
        )
    manifest = settings.resolve_path(settings.dataset_manifest_path)
    version = json.loads(manifest.read_text(encoding="utf-8")).get("dataset_version", "unknown")
    labels = HiddenLabels.load(REPOSITORY_GROUND_TRUTH)
    return DataSource(
        db_path=db_path,
        ground_truth_path=REPOSITORY_GROUND_TRUTH,
        seed=labels.random_seed,
        dataset_version=str(version),
        as_of=settings.as_of_date,
        origin="repository",
    )


def generated_source(seed: int, cache_dir: Path = DEFAULT_CACHE, customer_count: int | None = None) -> DataSource:
    """Generate (once, cached) a dataset for ``seed`` with the unchanged Phase 1 generator."""
    from data.generator.config import GeneratorConfig
    from data.generator.generate import generate_dataset

    suffix = f"seed-{seed}" + (f"-{customer_count}" if customer_count else "")
    root = cache_dir / suffix
    params: dict[str, Any] = {
        "seed": seed,
        "db_path": root / "northwind.duckdb",
        "parquet_dir": None,
        "metadata_dir": root / "metadata",
        "ground_truth_path": root / "injected_events.json",
        "strict_event_validation": False,  # other seeds are evaluated as generated, not rejected
    }
    if customer_count:
        params["customer_count"] = customer_count
    config = GeneratorConfig(**params)
    manifest_path = config.metadata_dir / "dataset_manifest.json"
    if not (config.db_path.exists() and config.ground_truth_path.exists() and manifest_path.exists()):
        root.mkdir(parents=True, exist_ok=True)
        generate_dataset(config)
    version = json.loads(manifest_path.read_text(encoding="utf-8")).get("dataset_version", "unknown")
    return DataSource(
        db_path=config.db_path,
        ground_truth_path=config.ground_truth_path,
        seed=seed,
        dataset_version=str(version),
        as_of=config.end_date,
        origin="generated",
        coverage_start=config.start_date,
    )


@dataclass
class EvalContext:
    """Everything a run needs. Only ``db`` (and ``as_of``) ever reach the production system."""

    source: DataSource
    db: DuckDBDatabase
    reference: Reference
    labels: HiddenLabels
    exposure: DataExposurePolicy = field(default_factory=default_exposure_policy)
    withheld_values: dict[str, set[str]] = field(default_factory=dict)
    customer_ids: set[str] = field(default_factory=set)

    @property
    def as_of(self) -> date:
        return self.source.as_of

    @classmethod
    def open(cls, source: DataSource) -> EvalContext:
        from tests.integration.reference_kpis import Reference, load_raw_tables

        tables = load_raw_tables(source.db_path)
        reference = Reference(tables)
        exposure = default_exposure_policy()
        withheld: dict[str, set[str]] = {}
        for policy_columns in (exposure.withheld_columns, exposure.pii_columns):  # both maps share table keys
            for table, columns in policy_columns.items():
                frame = tables.get(table)
                for column in columns:
                    if frame is not None and column in frame.columns:
                        withheld.setdefault(column, set()).update(str(v) for v in frame[column].dropna().unique())
        db = DuckDBDatabase(source.db_path, dataset_version=source.dataset_version, read_only=True)
        return cls(
            source=source,
            db=db,
            reference=reference,
            labels=HiddenLabels.load(source.ground_truth_path),
            exposure=exposure,
            withheld_values=withheld,
            customer_ids={str(c) for c in tables["customers"]["customer_id"]},
        )

    def close(self) -> None:
        self.db.close()
