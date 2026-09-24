"""Generate the Northwind Cloud dataset end to end.

Usage (from the repository root)::

    python -m data.generator.generate                  # full dataset, default paths
    python -m data.generator.generate --customers 100 --start 2026-03-01 --end 2026-08-31 \
        --db-path /tmp/small.duckdb --metadata-dir /tmp/meta --ground-truth /tmp/gt.json --no-parquet

Pipeline: entities -> acquisition -> lifecycle simulation (with injected events) ->
relational tables -> load into DuckDB -> data-quality validation -> event validation ->
manifest + checksums + machine-readable data dictionary (+ optional Parquet export).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import get_settings
from app.database.data_dictionary import data_dictionary_json
from app.database.factory import parse_duckdb_path
from app.database.loader import build_database, export_parquet
from data.generator.acquisition import non_marketing_signups, opening_signups, simulate_marketing
from data.generator.config import DATASET_VERSION, GeneratorConfig
from data.generator.entities import build_customers, create_sales_reps
from data.generator.event_validation import EventCheck, validate_events
from data.generator.events import GroundTruthLog
from data.generator.lifecycle import SimulationResult, simulate_lifecycle
from data.generator.manifest import (
    CHECKSUMS_FILENAME,
    MANIFEST_FILENAME,
    build_manifest,
    dataset_fingerprint,
    table_checksums,
    write_json,
)
from data.generator.relationships import (
    build_customers_table,
    build_daily_revenue,
    build_product_features,
    build_subscriptions,
    build_support_tickets,
    build_usage_events,
)
from data.generator.sales import build_opportunities
from data.generator.validation import CheckResult, validate_database

logger = logging.getLogger("data.generator")

DATA_DICTIONARY_FILENAME = "data_dictionary.json"


class GenerationError(RuntimeError):
    """Raised when generated data fails validation."""


@dataclass
class GenerationResult:
    config: GeneratorConfig
    row_counts: dict[str, int]
    manifest: dict[str, Any]
    checksums: dict[str, dict[str, Any]]
    fingerprint: str
    quality_checks: list[CheckResult]
    event_checks: list[EventCheck]
    ground_truth: dict[str, Any]
    timings: dict[str, float] = field(default_factory=dict)


def generate_frames(
    config: GeneratorConfig,
) -> tuple[dict[str, pd.DataFrame], SimulationResult, GroundTruthLog]:
    """Create all business tables in memory (nothing is written)."""
    rng = np.random.default_rng(config.seed)
    truth = GroundTruthLog()

    reps = create_sales_reps(rng, truth)
    campaigns, marketing_signups = simulate_marketing(config, rng, truth)
    other = non_marketing_signups(config, rng, config.new_customer_count - len(marketing_signups))
    new_signups = pd.concat([marketing_signups, other], ignore_index=True)
    customers = build_customers(rng, opening_signups(config, rng, config.opening_customer_count), new_signups)

    sim = simulate_lifecycle(config, customers, rng, truth)
    subscriptions = build_subscriptions(customers, sim)
    usage = build_usage_events(config, customers, subscriptions, sim, rng)
    revenue = build_daily_revenue(config, customers, subscriptions, usage)
    tickets = build_support_tickets(customers, sim, config)
    features = build_product_features(config, usage, sim, rng)
    opportunities = build_opportunities(config, customers, subscriptions, reps, sim.churn_date, rng)

    frames = {
        "customers": build_customers_table(customers, sim),
        "subscriptions": subscriptions,
        "usage_events": usage,
        "sales_opportunities": opportunities,
        "support_tickets": tickets,
        "marketing_campaigns": campaigns,
        "daily_revenue": revenue,
        "product_features": features,
    }
    return _normalise_types(frames), sim, truth


_DATE_COLUMNS = {
    "customers": ["signup_date"],
    "subscriptions": ["start_date", "end_date"],
    "usage_events": ["event_date"],
    "sales_opportunities": ["created_date", "close_date"],
    "marketing_campaigns": ["date"],
    "daily_revenue": ["date"],
    "product_features": ["date"],
}


def _normalise_types(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Coerce date columns to datetime64 so DuckDB receives a consistent type."""
    for table, cols in _DATE_COLUMNS.items():
        for col in cols:
            frames[table][col] = pd.to_datetime(frames[table][col]).astype("datetime64[s]")
    return frames


def generate_dataset(config: GeneratorConfig) -> GenerationResult:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    frames, _sim, truth = generate_frames(config)
    timings["generate_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    row_counts = build_database(config.db_path, frames)
    timings["load_s"] = time.perf_counter() - t1

    truth.write(config.ground_truth_path, seed=config.seed, dataset_version=DATASET_VERSION)
    ground_truth = truth.to_dict(seed=config.seed, dataset_version=DATASET_VERSION)

    t2 = time.perf_counter()
    quality = validate_database(config.db_path, config)
    events = validate_events(config.db_path, config.ground_truth_path, config)
    timings["validate_s"] = time.perf_counter() - t2

    failed_quality = [c for c in quality if not c.passed and c.severity == "error"]
    if failed_quality:
        details = "\n".join(f"  - {c.name}: {c.details}" for c in failed_quality)
        raise GenerationError(f"Data-quality validation failed:\n{details}")
    failed_events = [c for c in events if c.status == "failed"]
    if failed_events and config.strict_event_validation:
        details = "\n".join(f"  - {c.event_id} {c.check}: {c.details}" for c in failed_events)
        raise GenerationError(f"Injected-event validation failed:\n{details}")

    manifest = build_manifest(config, config.db_path)
    checksums = table_checksums(config.db_path)
    write_json(config.metadata_dir / MANIFEST_FILENAME, manifest)
    write_json(
        config.metadata_dir / CHECKSUMS_FILENAME,
        {
            "dataset_version": DATASET_VERSION,
            "random_seed": config.seed,
            "fingerprint_sha256": dataset_fingerprint(checksums),
            "tables": checksums,
        },
    )
    write_json(config.metadata_dir / DATA_DICTIONARY_FILENAME, data_dictionary_json())
    if config.parquet_dir is not None:
        export_parquet(config.db_path, config.parquet_dir)
    timings["total_s"] = time.perf_counter() - t0

    return GenerationResult(
        config=config,
        row_counts=row_counts,
        manifest=manifest,
        checksums=checksums,
        fingerprint=dataset_fingerprint(checksums),
        quality_checks=quality,
        event_checks=events,
        ground_truth=ground_truth,
        timings=timings,
    )


def _parse_args(argv: list[str] | None) -> GeneratorConfig:
    """CLI flags override environment settings (DATA_SEED, DATABASE_URL), which override defaults."""
    settings = get_settings()
    defaults = GeneratorConfig(seed=settings.data_seed, db_path=parse_duckdb_path(settings.database_url, settings))
    parser = argparse.ArgumentParser(description="Generate the Northwind Cloud synthetic dataset.")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--customers", type=int, default=defaults.customer_count)
    parser.add_argument("--start", type=date.fromisoformat, default=defaults.start_date, help="YYYY-MM-01")
    parser.add_argument("--end", type=date.fromisoformat, default=defaults.end_date, help="last day of a month")
    parser.add_argument("--db-path", type=Path, default=defaults.db_path)
    parser.add_argument("--parquet-dir", type=Path, default=defaults.parquet_dir)
    parser.add_argument("--no-parquet", action="store_true", help="skip the Parquet export")
    parser.add_argument("--metadata-dir", type=Path, default=defaults.metadata_dir)
    parser.add_argument("--ground-truth", type=Path, default=defaults.ground_truth_path)
    parser.add_argument(
        "--lenient-events",
        action="store_true",
        help="report (instead of fail on) injected-event checks, e.g. for tiny datasets",
    )
    args = parser.parse_args(argv)
    return GeneratorConfig(
        seed=args.seed,
        customer_count=args.customers,
        start_date=args.start,
        end_date=args.end,
        db_path=args.db_path,
        parquet_dir=None if args.no_parquet else args.parquet_dir,
        metadata_dir=args.metadata_dir,
        ground_truth_path=args.ground_truth,
        strict_event_validation=not args.lenient_events,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = _parse_args(argv)
    logger.info(
        "Generating %s customers, %s -> %s (seed %s)",
        config.customer_count,
        config.start_date,
        config.end_date,
        config.seed,
    )
    try:
        result = generate_dataset(config)
    except GenerationError as exc:
        logger.error("%s", exc)
        return 1

    passed = sum(c.passed for c in result.quality_checks)
    logger.info("Data-quality checks: %d/%d passed", passed, len(result.quality_checks))
    for check in result.quality_checks:
        if not check.passed:
            logger.warning("  [%s] %s: %s", check.severity, check.name, check.details)
    for event_check in result.event_checks:
        logger.info("  Event %s %-45s %s", event_check.event_id, event_check.check, event_check.status.upper())
    logger.info("Row counts: %s", result.row_counts)
    logger.info(
        "Database: %s | manifest: %s | fingerprint %s",
        config.db_path,
        config.metadata_dir / MANIFEST_FILENAME,
        result.fingerprint[:16],
    )
    logger.info("Timings: %s", {k: round(v, 1) for k, v in result.timings.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
