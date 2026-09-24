"""Load generated DataFrames into a fresh DuckDB database file."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from app.database.metadata import TABLES
from app.database.schema import create_table_sql, index_ddl, view_ddl

logger = logging.getLogger(__name__)


def build_database(db_path: Path, frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Create schema, insert every table (parents first), then indexes and views.

    Any existing file at ``db_path`` is replaced so a build is always from scratch.
    Returns the loaded row count per table.
    """
    expected = {t.name for t in TABLES}
    missing, extra = expected - frames.keys(), frames.keys() - expected
    if missing or extra:
        raise ValueError(f"Frame set does not match schema. Missing={missing}, unexpected={extra}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", ".wal"):
        stale = Path(str(db_path) + suffix)
        if stale.exists():
            stale.unlink()

    row_counts: dict[str, int] = {}
    con = duckdb.connect(str(db_path))
    try:
        for table in TABLES:
            con.execute(create_table_sql(table))
            frame = frames[table.name][list(table.column_names)]
            con.register("_staging", frame)
            columns = ", ".join(f'"{c}"' for c in table.column_names)
            con.execute(f'INSERT INTO "{table.name}" ({columns}) SELECT {columns} FROM _staging')
            con.unregister("_staging")
            (row_counts[table.name],) = con.execute(f'SELECT COUNT(*) FROM "{table.name}"').fetchone() or (0,)
            logger.info("Loaded %-20s %10d rows", table.name, row_counts[table.name])
        for statement in index_ddl():
            con.execute(statement)
        for statement in view_ddl():
            con.execute(statement)
        con.execute("CHECKPOINT")
    finally:
        con.close()
    return row_counts


def export_parquet(db_path: Path, output_dir: Path) -> list[Path]:
    """Export every business table to Parquet (useful for loading other databases later)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        for table in TABLES:
            target = output_dir / f"{table.name}.parquet"
            con.execute(f"COPY \"{table.name}\" TO '{target.as_posix()}' (FORMAT PARQUET)")
            written.append(target)
    finally:
        con.close()
    return written
