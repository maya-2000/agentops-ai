"""Create a ``Database`` from a DATABASE_URL."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings, get_settings
from app.database.base import Database
from app.database.duckdb_backend import DuckDBDatabase
from app.database.lineage import DatasetInfo

_DUCKDB_PREFIX = "duckdb:///"


def parse_duckdb_path(database_url: str, settings: Settings | None = None) -> Path:
    """Return the file path of a ``duckdb:///relative/or/absolute/path.duckdb`` URL."""
    if not database_url.startswith(_DUCKDB_PREFIX):
        raise ValueError(f"Not a DuckDB URL: {database_url!r}")
    raw = Path(database_url[len(_DUCKDB_PREFIX) :])
    settings = settings or get_settings()
    return settings.resolve_path(raw)


def get_database(database_url: str | None = None, *, read_only: bool = True) -> Database:
    """Open the configured business database.

    Supported: ``duckdb:///path``. PostgreSQL (``postgresql://``) is reserved for a later
    SQLAlchemy-backed implementation of the same ``Database`` protocol.
    """
    settings = get_settings()
    url = database_url or settings.database_url
    if url.startswith(_DUCKDB_PREFIX):
        manifest = settings.resolve_path(settings.dataset_manifest_path)
        version = DatasetInfo.from_manifest(manifest).dataset_version if manifest.exists() else "unknown"
        return DuckDBDatabase(parse_duckdb_path(url, settings), dataset_version=version, read_only=read_only)
    if url.startswith(("postgresql://", "postgres://")):
        raise NotImplementedError("The PostgreSQL backend is planned but not implemented yet; use a duckdb:/// URL.")
    raise ValueError(f"Unsupported DATABASE_URL scheme: {url!r}")
