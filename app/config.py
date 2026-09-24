"""Application settings loaded from environment variables (and an optional ``.env`` file)."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration. Secrets (added in later phases) must use ``SecretStr``."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "duckdb:///database/northwind_cloud.duckdb"
    dataset_manifest_path: Path = Path("data/metadata/dataset_manifest.json")
    as_of_date: date = date(2026, 8, 31)
    data_seed: int = 42
    log_level: str = "INFO"

    def resolve_path(self, path: Path) -> Path:
        """Resolve a project-relative path against the repository root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
