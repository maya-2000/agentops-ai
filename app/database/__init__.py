"""Database layer: schema metadata, backends, loading and lineage."""

from app.database.base import Database, QueryResult
from app.database.factory import get_database
from app.database.lineage import DatasetInfo, LineageRecord

__all__ = ["Database", "DatasetInfo", "LineageRecord", "QueryResult", "get_database"]
