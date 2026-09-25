"""Minimum SQL safety for the ``run_safe_sql`` tool (Phase 4). Formal guardrails are Phase 5.

A statement is accepted only if **all** of these hold (checked on the sqlglot AST, DuckDB dialect):

1. It parses, and it is exactly one statement.
2. The statement is a query: ``SELECT`` (optionally with CTEs) or a set operation of queries. No
   ``INSERT/UPDATE/DELETE/MERGE``, DDL (``CREATE/DROP/ALTER``), ``ATTACH/COPY/EXPORT/PRAGMA/SET/
   INSTALL/LOAD/CALL/DESCRIBE/SHOW`` and no ``SELECT ... INTO``.
3. Every table is an allow-listed business table or view (or a CTE of the same statement),
   referenced by a plain unqualified name. File paths, table functions (``read_csv``, ``glob``,
   ``duckdb_tables()``, ...) and other schemas or catalogs are rejected.
4. No function that reads files, the environment or system state (a deny-list of names and name
   prefixes).
5. Every column reference is a known column of an allow-listed table or an alias defined in the
   statement. PII-tagged columns cannot be selected, and ``SELECT *`` is refused on tables that
   contain them.
6. ``LIMIT`` (if present) is a literal integer. The tool always caps it at its row limit + 1, so
   truncation can be detected and reported.
7. Values are bound as named ``$parameters`` (scalars only). They are never formatted into the SQL.

Execution then goes through the Phase 2 ``QueryRunner`` on the read-only ``Database`` connection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

import sqlglot
from sqlglot import exp

from app.analytics.errors import InvalidRequestError
from app.database import metadata


class UnsafeSQLError(InvalidRequestError):
    """The SQL statement is not allowed."""

    code = "unsafe_sql"


_FORBIDDEN_FUNCTIONS = {
    "getenv",
    "current_setting",
    "glob",
    "load",
    "install",
    "system",
    "shell",
    "read_text",
    "read_blob",
    "query",
    "query_table",
    "json_execute_serialized_sql",
}
_FORBIDDEN_PREFIXES = (
    "read_",
    "write_",
    "copy",
    "duckdb_",
    "pragma_",
    "sqlite_",
    "postgres_",
    "mysql_",
    "iceberg_",
    "delta_",
    "parquet_",
    "csv_",
    "json_",
    "httpfs",
    "http_",
    "s3_",
)
_SET_OPERATIONS = (exp.Union, exp.Intersect, exp.Except)
_PARAMETER_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,40}$")
ParameterValue = str | int | float | bool | date | None


@dataclass(frozen=True)
class SQLPolicy:
    allowed_columns: dict[str, frozenset[str]]
    pii_columns: dict[str, frozenset[str]]

    @property
    def allowed_tables(self) -> frozenset[str]:
        return frozenset(self.allowed_columns)


def default_policy() -> SQLPolicy:
    """Business tables and views from the Phase 1 metadata; PII-tagged columns are withheld."""
    columns: dict[str, frozenset[str]] = {}
    pii: dict[str, frozenset[str]] = {}
    for table in metadata.TABLES:
        columns[table.name] = frozenset(c.name for c in table.columns if not c.pii)
        pii[table.name] = frozenset(c.name for c in table.columns if c.pii)
    for view in metadata.VIEWS:
        query = sqlglot.parse_one(view.sql, read="duckdb")
        assert isinstance(query, exp.Query), view.name
        columns[view.name] = frozenset(query.named_selects)
        pii[view.name] = frozenset()
    return SQLPolicy(allowed_columns=columns, pii_columns=pii)


@dataclass(frozen=True)
class ValidatedSQL:
    sql: str  # the statement with its LIMIT capped at row_limit + 1
    source_tables: list[str]
    parameters: dict[str, ParameterValue]


def validate_sql(
    sql: str, parameters: dict[str, Any] | None, row_limit: int, policy: SQLPolicy | None = None
) -> ValidatedSQL:
    """Validate a read-only query and return the executable statement (raises ``UnsafeSQLError``)."""
    policy = policy or default_policy()
    if not sql or not sql.strip():
        raise UnsafeSQLError("Empty SQL statement")
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError) as exc:
        raise UnsafeSQLError(f"SQL could not be parsed: {str(exc).splitlines()[0]}") from None
    if len(statements) != 1:
        raise UnsafeSQLError("Exactly one SQL statement is allowed")
    root = statements[0]
    if not isinstance(root, (exp.Select, *_SET_OPERATIONS)):
        raise UnsafeSQLError(f"Only SELECT queries are allowed (got {root.key.upper()})")

    for node in root.walk():
        if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter)):
            raise UnsafeSQLError(f"{node.key.upper()} is not allowed")
        if isinstance(node, (exp.Command, exp.Copy, exp.Pragma, exp.Set, exp.Attach)):
            raise UnsafeSQLError(f"{node.key.upper()} is not allowed")
        if isinstance(node, exp.Select) and node.args.get("into") is not None:
            raise UnsafeSQLError("SELECT ... INTO is not allowed")
        if isinstance(node, exp.Func):
            _check_function(node)

    cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
    tables = _check_tables(root, cte_names, policy)
    _check_columns(root, tables, cte_names, policy)
    bound = _check_parameters(root, parameters or {})
    limited = _cap_limit(root, row_limit)
    return ValidatedSQL(sql=limited.sql(dialect="duckdb"), source_tables=sorted(tables), parameters=bound)


def _check_function(node: exp.Func) -> None:
    name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
    if name in _FORBIDDEN_FUNCTIONS or name.startswith(_FORBIDDEN_PREFIXES):
        raise UnsafeSQLError(f"Function {name!r} is not allowed")


def _check_tables(root: exp.Expression, cte_names: set[str], policy: SQLPolicy) -> set[str]:
    used: set[str] = set()
    for table in root.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise UnsafeSQLError("Table functions and file paths are not allowed in FROM")
        if table.args.get("db") is not None or table.args.get("catalog") is not None:
            raise UnsafeSQLError(f"Qualified table names are not allowed: {table.sql(dialect='duckdb')}")
        name = table.name.lower()
        if name in cte_names:
            continue
        if name not in policy.allowed_tables:
            raise UnsafeSQLError(
                f"Table {table.name!r} is not allowed. Allowed tables: {', '.join(sorted(policy.allowed_tables))}"
            )
        used.add(name)
    if not used:
        raise UnsafeSQLError("The query must read at least one allowed business table")
    return used


def _check_columns(root: exp.Expression, tables: set[str], cte_names: set[str], policy: SQLPolicy) -> None:
    aliases = {a.alias.lower() for a in root.find_all(exp.Alias) if a.alias}
    aliases |= {a.alias.lower() for a in root.find_all(exp.TableAlias) if a.alias}
    for alias in root.find_all(exp.TableAlias):
        aliases |= {c.name.lower() for c in alias.columns}
    known = set().union(*(policy.allowed_columns[t] for t in policy.allowed_tables))
    pii = set().union(*(policy.pii_columns[t] for t in tables))
    for column in root.find_all(exp.Column):
        name = column.name.lower()
        if not name:
            continue
        if name in pii:
            raise UnsafeSQLError(f"Column {column.name!r} is PII-tagged and cannot be queried")
        if name not in known and name not in aliases and name not in cte_names:
            raise UnsafeSQLError(f"Unknown column {column.name!r}")
    for star in root.find_all(exp.Star):
        if isinstance(star.parent, exp.Count):
            continue
        if pii:
            raise UnsafeSQLError("SELECT * is not allowed on tables with PII-tagged columns; list the columns")


def _check_parameters(root: exp.Expression, parameters: dict[str, Any]) -> dict[str, ParameterValue]:
    names = [p.name for p in root.find_all(exp.Placeholder)]
    if any(not name or name == "?" or name.isdigit() for name in names):
        raise UnsafeSQLError("Use named $parameters, not positional placeholders")
    used = set(names)
    missing = used - parameters.keys()
    if missing:
        raise UnsafeSQLError(f"Missing values for parameters: {', '.join(sorted(missing))}")
    bound: dict[str, ParameterValue] = {}
    for name in sorted(used):
        value = parameters[name]
        if not _PARAMETER_NAME.match(name):
            raise UnsafeSQLError(f"Invalid parameter name {name!r}")
        if not isinstance(value, (str, int, float, bool, date)) and value is not None:
            raise UnsafeSQLError(f"Parameter {name!r} must be a scalar value")
        bound[name] = value
    return bound


def _cap_limit(root: exp.Expression, row_limit: int) -> exp.Expression:
    cap = row_limit + 1
    existing = root.args.get("limit")
    if existing is not None:
        value = existing.expression
        if not (isinstance(value, exp.Literal) and value.is_int):
            raise UnsafeSQLError("LIMIT must be a literal integer")
        cap = min(cap, int(value.this))
    if isinstance(root, (exp.Select, *_SET_OPERATIONS)):
        return root.limit(cap)
    raise UnsafeSQLError("Unsupported statement shape")  # pragma: no cover - rejected earlier
