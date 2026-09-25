"""SQL security for the ``run_safe_sql`` tool (Phase 4 minimum, hardened in Phase 5).

A statement is accepted only if **all** of these hold. They are checked on the sqlglot AST
(DuckDB dialect), and the first failure rejects the statement (fail closed):

1. **Size.** The SQL text is at most ``max_length`` characters.
2. **Shape.** It parses, and it is exactly one statement.
3. **Read-only.** The statement is a ``SELECT`` (optionally with non-recursive CTEs) or a set
   operation of selects. Refused anywhere in the tree:
   - ``INSERT/UPDATE/DELETE/MERGE``, ``CREATE/DROP/ALTER/TRUNCATE``;
   - ``ATTACH/DETACH/COPY/EXPORT/IMPORT/INSTALL/LOAD/CALL/SET/PRAGMA/DESCRIBE/SHOW/SUMMARIZE``;
   - ``SELECT ... INTO`` and ``WITH RECURSIVE``.
4. **Tables.** Every table is an approved business table or view from the data-exposure policy
   (or a CTE of the same statement), referenced by a plain unqualified name. Refused: file paths
   and URLs, table functions (``read_csv``, ``glob``, ``range``, ``duckdb_tables()``, ...),
   other schemas and catalogs, and system tables.
5. **Complexity.** At most ``max_joins`` joins, and every join needs an explicit ``ON`` or
   ``USING`` condition (no cartesian products). At most ``max_nesting_depth`` levels of nested
   subqueries, at most ``max_ctes`` CTEs, and at most ``max_parameters`` bound parameters.
6. **Functions.**
   - Functions known to the SQL parser are allowed, except a deny-list that reads files, the
     environment, the network, settings or catalog state.
   - Any other (unrecognised) function must be on an explicit allowlist of DuckDB analytic
     functions.
7. **Columns.** Every column is a known, exposed column of an approved relation or an alias
   defined in the statement.
   - PII-tagged columns (``sales_rep``) and columns withheld by the data-exposure policy
     (``company_name``, and any hidden-state name) are refused.
   - ``SELECT *`` is refused on relations that contain them.
8. **Rows.** ``LIMIT`` (if present) is a literal integer. It is always capped at the row limit
   + 1, so truncation is detected and reported.
9. **Values.** Values are bound as named ``$parameters`` (short scalars only). They are never
   formatted into the SQL.
10. **Regeneration.** The executed statement is regenerated from the validated syntax tree with
    all comments removed, so nothing unvalidated reaches the database.

Execution then goes through the Phase 2 ``QueryRunner`` on the read-only ``Database``
connection, with a statement timeout (``execution_deadline``). The read-only connection is a
second, independent layer: it would refuse a write even if validation had a bug.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

import sqlglot
from sqlglot import exp

from app.analytics.errors import InvalidRequestError
from app.security.data_policy import DataExposurePolicy, default_exposure_policy


class UnsafeSQLError(InvalidRequestError):
    """The SQL statement is not allowed."""

    code = "unsafe_sql"


@dataclass(frozen=True)
class SQLComplexityLimits:
    max_length: int = 4000
    max_joins: int = 4
    max_nesting_depth: int = 3
    max_ctes: int = 4
    max_parameters: int = 20
    max_parameter_chars: int = 200


_FORBIDDEN_FUNCTIONS = {
    "getenv",
    "current_setting",
    "current_database",
    "current_schema",
    "current_schemas",
    "current_catalog",
    "current_user",
    "current_version",
    "session_user",
    "user",
    "version",
    "getvariable",
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
    "generate_series",
    "range",
    "checkpoint",
    "force_checkpoint",
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
    "sniff_",
    "httpfs",
    "http_",
    "s3_",
    "enable_",
    "disable_",
)
# Functions the parser does not recognise are refused unless listed here (fail closed).
_ALLOWED_UNRECOGNISED_FUNCTIONS = frozenset(
    {
        "date_part",
        "datepart",
        "make_date",
        "last_day",
        "dayofweek",
        "dayofyear",
        "weekofyear",
        "yearweek",
        "epoch",
        "age",
        "quantile",
        "quantile_cont",
        "quantile_disc",
        "mode",
        "arg_max",
        "arg_min",
        "max_by",
        "min_by",
        "bool_and",
        "bool_or",
        "fsum",
        "favg",
        "sumkahan",
        "round_even",
        "sign",
        "isnan",
        "isfinite",
        "percent_rank",
        "cume_dist",
        "ntile",
        "nth_value",
        "starts_with",
        "ends_with",
        "contains",
        "format",
        "printf",
    }
)
_FORBIDDEN_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Command,
    exp.Copy,
    exp.Pragma,
    exp.Set,
    exp.Attach,
    exp.Detach,
    exp.Summarize,
    exp.Describe,
)
_SET_OPERATIONS = (exp.Union, exp.Intersect, exp.Except)
_PARAMETER_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,40}$")
ParameterValue = str | int | float | bool | date | None


@dataclass(frozen=True)
class SQLPolicy:
    allowed_columns: dict[str, frozenset[str]]  # exposed columns per approved relation
    pii_columns: dict[str, frozenset[str]]
    withheld_columns: dict[str, frozenset[str]]

    @property
    def allowed_tables(self) -> frozenset[str]:
        return frozenset(self.allowed_columns)

    @classmethod
    def from_exposure_policy(cls, policy: DataExposurePolicy) -> SQLPolicy:
        return cls(
            allowed_columns=dict(policy.exposed_columns),
            pii_columns=dict(policy.pii_columns),
            withheld_columns=dict(policy.withheld_columns),
        )


def default_policy() -> SQLPolicy:
    """Approved tables, views and columns from the data-exposure policy."""
    return SQLPolicy.from_exposure_policy(default_exposure_policy())


@dataclass(frozen=True)
class ValidatedSQL:
    sql: str  # the statement with its LIMIT capped at row_limit + 1
    source_tables: list[str]
    parameters: dict[str, ParameterValue]


def validate_sql(
    sql: str,
    parameters: dict[str, Any] | None,
    row_limit: int,
    policy: SQLPolicy | None = None,
    limits: SQLComplexityLimits | None = None,
) -> ValidatedSQL:
    """Validate a read-only query and return the executable statement (raises ``UnsafeSQLError``)."""
    policy = policy or default_policy()
    limits = limits or SQLComplexityLimits()
    if not isinstance(sql, str) or not sql.strip():
        raise UnsafeSQLError("Empty SQL statement")
    if len(sql) > limits.max_length:
        raise UnsafeSQLError(f"SQL statement is longer than {limits.max_length} characters")
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError) as exc:
        raise UnsafeSQLError(f"SQL could not be parsed: {str(exc).splitlines()[0][:120]}") from None
    if len(statements) != 1:
        raise UnsafeSQLError("Exactly one SQL statement is allowed")
    root = statements[0]
    if not isinstance(root, (exp.Select, *_SET_OPERATIONS)):
        raise UnsafeSQLError(f"Only SELECT queries are allowed (got {root.key.upper()})")

    for node in root.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise UnsafeSQLError(f"{node.key.upper()} is not allowed")
        if isinstance(node, exp.Select) and node.args.get("into") is not None:
            raise UnsafeSQLError("SELECT ... INTO is not allowed")
        if isinstance(node, exp.With) and node.args.get("recursive"):
            raise UnsafeSQLError("Recursive CTEs are not allowed")

    cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
    tables = _check_tables(root, cte_names, policy)
    _check_complexity(root, cte_names, limits)
    for node in root.find_all(exp.Func):
        _check_function(node)
    _check_columns(root, tables, cte_names, policy)
    bound = _check_parameters(root, parameters or {}, limits)
    limited = _cap_limit(root, row_limit)
    # Comments are dropped: the executed statement is exactly the validated syntax tree.
    executable = limited.sql(dialect="duckdb", comments=False)
    return ValidatedSQL(sql=executable, source_tables=sorted(tables), parameters=bound)


def _check_function(node: exp.Func) -> None:
    unrecognised = isinstance(node, exp.Anonymous)
    name = (node.name if unrecognised else node.sql_name()).lower()
    if name in _FORBIDDEN_FUNCTIONS or name.startswith(_FORBIDDEN_PREFIXES):
        raise UnsafeSQLError(f"Function {name!r} is not allowed")
    if unrecognised and name not in _ALLOWED_UNRECOGNISED_FUNCTIONS:
        raise UnsafeSQLError(f"Function {name!r} is not allowed (not on the function allowlist)")


def _check_tables(root: exp.Expression, cte_names: set[str], policy: SQLPolicy) -> set[str]:
    used: set[str] = set()
    for table in root.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise UnsafeSQLError("Table functions and file paths are not allowed in FROM")
        if table.args.get("db") is not None or table.args.get("catalog") is not None:
            raise UnsafeSQLError(f"Qualified table names are not allowed: {table.sql(dialect='duckdb')[:80]}")
        name = table.name.lower()
        if name in cte_names:
            continue
        if name not in policy.allowed_tables:
            raise UnsafeSQLError(
                f"Table {table.name[:60]!r} is not allowed. Allowed tables: {', '.join(sorted(policy.allowed_tables))}"
            )
        used.add(name)
    if not used:
        raise UnsafeSQLError("The query must read at least one allowed business table")
    return used


def _nesting_depth(node: exp.Expression) -> int:
    depth = 0
    parent = node.parent
    while parent is not None:
        if isinstance(parent, exp.Select):
            depth += 1
        parent = parent.parent
    return depth


def _check_complexity(root: exp.Expression, cte_names: set[str], limits: SQLComplexityLimits) -> None:
    joins = list(root.find_all(exp.Join))
    if len(joins) > limits.max_joins:
        raise UnsafeSQLError(f"At most {limits.max_joins} joins are allowed")
    for join in joins:
        if not join.args.get("on") and not join.args.get("using"):
            raise UnsafeSQLError("Every join needs an explicit ON or USING condition (no cartesian products)")
    if len(cte_names) > limits.max_ctes:
        raise UnsafeSQLError(f"At most {limits.max_ctes} CTEs are allowed")
    deepest = max((_nesting_depth(s) for s in root.find_all(exp.Select)), default=0)
    if deepest > limits.max_nesting_depth:
        raise UnsafeSQLError(f"Subqueries may be nested at most {limits.max_nesting_depth} levels deep")


def _check_columns(root: exp.Expression, tables: set[str], cte_names: set[str], policy: SQLPolicy) -> None:
    aliases = {a.alias.lower() for a in root.find_all(exp.Alias) if a.alias}
    aliases |= {a.alias.lower() for a in root.find_all(exp.TableAlias) if a.alias}
    for alias in root.find_all(exp.TableAlias):
        aliases |= {c.name.lower() for c in alias.columns}
    known = set().union(*(policy.allowed_columns[t] for t in policy.allowed_tables))
    pii = set().union(*(policy.pii_columns.get(t, frozenset()) for t in tables))
    withheld = set().union(*(policy.withheld_columns.get(t, frozenset()) for t in tables))
    for column in root.find_all(exp.Column):
        name = column.name.lower()
        if not name:
            continue
        if name in pii:
            raise UnsafeSQLError(f"Column {column.name!r} is PII-tagged and cannot be queried")
        if name in withheld:
            raise UnsafeSQLError(f"Column {column.name!r} is withheld by the data-exposure policy")
        if name not in known and name not in aliases and name not in cte_names:
            raise UnsafeSQLError(f"Unknown column {column.name[:60]!r}")
    for star in root.find_all(exp.Star):
        if isinstance(star.parent, exp.Count):
            continue
        if pii or withheld:
            raise UnsafeSQLError(
                "SELECT * is not allowed on tables with PII-tagged or withheld columns; list the columns"
            )


def _check_parameters(
    root: exp.Expression, parameters: dict[str, Any], limits: SQLComplexityLimits
) -> dict[str, ParameterValue]:
    names = [p.name for p in root.find_all(exp.Placeholder)]
    if any(not name or name == "?" or name.isdigit() for name in names):
        raise UnsafeSQLError("Use named $parameters, not positional placeholders")
    used = set(names)
    if len(used) > limits.max_parameters:
        raise UnsafeSQLError(f"At most {limits.max_parameters} parameters are allowed")
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
        if isinstance(value, str) and len(value) > limits.max_parameter_chars:
            raise UnsafeSQLError(f"Parameter {name!r} is longer than {limits.max_parameter_chars} characters")
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
