"""DDL generated from ``metadata.py``: tables, constraints, indexes and views."""

from __future__ import annotations

from app.database.metadata import TABLES, VIEWS, TableSpec, ViewSpec


def _quote(identifier: str) -> str:
    return f'"{identifier}"'


def create_table_sql(table: TableSpec) -> str:
    """Build a ``CREATE TABLE`` statement with NOT NULL, PK, FK and CHECK constraints."""
    lines: list[str] = []
    for col in table.columns:
        line = f"    {_quote(col.name)} {col.sql_type}"
        if not col.nullable:
            line += " NOT NULL"
        if col.allowed_values:
            values = ", ".join("'" + v.replace("'", "''") + "'" for v in col.allowed_values)
            line += f" CHECK ({_quote(col.name)} IN ({values}))"
        lines.append(line)

    pk_cols = ", ".join(_quote(c) for c in table.primary_key)
    lines.append(f"    PRIMARY KEY ({pk_cols})")
    for col in table.columns:
        if col.foreign_key is not None:
            fk = col.foreign_key
            lines.append(f"    FOREIGN KEY ({_quote(col.name)}) REFERENCES {_quote(fk.table)}({_quote(fk.column)})")
    body = ",\n".join(lines)
    return f"CREATE TABLE {_quote(table.name)} (\n{body}\n)"


def create_index_sql(table: TableSpec) -> list[str]:
    statements = []
    for columns in table.indexes:
        index_name = f"idx_{table.name}_{'_'.join(columns)}"
        cols = ", ".join(_quote(c) for c in columns)
        statements.append(f"CREATE INDEX {_quote(index_name)} ON {_quote(table.name)} ({cols})")
    return statements


def create_view_sql(view: ViewSpec) -> str:
    return f"CREATE VIEW {_quote(view.name)} AS\n{view.sql.strip()}"


def table_ddl() -> list[str]:
    return [create_table_sql(t) for t in TABLES]


def index_ddl() -> list[str]:
    return [stmt for t in TABLES for stmt in create_index_sql(t)]


def view_ddl() -> list[str]:
    return [create_view_sql(v) for v in VIEWS]
