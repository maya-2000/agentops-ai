"""Render the data dictionary (Markdown and JSON) from ``metadata.py``.

Regenerate the committed docs with::

    python -m app.database.data_dictionary
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT
from app.database.metadata import RELATIONSHIPS, SCHEMA_VERSION, TABLES, VIEWS, ColumnSpec, TableSpec

DICTIONARY_PATH = PROJECT_ROOT / "docs" / "data-dictionary.md"

_CONVENTIONS = """\
## Conventions

| Topic | Convention |
|---|---|
| Currency | All monetary values are in **SGD**. |
| Reporting window | 2024-09-01 to 2026-08-31 (24 full months). "Today" is **2026-08-31**. Customers acquired before the window keep their original `signup_date`. |
| Weeks | `usage_events` and `marketing_campaigns` use weeks starting on **Monday**; only weeks fully inside the window are included (2024-09-02 to 2026-08-30). |
| MRR | Contracted monthly recurring revenue per subscription record. Month-end MRR = records in force on the last day of the month (`v_monthly_mrr`). |
| Subscription history | `subscriptions` is versioned: every expansion or contraction closes the current record (`superseded`) and opens a new one the next day. |
| Churn date | `subscriptions.end_date` of the `churned` record = the last day of service. |
| Revenue recognition | Subscription revenue per day = MRR / days in that calendar month, so a full month recognises exactly the MRR. Usage revenue = API calls above the plan quota x SGD 6 per 1,000, spread evenly over the month's API-enabled days. |
| Revenue vs MRR | Revenue (`daily_revenue`) = recognised subscription + usage revenue for a period. MRR is a point-in-time recurring run-rate and excludes usage revenue. |
| Open items | Open opportunities have `close_date` NULL; open tickets have `resolved_at` and `resolution_time` NULL. |
| PII | Company and sales-rep names are synthetic. `sales_rep` is tagged as PII so later phases can mask it where needed. No contact names or e-mail addresses exist. |
"""

_EXCLUSIONS = """\
## What is deliberately *not* in the database

The synthetic data is produced by a simulation with a hidden customer-health mechanism and
seven injected business events (see `data/README.md`). To keep the database realistic and the
future evaluation honest:

- No hidden generator variable (customer health, engagement, adoption propensity, rep skill,
  API intensity) is stored in any table.
- There is no table describing the injected events. Their ground truth lives in
  `data/seeds/injected_events.json`, outside the database, for evaluation use only.
- The generator's validation checks for those events live in `data/generator/` and are not
  exposed to the agent.

A validation check (`exposure:*`) fails the build if the database contains any table, view or
column that is not documented here.
"""


def _mermaid_type(sql_type: str) -> str:
    return re.sub(r"\(.*\)", "", sql_type).upper()


def _key_marker(table: TableSpec, col: ColumnSpec) -> str:
    markers = []
    if col.name in table.primary_key:
        markers.append("PK")
    if col.foreign_key is not None:
        markers.append("FK")
    return ", ".join(markers)


def render_er_diagram() -> str:
    lines = ["erDiagram"]
    for rel in RELATIONSHIPS:
        parent_card = "|o" if rel.optional_parent else "||"
        link = ".." if rel.logical else "--"
        lines.append(f'    {rel.parent.upper()} {parent_card}{link}o{{ {rel.child.upper()} : "{rel.label}"')
    for table in TABLES:
        lines.append(f"    {table.name.upper()} {{")
        for col in table.columns:
            marker = _key_marker(table, col).replace(", ", ",")
            lines.append(f"        {_mermaid_type(col.sql_type)} {col.name}{' ' + marker if marker else ''}")
        lines.append("    }")
    return "\n".join(lines)


def _values(col: ColumnSpec) -> str:
    if col.allowed_values:
        return ", ".join(f"`{v}`" for v in col.allowed_values)
    return col.expected_values or ""


def _escape(text: str) -> str:
    return text.replace("|", "\\|")


def render_table(table: TableSpec) -> str:
    fks = [f"`{c.name}` -> `{c.foreign_key.table}.{c.foreign_key.column}`" for c in table.columns if c.foreign_key]
    out = [
        f"### `{table.name}`",
        "",
        f"**Business purpose:** {table.business_purpose}",
        "",
        f"**Grain:** {table.grain}",
        "",
        f"**Primary key:** {', '.join(f'`{c}`' for c in table.primary_key)}",
        "",
        f"**Foreign keys:** {'; '.join(fks) if fks else 'none'}",
        "",
        "| Column | Type | Nullable | Key | Allowed / expected values | Description | Business meaning |",
        "|---|---|---|---|---|---|---|",
    ]
    for col in table.columns:
        key = _key_marker(table, col)
        if col.pii:
            key = f"{key}, PII" if key else "PII"
        out.append(
            f"| `{col.name}` | {col.sql_type} | {'yes' if col.nullable else 'no'} | {key} | "
            f"{_escape(_values(col))} | {_escape(col.description)} | {_escape(col.business_meaning)} |"
        )
    return "\n".join(out)


def render_data_dictionary() -> str:
    sections = [
        "# Data Dictionary — Northwind Cloud",
        "",
        "<!-- Generated by `python -m app.database.data_dictionary` from app/database/metadata.py. "
        "Do not edit by hand. -->",
        "",
        f"Schema version **{SCHEMA_VERSION}**. Database: DuckDB (default `database/northwind_cloud.duckdb`).",
        "",
        "Northwind Cloud is a fictional B2B SaaS analytics company headquartered in Singapore. The database "
        "holds only business-observable data: what a real company would have in its billing, CRM, product "
        "telemetry, support and marketing systems.",
        "",
        "## Tables",
        "",
        "| Table | Grain | Primary key |",
        "|---|---|---|",
        *[f"| `{t.name}` | {t.grain} | {', '.join(f'`{c}`' for c in t.primary_key)} |" for t in TABLES],
        "",
        "## Entity-relationship diagram",
        "",
        "Solid lines are declared foreign keys; the dotted line is a logical (non-enforced) relationship.",
        "",
        "```mermaid",
        render_er_diagram(),
        "```",
        "",
        "## Relationships",
        "",
        "| Parent | Child | Join | Cardinality | Enforced |",
        "|---|---|---|---|---|",
        *[
            f"| `{r.parent}` | `{r.child}` | {r.join} | one {r.parent} to many {r.child}"
            f"{' (child may have no parent)' if r.optional_parent else ''} | {'no (logical)' if r.logical else 'FK'} |"
            for r in RELATIONSHIPS
        ],
        "",
        _CONVENTIONS,
        "## Table reference",
        "",
    ]
    for table in TABLES:
        sections += [render_table(table), ""]
    sections += ["## Analytical views", ""]
    sections += [
        "Views are descriptive building blocks. KPI definitions (churn rate, NRR, CAC, ...) are "
        "defined formally in the KPI framework, not in these views.",
        "",
        "| View | Description | Source tables |",
        "|---|---|---|",
        *[f"| `{v.name}` | {v.description} | {', '.join(f'`{s}`' for s in v.source_tables)} |" for v in VIEWS],
        "",
        _EXCLUSIONS,
    ]
    return "\n".join(sections).rstrip() + "\n"


def data_dictionary_json() -> dict[str, Any]:
    """Machine-readable dictionary (for tooling and, later, the agent's schema context)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "tables": [
            {
                "name": t.name,
                "business_purpose": t.business_purpose,
                "grain": t.grain,
                "primary_key": list(t.primary_key),
                "columns": [{k: v for k, v in asdict(c).items() if v not in (None, "", ())} for c in t.columns],
            }
            for t in TABLES
        ],
        "views": [
            {"name": v.name, "description": v.description, "source_tables": list(v.source_tables)} for v in VIEWS
        ],
        "relationships": [asdict(r) for r in RELATIONSHIPS],
    }


def write_data_dictionary(markdown_path: Path = DICTIONARY_PATH, json_path: Path | None = None) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_data_dictionary(), encoding="utf-8")
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(data_dictionary_json(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_data_dictionary(DICTIONARY_PATH, PROJECT_ROOT / "data" / "metadata" / "data_dictionary.json")
    print(f"Wrote {DICTIONARY_PATH.relative_to(PROJECT_ROOT)} and data/metadata/data_dictionary.json")
