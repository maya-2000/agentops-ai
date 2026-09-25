"""Data-exposure policy: which business data the agent may read, and in what form.

The agent reads data only through the approved business interfaces: the 12 tools over Phases 1-3
and, for ad-hoc questions, ``run_safe_sql``. This module states explicitly what those interfaces
may expose.

- **Approved tables and views** are listed by name here. A table added to the schema later is
  not exposed until it is added to this list (fail closed).
- **Columns** are the schema's columns for approved tables, minus withheld columns.
  - ``sales_opportunities.sales_rep`` is withheld because it is PII-tagged (a person's name).
  - ``customers.company_name`` is withheld because it identifies the account and analytics do
    not need it.
  - Any column whose name suggests hidden or generated state (health, latent, propensity,
    ground truth, injected, ...) or a credential is always withheld, even if it were added to
    the schema.
- **Customer-level output** (one row per customer) is capped at ``max_customer_rows``.
  Aggregated analytics are preferred.
- **PII in analytics output**: ``analyze_sales.rep_performance`` returns rep names because the
  question it answers is about reps. This is the only operation allowed to return a PII-tagged
  field, and rep names never enter evidence statements or prompts.

The generator's hidden customer-health process and the injected-event ground truth are not in
the database at all. They live in the generator and ``data/seeds``, which no agent interface
can reach. This policy additionally rejects any attempt to name them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from app.database import metadata

APPROVED_TABLES: frozenset[str] = frozenset(
    {
        "customers",
        "subscriptions",
        "usage_events",
        "sales_opportunities",
        "support_tickets",
        "marketing_campaigns",
        "daily_revenue",
        "product_features",
    }
)
APPROVED_VIEWS: frozenset[str] = frozenset(
    {"v_monthly_revenue", "v_monthly_mrr", "v_subscription_events", "v_campaign_summary"}
)
WITHHELD_COLUMNS: dict[str, frozenset[str]] = {
    "customers": frozenset({"company_name"}),
}
HIDDEN_STATE_PATTERN = re.compile(
    r"(health|latent|propensity|ground_?truth|injected|calibrat|hidden|generator|secret|password|token|api_?key)",
    re.IGNORECASE,
)
# Tool results that may contain a PII-tagged field, by operation, and the field names allowed.
PII_ALLOWED_OPERATIONS: dict[str, frozenset[str]] = {
    "analyze_sales.rep_performance": frozenset({"sales_rep"}),
}
CUSTOMER_LEVEL_OPERATIONS: frozenset[str] = frozenset({"get_customer_risk"})


@dataclass(frozen=True)
class DataExposurePolicy:
    exposed_columns: dict[str, frozenset[str]]
    pii_columns: dict[str, frozenset[str]]
    withheld_columns: dict[str, frozenset[str]] = field(default_factory=dict)

    @property
    def approved_relations(self) -> frozenset[str]:
        return frozenset(self.exposed_columns)

    def is_withheld(self, table: str, column: str) -> bool:
        return column in self.pii_columns.get(table, frozenset()) or column in self.withheld_columns.get(
            table, frozenset()
        )

    @property
    def all_pii_columns(self) -> frozenset[str]:
        return frozenset().union(*self.pii_columns.values())

    @property
    def all_withheld_columns(self) -> frozenset[str]:
        return frozenset().union(*self.pii_columns.values(), *self.withheld_columns.values())


def default_exposure_policy() -> DataExposurePolicy:
    """Build the policy from the Phase 1 schema metadata and the explicit lists above."""
    exposed: dict[str, frozenset[str]] = {}
    pii: dict[str, frozenset[str]] = {}
    withheld: dict[str, frozenset[str]] = {}
    for table in metadata.TABLES:
        if table.name not in APPROVED_TABLES:
            continue  # not approved: not exposed
        hidden = {c.name for c in table.columns if HIDDEN_STATE_PATTERN.search(c.name)}
        pii[table.name] = frozenset(c.name for c in table.columns if c.pii)
        withheld[table.name] = WITHHELD_COLUMNS.get(table.name, frozenset()) | hidden
        exposed[table.name] = frozenset(
            c.name for c in table.columns if c.name not in pii[table.name] and c.name not in withheld[table.name]
        )
    for view in metadata.VIEWS:
        if view.name not in APPROVED_VIEWS:
            continue
        query = sqlglot.parse_one(view.sql, read="duckdb")
        assert isinstance(query, exp.Query), view.name
        columns = frozenset(query.named_selects)
        pii[view.name] = frozenset()
        withheld[view.name] = frozenset(c for c in columns if HIDDEN_STATE_PATTERN.search(c))
        exposed[view.name] = columns - withheld[view.name]
    return DataExposurePolicy(exposed_columns=exposed, pii_columns=pii, withheld_columns=withheld)


def names_hidden_state(text: str) -> bool:
    """Whether an identifier names hidden or generated state (always withheld)."""
    return bool(HIDDEN_STATE_PATTERN.search(text))
