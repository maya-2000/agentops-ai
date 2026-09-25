# AgentOps: AI Business Intelligence & Decision Agent

An agentic AI analyst for a (fictional) B2B SaaS company. A business user asks a question in
plain language, such as *"Why did revenue decline last month?"*. The agent plans an
investigation, runs validated SQL and statistical tools against real data, checks its
evidence and returns an answer in which every number is traceable to a query. It keeps
observed facts separate from inference and says when the evidence is insufficient.

> **Status: Phase 2 of 9 complete.**
> See [`docs/implementation-plan.md`](docs/implementation-plan.md) for the full plan.

| Phase | Scope | Status |
|---|---|---|
| 0 | Implementation plan | ✅ |
| 1 | Synthetic data generator, DuckDB schema, data dictionary, manifest, lineage foundation, validation | ✅ |
| 2 | KPI framework and analytics engine (20 KPIs, 8 analytics modules, independent validation) | ✅ |
| 3 | Forecasting and anomaly detection | ⏳ |
| 4 | LangGraph agent, tools, evidence layer | ⏳ |
| 5 | Guardrails and security | ⏳ |
| 6 | MCP server | ⏳ |
| 7 | Evaluation framework (50+ benchmark questions) | ⏳ |
| 8 | FastAPI and Streamlit | ⏳ |
| 9 | Final QA and portfolio documentation | ⏳ |

## The data: Northwind Cloud

A reproducible simulation of a Singapore-headquartered B2B SaaS company (reporting currency
SGD), covering September 2024 to August 2026 ("today" = 2026-08-31):

- **Scale:** 5,000 customers across 4 regions, 13 countries, 3 segments and 4 plans, with 20
  sales reps.
- **Tables:** 8 business tables: customers, versioned subscriptions, weekly usage, CRM
  opportunities, support tickets, weekly campaign performance, daily revenue and daily
  feature adoption. About 2.8 million rows in total.
- **Hidden mechanism:** customer health drives usage, support load, churn, expansion and
  contraction. It is never stored in the database.
- **Injected events:** seven business events (e.g. an August 2026 churn wave among Singapore
  Enterprise accounts, a support-ticket spike, an inefficient campaign) serve as evaluation
  ground truth. That ground truth is kept outside the database.

Details: [`data/README.md`](data/README.md) · schema: [`docs/data-dictionary.md`](docs/data-dictionary.md)

## The analytics engine

The analytics layer is the **source of truth for business numbers**. Twenty KPIs (revenue,
MRR/ARR, growth, logo and revenue churn, retention, NRR, CAC, CLV, ARPU, AOV, conversion,
pipeline, win rate, sales cycle, ticket volume, resolution time, product adoption, customer
count) are defined once in a typed registry, with formula, SQL, unit, time grain,
interpretation, limitations and dependencies. They are computed from parameterised SQL through
the read-only database layer.
- **Results:** every result is typed and evidence-ready, carrying its SQL, bound parameters,
  source tables and lineage IDs.
- **Analytics modules:** revenue decomposition and MRR bridge, churn, cohorts, customer risk,
  sales, marketing, support and product analytics.
- **Validation:** all of it is checked against an independent pandas implementation.

```python
from app.database import get_database
from app.analytics.kpis import calculate_kpi, get_kpi_definition
from app.analytics import revenue

db = get_database()
calculate_kpi(db, "nrr", period="trailing_12_months")               # KPIResult with value + provenance
calculate_kpi(db, "revenue", period="last_month", dimension="region")
revenue.decompose_revenue_change(db, "segment", "last_month")        # reconciles to the total change
get_kpi_definition("cac").limitations
```

Details: [`docs/analytics.md`](docs/analytics.md) · KPI catalog: [`docs/kpi-catalog.md`](docs/kpi-catalog.md)

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env              # optional; defaults work without it

python -m data.generator.generate # build database/northwind_cloud.duckdb (~30 s)
pytest                            # full test suite (~65 s; builds its own datasets)
```

The test suite needs no API key, LLM, network access or pre-built database.

## Repository layout (so far)

```
app/
  config.py              settings from environment / .env
  database/              schema metadata (single source of truth), DDL, loader,
                         read-only DuckDB backend, lineage models, data-dictionary renderer
  analytics/             KPI registry + calculation engine (kpis/), periods, dimension allow-list,
                         revenue / customers / cohorts / risk / sales / marketing / support / product
data/
  generator/             reproducible synthetic-data pipeline (CLI: python -m data.generator.generate)
  metadata/              dataset manifest, checksums, machine-readable data dictionary
  seeds/                 injected-event ground truth (evaluation only; never exposed to the agent)
database/                DuckDB file (generated, git-ignored)
docs/                    implementation plan, data dictionary, analytics guide, KPI catalog
tests/                   unit and integration tests
```

## License

MIT, see [LICENSE](LICENSE).
