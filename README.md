# AgentOps: AI Business Intelligence & Decision Agent

An agentic AI analyst for a (fictional) B2B SaaS company. A business user asks a question in
plain language, such as *"Why did revenue decline last month?"*. The agent plans an
investigation, runs validated SQL and statistical tools against real data, checks its
evidence and returns an answer in which every number is traceable to a query. It keeps
observed facts separate from inference and says when the evidence is insufficient.

> **Status: Phase 6 of 9 complete.**
> See [`docs/implementation-plan.md`](docs/implementation-plan.md) for the full plan.

| Phase | Scope | Status |
|---|---|---|
| 0 | Implementation plan | ✅ |
| 1 | Synthetic data generator, DuckDB schema, data dictionary, manifest, lineage foundation, validation | ✅ |
| 2 | KPI framework and analytics engine (20 KPIs, 8 analytics modules, independent validation) | ✅ |
| 3 | Forecasting and anomaly detection | ✅ |
| 4 | LangGraph agent, tools, evidence layer | ✅ |
| 5 | Guardrails, security and reliability | ✅ |
| 6 | MCP integration | ✅ |
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

## Forecasting and anomaly detection

Deterministic monthly forecasting and anomaly detection for revenue, MRR, active customers,
support tickets and product adoption. Every series is built from the Phase 2 KPIs. Everything
uses **only information available at the cutoff** (business as-of date 2026-08-31).

- **Forecasting:** naive, seasonal-naive, moving-average and drift baselines, plus damped-trend
  exponential smoothing (ETS). Models are compared with rolling-origin backtests and must beat the
  naive baseline to be selected. Forecasts carry 95% prediction intervals whose historical
  coverage is reported.
- **Anomaly detection:** rolling z-score, robust IQR (Tukey fences) and forecast-residual
  detectors. Each month is judged only against the months before it, with documented severity
  thresholds, statistical direction and no causal claims.
- **Evidence:** every result carries its SQL, backtest folds, fitted parameters, thresholds and
  limitations. Leakage tests alter the future and confirm that nothing at the cutoff changes.

```python
from app.database import get_database
from app.forecasting import forecast_metric
from app.anomalies import detect_anomalies

db = get_database()
forecast_metric(db, "mrr", horizon=3)                              # ForecastResult with intervals + backtests
detect_anomalies(db, "support_ticket_volume", detector="iqr")      # AnomalyReport, ranked by |score|
```

Details: [`docs/forecasting.md`](docs/forecasting.md) · [`docs/anomaly-detection.md`](docs/anomaly-detection.md)

## The agent

A LangGraph state machine answers a business question in plain language. It understands the
question, validates it against the data, plans an investigation with 12 allow-listed tools,
executes the tools, turns their typed results into evidence and claims, validates them, and
writes an answer that is validated again.

- **The LLM never produces a business number.** Every number comes from the Phase 2/3 tools
  and carries query IDs, source tables and the calculation. The response validator rejects
  any number that is not in the evidence the text cites.
- **Facts and inference are kept apart.** Observed and calculated findings, inferences worded
  as "was concentrated in" / "coincided with", recommendations, caveats and assumptions are
  separate sections. Causal claims are rejected unless they are negated.
- **Controlled paths:** unsupported questions (e.g. stock prices) run no tools. Questions the
  data cannot answer (the future, before the data, "What caused churn?") say so. Failed tools,
  invalid plans and rejected drafts end in typed failure states.
- **Bounded:** at most 12 tool calls, 2 planning iterations and 2 retries per step, plus a
  wall-clock limit (all configurable).
- **SQL safety:** `run_safe_sql` accepts one read-only `SELECT` over allow-listed tables, with
  bound parameters and a capped row limit. It runs on a read-only connection.
- **Offline by default:** the deterministic model (`LLM_PROVIDER=deterministic`) needs no key
  or network. `LLM_PROVIDER=anthropic` uses Claude through the official SDK
  (`pip install -e ".[anthropic]"`, `ANTHROPIC_API_KEY`).

```python
from app.database import get_database
from app.agent import run_agent

result = run_agent(get_database(), "Why did revenue decline last month?")
result.status                    # "completed"
result.response.answer           # the direct answer, citing claims
result.response.key_findings     # each with evidence IDs
result.tool_trace                # every tool call with inputs, status, timing and query IDs
```

Details: [`docs/agent-architecture.md`](docs/agent-architecture.md)

## Security and reliability

Application-level security controls are implemented for the prototype. This is not
production-grade security: authentication, multi-tenancy and network exposure come with the
later phases. The principle is **the model can propose; the application decides.**

- **Untrusted input.**
  - The question is type- and size-checked, and secrets are redacted before anything sees it.
  - A deterministic prompt-injection screen blocks requests for secrets, hidden or ground-truth
    data, files, code, the system prompt, disabling checks or changing limits.
  - Instruction overrides are answered with reduced privileges (no ad-hoc SQL).
- **Authorization.** Every tool call is authorised twice, at planning and just before execution,
  against:
  - an explicit allowlist and the tools permitted for the validated intent;
  - validated arguments and the data-exposure policy;
  - the run budget.

  A policy denial is never retried.
- **SQL.**
  - Read-only, single `SELECT` over allow-listed tables and columns (PII withheld).
  - Complexity limits, bound parameters, a row cap with truncation flags.
  - A statement timeout and a read-only connection.
- **Bounded.** Tool calls, retries, SQL calls and rows, model calls, context, response length and
  time are all limited and configurable (`AGENT_*`).
- **Output.**
  - Tool outputs are validated before they become evidence, and evidence is sealed with SHA-256
    fingerprints.
  - The response is checked for unsupported numbers, contradicted directions, causal claims,
    forecasts stated as facts, anomalies judged as good or bad, and directive recommendations.
- **Audit.** Every decision is a typed `SecurityEvent` (INFO/WARNING/HIGH/CRITICAL). Errors reach
  users only as safe categories.
- **Ground truth.** The agent reaches data only through approved tools. A regression suite with a
  Python audit hook shows that no file, process or network access happens during agent runs.

Details: [`docs/security-architecture.md`](docs/security-architecture.md) · threats and residual risks: [`docs/security-threat-model.md`](docs/security-threat-model.md)

## MCP server

The twelve analytics tools are also available to any [MCP](https://modelcontextprotocol.io)
client (IDE and desktop assistants, other agents) as `agentops_get_kpi`,
`agentops_analyze_revenue`, …, `agentops_forecast_metric`, `agentops_detect_anomalies` and
`agentops_run_safe_sql`.

- **A thin adapter.** It uses the official MCP Python SDK over stdio. Each call goes through the
  same Phase 5 execution path as the agent's tool calls: authorization, argument and
  data-exposure policy, budget, deadlines, output validation and redaction. The adapter has no
  analytics, SQL or permissions of its own.
- **Structured, traceable results.** Responses keep the Phase 4 evidence and provenance: query
  IDs, source tables, calculation, period and filters. They label forecasts, anomaly scores and
  risk scores as such, and fail safely with categorised, sanitised errors.
- **No machine access.** There are no file, shell, Python or environment tools, and the
  ground-truth seed data is unreachable.

```bash
python -m app.mcp --list-tools    # the enabled tools
python -m app.mcp                 # stdio server (normally launched by the MCP client)
claude mcp add agentops -- "$PWD/.venv/bin/agentops-mcp"   # e.g. register it with Claude Code
```

Details, client configuration and examples: [`docs/mcp-architecture.md`](docs/mcp-architecture.md)

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env              # optional; defaults work without it

python -m data.generator.generate # build database/northwind_cloud.duckdb (~30 s)
pytest                            # full test suite (~2 min; builds its own datasets)
```

The test suite needs no API key, LLM, network access or pre-built database: the agent tests use
the deterministic offline model and a scripted test double.

## Repository layout (so far)

```
app/
  config.py              settings from environment / .env
  database/              schema metadata (single source of truth), DDL, loader,
                         read-only DuckDB backend, lineage models, data-dictionary renderer
  analytics/             KPI registry + calculation engine (kpis/), periods, dimension allow-list,
                         revenue / customers / cohorts / risk / sales / marketing / support / product
  timeseries/            monthly series from the KPIs (missing-data policy, cutoff-bounded queries)
  forecasting/           baselines + ETS, rolling-origin backtests, model selection, ForecastService
  anomalies/             rolling z-score, IQR and forecast-residual detectors, AnomalyService
  tools/                 12 typed tools over Phases 2-3, allow-listed registry, SQL safety
  evidence/              evidence and claim models, claim-evidence graph, evidence and response validators
  llm/                   provider-neutral LLM interface, prompts, output schemas, deterministic + Anthropic providers
  agent/                 LangGraph state machine, request validation, claim builders, responses, runner
  security/              limits, input guard, prompt-injection screen, tool authorization, plan validator,
                         data-exposure policy, output guard, budgets, retries, timeouts, redaction, audit events,
                         the secured tool executor shared by the agent and MCP
  mcp/                   MCP server: tool registry, adapters, schemas, error model, audit, stdio entry point
data/
  generator/             reproducible synthetic-data pipeline (CLI: python -m data.generator.generate)
  metadata/              dataset manifest, checksums, machine-readable data dictionary
  seeds/                 injected-event ground truth (evaluation only; never exposed to the agent)
database/                DuckDB file (generated, git-ignored)
docs/                    implementation plan, data dictionary, analytics guide, KPI catalog,
                         forecasting and anomaly-detection guides, agent architecture,
                         security architecture and threat model, MCP architecture
tests/                   unit, integration, security (adversarial, SQL attack, regression) and MCP tests
```

## License

MIT, see [LICENSE](LICENSE).
