# AgentOps: AI Business Intelligence & Decision Agent

An agentic AI analyst for a (fictional) B2B SaaS company. A business user asks a question in
plain language, such as *"Why did revenue decline last month?"*. The agent plans an
investigation, runs validated SQL and statistical tools against real data, checks its
evidence and returns an answer in which every number is traceable to a query. It keeps
observed facts separate from inference and says when the evidence is insufficient.

> **Status: Phase 9 complete: production hardening and deployment readiness.**
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
| 7 | Evaluation & benchmark suite (89 scenarios, deterministic grading, security, MCP parity) | ✅ |
| 8 | Product API and UI (FastAPI + Streamlit: typed answers, evidence, charts, trace) | ✅ |
| 9 | Production hardening and deployment (auth, rate limiting, logs, metrics, readiness, bounded runs, Docker, CI) | ✅ |
| 10 | Final QA and portfolio documentation | ⏳ |

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

Security is enforced inside the agent (Phase 5) and, since Phase 9, in front of it: bearer-token
authentication, rate limiting, request limits and safe errors on the API (see
[Production Setup](#production-setup) and [`docs/security.md`](docs/security.md)). There are no
user accounts or multi-tenancy. The principle is **the model can propose; the application decides.**

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

Details: [`docs/security.md`](docs/security.md) (overview) · [`docs/security-architecture.md`](docs/security-architecture.md) · threats and residual risks: [`docs/security-threat-model.md`](docs/security-threat-model.md)

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

## Evaluation and benchmark

`evals/` measures how reliably the system behaves. It is a harness outside the application:
production never imports it. It runs the real paths (the LangGraph agent, the shared secured
executor and the MCP server over the protocol) and grades their structured behaviour, never
their wording.

- **89 versioned scenarios** (`eval_v1`) cover KPIs, investigations, forecasts, anomalies,
  refusals, 10 prompt injections, compromised-model plans, an SQL attack benchmark, data
  exposure, MCP and evidence integrity. Difficulty levels are easy, medium, hard and adversarial.
- **Independent references.** Expected values come from the Phase 2/3 pandas and time-series
  references at run time, within documented tolerances. The scenarios hold no numbers and no
  answers.
- **Hidden ground truth stays hidden.** Only `evals/reference/` reads the injected-event labels.
  They are rewarded as observable findings (the country, segment, month...), never by name.
  Static and runtime tests prove that nothing reaches the agent, model, tools or MCP.
- **One execution path.** A probe on `app/security/execution.py` shows that the agent and MCP get
  the same authorization, deadlines, retries, output validation, budgets and security events.
  Twenty parity scenarios check that MCP returns exactly what the direct path returns.

Results of the deterministic run (local, synthetic data; not a production claim):

| | Phase 7 | After Phase 7.1 |
|---|---|---|
| Scenarios passed | 82 / 89 | 89 / 89 |
| Security, injection, SQL, exposure | 32 / 32; 0 security or data-exposure failures | 32 / 32; 0 failures |
| MCP parity | 100% (20 scenarios, 47 calls) | 100% |
| Evidence grounding / hallucinations / unsupported causal claims | 100% / 0% / 0% | 100% / 0% / 0% |
| Numerical accuracy | 92.3% (36 of 39 reference checks) | 100% (39 of 39) |
| Intent / parameter / tool selection | 97.3% / 97.1% / 96.6% | 100% / 100% / 100% |
| Refusal recall / false refusals | 100% / 2.5% | 100% / 0% |
| Multi-seed (seeds 7 and 2027) / critical suite | 22 / 22, 13 / 13 | 22 / 22, 13 / 13 |

**Reliability note (Phase 7.1).** The Phase 7 benchmark found seven real failures:

- a wrong comparison month;
- a level ranking where a change was asked for;
- a missing channel breakdown;
- a false refusal of a sales-rep question;
- a validator that read customer-ID digits as numbers;
- a validator that did not check a claim's metric against its evidence (two failures).

Each was traced to its root cause, fixed in the layer that owned it, and given a regression test.
Claims now carry a structured subject (metric, unit, period, comparison, dimension, filters) that
must match their evidence. The benchmark, its thresholds and the security policy were not changed
to get there. A full pass means the known failure modes are fixed, not that the agent is reliable
in general. Details: [`docs/phase-7-1-reliability.md`](docs/phase-7-1-reliability.md).

```bash
python -m evals.run                    # full benchmark, deterministic (no key, no network)
python -m evals.run --suite critical   # 13-scenario regression suite
python -m evals.run --category prompt_injection --multi-seed 7,2027
```

Metrics, thresholds, reports and limitations: [`docs/evaluation.md`](docs/evaluation.md)

## Running AgentOps

Ask a question in the browser (or over HTTP) and get:

- a concise answer;
- the key figures and the period compared;
- typed findings (observed, calculated, inferred, recommended);
- charts drawn from the evidence;
- forecast and anomaly details;
- an evidence and provenance table;
- the analysis trace.

Everything runs locally with the deterministic offline model: no API key, no network.

**1. Install** (Python 3.11+, pip):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"      # everything, including tests; or: pip install -e ".[api,ui]"
cp .env.example .env
echo "API_AUTH_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env
```

The API requires a bearer token, and the UI reads the same `.env` to send it.

**2. Generate the data** (once, about 30 s): `python -m data.generator.generate` builds
`database/northwind_cloud.duckdb`.

**3. Start the API** (terminal 1):

```bash
python -m app.api            # http://127.0.0.1:8000, OpenAPI docs at /docs (development)
```

**4. Start the UI** (terminal 2):

```bash
python -m app.ui             # http://localhost:8501
```

Or run both in containers: `docker compose up --build -d` (see [Production Setup](#production-setup)).

**Example request:**

```bash
export API_AUTH_TOKEN=...    # the value in .env
curl -s http://127.0.0.1:8000/api/v1/ask \
  -H "Authorization: Bearer $API_AUTH_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question": "What was revenue in July compared with June?"}'
```

It returns `outcome: "answered"`, the answer ("Revenue changed by +SGD 40,472 (+0.70%) from
2026-06 to 2026-07."), both periods, KPI values, claims, evidence with query IDs, the tool trace
and two chart specs. `GET /api/v1/health` (liveness) and `GET /api/v1/readiness` are public;
`GET /api/v1/capabilities` describes what the agent can analyse.

**Try these questions:**

- What was revenue in July compared with June?
- Which region had the largest revenue decline?
- Which acquisition channel has the highest CAC?
- What is our 3-month revenue forecast?
- Are there any unusual trends in support tickets?
- Why did support tickets increase?

The last one tests the premise. Tickets actually fell in August (-13.4%): the answer reports the
decrease rather than inventing an increase, and shows the June spike flagged by the anomaly check.

**Architecture.**

```
Streamlit UI (app/ui)        formats and draws; no data access, no calculations
      │ HTTP (JSON)
FastAPI (app/api)            bearer auth, rate limit, validation, request IDs, safe errors, logs, metrics, chart specs
      │ AgentRunner.run
LangGraph agent (app/agent)  understand → validate → plan → execute → evidence → validate → respond → validate
      │ plans of tool calls
Secured executor (app/security)  authorization, data-exposure policy, SQL safety, budgets, deadlines, output checks
      │
Tools (app/tools) → analytics / forecasting / anomalies (Phases 2–3) → read-only DuckDB
```

**Security boundary.** The API and UI are new entry points to the agent, not a path around it.
Every question goes through `AgentRunner.run`, so the Phase 5 controls apply unchanged. The API has
no SQL, tools, file access or settings of its own, and unknown request fields are rejected. No route
serves files, so the hidden ground truth (`data/seeds/`) is unreachable. Refusals are controlled
responses (HTTP 200, `outcome: "refused"`) without security internals. Errors are fixed messages
with a request ID: no stack traces, SQL, paths or secrets. Every endpoint except liveness and
readiness requires the bearer token, and the ask endpoints are rate-limited. The API tests repeat
the Phase 5/7 attacks over HTTP, including against the authenticated, rate-limited production
configuration.

**Evidence and provenance.** Every number in an answer comes from an evidence item. Each item is
produced by a tool call and carries the period, filters, source tables, calculation, query IDs and a
fingerprint. Each claim cites its evidence and is typed: observed fact, calculated result, inference
or recommendation. The UI shows all of it in "Evidence & Provenance"; charts and KPI cards reuse the
same numbers. Forecasts are labelled as estimates with their interval, model and backtest.
Anomalies are labelled as statistically unusual, which is not necessarily bad.

Details: [`docs/api.md`](docs/api.md) (contract, errors, request IDs, security, performance) and
[`docs/ui.md`](docs/ui.md) (page, refusal handling, testing).

## Production Setup

Phase 9 makes AgentOps deployable without changing its architecture (UI → API → agent → secured
tools → read-only DuckDB). Full guide: [`docs/deployment.md`](docs/deployment.md). Security model:
[`docs/security.md`](docs/security.md).

**Environment variables.** Everything is configured through the environment (or `.env`), validated
at start-up by `app/config.py`. [`.env.example`](.env.example) documents every setting with safe
placeholders, never real values. The important ones:

| Variable | Default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | `production` enables strict start-up checks (below) |
| `API_AUTH_TOKEN` | — | Bearer token, 32+ characters: `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `API_AUTH_MODE` | `token` | `disabled` only for local development; refused in production |
| `API_RATE_LIMIT` | `20/minute` | Per-client limit on `/ask` and `/ask/stream` (`N/second\|minute\|hour`) |
| `API_CORS_ORIGINS` | empty | Explicit browser origins; empty = none. Never `*` in production |
| `DATABASE_URL` | `duckdb:///database/northwind_cloud.duckdb` | Must be explicit in production |
| `API_REQUEST_TIMEOUT_SECONDS` | `150` | Per request; the run is cancelled at the limit (504) |
| `LOG_FORMAT` / `LOG_LEVEL` | `json` / `INFO` | Structured JSON lines on stderr |
| `UI_API_URL` | `http://127.0.0.1:8000` | Where the UI calls the API |

**Authentication.** Every `/api/v1` endpoint except `/health` and `/readiness` needs
`Authorization: Bearer <API_AUTH_TOKEN>`. Missing, malformed and wrong credentials get the same
401, which reveals nothing. Tokens are compared in constant time and never logged or shown in the
UI. The UI sends the token from its own environment.

**Rate limiting.** A sliding window per client on the ask endpoints (default 20 requests a
minute); excess requests get 429 with `Retry-After`. The limiter is in memory and per process,
which suits the single-process deployment. Bodies are capped at 16 KB (413), the ask endpoints
accept JSON only (415), questions are capped at 1,000 characters (422), and at most 4 requests
wait for the agent (503).

**API startup.** `python -m app.api` (`--check-config` validates and exits). With
`APP_ENV=production` the API refuses to start without:

- a token, and authentication cannot be disabled;
- a rate limit;
- an explicit `DATABASE_URL`;
- HTTPS-only, non-wildcard CORS origins;
- timeouts that nest (SQL ≤ tool ≤ run ≤ API ≤ UI).

Each problem is listed by setting name, never by value.

**UI startup.** `python -m app.ui` starts Streamlit headless with hardened options: XSRF
protection, no usage statistics, no uploads, and in production no error details.

**Docker startup.**

```bash
python -m data.generator.generate      # the database is mounted read-only, never baked into an image
echo "API_AUTH_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env
docker compose up --build -d           # UI http://127.0.0.1:8501 · API http://127.0.0.1:8000
API_AUTH_TOKEN=... python scripts/smoke_test.py --api-url http://127.0.0.1:8000 --ui-url http://127.0.0.1:8501
docker compose stop                    # graceful: in-flight runs finish or are cancelled, exit code 0
```

Two images from one multi-stage `Dockerfile`:

- **API image:** FastAPI plus the agent.
- **UI image:** Streamlit and the HTTP client only. It contains no database, agent or data code.

Both images, and the compose stack around them, are hardened:

- Both run as a non-root user (uid 10001), with a read-only root filesystem, no Linux capabilities
  and `no-new-privileges`.
- Ports are published on `127.0.0.1` only. Put a TLS-terminating reverse proxy in front for remote
  users.
- The images contain no secrets, tests, evaluation code or ground truth.

**Health and readiness.**

- `GET /api/v1/health` is liveness: the process serves HTTP.
- `GET /api/v1/readiness` answers 200 or 503 with named checks: configuration, database, agent,
  and accepting requests (it turns 503 while shutting down).

Both are public and contain no internals. The Docker healthcheck uses readiness.

**Logging.** One JSON object per line with timestamp, level, logger, request ID, route, status,
outcome, error code, durations, and the agent's run ID and tool names. The request ID links the
response, the API log and the agent's log lines. Never logged: questions, answers, headers,
tokens, API keys, client addresses, prompts, tool outputs, SQL error text, paths and tracebacks
(reduced to the exception type).

**Metrics.** `GET /api/v1/metrics` (authenticated) separates the following:

- answered, partial, refused, unsupported and insufficient-evidence outcomes;
- tool or planning failures;
- client errors, unauthorised and rate-limited requests;
- timeouts, unavailability and internal errors.

It also reports p50/p95 request and agent latency and run states (running, stopping, completed,
refused, failed, timeout, cancelled).

**Security considerations.**

- The API is an entry point to the agent, not a way around it. Every Phase 5 control still
  applies: injection screening, tool authorization, SQL safety, the data-exposure policy, budgets,
  deadlines and output validation.
- One shared service token, not user accounts. Keep the API private and add user sign-in at the
  proxy.
- Rotate the token by changing it and restarting.
- CI (`.github/workflows/ci.yml`) runs:
  - lint, format, types and the test suite;
  - the critical benchmark suite;
  - a Docker build and smoke test.

  It needs no secrets. The full benchmark and the multi-seed check run in `evaluation.yml`.

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
echo "API_AUTH_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env

python -m data.generator.generate # build database/northwind_cloud.duckdb (~30 s)
pytest                            # full test suite (builds its own datasets)
python -m evals.run --suite critical  # evaluation regression suite (~12 s)
python -m app.api                 # API (see "Running AgentOps")
python -m app.ui                  # web UI
docker compose up --build -d      # or both, in containers (see "Production Setup")
```

The test suite needs no API key, LLM, network access, browser or pre-built database. The agent
tests use the deterministic offline model and a scripted test double. The API and UI tests run the
app in-process (FastAPI's test client, Streamlit's `AppTest`).

## Repository layout

```
app/
  config.py              settings from environment / .env (validated; production start-up rules in app/api/config.py)
  logs.py                JSON log formatting (redaction, no tracebacks)
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
  api/                   FastAPI app: /ask, /ask/stream, /health, /readiness, /capabilities, /metrics; schemas,
                         agent service and run tracker, bearer auth and rate limiter, presenter and chart specs,
                         error model, request-ID and security-header middleware
  ui/                    Streamlit page, HTTP client, testable view models, rendering, hardened launcher
evals/                   evaluation harness (not part of the app): scenario datasets, independent references,
                         hidden-label mapping, runners, deterministic graders, metrics, thresholds, reports
data/
  generator/             reproducible synthetic-data pipeline (CLI: python -m data.generator.generate)
  metadata/              dataset manifest, checksums, machine-readable data dictionary
  seeds/                 injected-event ground truth (evaluation only; never exposed to the agent)
database/                DuckDB file (generated, git-ignored)
docs/                    implementation plan, data dictionary, analytics guide, KPI catalog,
                         forecasting and anomaly-detection guides, agent architecture,
                         security overview, architecture and threat model, MCP architecture, evaluation, API, UI,
                         deployment
scripts/smoke_test.py    smoke test for a running deployment (local or docker compose)
tests/                   unit, integration, security (adversarial, SQL attack, regression), MCP, evaluation,
                         API (contract, security, production, runtime, streaming, isolation, performance),
                         UI and deployment tests
Dockerfile               multi-stage: api and ui images (non-root, no data or secrets)
docker-compose.yml       the two services, hardened (read-only, no capabilities, localhost ports)
.github/workflows/       ci.yml (lint, types, tests, critical suite, Docker smoke test); evaluation.yml (full
                         benchmark, multi-seed)
```

## License

MIT, see [LICENSE](LICENSE).
