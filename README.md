# AgentOps AI: evidence-grounded business intelligence and decision agent

AgentOps AI is an AI analyst for a (fictional) B2B SaaS company. A business user asks a question in
plain language ("Why did revenue decline last month?") or states an objective ("Why is revenue growth
slowing?"). AgentOps plans the analysis, runs **deterministic analytics tools** against a read-only
database, turns every result into **evidence**, validates every claim against that evidence, and returns
either an answer or a **decision brief**. Every number is traceable to a query.

The language model interprets and drafts. **It never produces a business number, never runs a tool
itself, and cannot widen its own permissions.** The analytics engine is the only source of numbers.
Facts, calculations, inferences and recommendations are labelled separately. Correlation is never
presented as causation, and when the data cannot answer, the system says so.

> **Status:** final release, version **0.10.0**. It runs fully offline with the default deterministic
> model (no API key, no network); Claude can be used as the model through the official SDK.

## The problem it addresses

Generic LLM assistants answer business questions fluently, but their numbers and explanations cannot
be trusted:

- figures can be invented;
- correlations get reported as causes;
- a prompt can talk the assistant into reading data it should not.

AgentOps shows a different design:

- a deterministic analytics layer computes every number;
- an evidence layer records where each number came from;
- validators reject any statement the evidence does not support;
- one secured executor enforces permissions, budgets and data-exposure rules on every tool call, from
  every entry point.

## Key capabilities

| Area | What exists |
|---|---|
| **Analytics** | 20 registered KPIs (revenue, MRR/ARR, growth, logo and revenue churn, retention, NRR, CAC, CLV, ARPU, AOV, conversion, pipeline, win rate, sales cycle, ticket volume, resolution time, product adoption, customer count) and 8 analytics modules (revenue decomposition and MRR bridge, churn, cohorts, customer risk, sales, marketing, support, product), all checked against an independent pandas reference |
| **Forecasting and anomalies** | Baselines and damped ETS, with rolling-origin backtests and 95% intervals; rolling z-score, IQR and forecast-residual detectors; no look-ahead past the cutoff |
| **Agent** | A LangGraph state machine that runs understand → validate → plan → execute → evidence → validate → respond → validate, using 12 allow-listed tools |
| **Safe SQL** | One read-only `SELECT` over allow-listed tables and columns, with bound parameters, a row cap and a timeout, on a read-only connection |
| **Evidence and claims** | Fingerprinted evidence with provenance (query IDs, source tables, calculation, period, filters). Claims are typed and carry a structured subject that must match their evidence. Response validation checks numbers, KPI names, direction, causal wording and labels |
| **Investigations** | Multi-step investigations from six fixed templates. Cross-finding validation, rule-based driver analysis (contributions, co-movements, contradictions, associations) and grounded recommendations feed a decision brief. Budgets are enforced in code |
| **Security** | Input guard, prompt-injection screen, tool authorisation, data-exposure policy, budgets, deadlines, output validation, audit events; bearer-token authentication and rate limiting on the API |
| **Interfaces** | FastAPI (`/ask`, `/investigations`, streaming variants, health, readiness, capabilities, metrics); a Streamlit UI with *Ask* and *Investigate* modes; an MCP server (stdio) exposing the 12 tools |
| **Evaluation** | eval_v1 (89 single-question, security and MCP scenarios) and eval_v2 (77 investigation scenarios), graded deterministically against independent references, including multi-seed runs |
| **Deployment** | Two hardened Docker images (API, UI) and docker compose; production start-up checks; CI for lint, types, tests, critical benchmarks and a Docker smoke test |

## Architecture

```
Browser ─▶ Streamlit UI (app/ui)            formats and draws; no data access, no calculations
             │ HTTP + bearer token
           FastAPI (app/api)                 auth · rate limit · request limits · request IDs · safe errors · logs · metrics
             │ /ask                          │ /investigations
           LangGraph agent (app/agent)       Investigator (app/investigation)
             │ proposed tool calls           │ template steps (the model never chooses a tool)
MCP ─stdio─▶ SecuredToolExecutor (app/security): authorise → deadline → execute → retry → validate output → budget → audit
             │
           12 tools (app/tools) → analytics · forecasting · anomalies (deterministic) → read-only DuckDB
             │
           evidence (fingerprinted) → claims → evidence and response validators → answer / decision brief
```

The UI, API and MCP server are entry points, not paths around the agent's controls. Every tool call,
from any of them, goes through the same executor. The full description, including the request and
investigation lifecycles, is in [`docs/final-architecture.md`](docs/final-architecture.md).

## Example questions

These all work with the default dataset and model. [`docs/demo.md`](docs/demo.md) has a 5–10 minute
demo with example outputs.

| Kind | Ask this | What it shows |
|---|---|---|
| KPI | What was revenue last month? | A relative period, one KPI tool call, provenance |
| Comparison | What was revenue in July 2026 compared with June 2026? | Explicit periods and a calculated change |
| Breakdown | Which region had the largest revenue decline last month? | A change decomposition (member changes reconcile to the total) |
| Forecast | What is our 3-month revenue forecast? | A backtested model, a 95% interval, labelled as an estimate |
| Anomaly | Are there any unusual trends in support tickets? | Flagged months with direction, not judged good or bad |
| Customer risk | Which customers are at risk? | Risk bands by customer ID only; withheld fields never shown |
| False premise | Why did support tickets increase? | Tickets actually fell: the answer says so |
| Investigation | Why is revenue growth slowing? | Plan, drivers, contradictions, grounded recommendations |
| Management brief | Give me a management brief on the current state of the business. | Sections only where data supports them |
| Causality | Did the price increase cause churn? | `insufficient_evidence`, with the observed findings |

## Evidence and trust model

| Label | Meaning |
|---|---|
| **Observed** | Read directly from recorded data |
| **Calculated** | Computed from recorded data by a registered calculation |
| **Inferred** | A reading of the evidence ("was concentrated in"), not an observed fact |
| **Recommended** | A suggested next step that cites the findings it rests on; not a finding |

**Numbers.** Every number shown comes from an evidence item produced by a tool call. Each item carries
the period, filters, source tables, calculation, query IDs and a SHA-256 fingerprint. A text whose
numbers are not in the evidence it cites fails validation.

**Causes.** Causal wording ("caused by", "because of", "due to", "led to", "resulted from") is rejected
unless negated. Investigation drivers are typed as accounting contributions, same-period co-movements,
associations, contradictions or context, never as causes. Questions that ask for a cause get
`insufficient_evidence`, together with what was observed.

**Presentation.** Forecasts are labelled estimates with intervals and backtests. Anomalies are
"statistically unusual", not good or bad. A failed, refused or stopped run returns a controlled
outcome, never an unvalidated guess.

## Investigations

`/ask` answers one question. An investigation answers an objective. It works like this:

1. It chooses one of six templates: revenue, customer, sales, product and support, general, or
   management brief.
2. It runs the template's steps through the secured executor, reusing identical calls and never
   running ad-hoc SQL.
3. It validates the findings against each other, derives drivers and recommendations by rule, and
   writes a decision brief.

Example, abridged from the default dataset:

```
Plan            ✓ revenue change · ✓ region/segment decomposition · ✓ MRR bridge · ✓ churn, NRR · ✓ win rate, pipeline
                ✓ usage before churn · ✓ revenue anomalies · ✓ drill-down into the region that concentrates the change
Findings        Revenue changed by -SGD 56,286 (-0.97%) from 2026-07 to 2026-08. (calculated)
Drivers         Region APAC: 92.1% of the gross decline (contributes to) · Country Singapore within APAC: 83.2%
                Logo churn up, NRR down, pipeline down in the same period (moved in line)
Contradicting   Win rate rose: the evidence does not point to it
Recommendations Investigate the Singapore accounts within APAC behind the revenue decline (rests on F16, F17)
Uncertainty     Drivers are accounting contributions and same-period co-movements, not established causes
Evidence        41 evidence items from 11 tool calls
```

Budgets are enforced in code: 14 steps, 16 tool calls, 120 s, 400 evidence items and 12,000 output
characters. A stop is reported as "Investigation stopped because the analysis budget was reached.",
never as complete. Details: [`docs/investigations.md`](docs/investigations.md).

## Security

The design principle is: **the model can propose; the application decides.**

- **Input.** Questions are type- and size-checked, and secrets are redacted before anything sees them.
  A deterministic prompt-injection screen blocks requests for secrets, the system prompt, hidden or
  ground-truth data, files, code execution or rule changes, before any model or tool call.
- **Authorisation.** Every tool call is authorised against:
  - the allow-list and the tools permitted for the validated intent;
  - validated arguments;
  - the data-exposure policy (withheld and personal columns, restricted breakdowns);
  - the budget.
- **SQL.** One read-only `SELECT` over allow-listed tables and columns, with complexity limits, a row
  cap and a statement timeout.
- **Bounds.** Tool calls, retries, SQL, model calls, context, response size, wall clock and the
  investigation budgets are all capped. Runs are cancelled on timeout, client disconnect and shutdown.
- **Output.** Tool outputs are validated before they become evidence. Evidence is fingerprinted.
  Responses are validated against their evidence.
- **API.**
  - Access: bearer-token authentication with constant-time comparison and a uniform 401, and one
    per-client rate limit (default 20/minute) shared by `/ask` and `/investigations`.
  - Requests: JSON only, body and question limits, unknown fields rejected.
  - Traceability: request IDs, security headers, safe error envelopes.
  - Logs: structured, with no questions, answers, tokens or tracebacks.
- **Deployment.** Containers run non-root, with a read-only root filesystem, no Linux capabilities and
  localhost-only ports. The database is mounted read-only. Images contain no secrets, data or ground
  truth. `APP_ENV=production` refuses an insecure configuration.
- **Ground truth.** The injected-event labels are read only by the evaluation harness. Audit-hook
  tests show that agent and investigation runs touch no files, processes or network.

Details: [`docs/security.md`](docs/security.md), [`docs/security-architecture.md`](docs/security-architecture.md),
[`docs/security-threat-model.md`](docs/security-threat-model.md).

## MCP server

The 12 analytics tools are available to any [MCP](https://modelcontextprotocol.io) client as
`agentops_get_kpi`, …, `agentops_run_safe_sql`, over stdio with the official Python SDK. The server is
a thin adapter: every call goes through the same secured executor, and results keep their evidence and
provenance. There are no file, shell or code tools. Investigations are deliberately not exposed as an
MCP tool; they are a workflow built from these same calls.

```bash
python -m app.mcp --list-tools
claude mcp add agentops -- "$PWD/.venv/bin/agentops-mcp"   # e.g. register it with Claude Code
```

Details: [`docs/mcp-architecture.md`](docs/mcp-architecture.md).

## Evaluation

`evals/` is an evaluation harness outside the application (production never imports it). It runs the
real paths and grades structured behaviour against independent references, not wording. Final
deterministic results on the seed-42 dataset:

| Benchmark | Result |
|---|---|
| eval_v1: single questions, security, MCP (89 scenarios) | **89/89**. Intent, parameter and tool selection, numerical accuracy (39 reference checks), grounding and claim support all at 100%; hallucinations, unsupported causal claims and false refusals at 0% |
| Security benchmark (in eval_v1) | 32/32: 10 security, 10 prompt injection, 6 SQL attack, 6 data exposure |
| MCP (in eval_v1) | 23/23; direct-vs-MCP parity 100% over 20 scenarios (47 calls) |
| eval_v2: investigations (77 scenarios, 18 categories) | **77/77**. Evidence completeness and identity, driver correctness, recommendation grounding, causal safety and budget compliance all at 100% |
| Critical suites | eval_v1 13/13, eval_v2 12/12 |
| Multi-seed (seeds 7 and 2027) | eval_v1 22/22 runs, eval_v2 24/24 runs |

**What this does and does not show.** These are regression benchmarks for the behaviours they encode,
on one synthetic company, with the deterministic model. They are not a claim of general reliability or
production accuracy. The Phase 7 baseline was 82/89; seven real failures were fixed with regression
tests in Phase 7.1, without changing the benchmark. The test suite (`pytest`) adds about 3,000 unit,
integration, security, MCP, API, UI, deployment and grader tests. Details, metrics, thresholds and
limitations: [`docs/evaluation.md`](docs/evaluation.md).

## Running locally

Requirements: Python 3.11+. Docker is optional. Everything runs offline with the deterministic model.

```bash
git clone https://github.com/maya-2000/agentops-ai.git && cd agentops-ai
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                        # app, API, UI, tests and tooling
cp .env.example .env
echo "API_AUTH_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env

python -m data.generator.generate              # build database/northwind_cloud.duckdb (seed 42, ~30 s)
pytest                                         # the test suite (builds its own datasets; no key, no network)
python -m evals.run --suite critical           # eval_v1 critical suite
python -m evals.run --dataset eval_v2 --suite critical   # eval_v2 critical suite

python -m app.api                              # API on http://127.0.0.1:8000 (OpenAPI docs at /docs in development)
python -m app.ui                               # UI on http://localhost:8501 (reads the same .env for the token)
```

Example request:

```bash
export API_AUTH_TOKEN=...   # the value in .env
curl -s http://127.0.0.1:8000/api/v1/ask -H "Authorization: Bearer $API_AUTH_TOKEN" \
  -H 'Content-Type: application/json' -d '{"question": "What was revenue in July 2026 compared with June 2026?"}'
curl -s http://127.0.0.1:8000/api/v1/investigations -H "Authorization: Bearer $API_AUTH_TOKEN" \
  -H 'Content-Type: application/json' -d '{"objective": "Why is revenue growth slowing?"}'
```

Docker (the database is mounted read-only, never baked into an image):

```bash
python -m data.generator.generate
docker compose up --build -d                   # UI http://127.0.0.1:8501 · API http://127.0.0.1:8000
API_AUTH_TOKEN=... python scripts/smoke_test.py --api-url http://127.0.0.1:8000 --ui-url http://127.0.0.1:8501
docker compose stop                            # graceful: in-flight runs finish or are cancelled
```

Behind a TLS-inspecting proxy, pass its CA bundle to the image build as the optional `pip_ca` secret
(see the `Dockerfile` header).

Main settings (all in [`.env.example`](.env.example), validated at start-up):

| Variable | Default | Purpose |
|---|---|---|
| `API_AUTH_TOKEN` | — | Bearer token (32+ characters), sent by the UI from the same `.env` |
| `APP_ENV` | `development` | `production` enables strict start-up checks |
| `API_RATE_LIMIT` | `20/minute` | Per-client limit shared by `/ask` and `/investigations` |
| `LLM_PROVIDER` | `deterministic` | `anthropic` uses Claude (`pip install -e ".[anthropic]"`, `ANTHROPIC_API_KEY`) |
| `AGENT_MAX_*` / `AGENT_MAX_INVESTIGATION_*` | see `.env.example` | Agent and investigation budgets |
| `DATABASE_URL` | `duckdb:///database/northwind_cloud.duckdb` | Must be explicit in production |

Deployment guide (configuration, auth, rate limiting, health, logging, metrics, shutdown):
[`docs/deployment.md`](docs/deployment.md).

## Limitations

- **Data.** The data is synthetic, from one company. Results describe Northwind Cloud, not other
  datasets.
- **Understanding.** The deterministic model covers a finite set of phrasings. Unusual wording may be
  asked to clarify, or may be mapped to a neighbouring intent or investigation template. With Claude
  configured, understanding is the model's, and it is validated in the same way.
- **Investigations.** They come from six fixed templates; objectives outside them are unsupported.
  Drivers are accounting shares and co-movements over one pair of periods: there are no significance
  tests and no confounder control.
- **Deployment.** One process with one read-only database connection: runs are serialised, rate
  limits and metrics are in memory, and nothing is persisted (no stored questions, runs or
  investigations).
- **Access.** One shared service token, with no user accounts. TLS and user sign-in belong in a reverse
  proxy.
- **Benchmarks.** They are regression floors on this dataset, not general reliability claims.

## Deferred improvements (out of scope)

These are deliberately **not** part of this project:

- **Model choice.** Model-based (rather than template-based) investigation planning; an LLM-mode
  benchmark as a release gate.
- **Scale and persistence.** Stored investigations and a status endpoint; multi-replica deployment with
  shared rate limits.
- **Access.** User accounts and per-user attribution.
- **Data.** Connectors to real business data sources.
- **Analytics.** Statistical tests for driver significance.

## Documentation

| Topic | Document |
|---|---|
| Final architecture, request and investigation lifecycles | [`docs/final-architecture.md`](docs/final-architecture.md) |
| 5–10 minute demo | [`docs/demo.md`](docs/demo.md) |
| Investigations | [`docs/investigations.md`](docs/investigations.md) |
| Evaluation | [`docs/evaluation.md`](docs/evaluation.md) · [`docs/phase-7-1-reliability.md`](docs/phase-7-1-reliability.md) |
| Agent | [`docs/agent-architecture.md`](docs/agent-architecture.md) |
| Security | [`docs/security.md`](docs/security.md) · [`docs/security-architecture.md`](docs/security-architecture.md) · [`docs/security-threat-model.md`](docs/security-threat-model.md) |
| API and UI | [`docs/api.md`](docs/api.md) · [`docs/ui.md`](docs/ui.md) |
| MCP | [`docs/mcp-architecture.md`](docs/mcp-architecture.md) |
| Deployment | [`docs/deployment.md`](docs/deployment.md) |
| Data, KPIs and analytics | [`data/README.md`](data/README.md) · [`docs/data-dictionary.md`](docs/data-dictionary.md) · [`docs/kpi-catalog.md`](docs/kpi-catalog.md) · [`docs/analytics.md`](docs/analytics.md) |
| Forecasting and anomalies | [`docs/forecasting.md`](docs/forecasting.md) · [`docs/anomaly-detection.md`](docs/anomaly-detection.md) |
| Plan and phase history | [`docs/implementation-plan.md`](docs/implementation-plan.md) |

## The data: Northwind Cloud

A reproducible simulation of a Singapore-headquartered B2B SaaS company (reporting currency SGD),
covering September 2024 to August 2026 ("today" is 2026-08-31):

- **Scale:** 5,000 customers across 4 regions, 13 countries, 3 segments and 4 plans, with 20 sales reps.
- **Tables:** 8 business tables (customers, subscriptions, usage, CRM opportunities, support tickets,
  campaigns, daily revenue and feature adoption), about 2.8 million rows in total.
- **Hidden mechanism:** a latent customer-health variable drives usage, support load, churn and
  expansion, but is never stored.
- **Injected events:** seven business events serve as evaluation ground truth. They are kept outside
  the database and are never visible to the agent.

## Repository layout

```
app/
  database/        schema metadata, DDL, loader, read-only DuckDB backend, lineage
  analytics/       KPI registry and engine, periods, dimensions, analytics modules
  timeseries/      monthly series from the KPIs (cutoff-bounded)
  forecasting/     baselines, ETS, backtests, model selection
  anomalies/       rolling z-score, IQR and forecast-residual detectors
  tools/           the 12 typed tools, allow-listed registry, SQL safety
  evidence/        evidence and claim models, graph, evidence and response validators
  llm/             provider-neutral interface, prompts, schemas, deterministic and Anthropic providers
  agent/           LangGraph state machine, request validation, claim builders, runner
  investigation/   templates, planner, step execution, cross-finding validation, drivers, recommendations, briefs
  security/        limits, input guard, injection screen, authorisation, data-exposure policy, budgets, secured executor
  mcp/             MCP server (stdio): registry, adapters, schemas, errors
  api/             FastAPI app, schemas, service, auth and rate limiting, presenters, chart specs
  ui/              Streamlit page, HTTP client, view models, rendering, launcher
evals/             evaluation harness: eval_v1 and eval_v2 datasets, references, runners, graders, metrics, reports
data/              reproducible synthetic-data generator, metadata, evaluation-only ground truth
docs/              documentation (see the table above)
scripts/           smoke_test.py for a running deployment
tests/             unit, integration, security, MCP, API, UI, deployment and evaluation tests
Dockerfile, docker-compose.yml, .github/workflows/ (ci.yml, evaluation.yml)
```

## Development history

The project was built in phases, each ending with tests, benchmarks and documentation:

- **Phases 0–4:** plan, data generator, KPI and analytics engine, forecasting and anomaly detection,
  and the LangGraph agent with its tools and evidence layer.
- **Phase 5:** security and guardrails.
- **Phase 6:** MCP.
- **Phases 7 and 7.1:** the evaluation benchmark, and the reliability fixes it drove.
- **Phase 8:** API and UI.
- **Phase 9:** production hardening and Docker.
- **Phase 10:** investigations and decision briefs.
- **Phase 11:** final polish, documentation and release validation.

Notes for every phase: [`docs/implementation-plan.md`](docs/implementation-plan.md).

## License

MIT, see [LICENSE](LICENSE).
