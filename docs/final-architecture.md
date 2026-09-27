# AgentOps AI: final architecture

This is the architecture of the finished system (version 0.10.0), as implemented. Each section links to
the detailed document for its layer.

## 1. System overview

AgentOps AI is an evidence-grounded business-intelligence and decision-intelligence agent for a
fictional B2B SaaS company. A user asks a question ("What was revenue in July compared with June?") or
states an objective ("Why is revenue growth slowing?"). The system plans analyses, runs them with
deterministic tools over a read-only database, turns the results into evidence, validates every claim
against that evidence, and returns an answer or a decision brief. Every number in either is traceable
to a query.

The design rule throughout is **the model can propose; the application decides.** A language model (a
deterministic offline model by default, or Claude when configured) interprets the question and, for
`/ask`, proposes a plan. It never produces a business number, never runs a tool directly, and cannot
widen its own permissions.

```
                ┌──────────────────────┐        ┌────────────────────────────┐
  Browser ────▶ │ Streamlit UI         │  HTTP  │ FastAPI (app/api)          │
                │ app/ui (no data,     │ ─────▶ │ auth · rate limit · limits │
                │ no calculations)     │ bearer │ request IDs · safe errors  │
                └──────────────────────┘ token  └──────────┬─────────────────┘
                                                  /ask     │     /investigations
                                    ┌──────────────────────┴───────────────────┐
                                    ▼                                          ▼
                     ┌───────────────────────────┐          ┌─────────────────────────────────┐
                     │ LangGraph agent           │          │ Investigator                    │
                     │ app/agent                 │          │ app/investigation               │
                     │ one question → one answer │          │ one objective → decision brief  │
                     └─────────────┬─────────────┘          └────────────────┬────────────────┘
                                   │  tool calls (proposed)                  │ template steps
                                   ▼                                         ▼
   MCP client ─stdio─▶ app/mcp ─▶ ┌──────────────────────────────────────────────────────────┐
                                  │ SecuredToolExecutor (app/security/execution.py)          │
                                  │ authorize → deadline → execute → retry → validate output │
                                  │ → charge budget → audit event                            │
                                  └──────────────────────────┬───────────────────────────────┘
                                                             ▼
                                  ┌──────────────────────────────────────────────────────────┐
                                  │ 12 allow-listed tools (app/tools)                        │
                                  │ → analytics · forecasting · anomalies (app/analytics …)  │
                                  └──────────────────────────┬───────────────────────────────┘
                                                             ▼
                                               read-only DuckDB (Northwind Cloud)
                   results ─▶ evidence (fingerprinted) ─▶ claims ─▶ validators ─▶ answer / brief
```

## 2. Data layer

- **Dataset:** Northwind Cloud, a reproducible synthetic Singapore B2B SaaS company. It covers 5,000
  customers across 8 tables (about 2.8 million rows), from September 2024 to August 2026, with a
  business as-of date of 2026-08-31. The generator is `data/generator` (seed 42 by default); it writes
  a DuckDB file, a manifest with checksums, and a data dictionary.
- **Access:** the application opens the database **read-only**, through one connection (`app/database`).
- **Hidden mechanism:** a latent customer-health variable drives the data but is never stored.
- **Ground truth:** seven injected business events are recorded in `data/seeds/injected_events.json`.
  They are for evaluation only: no application code reads them, and static and runtime tests enforce
  this.

Details: [data-dictionary.md](data-dictionary.md), `data/README.md`.

## 3. Analytics layer

The analytics layer is the **only source of business numbers**:

- **KPIs:** a typed registry of 20 KPIs, each with formula, SQL, unit, grain, limitations and
  dependencies.
- **Analytics modules:** revenue decomposition and MRR bridge, churn, cohorts, customer risk, sales,
  marketing, support and product.
- **Forecasting:** baselines and damped ETS, with rolling-origin backtests and a naive-baseline gate.
- **Anomaly detection:** rolling z-score, IQR and forecast-residual detectors, cutoff-bounded (no
  look-ahead).

Every result carries its SQL, parameters, source tables and lineage IDs. Everything is checked against
an independent pandas reference.

The 12 **tools** in `app/tools` are the only interface the agent, the investigator and MCP have to this
layer: `get_kpi`, `analyze_revenue`, `analyze_customers`, `analyze_sales`, `analyze_marketing`,
`analyze_support`, `analyze_product`, `get_cohort_analysis`, `get_customer_risk`, `forecast_metric`,
`detect_anomalies` and `run_safe_sql`. `run_safe_sql` is one read-only `SELECT` over allow-listed
tables and columns.

Details: [analytics.md](analytics.md), [kpi-catalog.md](kpi-catalog.md), [forecasting.md](forecasting.md),
[anomaly-detection.md](anomaly-detection.md).

## 4. Agent runtime (single questions)

`app/agent` is a LangGraph state machine:

```
question_received → understand_question → validate_request → plan_investigation → execute_tools
  → collect_evidence → (plan again, at most 2 iterations) → validate_evidence → generate_response
  → validate_response → done
```

It has typed terminal states: `unsupported_request`, `insufficient_evidence`, `planning_failure`,
`tool_error` and `validation_failure`. When a budget, deadline or cancellation stops a run, the run is
flagged `limit_reached` and returns a controlled response.

- **Understanding.** The model's understanding output (intent, metric, periods, dimensions, filters)
  is checked by the central request validation. Periods outside the data are resolved or rejected, and
  ambiguous periods are clarified.
- **Plans.** A plan is a list of proposed tool calls. The plan validator checks each one, and the
  secured executor authorises each one again before it runs.
- **Drafts.** Response drafts cite claim IDs. A draft is validated for supported numbers, KPI names,
  direction, causal wording, forecast certainty and labels, and is regenerated or failed if it does
  not pass.
- **Models.** `LLM_PROVIDER=deterministic` (the default) is an offline rule-based model: no key, no
  network, fully reproducible. `LLM_PROVIDER=anthropic` uses Claude through the official SDK.

Details: [agent-architecture.md](agent-architecture.md).

## 5. Investigation runtime (objectives)

`app/investigation` answers an objective with several analytical steps. It reuses the agent's runtime:
the same input guard, understanding step, request validation, secured executor, evidence builder and
validators. It adds orchestration only.

- **Plan.** One of six fixed templates (revenue, customer, sales, product & support, general,
  management brief) is chosen by brief cues, then the metric, then the intent. Each step names one
  tool and the intent it is authorised under.
- **Steps.** Conditions and evidence-bound arguments are closed sets evaluated in code. An identical
  call is reused, never run twice.
- **Model role.** The model never chooses a tool: its only output is the understanding, which selects
  a template.
- **Budgets.** Steps, tool calls, runtime, evidence and output size are capped in code. A stop is
  reported as `budget_exhausted`, never `completed`.
- **Synthesis.** Findings are validated against each other; drivers (§15) and grounded
  recommendations are derived by rules; the decision brief is composed and size-capped.

Details: [investigations.md](investigations.md).

## 6. Evidence layer

`app/evidence` holds `Evidence`, `Claim` and the `EvidenceGraph`:

- **Evidence.** Each tool result becomes evidence items: value, unit, period, comparison, filters,
  dimension, source tables, calculation, query IDs and the tool call. Each item is sealed with a
  SHA-256 fingerprint when it enters the graph.
- **Claims.** Claims are built by rule-based claim builders from evidence, never from model text. Each
  claim is typed (observed fact, calculated result, inference, recommendation) and has a structured
  subject: metric, unit, period, comparison, dimension, member and filters.
- **Evidence validator.** Checks provenance, successful calls, numbers, direction, the claim's subject
  against its evidence (the Phase 7.1 identity rules), and causal wording.
- **Response validator.** Checks every text against the claims it cites.

Details: [agent-architecture.md](agent-architecture.md) (evidence sections),
[phase-7-1-reliability.md](phase-7-1-reliability.md).

## 7. Security and execution layer

`app/security` contains the input guard (types, sizes, secret redaction) and a deterministic
prompt-injection screen. Every tool call from the agent, the investigator and MCP goes through one
`SecuredToolExecutor`:

1. **Authorisation:** the allow-list, the tools permitted for the validated intent, argument
   validation, and the data-exposure policy (withheld and personal columns, restricted breakdowns).
2. **Deadline** and per-tool timeout.
3. **Execution.**
4. **Bounded retries.**
5. **Output validation.**
6. **Budget charge:** tool calls, SQL calls and rows, retries, model calls, context and response size,
   wall clock.
7. **Audit:** a typed `SecurityEvent`.

SQL is parsed and restricted to one read-only `SELECT` with complexity limits, bound parameters, a row
cap and a statement timeout.

Details: [security.md](security.md), [security-architecture.md](security-architecture.md),
[security-threat-model.md](security-threat-model.md).

## 8. MCP integration

`app/mcp` is a stdio MCP server (official Python SDK) exposing the same 12 tools as `agentops_*`. It is
a thin adapter with no analytics, SQL or permissions of its own. Each call goes through the same
`SecuredToolExecutor`, and results keep their evidence and provenance.

MCP parity is part of the benchmark: 20 eval_v1 scenarios check that MCP returns exactly what the
direct path returns, and eval_v2 checks each investigation step through MCP. Investigations are
deliberately **not** an MCP tool (see [investigations.md §10](investigations.md#10-api)).

Details: [mcp-architecture.md](mcp-architecture.md).

## 9. API layer

`app/api` is a FastAPI application. Its endpoints:

| Endpoint | Access |
|---|---|
| `POST /api/v1/ask`, `/ask/stream` | Token; rate-limited |
| `POST /api/v1/investigations`, `/investigations/stream` | Token; rate-limited (same quota as `/ask`) |
| `GET /api/v1/health` | Public (liveness) |
| `GET /api/v1/readiness` | Public |
| `GET /api/v1/capabilities` | Token |
| `GET /api/v1/metrics` | Token |

The API adds a transport and nothing else:

- **Access control:** bearer-token authentication with constant-time comparison, and a per-client
  sliding-window rate limit.
- **Request limits:** JSON only, a body limit, question and objective limits, and unknown fields
  rejected.
- **Traceability and errors:** request IDs, safe error envelopes, security headers, structured JSON
  logs without content, and in-process metrics.
- **Running work:** one worker thread with a queue bound, a request timeout that cancels the run, and
  graceful shutdown.

The API has no SQL, tools or permissions of its own.

Details: [api.md](api.md), [deployment.md](deployment.md).

## 10. UI layer

`app/ui` is a single Streamlit page and a thin HTTP client of the API. It never opens the database or
imports agent code.

- **Modes.** Two modes: *Ask a question* and *Investigate a business issue*.
- **Rendering.** It renders what the API returns: the answer or decision brief, typed findings, KPI
  cards, Vega-Lite charts, forecast and anomaly panels, the evidence and provenance table, and the
  analysis trace.
- **Safety.** All text is escaped. Session history is in memory, bounded and redacted.

Details: [ui.md](ui.md).

## 11. Evaluation layer

`evals/` is outside the application: production never imports it. It runs the real paths (the agent,
the investigator, the API, the UI view models, the secured executor and the MCP server) and grades
structured behaviour against independent references:

- **eval_v1:** 89 single-question, security and MCP scenarios.
- **eval_v2:** 77 investigation scenarios.

The hidden labels are read only in `evals/reference`.

Details: [evaluation.md](evaluation.md).

## 12. Deployment

- **Images.** A multi-stage `Dockerfile` builds two images:
  - **API image:** the agent and FastAPI.
  - **UI image:** only Streamlit, httpx and `app/ui`; it contains no database, agent or data code.
- **Compose.** `docker-compose.yml` runs both containers:
  - as a non-root user (uid 10001), with a read-only root filesystem, no Linux capabilities and
    `no-new-privileges`;
  - with ports published on 127.0.0.1 only;
  - with the database mounted read-only;
  - with readiness-based healthchecks.
- **Production start-up.** `APP_ENV=production` refuses to start with an insecure configuration: no
  token, rate limit off, wildcard or plain-HTTP CORS, an implicit database path, or timeouts that do
  not nest.
- **CI.** `ci.yml` runs lint, format, types, pytest, the critical suites of both benchmarks, and a
  Docker build and smoke test. `evaluation.yml` runs the full benchmarks and the multi-seed checks.

Details: [deployment.md](deployment.md).

## 13. Request lifecycle (`/ask`)

1. The client sends `POST /api/v1/ask` with a bearer token.
2. The middleware assigns a request ID, enforces the body limit and security headers, authenticates,
   checks the content type and applies the rate limit.
3. The route validates the question (not empty, not too long).
4. `AgentService` queues the run on the worker, with a deadline and a cancel event.
5. `AgentRunner.run` screens the question: blocked requests are refused here, before any model call.
6. The model understands the question, and the request is validated.
7. The model proposes a plan. The plan validator checks it and the secured executor runs each call.
8. Results become fingerprinted evidence, then claims, then pass evidence validation.
9. The model drafts the answer citing claim IDs. The draft is validated, and regenerated or failed if
   it does not pass.
10. The presenter copies the answer, claims, evidence, trace and chart specs into `AskResponse`. It
    recomputes nothing.
11. The request is logged (IDs, outcome, durations, never content) and the metrics are updated.

## 14. Investigation lifecycle (`/investigations`)

1. Same transport as §13; the objective shares the question limits and the rate-limit quota.
2. **Screen**, then the agent's **understanding** step and request validation.
3. **Plan:** a template's steps, with default periods recorded as assumptions. The plan is streamed.
4. **Execute:** each step goes through dependencies, conditions, bindings, reuse and budget checks,
   then the `SecuredToolExecutor` under the template's intent, never with ad-hoc SQL.
5. **Validate:** the claims pass the evidence validator; findings, relationships, drivers and
   recommendations pass cross-finding validation (removed or downgraded, never repaired).
6. **Synthesise:** the decision brief (validated summary, key findings, drivers, contradictions, risks,
   recommendations, uncertainty) is composed and size-capped.
7. The presenter reuses `/ask`'s views for the trace, KPIs, charts, forecasts and anomalies.

## 15. Trust and validation model

| Label | Meaning | Where it comes from |
|---|---|---|
| **Observed** | Read directly from recorded data | A claim builder over an observed evidence item |
| **Calculated** | Computed from recorded data by a registered calculation | A claim builder over calculated evidence (KPIs, changes, shares) |
| **Inferred** | A reading of the evidence, not an observed fact ("was concentrated in") | Rule-based inference builders, worded as inference |
| **Recommended** | A suggested next step, not a finding | Rule-based recommendation builders that cite their findings |

- Numbers come only from tools. A number in any text that is not in the cited evidence fails
  validation.
- Causal wording ("caused by", "because of", "due to", "led to", …) is rejected unless negated.
- Causal questions return `insufficient_evidence` with the observed findings.
- Drivers are typed as contributions, co-movements, associations, contradictions or context, never as
  causes.
- Forecasts are labelled estimates with intervals and backtests. Anomalies are "statistically
  unusual", not good or bad.
- Evidence is fingerprinted, and claims carry a structured subject that must match their evidence.
- Nothing unvalidated is shown: a failed or stopped run returns a controlled outcome, not a partial
  guess.
