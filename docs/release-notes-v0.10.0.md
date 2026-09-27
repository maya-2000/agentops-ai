# AgentOps AI v0.10.0

The final release of AgentOps AI, an evidence-grounded business intelligence and decision intelligence
agent.

## What it is

AgentOps AI turns business questions in plain language into evidence-backed analytics and multi-step
investigations. Three layers share the work:

- **The language model** interprets the question and orchestrates the work.
- **Deterministic analytics tools** compute every business number.
- **Validators** check every claim against recorded evidence before it is shown.

It runs fully offline with the default deterministic model. Claude can be used through the official
Anthropic SDK.

## Capabilities

- **Questions (`POST /ask`)** about the fictional company's data:
  - KPIs, comparisons and breakdowns;
  - customer risk;
  - forecasts and anomaly checks.
- **Investigations (`POST /investigations`):**
  - six templates: revenue, customer, sales, product and support, general, management brief;
  - multi-step plans, with findings validated against each other;
  - rule-based drivers: accounting contributions, co-movements, associations and contradictions;
  - grounded recommendations and a decision brief.
- **Analytics:**
  - 20 registered KPIs and 8 analytics modules, checked against an independent pandas reference;
  - forecasting with baselines and damped ETS, chosen by backtest, with 95% intervals;
  - rolling z-score, IQR and forecast-residual anomaly detectors.
- **Interfaces:**
  - a FastAPI service with NDJSON progress streams, health, readiness and metrics;
  - a Streamlit UI with *Ask* and *Investigate* modes;
  - an MCP server (stdio) exposing the 12 tools as `agentops_*`.

## Architecture highlights

- A LangGraph state machine runs single questions: understand, validate, plan, execute, collect
  evidence, validate evidence, respond, validate the response.
- The investigator reuses the same runtime components and runs template steps.
- **One path to the data.** Every tool call, from the agent, the investigator or MCP, goes through
  `SecuredToolExecutor`: authorize, deadline, execute, retry, validate output, charge budget, audit.
- **Deterministic analytics** over a read-only DuckDB database are the only source of numbers.

See [final-architecture.md](https://github.com/maya-2000/agentops-ai/blob/main/docs/final-architecture.md) and
[engineering-decisions.md](https://github.com/maya-2000/agentops-ai/blob/main/docs/engineering-decisions.md).

## Evidence and validation model

- **Evidence.** Every tool result becomes an evidence item with provenance (query IDs, source tables,
  calculation, period, filters) and a SHA-256 fingerprint.
- **Claims.** Every claim is labelled *Observed*, *Calculated*, *Inferred* or *Recommended*, and cites
  its evidence.
- **Validation.** Validators check that a claim's numbers, direction, metric, period, dimension and
  unit match its evidence. Claims that fail are removed.
- **No causal claims.** Causal wording is rejected. Questions that ask for a cause return
  `insufficient_evidence`, together with the observed findings.

## Security

- **API access:**
  - bearer-token authentication, checked in constant time;
  - a per-client rate limit shared by `/ask` and `/investigations`;
  - JSON-only requests with size limits.
- **Screening and authorization:**
  - a prompt-injection screen and an input guard run before any model or tool call;
  - tool authorization checks the allow-list, the intent, the arguments and the data-exposure policy.
- **SQL:** one read-only `SELECT` over allow-listed tables and columns, with complexity limits, a row
  cap and a timeout.
- **Budgets:**
  - agent budgets cap tool calls, retries, SQL, model calls, context, response size and time;
  - investigation budgets cap runs at 14 steps, 16 tool calls, 120 s, 400 evidence items and 12,000
    output characters.
- **Logging and ground truth:**
  - structured logs contain no questions, answers, tokens or tracebacks;
  - the evaluation ground truth is isolated from the application.

## MCP

- `python -m app.mcp` (or `agentops-mcp`) serves the 12 analytics tools over stdio, using the official
  Python SDK.
- Calls go through the same secured executor and return the same evidence and provenance as direct
  calls.
- There are no file, shell or code tools.

## Docker

- Two images, built in a multi-stage Dockerfile: `agentops-api:0.10.0` and `agentops-ui:0.10.0`.
- docker compose runs them with:
  - a non-root user, a read-only root filesystem and all Linux capabilities dropped;
  - `no-new-privileges` and ports bound to localhost only;
  - health checks, a read-only data mount and graceful shutdown.
- Images contain no secrets and no data.

## Validation results

These are project validation and regression results on the synthetic dataset. They are not a general
measure of AI reliability.

| Check | Result |
|---|---|
| Test suite (pytest) | 2,988 passed, 0 failed, 0 skipped |
| Static checks | ruff, ruff format and mypy clean |
| eval_v1: single questions, security, MCP (89 scenarios) | 89/89; critical suite 13/13; multi-seed 22/22 runs |
| eval_v2: investigations (77 scenarios, 18 categories) | 77/77; critical suite 12/12; multi-seed 24/24 runs |
| Security benchmark | 32/32 |
| MCP benchmark | 23/23; direct-vs-MCP parity 100% (20 scenarios, 47 calls) |
| Docker smoke test | 15/15; both services healthy |

## Known limitations

- **Data.** Synthetic data from one company.
- **Language coverage.** The deterministic model understands a limited set of phrasings.
- **Investigations.** Six fixed templates. Drivers are accounting shares and co-movements, not
  significance-tested causes.
- **Deployment.** A single process: rate limits and metrics are held in memory, and nothing is stored.
- **Access.** One shared service token, with no user accounts.
- **Benchmarks.** They are regression checks on this dataset.

## Getting started

See the [README](https://github.com/maya-2000/agentops-ai/blob/main/README.md#quick-start) and the [demo guide](https://github.com/maya-2000/agentops-ai/blob/main/docs/demo.md).
