# MCP architecture (Phase 6)

AgentOps exposes its deterministic analytics tools through the
[Model Context Protocol](https://modelcontextprotocol.io) (MCP), so that any MCP client (an IDE
assistant, a desktop assistant or another agent) can discover and call them. The server is a
thin adapter over the existing layers. It adds no analytics, no SQL and no security rules of its
own: every call goes through the same Phase 5 execution path as the LangGraph agent's tool calls.

- Code: [`app/mcp/`](../app/mcp)
- Tests: [`tests/mcp/`](../tests/mcp)
- SDK: the official MCP Python SDK (`mcp` 2.x), low-level `Server`, stdio transport

Contents:

1. [What MCP is doing in AgentOps](#1-what-mcp-is-doing-in-agentops)
2. [Why MCP is an adapter layer](#2-why-mcp-is-an-adapter-layer)
3. [MCP server architecture](#3-mcp-server-architecture)
4. [Tool catalog](#4-tool-catalog)
5. [Tool schemas](#5-tool-schemas)
6. [Authentication and authorization boundary](#6-authentication-and-authorization-boundary)
7. [Phase 5 security integration](#7-phase-5-security-integration)
8. [Evidence preservation](#8-evidence-preservation)
9. [Error model](#9-error-model)
10. [Transport](#10-transport)
11. [Local setup (quickstart)](#11-local-setup-quickstart)
12. [Client usage](#12-client-usage)
13. [Security limitations](#13-security-limitations)
14. [Example requests and responses](#14-example-requests-and-responses)
15. [Future integration possibilities](#15-future-integration-possibilities)

Then: [MCP and LangGraph](#mcp-and-langgraph), [configuration](#configuration),
[performance](#performance) and [testing](#testing).

---

## 1. What MCP is doing in AgentOps

MCP is the **standard external tool interface**. It lets a client do three things:

- **Discover** the twelve read-only analytics capabilities (`tools/list`), each with a complete
  description, a typed input schema and a typed output schema.
- **Invoke** them (`tools/call`) with validated arguments.
- **Receive** structured results that keep the Phase 4 evidence and provenance: which tool
  calculated what, from which tables and queries, for which period and filters, and with which
  limitations.

MCP does not reason, plan or write answers. In AgentOps these jobs belong to the LangGraph
agent (see [MCP and LangGraph](#mcp-and-langgraph)).

## 2. Why MCP is an adapter layer

The business logic already exists and is tested:

- the Phase 2 KPI and analytics services;
- the Phase 3 forecasting and anomaly services;
- the Phase 4 tools and evidence builder;
- the Phase 5 security controls.

A second implementation behind MCP would drift from the first, and every drift would be a
correctness or security bug. So the MCP layer only translates between the protocol and the
existing interfaces:

| Concern | Owned by | MCP layer does |
|---|---|---|
| KPI, revenue, churn, forecast, anomaly and risk formulas | Phase 2/3 services | Nothing (no imports of the services; tested statically) |
| Tool inputs and handlers | Phase 4 `ToolDefinition`s | Exposes the same input models as schemas |
| Authorization, argument policy, budget, deadlines, retries, output validation | Phase 5 `SecuredToolExecutor` | Calls it |
| SQL parsing and validation | `app/tools/sql_safety.py` | Nothing (only `agentops_run_safe_sql` reaches SQL, through the tool) |
| Evidence and provenance | Phase 4 evidence builder and validator | Calls them and returns the items unchanged |
| Redaction, error sanitisation, audit events | Phase 5 | Calls them; maps error codes to MCP categories |

Phase 6 made one refactor to guarantee this. The secured tool-execution sequence that used to be
inline in the agent's `execute_tools` node is now `app/security/execution.py`
(`SecuredToolExecutor`), and both the agent and the MCP adapter call it. The agent's behaviour
is unchanged: all earlier tests pass unmodified.

## 3. MCP server architecture

```
MCP client ──(JSON-RPC over stdio)──▶ app/mcp/server.py        low-level mcp.server.Server
                                        │  tools/list  → MCPToolRegistry.tools()
                                        │  tools/call  → worker thread, one call at a time
                                        ▼
                                      app/mcp/adapters.py      MCPToolService.call()
                                        │  1. name and request-size checks, new run ID
                                        │  2. parameter screen (audit only; parameters are data)
                                        ▼
                                      app/security/execution.py  SecuredToolExecutor (Phase 5)
                                        │  authorize → deadline → execute → retry → validate output → charge budget
                                        ▼
                                      app/tools/registry.py    Phase 4 tool → Phase 2/3 service / safe SQL
                                        ▼
                                      app/evidence/            build_evidence + validate_evidence (Phase 4/5)
                                        ▼
                                      app/mcp/schemas.py       MCPToolOutput → mask withheld fields,
                                                               redact secrets, bound size → CallToolResult
```

| Module | Responsibility |
|---|---|
| `registry.py` | The single catalogue: the twelve `agentops_*` tools, each bound to its Phase 4 definition, with its intent, output kind, composed description, schemas and version metadata |
| `schemas.py` | `MCPToolOutput`, the response envelope and `outputSchema` of every tool; `ForecastView` and `AnomalyReportView` (defined in `app/tools/views.py` since Phase 8, shared with the HTTP API, and re-exported here) |
| `adapters.py` | `MCPToolService`: one call in, one validated, redacted, size-bounded output out; request-scoped state |
| `errors.py` | The MCP error categories and fixed messages, derived from the Phase 5 categories |
| `config.py` | `MCPServerConfig` from settings |
| `audit.py` | The `agentops.mcp` audit log |
| `server.py` | The low-level `Server`, its lifespan, and `serve_stdio` |
| `__main__.py` | `python -m app.mcp` / `agentops-mcp` |

**Why the low-level server.** The SDK's high-level server (`MCPServer`, formerly `FastMCP`)
derives schemas from function signatures. AgentOps already has its input schemas as Pydantic
models, so the low-level server exposes those models directly and nothing is declared twice.

**Lifecycle.**

- *Startup.* `create_server` builds the tool registry and the Phase 5 policy eagerly, so a
  misconfiguration fails before anything is opened. The lifespan then opens the database
  read-only through the existing `get_database()` abstraction. `python -m app.mcp` first checks
  that the database exists and exits with a clear message if it does not.
- *Shutdown.* When the client closes the connection, the lifespan closes the database. Both
  events are audited.
- *Resources.* No background threads or processes are started. The only long-lived resource is
  the one read-only database handle, owned by the lifespan.

**Concurrency.** The DuckDB connection is not safe for concurrent use. Each `tools/call` runs
in a worker thread, so it never blocks the event loop, and under an `anyio.Lock`, so calls run
one at a time. Everything else is request-scoped:

- a new run ID;
- a new budget (`RunBudget` / `BudgetUsage`);
- a new evidence graph;
- its own security events and audit record.

The service holds only immutable configuration, the policies and the database handle. There is
no global mutable state.

## 4. Tool catalog

Names are stable: `agentops_` followed by the Phase 4 tool name. The internal names
(`get_kpi`, …) are not callable over MCP.

| MCP tool | Phase 4 tool | Output kind | Required inputs | Authorized under intent |
|---|---|---|---|---|
| `agentops_get_kpi` | `get_kpi` | observed | `kpi` | `kpi_lookup` |
| `agentops_analyze_revenue` | `analyze_revenue` | observed | `operation` | `revenue_investigation` |
| `agentops_analyze_customers` | `analyze_customers` | observed | `operation` | `customer_investigation` |
| `agentops_analyze_sales` | `analyze_sales` | observed | `operation` | `sales_analysis` |
| `agentops_analyze_marketing` | `analyze_marketing` | observed | `operation` | `marketing_analysis` |
| `agentops_analyze_support` | `analyze_support` | observed | `operation` | `support_analysis` |
| `agentops_analyze_product` | `analyze_product` | observed | `operation` | `product_analysis` |
| `agentops_get_cohort_analysis` | `get_cohort_analysis` | observed | none | `customer_investigation` |
| `agentops_get_customer_risk` | `get_customer_risk` | risk score | none | `customer_investigation` |
| `agentops_forecast_metric` | `forecast_metric` | forecast | `metric` | `forecast` |
| `agentops_detect_anomalies` | `detect_anomalies` | anomaly | `metric` | `anomaly_detection` |
| `agentops_run_safe_sql` | `run_safe_sql` | ad-hoc query | `sql` | `mixed_investigation` |

Each description is composed from the Phase 4 definition and the live vocabularies (KPI keys,
series metrics, detectors, dimensions, approved tables). It states:

- what the tool does, and when to use it and when not;
- the required and optional inputs;
- the supported dimensions;
- what the output means and whether it is observed data, a forecast, an anomaly score or a
  rule-based score;
- the limitations;
- the safety constraints.

Run `python -m app.mcp --list-tools` to see the enabled catalogue.

There are no file, directory, shell, Python, environment or database-file tools, and none can be
registered at runtime. The catalogue is a fixed tuple, and a test fails if a name suggesting
machine access ever appears.

## 5. Tool schemas

**Inputs.** Each tool's `inputSchema` is its Phase 4 Pydantic input model's JSON schema, exactly.
The schemas have `additionalProperties: false`, and they include the enums (`operation`,
`detector`, `transform`, `grain`, `min_band`), bounds (`limit`, `max_rows`, `horizon`,
`confidence_level`) and date formats. Before any tool runs:

- the MCP layer checks the tool name, that the arguments are an object, and the request size;
- the Phase 5 authorization then validates the arguments against the same model (so unknown
  fields are rejected) and applies the central argument policy.

Malformed input is rejected before business logic; a test proves this with spy handlers for
all twelve tools.

**Output.** Every tool returns an `MCPToolOutput` as `structuredContent`. The same JSON is also
sent as a text block, for clients without structured-content support. MCP clients validate the
structured content against the published `outputSchema`, and the tests do too.

| Field | Meaning |
|---|---|
| `schema_version` | Envelope version (`1.0`) |
| `tool_name` | The MCP tool that was called (`unknown` if the name was not a valid identifier) |
| `request_id` | The run ID: the correlation key for the audit log and security events |
| `status` | `ok`, `no_data`, `insufficient_data`, `insufficient_history` (the tool's own status), or `error` |
| `output_kind` | `observed`, `risk_score`, `forecast`, `anomaly` or `ad_hoc_query` |
| `result_type`, `result` | The Phase 2/3 typed result, serialised, with two changes. The internal SQL text of analytics queries is omitted (query IDs, row counts and lineage stay). Withheld fields are masked by the data-exposure policy |
| `forecast` | Forecasts only: metric, forecast period, cutoff, horizon, model, points (predicted value, interval bounds), interval method and level, backtest and naive-baseline metrics |
| `anomalies` | Anomaly detection only: metric, detector, window, threshold, evaluation window, severity counts, and each flagged month's observed and expected values, deviation, score, severity, direction and explanation |
| `evidence` | Phase 4 evidence items (see §8) |
| `provenance` | Tool, source layer, operation, call ID, canonical arguments, query IDs, source tables, calculation, execution time, attempts, dataset version, as-of date, toolset version |
| `query_id` | The last query ID of the call |
| `warnings` | For example: forecast/anomaly/risk labelling, evidence-status notes, truncation, retries |
| `limitations` | The tool's own limitations |
| `error` | `category`, `code`, fixed `message`, sanitised `detail` (argument errors only), `retryable` |
| `truncated` | Parts were dropped to respect the response size limit (named in `warnings`) |

**Versioning.** Each tool carries simple metadata in `_meta`: `io.agentops/version`,
`io.agentops/toolset_version`, `io.agentops/output_kind` and `io.agentops/source_layer`. A future
incompatible change would bump the tool version, and add a new name if both versions must be
served. There is no framework beyond this.

## 6. Authentication and authorization boundary

- **Authentication.** There is none in Phase 6, by design. The server speaks stdio only, so it
  runs as a child process of the local user's MCP client, with that user's permissions. There is
  no network listener. Authentication, OAuth and multi-tenant permissions belong to a later
  networked deployment.
- **Authorization.** This is the Phase 5 `ToolAuthorizationPolicy` with the Phase 5 permission
  table. MCP has no permission table of its own:
  - Each MCP tool declares the intent it is authorized under (table above). The registry checks,
    at build time, that `INTENT_TOOL_PERMISSIONS` permits the tool for that intent, and the
    policy checks it again on every call.
  - Tools switched off for MCP (`MCP_ENABLED_TOOLS`) or for the agent (`AGENT_DISABLED_TOOLS`)
    go into the policy's disabled set. They are neither listed nor callable.
  - `MCP_SQL_ENABLED=false` unlists `agentops_run_safe_sql` and also revokes the SQL privilege
    in the authorization context, the same mechanism the agent uses for flagged questions.
  - Nothing a client sends can change a permission. MCP request metadata (`_meta`) is ignored,
    and argument fields such as `role`, `intent`, `limits` or `_meta` are unknown fields and
    are rejected.

## 7. Phase 5 security integration

The path of every call:

```
MCP request
  → MCP input checks (tool name, argument object, request size ≤ MCP_MAX_REQUEST_BYTES)
  → Phase 5 authorization (allowlist → enabled → intent → SQL privilege
                           → typed arguments → argument/data-exposure policy → budget → prerequisites)
  → Phase 4 tool under the Phase 5 tool deadline (SQL under its own statement deadline)
  → deterministic Phase 2/3 service or safe SQL
  → Phase 5 retry policy (transient database errors only)
  → Phase 5 tool-output validation (provenance, finite numbers, dates, hidden-state / PII keys,
                                    KPI, forecast, anomaly and SQL-result checks)
  → Phase 4 evidence + Phase 5 evidence validation (fingerprints, integrity)
  → data-exposure masking, secret redaction, response size bound
  → MCP response
```

| Phase 5 control | How MCP inherits it |
|---|---|
| Authorization and argument policy | `SecuredToolExecutor` → `ToolAuthorizationPolicy.authorize` |
| SQL safety | `run_safe_sql` → `validate_sql`: SELECT-only, table and column allowlists, withheld and PII columns, function allowlist, no files, system tables or extensions, complexity and length limits, bound parameters, row cap, statement timeout, read-only connection |
| Resource budget | A fresh `RunBudget` per request. One tool call; SQL rows ≤ `AGENT_SQL_ROW_LIMIT`; retries ≤ `AGENT_MAX_RETRIES`; time ≤ `AGENT_TOOL_TIMEOUT_SECONDS`; customer-level rows ≤ `AGENT_MAX_CUSTOMER_ROWS` |
| Output validation | `ToolOutputValidator` before evidence is built; a rejected output is discarded |
| Secret redaction | Every string in the response is passed through `redact_value`; the audit log is redacted too |
| Error sanitisation | Fixed messages; `sanitize_detail` for the argument feedback; no exception text |
| Data exposure | `mask_withheld_fields` (in `app/security/data_policy.py`) applies the same withheld / PII column lists to every result that leaves the process. `company_name` is masked; the only PII allowed is where Phase 5 allows it (`analyze_sales.rep_performance`) |
| Prompt injection | Parameters are data: they are validated like any value and never interpreted. The Phase 5 injection detector additionally flags instruction-like text in parameters as a `suspicious_prompt` audit event, without changing the decision |
| Audit | Phase 5 `SecurityEvent`s (`tool_authorized`, `tool_denied`, `sql_rejected`, …) with the request's run ID, plus one `agentops.mcp` record per call |

**Ground truth.** `data/seeds/injected_events.json` and the generator's hidden state are not in
the database, and no MCP tool reads files. The SQL tool refuses file functions, file paths as
tables, and hidden-state column or table names. A test reads the generated ground truth and
checks that none of its event names or descriptions appear in any tool's output.

**Audit record.** One per call, on the `agentops.mcp` logger (stderr), for example:

```json
{"event": "tool_call", "request_id": "R-…", "tool_name": "agentops_get_kpi", "mcp_request_id": 3,
 "validation": "passed", "authorization": "allowed", "execution": "succeeded", "status": "ok",
 "error_category": null, "evidence_ids": ["E1"], "query_ids": ["Q-…"], "execution_time_ms": 16.4,
 "request_bytes": 35, "response_bytes": 5180, "truncated": false, "security_events": 1,
 "highest_severity": "INFO"}
```

The `request_id` equals the `run_id` of the call's security events and the `request_id` in the
response. A trace can therefore be followed from the MCP request through authorization, tool
execution and evidence to the response. Arguments, SQL text, rows and secrets are never logged.

## 8. Evidence preservation

Successful calls return the Phase 4 evidence items unchanged. Each item has:

- its ID (`E1`, …) and type (`observed`, `calculated`, `forecast`, `anomaly`, `derived`);
- a neutral statement;
- metric, value and unit;
- period and comparison period;
- filters and dimension;
- tool, call ID, operation and canonical input arguments;
- query IDs, source tables and calculation;
- execution timestamp and limitations;
- a SHA-256 fingerprint of its content.

A client can recompute the fingerprint to confirm that an item was not modified in transit.
Evidence is validated before it is sent (integrity, provenance). Forecast and anomaly results keep
their method metadata both in the evidence `details` and in the dedicated `forecast` /
`anomalies` views. Nothing is flattened into an untraceable string.

## 9. Error model

Errors are MCP tool results with `isError: true`, not protocol errors. The client, often a
model, can therefore read the error and correct its call.

| Category | Typical codes | Message |
|---|---|---|
| `INVALID_ARGUMENT` | `invalid_arguments`, `invalid_period`, `invalid_date_range`, `invalid_filter_value`, `unsupported_dimension`, `oversized_input` | The request arguments are invalid for this tool. *(plus a sanitised `detail`)* |
| `UNSUPPORTED_REQUEST` | `unsupported_kpi`, `unsupported_metric`, `unsupported_method` | The requested KPI is not supported. / The requested analysis is not supported. |
| `UNAUTHORIZED_TOOL` | `unknown_tool`, `tool_disabled`, `tool_not_permitted`, `sql_not_permitted` | The requested operation was rejected by the data-access policy. |
| `UNSAFE_QUERY` | `unsafe_sql`, `data_policy` | The requested operation was rejected by the data-access policy. |
| `RESOURCE_LIMIT` | `budget_exceeded`, `request_too_large`, `response_too_large` | The analysis exceeded the configured execution limit. |
| `TIMEOUT` | `timeout` | The analysis exceeded the configured execution limit. |
| `TOOL_FAILURE` | `database_error` (retryable), `calculation_error`, `analytics_error` | The analysis could not be completed because the analytics tool failed. |
| `VALIDATION_FAILURE` | `invalid_tool_output`, `evidence_integrity_failed` | The analysis could not be completed. |
| `INTERNAL_ERROR` | `internal_error`, any unknown code (fail closed) | The analysis could not be completed. |

The mapping is derived from the Phase 5 error categories, with a few codes refined. Only argument
errors carry a `detail`, which is:

- the first line only;
- with secrets redacted and paths replaced;
- with control characters removed;
- at most 240 characters.

Security rejections never explain which rule matched. That stays in the audit trail. Stack
traces, filesystem paths, SQL internals, raw database exceptions and environment values never
reach the client.

## 10. Transport

The server uses **stdio**, the SDK's local-development transport. The client starts
`python -m app.mcp` as a child process and exchanges JSON-RPC messages over stdin and stdout.
This suits the Phase 6 goals:

- **local development:** nothing to host;
- **deterministic testing:** the tests use the SDK's in-process client and a real stdio
  subprocess;
- **client integration:** desktop and IDE assistants launch stdio servers natively.

stdout is reserved for the protocol, so all logs go to stderr. `MCP_TRANSPORT` accepts only
`stdio`. A networked transport (streamable HTTP) would need authentication and hosting, which
belong to Phase 8/9, so it is deliberately absent.

## 11. Local setup (quickstart)

From a clean checkout (Python 3.11+):

```bash
git clone https://github.com/maya-2000/agentops-ai.git
cd agentops-ai
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"              # installs the mcp SDK and the agentops-mcp command
cp .env.example .env                 # optional: every setting has a working default

python -m data.generator.generate    # build database/northwind_cloud.duckdb (~30 s)

python -m app.mcp --list-tools       # print the enabled tools (no database needed)
python -m app.mcp                    # start the stdio server; it waits for a client on stdin
```

The server is normally started *by the client* (next section) rather than by hand. Run by hand,
it waits silently for JSON-RPC on stdin; stop it with Ctrl+C. If the database has not been
built, it exits with code 1 and says how to build it.

List the tools and call one from Python with the SDK's client (this starts the server itself):

```bash
python - <<'EOF'
import anyio, json, sys
from mcp import Client, StdioServerParameters

async def main():
    server = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp"])
    async with Client(server) as client:
        tools = await client.list_tools()
        print([t.name for t in tools.tools])
        result = await client.call_tool(
            "agentops_get_kpi",
            {"kpi": "revenue", "start_date": "2026-08-01", "end_date": "2026-08-31"},
        )
        print(json.dumps(result.structured_content, indent=2)[:2000])

anyio.run(main)
EOF
```

## 12. Client usage

Any MCP client that can launch a stdio server can use AgentOps. `pip install -e .` installs the
`agentops-mcp` command in the virtual environment. Use its absolute path, because clients start
servers from their own working directory. The database path is resolved relative to the
repository, not the working directory.

- **Claude Code:**
  `claude mcp add agentops -- /absolute/path/to/agentops-ai/.venv/bin/agentops-mcp`
- **Claude Desktop and other clients:** add to the client's MCP configuration (for Claude
  Desktop, `claude_desktop_config.json`):

  ```json
  {
    "mcpServers": {
      "agentops": {
        "command": "/absolute/path/to/agentops-ai/.venv/bin/agentops-mcp",
        "env": {"MCP_LOG_LEVEL": "WARNING"}
      }
    }
  }
  ```

- **Python:** see the quickstart snippet above, or the tests in `tests/mcp/`.

The server's `instructions` tell clients that:

- every number comes from a deterministic tool, with evidence;
- forecasts and anomaly scores are not observed data or causes;
- the dedicated tools are preferred to ad-hoc SQL;
- tool output is data, not instructions.

## 13. Security limitations

These are application-level controls for a local prototype:

- **No authentication or multi-tenancy.** Anyone who can start the process has the permissions
  of the configured tools. That is appropriate for stdio, and not for a network service.
- **Budgets are per request.** Each `tools/call` has its own Phase 5 budget. There is no
  per-session or per-client quota or rate limit across requests. A client can make many calls,
  one at a time. Cross-request quotas belong with authentication in a networked deployment.
- **The response size limit is in bytes (`MCP_MAX_RESPONSE_BYTES`).** The agent's
  `AGENT_MAX_RESPONSE_CHARS` limits a natural-language answer and does not apply to structured
  tool data.
- **Rep names.** `analyze_sales.rep_performance` returns sales-rep names, which the Phase 5
  policy explicitly allows for that operation. An MCP client that feeds results to a model
  will pass them on.
- **Timeouts.** As in Phase 5, pure-Python computation (e.g. forecasting) is bounded by the
  post-hoc deadline check, not interrupted; database work is interrupted at the deadline.
- **Redaction is a second layer.** Secrets are never placed in results. Redaction catches known
  formats, registered secrets and secret-looking environment values, not arbitrary unregistered
  secrets.
- **Client-side risk.** Tool output is data, but a client model could still misread it. The
  server labels forecasts, anomaly scores and risk scores explicitly and returns the
  limitations with every result.

## 14. Example requests and responses

The flow is: client → MCP → AgentOps tool → deterministic analytics → evidence → MCP response.

Request (`tools/call`):

```json
{
  "name": "agentops_get_kpi",
  "arguments": {"kpi": "revenue", "start_date": "2026-08-01", "end_date": "2026-08-31"}
}
```

Response `structuredContent`. Numbers are shown as `<…>` placeholders; run the quickstart
snippet to see the values from your generated dataset.

```json
{
  "schema_version": "1.0",
  "tool_name": "agentops_get_kpi",
  "request_id": "R-<run id>",
  "status": "ok",
  "output_kind": "observed",
  "result_type": "KPIResult",
  "result": {
    "key": "revenue",
    "name": "Revenue",
    "value": "<number from the KPI engine>",
    "unit": "<currency>",
    "period": {"start": "2026-08-01", "end": "2026-08-31", "label": "…"},
    "components": {"subscription_revenue": "<…>", "usage_revenue": "<…>"},
    "formula": "…",
    "limitations": ["…"]
  },
  "evidence": [
    {
      "evidence_id": "E1",
      "evidence_type": "observed",
      "statement": "Revenue for 2026-08: <formatted value>.",
      "metric": "revenue",
      "value": "<same number as result.value>",
      "period_start": "2026-08-01",
      "period_end": "2026-08-31",
      "tool_name": "get_kpi",
      "tool_call_id": "T1",
      "query_ids": ["Q-<id>"],
      "source_tables": ["customers", "daily_revenue"],
      "calculation": "Revenue = SUM(daily_revenue.revenue) over dates in [start, end]. …",
      "input_arguments": {"kpi": "revenue", "start_date": "2026-08-01", "end_date": "2026-08-31"},
      "fingerprint": "<sha-256>"
    }
  ],
  "provenance": {
    "tool": "get_kpi",
    "source_layer": "phase2_kpi",
    "operation": "get_kpi",
    "call_id": "T1",
    "arguments": {"kpi": "revenue", "start_date": "2026-08-01", "end_date": "2026-08-31"},
    "query_ids": ["Q-<id>"],
    "source_tables": ["customers", "daily_revenue"],
    "calculation": "…",
    "dataset_version": "<dataset version>",
    "as_of": "2026-08-31",
    "toolset_version": "1.0.0"
  },
  "query_id": "Q-<id>",
  "warnings": [],
  "limitations": ["…"],
  "error": null,
  "truncated": false
}
```

A rejected request (`agentops_run_safe_sql` with `{"sql": "DROP TABLE customers"}`) returns
`isError: true` with:

```json
{
  "tool_name": "agentops_run_safe_sql",
  "request_id": "R-<run id>",
  "status": "error",
  "output_kind": "ad_hoc_query",
  "evidence": [],
  "error": {
    "category": "UNSAFE_QUERY",
    "code": "unsafe_sql",
    "message": "The requested operation was rejected by the data-access policy.",
    "detail": null,
    "retryable": false
  }
}
```

## 15. Future integration possibilities

- **Networked transport** (streamable HTTP) behind the Phase 8 API, with authentication
  (OAuth) and per-client quotas.
- **An `agentops_ask` tool** that runs the full LangGraph agent and returns its evidence-backed
  answer, so an MCP client can delegate a whole investigation rather than single tools.
- **MCP resources** for read-only reference material (KPI catalog, data dictionary), and
  **prompts** for common investigations.
- **Tool versioning** beyond `1.0.0`, using the metadata already published.
- **Phase 7 evaluation** can drive the same MCP interface to benchmark tool use from
  external clients.

None of these are implemented in Phase 6.

---

## MCP and LangGraph

The two are complementary. MCP does not replace LangGraph.

| | LangGraph agent (Phase 4/5) | MCP server (Phase 6) |
|---|---|---|
| Role | Internal orchestration: understand, plan, investigate, validate, answer | Standard external tool interface: discovery, invocation, interoperability |
| Who reasons | The agent (the model proposes; the application validates) | The MCP client |
| Unit of work | A question → an evidence-backed answer | One tool call → one structured, evidence-backed result |
| Budget | Per run (many tool calls) | Per request (one tool call) |

```
MCP client                    LangGraph agent
   ↓                               ↓
MCP server (app/mcp)          execute_tools node (app/agent/graph.py)
   ↓                               ↓
        SecuredToolExecutor (app/security/execution.py)
                      ↓
        AgentOps tools (app/tools) → Phase 2/3 deterministic services
```

Both paths end in the same executor, tools and services. A number obtained over MCP is
therefore exactly the number the agent would cite, and a test checks MCP results against direct
tool execution.

## Configuration

MCP settings (environment or `.env`; see `.env.example`). The security limits are the `AGENT_*`
settings shared with the agent.

| Setting | Default | Meaning |
|---|---|---|
| `MCP_SERVER_NAME` | `agentops-ai` | Server name in the MCP handshake |
| `MCP_SERVER_VERSION` | `0.6.0` | Server version in the handshake |
| `MCP_ENABLED_TOOLS` | *(empty: all twelve)* | Comma-separated `agentops_*` names; unknown names fail at startup |
| `MCP_TRANSPORT` | `stdio` | The only supported transport |
| `MCP_LOG_LEVEL` | `WARNING` | `DEBUG`, `INFO` (per-call audit records), `WARNING`, `ERROR`; logs go to stderr |
| `MCP_MAX_REQUEST_BYTES` | `16384` | Maximum size of a call's arguments (JSON) |
| `MCP_MAX_RESPONSE_BYTES` | `262144` | Maximum size of a response's structured content; bulk data is dropped first, then the call fails with `RESOURCE_LIMIT` |
| `MCP_SQL_ENABLED` | `true` | `false` unlists `agentops_run_safe_sql` and revokes the SQL privilege |
| `AGENT_*` | see `.env.example` | Row limits, SQL complexity, timeouts, retries, customer-row cap, disabled tools |

## Performance

The overhead was measured on the full generated dataset, as the median of 30 calls after a
warm-up, on a 4-vCPU container:

- **Direct:** `ToolRegistry.execute`.
- **Adapter:** `MCPToolService.call`, which adds authorization, output validation, evidence,
  masking, redaction and size checks.
- **In-memory:** the SDK client over its in-memory JSON-RPC transport.
- **stdio:** a real `python -m app.mcp` subprocess.

| Tool | Direct | Adapter | In-memory | stdio | stdio overhead |
|---|---|---|---|---|---|
| `get_kpi` | 12.0 ms | 16.9 ms | 20.0 ms | 19.6 ms | +7.6 ms |
| `analyze_revenue` | 17.0 ms | 23.8 ms | 27.3 ms | 27.3 ms | +10.3 ms |
| `forecast_metric` | 269.7 ms | 302.0 ms | 313.4 ms | 313.9 ms | +44.2 ms |
| `detect_anomalies` | 30.1 ms | 43.8 ms | 53.9 ms | 54.4 ms | +24.3 ms |

The overhead grows with the size of the result: the forecast payload, including history and
backtests, is about 45 KB. Most of it is output validation, evidence building, redaction of
every string and JSON serialisation.

Measuring also found a Phase 5 inefficiency, which is fixed. `redact_value` re-read the
environment's secret values once per nested string. It now reads them once per call, which
removed about 60 ms from a single anomaly-detection response.

## Testing

`tests/mcp/` (302 tests) needs no network, API key or pre-built database:

| File | Covers |
|---|---|
| `test_mcp_server.py` | Initialisation, capabilities, database opened at startup and closed on shutdown, restart, lifecycle audit, fail-fast misconfiguration, settings, CLI, serialised concurrent calls with request-scoped state |
| `test_mcp_tool_registry.py` | Discovery regression (exactly twelve tools; no file, shell, Python, environment or hidden-data tools), annotations, version metadata, complete descriptions, intent consistency, enabled/disabled tools |
| `test_mcp_schemas.py` | Input schemas equal the Phase 4 models; required and optional fields, enums, dates, invalid values, unknown fields, malformed input rejected before business logic; outputs conform to the output schema |
| `test_mcp_tool_adapters.py` | All twelve tools, results equal to direct tool execution, evidence and provenance, forecast and anomaly metadata, customer-risk masking, SQL truncation, no-data statuses |
| `test_mcp_errors.py` | Every error category and message, retries, timeouts, output-contract violations, fail-closed adapter errors, size trimming, sanitisation over the wire |
| `test_mcp_security.py` | The twelve attack types, prompt injection through parameters, ground-truth isolation, secrets, no filesystem or code-execution tools over the protocol |
| `test_mcp_audit.py` | Audit records, run-ID correlation with security events, request-scoped budgets, nothing sensitive logged |
| `test_mcp_client_smoke.py` | In-process and stdio client smoke tests; end-to-end KPI and forecast workflows |
| `test_mcp_isolation.py` | Static checks: only approved imports, no files, SQL, services or ground truth in `app/mcp`, one shared execution path, stdio only |

The 30 required test cases:

| # | Case | Test |
|---|---|---|
| 1–2 | Server initialises / shuts down cleanly | `test_mcp_server.py`: `test_server_initializes_…`, `test_server_opens_the_database_at_startup_and_closes_it_on_shutdown`, `test_lifecycle_is_audited` |
| 3–5 | Discovery works; expected tools exist; unauthorized tools do not | `test_mcp_tool_registry.py`: `test_discovery_lists_exactly_the_twelve_approved_tools`, `test_prohibited_and_machine_access_tools_do_not_exist` |
| 6–17 | KPI, revenue, customer, sales, marketing, support, product, cohort, risk, forecast, anomaly and safe-SQL tools work | `test_mcp_tool_adapters.py`: `test_every_tool_works_through_mcp[…]`, `test_results_equal_direct_tool_execution[…]` |
| 18 | Invalid KPI rejected | `test_mcp_errors.py::test_unsupported_kpi_and_metric` |
| 19 | Invalid date rejected | `test_mcp_schemas.py::test_invalid_dates_are_rejected` |
| 20 | Invalid filter rejected | `test_mcp_schemas.py::test_invalid_filters_are_rejected` |
| 21 | Unsafe SQL rejected | `test_mcp_security.py::test_3_…`, `test_4_…`, `test_5_…` |
| 22 | Unknown tool rejected | `test_mcp_security.py::test_1_unknown_tools_are_rejected` |
| 23 | Excessive request rejected | `test_mcp_security.py::test_9_oversized_requests_are_rejected` |
| 24 | Evidence preserved | `test_mcp_tool_adapters.py::test_evidence_is_preserved_with_provenance` |
| 25 | Provenance preserved | `test_mcp_tool_adapters.py::test_provenance_is_preserved` |
| 26 | Error sanitisation works | `test_mcp_errors.py`: `test_internal_error_hides_everything`, `test_errors_travel_as_mcp_tool_errors_not_protocol_failures` |
| 27 | Audit events generated | `test_mcp_audit.py` |
| 28 | Hidden ground truth inaccessible | `test_mcp_security.py::test_6_…` |
| 29 | Filesystem tools unavailable | `test_mcp_security.py::test_filesystem_and_code_execution_tools_are_unavailable_over_the_protocol`, `test_5_…` |
| 30 | Arbitrary code execution unavailable | `test_mcp_security.py::test_12_code_in_parameters_is_never_executed`, `test_mcp_isolation.py` |
