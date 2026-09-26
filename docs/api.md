# AgentOps HTTP API (Phase 8)

The API makes the evidence-backed agent usable over HTTP. It is a thin transport: every question
is answered by the existing Phase 4 agent (`AgentRunner.run`), whose tool calls go through the Phase
5 secured executor. The API adds a typed contract, safe errors, request IDs, progress events and
chart specifications. It adds no analytics, SQL, planning or permissions of its own.

- Code: `app/api/` (FastAPI). Start it with `python -m app.api` (or `agentops-api`).
- Interactive schema: `http://127.0.0.1:8000/docs` (OpenAPI at `/openapi.json`).
- The web UI (`app/ui/`, [ui.md](ui.md)) is a client of this API.

**Local development service.** There is no authentication, and it is not meant to be exposed on a
network. It binds to `127.0.0.1` by default.

## 1. Architecture assessment (Step 0)

Before implementation, the repository was inspected to find the pieces the API must reuse rather
than rebuild:

| Question | Answer in the code base |
|---|---|
| Agent entry point | `AgentRunner.run(question)` in `app/agent/runner.py`. It returns an `AgentRunResult`: status, `AgentResponse`, understanding, `ValidatedRequest`, plans, claims, evidence, tool trace, LLM calls, security events, budget usage, timing. |
| Evidence representation | `Evidence` in `app/evidence/models.py`: value, unit, display value, period, comparison, filters, dimension, attributes, tool call, query IDs, source tables, calculation, limitations and a SHA-256 fingerprint. |
| Claim representation | `Claim`: `observed_fact` / `calculated_result` / `inference` / `recommendation`, support status, kind, `primary`, cited evidence IDs and a structured `ClaimSubject`. |
| Tool trace | `ToolCallRecord` (developer trace) and `ToolTraceEntry` (user-safe summary with a fixed error category, built by `trace_entries`). |
| Security boundary | Inside the agent: the input guard and injection screen, tool authorization, the plan validator, the data-exposure policy, budgets, deadlines and output validation, all in `SecuredToolExecutor` (`app/security/execution.py`), shared with the MCP server. |
| Forecast and anomaly outputs | `ForecastResult` and `AnomalyReport`, with MCP views (`ForecastView`, `AnomalyReportView`) that copy their key fields. |
| Configuration | `app/config.py` (`Settings`, environment variables, `.env`); only `app/security/redaction.py` reads the environment. |
| Constraints found | One read-only DuckDB connection, which is never used concurrently (the MCP server serialises calls with a lock). Raw tool results can contain withheld fields (`company_name` in customer-risk rows); evidence never does. |

Decisions that follow from it:

1. The API calls `AgentRunner.run` and nothing else. One runner and one database connection are
   built at start-up; runs are serialised on one worker thread.
2. The response serialises the existing domain objects (`AgentResponse`, `Claim`, `Evidence`,
   `ValidatedRequest`). New models are views that select or copy from them.
3. The forecast and anomaly views moved from `app/mcp/schemas.py` to `app/tools/views.py`, so the
   MCP server and the API share one definition. `app/mcp/schemas.py` re-exports them; the MCP
   output is unchanged.
4. Raw tool results stay in-process. `AgentRunResult.tool_results` is excluded from serialisation
   and `repr`. The API reads only forecast and anomaly results from it (series and intervals for the
   charts), and only when the run produced evidence from them.

## 2. Request flow

```
Streamlit UI / curl ──HTTP──▶ FastAPI (app/api)
                               │  RequestContextMiddleware: request ID, 413 body limit, security headers, request log
                               │  routes/ask.py: validate the question (empty, too long)
                               ▼
                             AgentService (service.py): one worker thread, lock, timeout, queue bound
                               ▼
                             AgentRunner.run(question, run_id=request_id, on_progress=…)   (Phase 4 graph)
                               ▼
                             SecuredToolExecutor → tool registry → Phase 2/3 services → read-only DuckDB
                               ▼
                             presenter.py + visualizations.py: AskResponse (copies only; nothing recomputed)
```

The API never opens files, reads the environment, queries the database (apart from the health
probe's `list_tables` metadata query and the agent's own coverage lookup at start-up), calls a tool
or imports the MCP server. `tests/api/test_api_isolation.py` checks this statically.

## 3. Endpoints

| Method and path | Purpose | Success |
|---|---|---|
| `POST /api/v1/ask` | Ask a question; returns an `AskResponse` | 200 for every agent outcome (see §4) |
| `POST /api/v1/ask/stream` | The same, as NDJSON progress events followed by one result or error event | 200 (request errors are returned before the stream starts) |
| `GET /api/v1/health` | Readiness: status, version, agent and database availability, dataset version, as-of date, provider name | 200 (`ok` or `degraded`), 503 (`unavailable`) |
| `GET /api/v1/capabilities` | KPIs, forecast and anomaly metrics, detectors, dimensions, analyses, outcomes, limits, example questions, what is not supported | 200 |
| `GET /api/v1/metrics` | In-process counters since start-up: requests, outcomes, status codes, agent time, API overhead | 200 |

### `POST /api/v1/ask`

Request:

```json
{
  "question": "What was revenue in July compared with June?",
  "request_id": "optional-client-id",
  "session_id": "optional-session-id"
}
```

- `question`: a string of 1–10,000 characters, not blank. The configured agent limit
  (`AGENT_MAX_QUESTION_CHARS`, 1,000 by default) is checked before the run (422 above it), and the
  agent's input guard checks it again.
- `request_id` / `session_id`: optional tokens of letters, digits and `._:-` (at most 64
  characters, starting alphanumeric). No other fields are accepted (`extra="forbid"`), so a request
  cannot carry settings such as `max_tool_calls`.

Response (`AskResponse`, abridged):

```json
{
  "schema_version": "1.0",
  "request_id": "R-3f2a9c1d7e44",
  "session_id": null,
  "status": "completed",
  "outcome": "answered",
  "question": "What was revenue in July compared with June?",
  "answer": "Revenue changed by +SGD 40,472 (+0.70%) from 2026-06 to 2026-07.",
  "response": { "...": "the agent's AgentResponse: answer, findings, caveats, assumptions, cited evidence, user-safe tool trace" },
  "scope": { "intent": "period_comparison", "period": {"label": "2026-07", "...": "..."}, "comparison_period": {"label": "2026-06"} },
  "period": {"start": "2026-07-01", "end": "2026-07-31", "label": "2026-07"},
  "comparison_period": {"start": "2026-06-01", "end": "2026-06-30", "label": "2026-06"},
  "kpis": [{"evidence_id": "E3", "label": "Revenue change", "display_value": "SGD 40,472", "percentage_change": 0.0070, "...": "..."}],
  "claims": [{"claim_id": "C1", "claim_type": "calculated_result", "primary": true, "evidence_ids": ["E3"], "...": "..."}],
  "evidence": [{"evidence_id": "E1", "statement": "Revenue for 2026-07: SGD 5,809,163.", "query_ids": ["Q-…"], "source_tables": ["daily_revenue"], "calculation": "…", "fingerprint": "…"}],
  "trace": [{"step": 1, "tool_name": "analyze_revenue", "purpose": "Compare revenue between the periods.", "status": "ok", "execution_time_ms": 21.8, "evidence_ids": ["E1", "E2", "E3"]}],
  "forecasts": [],
  "anomalies": [],
  "visualizations": [{"chart_id": "V1", "kind": "kpi_card", "...": "..."}, {"chart_id": "V2", "kind": "comparison", "...": "..."}],
  "refusal": null,
  "run": {"run_id": "R-3f2a9c1d7e44", "llm_provider": "deterministic", "tool_calls": 1, "agent_time_ms": 62.0, "stages": [{"stage": "question_received", "label": "Screening the question", "duration_ms": 3.3, "ok": true}]},
  "api_time_ms": 63.1
}
```

| Field | Content | Source |
|---|---|---|
| `status` | The agent's own status (`completed`, `insufficient_evidence`, `unsupported_request`, `tool_error`, `validation_failure`, `planning_failure`) | `AgentRunResult.status` |
| `outcome` | `answered`, `partial`, `refused`, `unsupported`, `insufficient_evidence` or `failed` (§4) | derived from the status and the kind of denial |
| `response` | The validated `AgentResponse`, unchanged | agent |
| `scope`, `period`, `comparison_period` | Intent, metric, periods, dimensions, filters, horizon and assumptions | `ValidatedRequest` |
| `claims`, `evidence` | Every claim and evidence item of the run, unchanged | evidence graph |
| `kpis` | Headline evidence (observed or calculated, not a breakdown member), primary-cited first; value and display value copied | evidence |
| `trace` | Each tool call: the plan step's purpose, status, time, attempts, query and evidence IDs, and a fixed error category | `ToolCallRecord` + `trace_entries` |
| `forecasts` | `ForecastView` (horizon, points with interval bounds, model, cutoff, confidence level, backtest and baseline error metrics) + limitations + the forecast notice | `ForecastResult` of a call that produced evidence |
| `anomalies` | `AnomalyReportView` (detector, window, threshold, severity counts, flagged months with observed, expected, deviation, score, severity and direction) + limitations + the anomaly notice | `AnomalyReport` of a call that produced evidence |
| `visualizations` | Chart specs (§6) | evidence, forecast and anomaly results |
| `refusal` | `{kind, message}` for refused and unsupported questions; `kind` is `policy`, `out_of_scope` or `invalid_input`, and `message` is the agent's own fixed text | status + security events |
| `run` | Run ID (= request ID), provider and model names, tool calls, retries, agent time, graph nodes and their durations | runner + service |

Never included: prompts, model reasoning, raw tool results, security events and their pattern
names, stack traces, SQL error text, file paths, environment values and secrets.

### `POST /api/v1/ask/stream`

Same request body. The response is `application/x-ndjson`, one JSON object per line:

```
{"type":"progress","request_id":"R-…","stage":"question_received","label":"Screening the question","elapsed_ms":3.2}
{"type":"progress","request_id":"R-…","stage":"understand_question","label":"Understanding the question","elapsed_ms":5.9}
…
{"type":"result","request_id":"R-…","data":{ …the same AskResponse as /ask… }}
```

or, if the run fails after the stream started (for example the API timeout):

```
{"type":"error","request_id":"R-…","status_code":504,"error":{"code":"timeout","message":"…","retryable":true}}
```

Progress uses LangGraph's `stream` API through an optional `on_progress` callback on
`AgentRunner.run`. The callback receives node names only, never state. Without a callback the runner
calls `invoke` exactly as before, so the benchmark path is unchanged. The API always passes a
callback, because it records stage durations for the `run.stages` trace.
`tests/api/test_api_streaming.py` shows that a streamed run concludes exactly like an invoked one:
same status, answer, claims, evidence, tool calls, transitions and security events. It also shows
that a failing observer cannot change a run.

## 4. Outcomes, status codes and errors

A refusal is a controlled response, not an error. Everything the agent produces is returned with
HTTP 200:

| Agent status | `outcome` | Meaning |
|---|---|---|
| `completed` | `answered` | Validated answer with evidence |
| `validation_failure` | `partial` | The explanation failed validation; only validated findings are shown |
| `unsupported_request` + prompt-injection denial | `refused` (`refusal.kind = policy`) | Asked for secrets, prompts, hidden data, files, code execution or rule changes |
| `unsupported_request` + input rejection | `refused` (`invalid_input`) | The question could not be accepted as written |
| `unsupported_request` otherwise | `unsupported` (`out_of_scope`) | Outside the dataset or the supported analyses |
| `insufficient_evidence` | `insufficient_evidence` | Missing data, out-of-range period, a needed clarification, or a limit reached |
| `tool_error`, `planning_failure` | `failed` | Required tools failed, or no valid plan could be made |

API errors are failures to produce such a response. Every error body has the same shape, and the
message is a fixed string:

```json
{"request_id": "R-…", "error": {"code": "empty_question", "message": "The question is empty. …", "retryable": false, "issues": []}}
```

| HTTP | `code` | When |
|---|---|---|
| 400 | `malformed_request` | The body is not valid JSON |
| 422 | `invalid_request` | Schema violation: missing or wrongly typed field, unknown field, bad `request_id`. `issues` lists `{location, message}`; the submitted value is never echoed |
| 422 | `empty_question` | Empty or whitespace-only question |
| 422 | `question_too_long` | Longer than `AGENT_MAX_QUESTION_CHARS` |
| 413 | `request_too_large` | Body above `API_MAX_REQUEST_BYTES` (declared or streamed) |
| 404 / 405 | `not_found` / `method_not_allowed` | Unknown route or method. No static files are served |
| 503 | `busy` | More than `API_MAX_PENDING_REQUESTS` requests waiting (`Retry-After: 5`) |
| 503 | `agent_unavailable` | The database or agent could not be initialised at start-up (`Retry-After: 5`) |
| 504 | `timeout` | The run exceeded `API_REQUEST_TIMEOUT_SECONDS` |
| 500 | `internal_error` | Anything unexpected. Only the exception class name goes to the log |

## 5. Request IDs, logs and metrics

- **Request ID.** Taken from the body's `request_id`, else from a valid `X-Request-ID` header, else
  generated (`R-` + 12 hex digits). It becomes the agent run ID. So it appears in the
  `X-Request-ID` response header, `request_id` and `run.run_id` in the body, every error body and
  stream event, and every `agentops.agent`, `agentops.security` and `agentops.api` log line of the
  request. Tool calls (`T1`, `T2`, …) and evidence items (`E1`, …) are numbered within that run.
  An invalid header is replaced, never echoed.
- **Request log.** Each request produces one JSON line on the `agentops.api` logger, built from an
  allow-list of keys: request and session IDs, method, path, status code, agent status, outcome,
  error code, exception class name, tool calls, evidence and claim counts, and durations. It never
  contains the question, answer, evidence values, headers, client addresses, prompts, environment
  values or secrets. `python -m app.api` turns off uvicorn's access log, because that log records
  client addresses.
- **Headers.** Every response carries `Cache-Control: no-store` and `X-Content-Type-Options:
  nosniff`.
- **Metrics.** `GET /api/v1/metrics` returns in-memory counters since start-up (nothing is
  persisted).

## 6. Visualization specs

`visualizations` is a renderer-neutral list of `VisualizationSpec` (`chart_id`, `kind`, `title`,
`subtitle`, `metric`, `unit`, `fields` with roles, `rows`, `evidence_ids`, `source`, `notes`). The UI
turns each spec into Vega-Lite. Any other client can draw the same rows.

| `kind` | Built when | Rows copied from |
|---|---|---|
| `kpi_card` | Headline evidence exists | `kpis` (value, display value, period, comparison, percentage change) |
| `comparison` | A change evidence item states both levels (`current_value` and `comparison_value` or `previous_value`) | that evidence item's attributes |
| `bar` + `table` | Two or more members of one breakdown (same tool call, dimension and metric) | each member's value, or the first numeric attribute every member states; in the tool's own order |
| `time_series` | Three or more monthly evidence items of one metric | their values |
| `forecast` | A forecast result produced evidence | history points (actual) + forecast points with lower and upper bounds |
| `anomaly` | An anomaly report produced evidence | the monthly series + each scored month's expected value, bounds, flag, severity, direction and score |

Rules:

- Nothing is recomputed, rescaled or aggregated.
- Every spec lists the evidence it depicts, and the tests check that each number matches its source.
- No chart is produced when the evidence does not support one: refusals and unsupported questions
  have none.
- Forecast and anomaly specs carry their notices: "an estimate, not observed data" and "not
  necessarily bad, and it does not explain the cause".

## 7. Security boundary

The API is a new entry point to the agent, not a new path around it:

- **Same controls.** Questions go to `AgentRunner.run`. The input guard, the injection screen,
  tool authorization, plan validation, SQL safety, the data-exposure policy, budgets, deadlines,
  output validation and redaction all apply unchanged. The API cannot raise a limit, because
  unknown request fields are rejected.
- **No data access of its own.** No SQL, no tool calls, no file reads, no environment reads and no
  MCP imports (checked statically). No route serves files, so `data/seeds/injected_events.json`
  and every other file are unreachable.
- **No internals in responses.** Evidence (never raw rows) is returned, so withheld fields such as
  `company_name` never appear. Customer-level results name customer IDs only. Tool errors appear as
  fixed categories. Validation errors never echo the submitted value, and unexpected exceptions
  become `internal_error`.
- **Regression tests** (`tests/api/test_api_security.py`) cover, over HTTP:
  - prompt injection (`/ask` and `/ask/stream`);
  - unsafe SQL and file-reading SQL from a compromised model;
  - unregistered, smuggled, out-of-intent and disabled tools;
  - hidden ground truth: its text does not appear in any response or log, and no data file is
    opened during requests (an audit hook records file opens);
  - customer-name exposure;
  - malformed tool outputs;
  - tool-call and wall-clock budgets, and the API timeout;
  - concurrency bounds;
  - output validation with a fabricated model draft;
  - secret redaction and log hygiene.
- **Fixed during Phase 8.** A tool handler that returned something other than a typed output used
  to crash the whole agent run (the API returned a safe 500). `ToolRegistry.execute` now fails the
  call closed with `invalid_tool_output`, like any other invalid output. Regression tests cover
  this in `tests/unit/test_tool_registry.py` and over the API.

## 8. Concurrency and timeouts

- One `AgentRunner` and one read-only database connection per process, and one worker thread.
  Runs are serialised, and a lock also guards the health probe. `python -m app.api` starts a
  single uvicorn worker.
- `API_REQUEST_TIMEOUT_SECONDS` (default 150 s, above `AGENT_MAX_RUN_SECONDS`) bounds the wait.
  After it, the client gets 504. A queued run is cancelled before it starts. A run that has already
  started cannot be interrupted, because Python threads cannot be killed: it finishes in the
  background under the agent's own limits (run wall clock, per-tool and SQL timeouts, budgets), and
  its result is discarded.
- At most `API_MAX_PENDING_REQUESTS` (default 4) requests wait at once; more get 503 `busy`.

## 9. Configuration

All settings live in `app/config.py` and can be set as environment variables or in `.env` (see
`.env.example`). The API adds only transport settings; the agent keeps its `AGENT_*` limits and
`LLM_*` provider settings.

| Variable | Default | Meaning |
|---|---|---|
| `API_HOST` | `127.0.0.1` | Bind address (keep it local: there is no authentication) |
| `API_PORT` | `8000` | Port |
| `API_REQUEST_TIMEOUT_SECONDS` | `150` | Wall-clock limit per request |
| `API_MAX_REQUEST_BYTES` | `16384` | Largest accepted body |
| `API_MAX_PENDING_REQUESTS` | `4` | Waiting requests before 503 |
| `UI_API_URL` | `http://127.0.0.1:8000` | Where the UI sends questions |
| `UI_REQUEST_TIMEOUT_SECONDS` | `180` | How long the UI waits |

No secrets are needed with the default deterministic provider. With `LLM_PROVIDER=anthropic`, the
key is read from `ANTHROPIC_API_KEY` as before and is never logged or returned.

## 10. Performance

Measured locally on the generated dataset (deterministic provider, warm process, median of 7
runs, milliseconds). "Direct" is `AgentRunner.run` (the benchmark path, `invoke`). "API route" is
the time inside the route. "In-process" is the FastAPI test client. "HTTP" is a real uvicorn server
called with httpx on localhost.

| Question | Direct | Agent (API run) | API route | In-process client | HTTP (uvicorn) | Present + serialise | Body |
|---|---:|---:|---:|---:|---:|---:|---:|
| Revenue in July vs June | 31.5 | 30.2 | 30.8 | 32.6 | 42.5 | 0.3 | 11.7 KB |
| Region with the largest revenue decline | 49.7 | 48.4 | 49.1 | 50.9 | 57.5 | 0.5 | 21.2 KB |
| Channel with the highest CAC | 23.7 | 23.2 | 23.7 | 25.2 | 33.0 | 0.5 | 18.1 KB |
| 3-month revenue forecast | 279.7 | 286.7 | 287.4 | 288.9 | 300.4 | 0.5 | 20.2 KB |
| Unusual trends in support tickets | 86.3 | 77.2 | 78.3 | 80.8 | 81.1 | 0.8 | 39.1 KB |
| Why support tickets increased | 123.2 | 122.1 | 123.2 | 125.7 | 126.7 | 1.0 | 56.6 KB |
| Prompt injection (refused) | 2.4 | 2.4 | 2.8 | 3.8 | 5.5 | 0.0 | 2.2 KB |
| Out-of-scope question | 3.9 | 3.4 | 3.7 | 4.6 | 6.5 | 0.0 | 1.8 KB |

Medians across the eight questions:

- The in-process API adds 1.3 ms over a direct run, and real HTTP on localhost adds 5.6 ms.
- The streamed (progress) path costs the same as `invoke` (+0.1 ms).
- Building and serialising the response takes 0.5 ms.

Agent time dominates (the forecast's model selection and backtests take about 280 ms). With a
network LLM provider, model latency would dominate instead.
`tests/api/test_api_performance.py` guards these properties with generous bounds.

## 11. Known limitations

- **No authentication, rate limiting or TLS.** It is a local development service.
- **One run at a time per process.** Throughput is bounded by the agent. A run that exceeds the API
  timeout keeps its worker until the agent's own limits end it.
- **Metrics are in-process** and reset on restart. Nothing is persisted, including questions and
  answers.
- **Progress events report finished graph stages**, not partial answers. With the deterministic
  provider a run takes milliseconds, so progress matters mainly with a network model.
- **Evidence `input_arguments` are returned as provenance.** For an ad-hoc SQL step that includes
  the validated read-only `SELECT`. It is the agent's own query over allow-listed tables, never
  database error text.
- **The chart set is fixed** by the evidence shapes above. Questions whose evidence has no
  chartable shape get tables and evidence only.
