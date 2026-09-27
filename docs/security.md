# AgentOps security (Phase 9)

This is the security model of the deployable application: what protects the HTTP entry points,
the agent, the data and the operators' secrets. It summarises and links the detailed documents:

- [security-architecture.md](security-architecture.md): the Phase 5 controls inside the agent.
- [security-threat-model.md](security-threat-model.md): threats and residual risks.
- [mcp-architecture.md](mcp-architecture.md) §6–7: the MCP boundary.
- [deployment.md](deployment.md): how to run it safely.

The principle has not changed since Phase 5: **the model can propose; the application decides.**
Phase 9 adds the layer in front of the agent (authentication, rate limiting, request hardening)
and the operational rules (configuration, secrets, logging, containers). It does not add a second
path to the data.

```
client ─▶ [transport]  security headers · body size · content type · request ID
        ─▶ [access]     bearer token (401) · rate limit (429)
        ─▶ [contract]   typed request, unknown fields rejected, question length (422)
        ─▶ [agent]      input guard · injection screen · plan validation · tool authorization ·
                        SQL safety · data-exposure policy · budgets · deadlines · output validation
        ─▶ [data]       12 allow-listed tools over a read-only DuckDB connection
```

## 1. API authentication

- **Scheme.** `Authorization: Bearer <token>`, compared in constant time with `API_AUTH_TOKEN`
  (`app/api/security.py`). Every `/api/v1` endpoint requires it except liveness and readiness,
  which return no business data.
- **Uniform failure.** Missing, malformed and wrong credentials get the same 401 body ("Missing or
  invalid credentials.") and `WWW-Authenticate: Bearer`. The response never says which check
  failed. Unknown `/api/v1` paths also answer 401, so routes cannot be discovered without a token.
- **Before everything else.** Authentication runs in middleware before routing, before the body
  is read and before rate limiting. An unauthenticated request cannot reach validation or the
  agent, and cannot use up an authenticated client's quota.
- **Strength.** At least 32 characters, no whitespace, validated at start-up.
  `python -c "import secrets; print(secrets.token_urlsafe(32))"` gives 256 bits.
- **Secure by default.** Token authentication is the default in every environment. Without a
  token the API does not start. `API_AUTH_MODE=disabled` must be set explicitly, and is refused
  when `APP_ENV=production`.
- **Scope.** One service token authenticates the UI and operators to the API. There are no user
  accounts or roles. User authentication belongs in the reverse proxy in front of the UI
  ([deployment.md §4](deployment.md#4-authentication)).

## 2. Authorization boundary

Authentication decides who may call the API. What a call may do is decided inside the agent,
exactly as before Phase 9:

- The API and the UI are entry points to `AgentRunner.run`, never a path around it. The API has no
  SQL, tool calls, file reads or environment reads of its own (`tests/api/test_api_isolation.py`
  checks this statically, and `test_the_api_still_has_no_path_to_raw_sql` over HTTP).
- A request cannot change a limit or a permission: the request schema accepts only `question`,
  `request_id` and `session_id` (`extra="forbid"`). `AGENT_*` limits are read once at start-up.
- Every tool call is authorised twice: when the plan is validated and again just before execution.
  The checks cover the allowlist, the tools permitted for the validated intent, argument
  validation, the SQL privilege, the data-exposure policy and the run budget
  ([security-architecture.md §6–7](security-architecture.md#7-tool-authorization-appsecurityauthorizationpy)).
  A policy denial is final and never retried.
- The UI has no data access at all. Its container image does not contain the agent, the tools,
  DuckDB or any data file, and it reaches the API only over HTTP with the token.
- The MCP server uses the same secured executor, so the MCP and HTTP paths enforce the same
  policy (MCP parity is part of the benchmark).

## 3. Secure execution

- **Bounded runs.** Tool calls, retries, planning iterations, SQL calls and rows, model calls,
  context size, response length and wall-clock time are all limited
  ([security-architecture.md §9](security-architecture.md#9-resource-limits-appsecuritylimitspy-budgetpy-timeoutspy-retrypy-contextpy)).
  Timeouts nest: SQL (10 s) ≤ tool (30 s) ≤ run (120 s) ≤ API request (150 s) ≤ UI (180 s).
  Production refuses a configuration where they do not.
- **Cancellation.** When the API timeout fires, a client disconnects or the service shuts down,
  the run's cancel event is set. The graph stops before its next step, DuckDB queries are
  interrupted, and model calls wait no longer than the time left. A stopped run returns the
  controlled "limit reached" response, never a partial answer.
- **Serialised access.** One worker thread and one read-only database connection. At most
  `API_MAX_PENDING_REQUESTS` requests wait; more get 503 `busy`.
- **Fail closed.** A malformed tool output, a failed validation or an exhausted budget ends the
  step or the run with a safe category. Nothing unvalidated becomes evidence.

## 4. Prompt-injection controls

- A deterministic screen runs before any model call. It blocks requests for secrets, prompts,
  hidden or ground-truth data, files, code execution, disabling checks or changing limits.
  Instruction overrides are answered with reduced privileges (no ad-hoc SQL for that run).
- The question reaches the model only as escaped, delimited data, never as instructions.
- Refusals are controlled responses: HTTP 200, `outcome: "refused"`, a fixed explanation. They
  include no pattern names, categories or other screening internals, so a response does not tell
  an attacker which rule fired.
- Model output (understanding, plan, draft) is untrusted until validated, and is never executed.

Details: [security-architecture.md §5](security-architecture.md#5-prompt-injection-handling-appsecurityinjectionpy-appllmpromptspy).

## 5. SQL controls

- Ad-hoc SQL exists only as one tool (`run_safe_sql`), available only to intents that need it and
  withdrawn after an instruction override. It can be switched off entirely (`AGENT_DISABLED_TOOLS`).
- Statements are parsed and must be a single read-only `SELECT` over allow-listed tables and
  columns. DDL, DML, file, network, extension and settings statements and functions are refused.
  Joins, nesting, CTEs and parameters are limited.
- The executed statement is regenerated from the validated syntax tree, with bound parameters, a
  row cap and truncation flags, a statement timeout, and a read-only connection as an independent
  second layer.
- The HTTP API never accepts SQL. A compromised or confused model cannot bypass these checks
  either, which the security tests show over HTTP.

Details: [security-architecture.md §8](security-architecture.md#8-sql-security-apptoolssql_safetypy-appsecuritydata_policypy).

## 6. Data-exposure controls

- **Minimum data.** Responses carry evidence (values, periods, provenance), never raw rows.
  Withheld and PII columns (company names, sales-rep names) cannot be queried, and customer-level
  results name customer IDs only. Per-customer and per-person breakdowns are denied, and
  customer-level output is capped.
- **Ground truth.** The injected-event labels (`data/seeds/`) are used only by the evaluation
  harness. They are not in the database, not in either container image and not mounted, and no
  route serves files. An audit hook in the tests records every file, process and network access
  during agent and API runs.
- **No internals in responses.** No traceback, file path, SQL error text, exception class name,
  environment value, prompt or model reasoning appears in any response. Errors are a fixed
  envelope: `{request_id, error: {code, message, retryable, issues, request_id}}`. Validation
  issues name the field, never the submitted value.
- **Browser hardening.** Every response carries `X-Content-Type-Options: nosniff`,
  `Cache-Control: no-store`, `Referrer-Policy: no-referrer` and `X-Frame-Options: DENY`, plus a
  deny-all Content-Security-Policy on API responses. CORS is off unless origins are listed
  explicitly. No credentials are allowed, and production refuses `*` and plain-HTTP origins.
- **UI rendering.** Every dynamic text is escaped before it reaches Streamlit markdown. No HTML is
  rendered, and in production Streamlit shows no exception details.

## 7. Secrets handling

| Secret | Where it lives | Protections |
|---|---|---|
| `API_AUTH_TOKEN` | Environment or `.env` (git-ignored); compose passes it from `.env` | `SecretStr`; not in `repr`; not in validation errors; registered for redaction; the UI never displays it |
| `ANTHROPIC_API_KEY` (optional) | Environment or `.env` | `SecretStr`; registered for redaction by the LLM factory; never in prompts, logs or responses |

- **Never in the repository or images.** `.env.example` holds empty placeholders only (tested).
  The Docker build context is an allow-list (`pyproject.toml`, `README.md`, `app/`), so `.env`
  cannot be copied into an image. The Dockerfile declares no secret, and the images and compose
  file are tested for this.
- **CI needs none.** The workflows run the deterministic model and create an ephemeral, masked
  token for the container smoke test. No repository secret is referenced.
- **Errors never echo values.** Invalid settings are reported as `NAME: reason` without the value
  (`hide_input_in_errors`). Start-up problems name settings, never their values.
- **Rotation.** Change the value and restart the services. Nothing caches the token outside the
  process.

## 8. Rate limiting and request limits

| Control | Limit | Response |
|---|---|---|
| Rate limit on `/ask`, `/ask/stream` (per client, sliding window) | `API_RATE_LIMIT`, default `20/minute`; must be on in production | 429 `rate_limited` + `Retry-After` |
| Body size (declared or streamed) | `API_MAX_REQUEST_BYTES` (16 KB) | 413 `request_too_large` |
| Content type on the ask endpoints | `application/json` only | 415 `unsupported_media_type` |
| Question length | `AGENT_MAX_QUESTION_CHARS` (1,000) | 422 `question_too_long` |
| Request and session IDs | 64 safe characters | body: 422; header: replaced, never echoed |
| Waiting requests | `API_MAX_PENDING_REQUESTS` (4) | 503 `busy` + `Retry-After` |
| Wall clock | `API_REQUEST_TIMEOUT_SECONDS` (150 s) | 504 `timeout`, run cancelled |

The limiter is in memory, per process and bounded (`API_RATE_LIMIT_MAX_CLIENTS`). It limits
cost and abuse for the single-process deployment; it is not a DDoS defence. Put volumetric
protection in front, at the proxy or the network edge.

## 9. Logging policy

Logs are for operators and must be safe to ship to a log store:

- **Structured and allow-listed.** One JSON object per line. Request lines are built from a fixed
  set of keys: request and session ID, method, route template, status, outcome, agent status,
  error code, counts and durations. Service events carry reasons and counts. Agent and security
  events carry tool names, decisions and fixed categories.
- **Never logged:** questions and answers, headers (including `Authorization`), tokens and API
  keys, environment values, client IP addresses (uvicorn's access log is off), prompts, model
  reasoning, full tool outputs, SQL error text, file paths, ground truth, and customer PII.
- **Defence in depth.** Every logged value passes secret redaction, and registered secrets (the
  token, the API key) are replaced wherever they appear. Library messages are redacted, stripped
  of paths and truncated. Exceptions are reduced to their class name, including tracebacks that a
  library hands over as text.
- **Correlation without content.** The request ID links a response, its log lines and its agent
  run, so an incident can be traced without logging what was asked.

Verified in tests (`test_the_token_never_reaches_responses_or_logs`, the log-hygiene tests in
`tests/api/test_api_security.py` and `test_api_runtime.py`) and in the container smoke test: of
189 API log lines, none contained the token or question text, and all were JSON.

## 10. Container and host

- Both images run as an unprivileged user (uid 10001). Compose adds a read-only root filesystem,
  `cap_drop: ALL`, `no-new-privileges` and `tmpfs` scratch space.
- The data is mounted read-only into the API only. The UI has no volume.
- Ports are published on `127.0.0.1`. Remote access goes through a TLS-terminating reverse proxy.
- Production mode turns off the OpenAPI docs, Streamlit's error details and its developer toolbar.
- `pip-audit` reported no known vulnerabilities in the 96 installed packages at the time of
  Phase 9. The CI installs from `pyproject.toml` ranges, so the audit should be repeated when
  dependencies change.

## 11. Verification

| Area | Evidence |
|---|---|
| Authentication, rate limiting, CORS, headers, request IDs, content types, error envelopes, configuration rules | `tests/api/test_api_production.py` |
| Injection, unsafe SQL, unauthorised tools, ground truth, customer names, malformed output, tool timeout, budget, oversized requests, over the production (authenticated, rate-limited) app | `tests/api/test_api_security_production.py` |
| Cancellation, shutdown, run tracking, metrics, log format | `tests/api/test_api_runtime.py` |
| Phase 5–8 attacks over HTTP; static isolation of the API and UI | `tests/api/test_api_security.py`, `tests/api/test_api_isolation.py` |
| Agent-level security regression (adversarial, SQL attacks, audit hook) | `tests/security/` |
| Images, compose hardening, workflows without secrets | `tests/deploy/test_deployment.py` |
| The running deployment | `scripts/smoke_test.py` (CI `docker` job) |
| Benchmark security, data-exposure and injection scenarios, MCP parity | `python -m evals.run` ([evaluation.md](evaluation.md)) |

## 12. Residual risks

- **One shared token.** Anyone holding it has full API access; there is no per-user attribution.
  Keep it in a secret store, rotate it, and keep the API off the public internet.
- **Per-process limits.** Rate limits reset on restart and are not shared across replicas. All UI
  users share the UI container's quota.
- **The screen is pattern-based.** Injection that evades it still meets tool authorization, the
  SQL checks, the data-exposure policy and output validation. Those decide what can happen, not
  the screen.
- **TLS is external.** The services speak plain HTTP and rely on the proxy and network placement.
- **A network model provider** (`LLM_PROVIDER=anthropic`) sends questions and evidence summaries to
  the provider. The deterministic default sends nothing anywhere.
