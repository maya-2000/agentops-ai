# Deploying AgentOps (Phase 9)

AgentOps runs as two processes on one host: the **API** (FastAPI with the agent inside it) and the
**UI** (Streamlit, a client of the API). The architecture is the one built in Phases 4–8. Phase 9
adds what is needed to run it outside a developer's terminal: authentication, rate limiting, safe
configuration, structured logs, metrics, health and readiness, bounded runs, graceful shutdown,
containers and CI.

```
Browser ──▶ UI container (Streamlit, app/ui)          no database, no agent code, no data files
               │ HTTP + Authorization: Bearer <token>
               ▼
            API container (FastAPI, app/api)           auth · rate limit · validation · request IDs · logs · metrics
               │ AgentRunner.run(question, run_id, cancel, deadline)
               ▼
            LangGraph agent → secured tool executor → tools → analytics → DuckDB (read-only mount)
```

There is deliberately no Kubernetes, Redis, Postgres, message queue or cloud-specific code. One API
process owns one read-only DuckDB connection and serialises agent runs. That is the right size for
this workload: a run takes tens to hundreds of milliseconds with the deterministic model. See
[Known limitations](#12-known-limitations) for what that means for scaling.

Related documents: [security.md](security.md) (the security model), [api.md](api.md) (the HTTP
contract) and [ui.md](ui.md) (the page).

## 1. Local development

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m data.generator.generate                 # once: database/northwind_cloud.duckdb (~30 s)

cp .env.example .env
echo "API_AUTH_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env

python -m app.api                                 # terminal 1: http://127.0.0.1:8000
python -m app.ui                                  # terminal 2: http://localhost:8501
```

- The API and the UI read the same `.env`, so the UI sends the token without further setup. (The
  appended line overrides the empty `API_AUTH_TOKEN=` of the example: the last value wins.)
- `python -m app.api --check-config` validates the configuration and exits: 0 when the API would
  start, 2 with the list of problems otherwise.
- `python -m app.ui` launches Streamlit with hardened flags (headless, no usage statistics, XSRF
  protection, no file watcher, 1 MB upload cap; in production error details and the developer
  toolbar are hidden). `streamlit run app/ui/main.py` still works for development.
- For throwaway local experiments, `API_AUTH_MODE=disabled` turns authentication off (the start-up
  log line records `"auth": "disabled"`). The API refuses that setting when `APP_ENV=production`.

Calling the API by hand:

```bash
export API_AUTH_TOKEN=...   # the value from .env
curl -s http://127.0.0.1:8000/api/v1/ask \
  -H "Authorization: Bearer $API_AUTH_TOKEN" -H 'Content-Type: application/json' \
  -d '{"question": "What was revenue in July compared with June?"}'
```

## 2. Docker deployment

Files: `Dockerfile` (multi-stage, two targets), `docker-compose.yml`, `.dockerignore`.

```bash
python -m data.generator.generate                 # the data is mounted, never baked into an image
echo "API_AUTH_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env
docker compose up --build -d                      # UI http://127.0.0.1:8501, API http://127.0.0.1:8000
docker compose ps                                 # both services "healthy"
API_AUTH_TOKEN=... python scripts/smoke_test.py --api-url http://127.0.0.1:8000 --ui-url http://127.0.0.1:8501
docker compose stop                               # SIGTERM: graceful shutdown (§11)
docker compose down
```

`docker compose` refuses to start without `API_AUTH_TOKEN` (`${API_AUTH_TOKEN:?…}`).

**Images.**

| | `agentops-api:0.9.0` (target `api`) | `agentops-ui:0.9.0` (target `ui`) |
|---|---|---|
| Contents | `app/` + the `api` extra (FastAPI, uvicorn, LangGraph, DuckDB, analytics stack) in a virtualenv | `app/__init__.py`, `app/config.py`, `app/ui/` + Streamlit, httpx, pydantic |
| Not included | tests, evals, data, `data/seeds/` (ground truth), `.env`, dev tools, Streamlit | the agent, tools, analytics, DuckDB and every data file: the UI cannot open the database |
| User | `agentops` (uid 10001), non-root | `agentops` (uid 10001), non-root |
| Command | `python -m app.api` (exec form, so Python receives SIGTERM) | `python -m app.ui` |
| Healthcheck | `GET /api/v1/readiness` | `GET /_stcore/health` |
| Defaults | `APP_ENV=production`, `LOG_FORMAT=json`, `DATABASE_URL=duckdb:////data/database/northwind_cloud.duckdb` | `APP_ENV=production`, `UI_API_URL=http://api:8000` |

The build context is an allow-list (`.dockerignore` starts with `*` and re-admits only
`pyproject.toml`, `README.md` and `app/`), so a stray `.env`, database or seed file can never reach an
image. No secret is declared in the Dockerfile. A corporate TLS-inspecting proxy can be trusted
at build time with an optional BuildKit secret (`--secret id=pip_ca,src=<ca-bundle>`), which is not
stored in any layer.

**Compose hardening** (both services): read-only root filesystem, `cap_drop: [ALL]`,
`no-new-privileges`, `tmpfs` for `/tmp` (and Streamlit's home directory), ports published on
`127.0.0.1` only, `restart: unless-stopped`. Only the API mounts data, read-only:
`./database → /data/database:ro` and `./data/metadata → /data/metadata:ro`. The UI starts after the
API is healthy.

**Remote users.** Put a TLS-terminating reverse proxy (nginx, Caddy, a cloud load balancer) in front
of the UI, and set `UI_PUBLIC_ADDRESS` to the public host name. The API needs no public exposure: the
UI calls it on the compose network. If browsers or other services must call the API directly,
expose it through the same proxy and list their origins in `API_CORS_ORIGINS`.

## 3. Environment configuration

All settings are environment variables (or `.env`), declared and validated in `app/config.py`.
`.env.example` lists every one with a comment. `APP_ENV` selects the rule set:

| | `development` (default) / `test` | `production` |
|---|---|---|
| Authentication | token (default), or `disabled` | token required; `disabled` is refused |
| `API_AUTH_TOKEN` | required when auth is on | required |
| Rate limit | on (default `20/minute`), may be `off` | must be on |
| CORS origins | any explicit origins | no `*`, no `http://` |
| `DATABASE_URL` | default location if unset | must be set explicitly |
| Timeouts | as set | must nest: SQL ≤ tool ≤ agent run ≤ API request ≤ UI request |
| OpenAPI docs (`/docs`) | on | off (unless `API_DOCS_ENABLED=true`) |
| Streamlit | error details, full toolbar | no error details, viewer toolbar |

The API refuses to start while any rule is broken. It logs `configuration_rejected` with the
setting names (never their values) and exits. `--check-config` prints the same list.

Phase 9 settings:

| Variable | Default | Meaning |
|---|---|---|
| `APP_ENV` | `development` | `development`, `test` or `production` |
| `LOG_FORMAT` | `json` | `json` (one object per line) or `text` |
| `LOG_LEVEL` | `INFO` | Root log level |
| `API_AUTH_MODE` | `token` | `token` or `disabled` (development only) |
| `API_AUTH_TOKEN` | — | Bearer token, at least 32 characters, no whitespace |
| `API_RATE_LIMIT` | `20/minute` | Per-client limit on `/ask` and `/ask/stream`: `N/second`, `N/minute`, `N/hour` or `off` |
| `API_RATE_LIMIT_MAX_CLIENTS` | `10000` | Clients tracked by the limiter (least recently seen are dropped first) |
| `API_CORS_ORIGINS` | empty | Comma-separated `scheme://host[:port]` origins; empty = no cross-origin access |
| `API_DOCS_ENABLED` | empty | Empty = on except in production |
| `API_METRICS_ENABLED` | `true` | `false` makes `/api/v1/metrics` a 404 |
| `API_REQUEST_TIMEOUT_SECONDS` | `150` | Per-request wall clock (504 after it; the run is cancelled) |
| `API_MAX_REQUEST_BYTES` | `16384` | Largest accepted body (413) |
| `API_MAX_PENDING_REQUESTS` | `4` | Waiting requests before 503 `busy` |
| `API_SHUTDOWN_GRACE_SECONDS` | `10` | Time in-flight runs get on SIGTERM |
| `UI_HOST` / `UI_PORT` | `127.0.0.1` / `8501` | Where `python -m app.ui` listens |
| `UI_PUBLIC_ADDRESS` | `localhost` | The host name users browse to |
| `UI_HISTORY_LIMIT` | `20` | Questions kept per browser session, in memory |

Why `20/minute`: a deterministic run takes 5–500 ms, so an interactive user rarely asks more than a
few questions a minute. Twenty allows bursts (a demo, example questions clicked in a row) and still
bounds a client to about a third of a worker's capacity for the slowest questions (forecasts). With
a network model, where a run costs money and seconds, a lower limit may suit better.

Secrets (`API_AUTH_TOKEN`, `ANTHROPIC_API_KEY`) come only from the environment. They are held as
`SecretStr`, never printed by `repr`, never included in validation errors
(`hide_input_in_errors`), and registered for log redaction.

## 4. Authentication

- Every `/api/v1/*` endpoint requires `Authorization: Bearer <API_AUTH_TOKEN>`, except
  `/api/v1/health` and `/api/v1/readiness` (orchestrator probes).
- The comparison is constant-time (`hmac.compare_digest`). Missing, malformed (`Basic …`, extra
  spaces, empty token) and wrong credentials all get the same response: `401`,
  `WWW-Authenticate: Bearer`, and `{"error": {"code": "unauthorized", "message": "Missing or invalid
  credentials."}}`. Unknown `/api/v1` paths also answer 401 before routing, so the API cannot be
  explored without a token.
- The check runs in middleware before the body is read or validated, so an unauthenticated request
  costs almost nothing and never reaches the agent.
- The UI sends the token from its own environment. It never shows it, and the client's `repr`
  reports only `authenticated=True`.
- Rotation: change `API_AUTH_TOKEN` and restart both services (`docker compose up -d`).

It is one shared service token, not user accounts: it authenticates the UI (and operators) to the
API. User sign-in, if needed, belongs in the reverse proxy in front of the UI.

## 5. Rate limiting

- `POST /api/v1/ask` and `/ask/stream` are limited per client with a sliding window: at most N
  requests in any window (not a fixed per-minute bucket, so there is no burst at window edges).
- The 21st request within a minute gets `429`, `{"error": {"code": "rate_limited", …}}` and
  `Retry-After` (seconds until a slot frees). Rejected requests do not consume the quota.
- The check comes after authentication, so unauthenticated floods are answered with 401 and cannot
  exhaust a real client's quota.
- The client is the peer address. Behind a reverse proxy on the same host, uvicorn takes the client
  address from `X-Forwarded-For` for proxies it trusts (`FORWARDED_ALLOW_IPS`, default
  `127.0.0.1`). In the compose setup every UI user reaches the API from the UI container, so they
  share one quota: size `API_RATE_LIMIT` for the whole UI, or rate-limit users at the proxy.
- State is in memory and per process, bounded to `API_RATE_LIMIT_MAX_CLIENTS` entries. It resets on
  restart and is not shared between processes. That is sufficient for the single-process
  deployment; several API replicas would need a shared store (for example Redis), which is out of
  scope.

Other request limits: `application/json` only on the ask endpoints (415), `API_MAX_REQUEST_BYTES`
(413, declared or streamed), `AGENT_MAX_QUESTION_CHARS` (422), request and session IDs of at most 64
safe characters, unknown fields rejected, and `API_MAX_PENDING_REQUESTS` (503 `busy`).

## 6. Health and readiness

| Endpoint | Auth | Meaning | Response |
|---|---|---|---|
| `GET /api/v1/health` | public | **Liveness**: the process serves HTTP. Checks nothing else, so a database problem never causes a restart loop | `200 {"status": "ok", "version": "0.9.0"}` |
| `GET /api/v1/readiness` | public | **Readiness**: requests can be served now | `200` or `503`, `{"status": "ready" \| "not_ready", "version", "checks": {"configuration", "database", "agent", "accepting_requests"}}` |

Readiness fails when the database could not be opened or stops answering a metadata query, when the
agent could not be built, or while the service drains on shutdown. It returns named booleans only:
no paths, settings or error text. The Docker healthcheck, `docker compose up --wait` and the UI's
sidebar ("API ready" / "API not ready") use it.

## 7. Logging

Logs go to stderr, one JSON object per line (`LOG_FORMAT=json`), for `docker logs` or any collector:

```json
{"timestamp": "2026-09-26T15:50:55.617+00:00", "level": "INFO", "logger": "agentops.api", "auth": "token", "environment": "production", "event": "service_started", "rate_limit": "20/60s", "version": "0.9.0"}
{"timestamp": "2026-09-26T15:51:04.681+00:00", "level": "INFO", "logger": "agentops.agent", "event": "transition", "node": "question_received", "route": "understand_question", "run_id": "R-e843e53d2e95"}
{"timestamp": "2026-09-26T15:51:04.743+00:00", "level": "INFO", "logger": "agentops.api", "agent_status": "completed", "agent_time_ms": 65.3, "claim_count": 3, "duration_ms": 67.7, "endpoint": "/api/v1/ask", "event": "http_request", "evidence_count": 3, "method": "POST", "outcome": "answered", "request_id": "R-e843e53d2e95", "status_code": 200, "tool_calls": 1}
```

- **One request, one ID.** The request ID (a valid `X-Request-ID`, else generated) is the agent's
  run ID. It appears in the response header and body, in errors, and in every `agentops.api`,
  `agentops.agent` and `agentops.security` line of that request. An invalid or oversized header is
  replaced, never echoed.
- **Fields.** Request lines carry the method, the route template (`/api/v1/ask`, never a raw path),
  status, outcome, agent status, error code, counts and durations. Service lines cover start,
  stop, unavailability (with a reason such as `database_missing` and a hint) and rejected
  configuration. Tool calls, security decisions and failures are logged by the agent with the tool
  name and a fixed error category.
- **Never logged:** the question or answer, headers (including `Authorization`), tokens, API keys,
  environment values, client addresses (uvicorn's access log is off), prompts, model reasoning,
  tool outputs, SQL error text, file paths and ground truth. Request lines are built from an
  allow-list of keys. Every value passes through secret redaction, and plain-text messages from
  libraries are redacted, stripped of paths and truncated. Exceptions are reduced to their class
  name: no tracebacks.

`LOG_FORMAT=text` gives the classic one-line format for a terminal.

## 8. Metrics

`GET /api/v1/metrics` (authenticated; `API_METRICS_ENABLED=false` turns it off) returns in-process
counters since start-up:

- `summary`: expected outcomes kept apart from errors: `answered`, `partial`, `refused`,
  `unsupported`, `insufficient_evidence`, `failed` (tool or planning failure); `client_errors`,
  `unauthorized`, `rate_limited`; `timeouts`, `unavailable` (busy, starting, draining),
  `internal_errors`.
- `by_outcome`, `by_status_code`, `by_error_code`.
- `request_latency` and `agent_latency`: count, mean, p50, p95 and max over the last 1,000 ask
  requests.
- `runs`: `queued`, `running` and `stopping` now, and how runs ended: `completed` (the agent
  concluded, including insufficient-evidence and partial answers), `refused` (refused or
  unsupported), `failed`, `timeout` and `cancelled` (client gone or shutdown).
- `uptime_seconds`, `requests_total`, `in_flight`, and average agent time and API overhead.

A refusal is an expected outcome, not an error: it counts under `refused`, never under
`internal_errors`. Metrics carry no question-derived labels, so they cannot leak content. They are
JSON rather than Prometheus text. A scraper can convert them, and the service stays
dependency-free.

## 9. Data lifecycle

- **Build.** `python -m data.generator.generate` writes `database/northwind_cloud.duckdb` and
  `data/metadata/dataset_manifest.json` (dataset version, checksums). It is deterministic
  (`DATA_SEED`, `AS_OF_DATE`). Nothing is generated inside a container.
- **Serve.** The API opens the file read-only, once, at start-up. One connection is shared by all
  runs, one run at a time; readiness probes use it only when no run holds it. In Docker the
  directory is mounted read-only (`:ro`), and the root filesystem is read-only too.
- **Missing data.** If the file is absent, the API logs `service_unavailable` with reason
  `database_missing` and the hint `python -m data.generator.generate`. In production the process
  then exits, so the container shows as unhealthy and restarts instead of serving 503s silently.
  In development the API starts, readiness reports `database: false`, and `/ask` answers 503
  `agent_unavailable`.
- **Refresh.** Regenerate (or copy in) a new file and restart the API. The dataset version in
  `/capabilities` and the UI sidebar comes from the manifest.
- **Ground truth.** `data/seeds/` holds the injected-event labels used only by the evaluation
  harness. It is excluded from both images and from the compose mounts, and no route serves files.

There is no database server and no migration step: DuckDB is an embedded, read-only analytical
store here. Postgres would add operations without adding a capability this workload uses.

## 10. Security

The full model is in [security.md](security.md). In short:

- The API and UI are entry points to the agent, never a path around it. Every question goes
  through `AgentRunner.run` and the Phase 5 secured executor: injection screening, tool
  authorization, SQL safety, the data-exposure policy, budgets, deadlines and output validation.
- Authentication, rate limiting, content-type and size checks happen before the agent is reached.
- Responses carry fixed error messages, a request ID and security headers
  (`X-Content-Type-Options: nosniff`, `Cache-Control: no-store`, `Referrer-Policy: no-referrer`,
  `X-Frame-Options: DENY`, and `Content-Security-Policy: default-src 'none'; frame-ancestors
  'none'` outside the docs pages). No traceback, path, SQL, class name or environment value
  appears in any response.
- Containers run as non-root with a read-only filesystem and no capabilities, and contain no
  secrets or ground truth.
- TLS is terminated by a reverse proxy; the services themselves speak plain HTTP on `127.0.0.1` or
  the compose network.

## 11. Shutdown

On SIGTERM or SIGINT (`docker stop`, `docker compose stop`, Ctrl-C):

1. uvicorn stops accepting connections and gives in-flight requests `API_SHUTDOWN_GRACE_SECONDS`
   (default 10 s) to finish. Requests still open after that are cancelled, and so are their runs.
2. The service drains. Readiness turns `not_ready`, and an ask request that still arrives gets
   `503 shutting_down` with `Retry-After`. Any run still going gets up to the grace period again to
   end.
3. After that, the remaining runs are cancelled cooperatively. Each stops at its next step with the
   controlled "limit reached" response: claims and evidence that were not yet validated are
   dropped, so there is never a partial answer.
4. The worker thread and the read-only database connection are closed, and `service_stopped` is
   logged. The process exits with code 0.

Compose gives the API 20 s (`stop_grace_period`) before SIGKILL. That covers the grace period
plus a cancelled run's last step, unless a single tool step runs close to its own 30 s timeout.
In the smoke test both containers stop in about 1.5 s with exit code 0.

**Timeouts and runs.** Limits nest from the inside out:

- the SQL statement timeout (10 s) sits inside the tool timeout (30 s);
- the tool timeout sits inside the agent run (120 s);
- the agent run sits inside the API request (150 s);
- the API request sits inside the UI wait (180 s).

When the API timeout fires, the client gets 504 at once. The run's cancel event is then set: the
graph stops before its next node, DuckDB queries are interrupted at the deadline, and model calls
wait no longer than the time left. So a timed-out run releases the worker within one step, instead
of running to its own limit as in Phase 8. A client that disconnects cancels its run the same way.
Runs are tracked in memory only as counters (queued, running, stopping, final states). There is no
`/runs` endpoint: runs are synchronous and bounded, so there is nothing to poll.

## 12. Known limitations

- **One API process, one run at a time.** DuckDB is opened read-only by one connection, and runs
  are serialised. Throughput is roughly 3–35 questions per second with the deterministic model,
  far lower with a network model. Scaling out means several API containers, each with its own
  mounted copy, behind a load balancer. The in-memory rate limit and metrics would then be per
  replica.
- **Rate limits and metrics are in memory** and reset on restart. All UI users share the UI
  container's quota (§5).
- **One shared token, no user accounts or roles.** Put user authentication in the reverse proxy.
- **No TLS in the containers.** Terminate TLS at a proxy; ports bind to `127.0.0.1` by default.
- **Cooperative cancellation** stops a run between steps: a single tool step that is already
  running finishes first, bounded by its own tool timeout (and the SQL interrupt for queries).
- **Session history is per browser tab and in memory** (at most `UI_HISTORY_LIMIT` entries, each
  truncated). There is no long-term memory, by design.
- **Images are about 950 MB (API) and 840 MB (UI) uncompressed** (216 MB and 192 MB compressed),
  mostly the scientific Python stack (NumPy, pandas, statsmodels, DuckDB).
- **CI runs the critical suite on every pull request.** The full 89-scenario benchmark and the
  multi-seed check run on `main`, weekly and on demand (`evaluation.yml`), to keep pull-request CI
  under a few minutes.
