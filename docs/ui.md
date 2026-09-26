# AgentOps web UI (Phases 8–9)

A Streamlit page where a business user asks a question and sees the answer, the evidence behind
it, and how it was produced. The UI is a client of the [HTTP API](api.md).

```
Browser ─▶ Streamlit page (app/ui/main.py) ─HTTP─▶ AgentOps API ─▶ agent ─▶ secured tools ─▶ data
```

The UI never opens the database, never imports the agent, tools, analytics, evidence or security
code, and never reads data files. It never calculates a business number: it formats and draws what
the API returns. `tests/api/test_api_isolation.py` checks these rules statically.

## Running it

```bash
pip install -e ".[dev]"             # or: pip install -e ".[api,ui]"
python -m data.generator.generate   # once: builds database/northwind_cloud.duckdb
cp .env.example .env
echo "API_AUTH_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env
python -m app.api                   # terminal 1: the API on http://127.0.0.1:8000
python -m app.ui                    # terminal 2: the UI on http://localhost:8501
```

- The UI finds the API at `UI_API_URL` (default `http://127.0.0.1:8000`) and waits up to
  `UI_REQUEST_TIMEOUT_SECONDS` (default 180) for an answer.
- It sends `Authorization: Bearer <API_AUTH_TOKEN>` from its own environment. It never displays
  the token, and the client's `repr` hides it.
- If the API is not running, the sidebar shows "API not reachable", and asking a question explains
  how to start it. If the token is missing or wrong, the sidebar says the UI is not authorised.

`python -m app.ui` is the Phase 9 launcher: Streamlit in headless mode, bound to
`UI_HOST:UI_PORT`, with these settings:

- XSRF protection on;
- no usage statistics, and no public-IP lookup (`UI_PUBLIC_ADDRESS` names the address users
  browse to);
- no file watcher, and uploads capped at 1 MB.

With `APP_ENV=production` it also hides exception details and the developer toolbar. In Docker it
runs in its own image, which contains only `app/config.py` and `app/ui/`
([deployment.md §2](deployment.md#2-docker-deployment)). `streamlit run app/ui/main.py` still works
for development.

## What the page shows

| Section | Content | From the response |
|---|---|---|
| Header | "AgentOps AI", "Evidence-backed AI Business Intelligence" and a one-line description | — |
| Sidebar | API status ("API ready", "API not ready" with the failed checks, "API not reachable"), an authorisation error when the token is missing or wrong, data as-of date, dataset version, model provider, API version; a "Clear history" button | `GET /readiness`, `GET /capabilities` |
| Question | A large text box and an **Analyze** button; clickable example questions | `GET /capabilities` (`example_questions`), with a built-in fallback list |
| Progress | A status box that follows the agent's stages ("Understanding the question…", "Running analysis tools…") | `/ask/stream` progress events |
| Answer | The answer, the period (and comparison period), the request ID, caveats and assumptions | `answer`, `period`, `comparison_period`, `response.caveats`, `response.assumptions` |
| Key figures | KPI cards: value, period, and the percentage change for changes (shown with its sign, never coloured as good or bad) | `kpi_card` spec |
| Findings | One card per claim, primary claims first and in bold. A badge gives the type: **Observed**, **Calculated**, **Inferred**, **Recommended**. Forecast and anomaly claims get a second badge (**Forecast**, **Forecast quality**, **Anomaly check**). Each card shows its support status and cited evidence | `claims` |
| Forecast | A notice that a forecast is an estimate, not observed data. Horizon, model, data cut-off and interval; the forecast points with lower and upper bounds; backtest error, interval coverage and the naive baseline; limitations | `forecasts` |
| Anomalies | A notice that an anomaly is not necessarily bad and does not explain the cause. Detector, window and threshold; each flagged month with observed and expected values, deviation, score, severity and direction ("Above expected" or "Below expected") | `anomalies` |
| Charts | Vega-Lite charts drawn from the chart specs: period comparison, breakdown bars (the tool's order; two neutral colours for sign), monthly series, forecast (history + forecast + interval band), anomaly (series, expected line and range, markers on flagged months). Breakdown tables sit in expanders | `visualizations` |
| Evidence & Provenance | A table of every evidence item: statement, metric, value, period, comparison, dimension, filters, source tables, calculation, query IDs and tool | `evidence` |
| Analysis Trace | The agent's stages with ✓ (or ■ where the run stopped) and durations. Under "Running analysis tools", each tool call with ✓/✗, time and the plan's purpose, plus totals | `run.stages`, `trace` |
| Earlier in this session | Previous questions with their outcome and answer | session state |

What is deliberately not shown: prompts, model reasoning, raw tool arguments, security events and
their pattern names, and anything the API does not return.

### Refusals, uncertainty and errors

| Outcome | Shown as |
|---|---|
| `refused` | Red: **"I can't answer that safely from the available data."** The agent's explanation and a one-line reason at a user level ("The request asked for something the agent is not permitted to do"), then what to ask instead. No evidence, charts or findings |
| `unsupported` | Blue: "I can't answer that from the available data." What AgentOps can analyse |
| `insufficient_evidence` | Amber: "There is not enough evidence for a reliable answer." How to make the question answerable (name the metric and period, or ask a narrower question) |
| `partial` | Amber: only validated findings are shown |
| `failed` | Red, with the request ID for the logs |
| API errors | Unreachable, timeout, busy, not authorised, rate-limited, empty or too-long question: a short title, the API's fixed message and a hint (for example "Set the same API_AUTH_TOKEN for the UI and the API", "Wait a moment, then ask again") |

Security implementation details (patterns, screening categories, policies) are never shown.

## Session history

Questions and answers of the current browser session are kept in `st.session_state`. Memory is
bounded:

- at most `UI_HISTORY_LIMIT` entries (default 20), newest first;
- question and answer texts are truncated to 1,000 characters;
- when the API answered, the stored question is its redacted copy, so a secret pasted into a question
  is not kept (after a failed request, the typed text is kept, as it is still in the text box).

Nothing is written to disk or sent anywhere else. Closing the tab or pressing "Clear history"
discards them. There is no long-term memory, and earlier questions are not sent back to the agent
as context. Each browser session sends a random `session_id` (`ui-…`) with its
requests, so the API log lines of one session can be correlated. The API stores nothing per
session.

## Code layout and testing

| Module | Role | Tests |
|---|---|---|
| `app/ui/client.py` | `AgentOpsClient(base_url, timeout=…, token=…)`: `health`, `readiness`, `capabilities`, `ask`, `ask_stream`. Sends the bearer token. Failures become `APIFailure(kind, message, code, request_id)`; only the API's own error envelope is shown | `tests/ui/test_ui_client.py`: mock transport and the real API in-process; `tests/ui/test_ui_production.py`: token handling, the page against the authenticated production API, bounded history, the launcher flags |
| `app/ui/view_models.py` | Pure functions from response JSON to what is shown: `answer_view`, `claim_items`, `evidence_rows`, `trace_rows`, `kpi_cards`, `vega_lite`, `table_rows`, `forecast_panel`, `anomaly_panel`, `error_view`, `history_entry`, formatting and markdown escaping | `tests/ui/test_ui_view_models.py`: run on real API responses, including forecast, anomaly and investigation answers from the full dataset |
| `app/ui/render.py` | Streamlit layout of the view models | exercised by the page tests |
| `app/ui/main.py` | The page: header, sidebar, question form, examples, progress, history | `tests/ui/test_ui_app.py`: `streamlit.testing.v1.AppTest` runs the page headless against the real API in-process (answer, refusal, unsupported, history, unreachable API) |

The tests need no browser, server, network or API key.

**Rendering safety.** Every dynamic text (question, answer, claims, evidence statements, notices)
is escaped before it reaches Streamlit markdown, so it is shown literally: no links, images, HTML,
LaTeX or Streamlit directives. Charts are Vega-Lite specifications built from the API's rows. No
`unsafe_allow_html` is used.

## Known limitations

- **One page, no user sign-in.** The UI authenticates to the API with the service token. Users
  are authenticated, if needed, by a reverse proxy in front of it. The Phase 0 plan's multipage
  dashboard (KPI dashboard, anomaly monitor, evaluation page) is not built: every view answers a
  question through the agent.
- **All UI users share one API rate-limit quota**, because the API sees the UI as one client.
  Size `API_RATE_LIMIT` for the whole UI.
- **History lasts for the browser session** and is not persisted (by design).
- **Chart types are those the API provides** (§6 of [api.md](api.md)). Questions whose evidence
  has no chartable shape show tables and evidence only.
- **The progress box reports finished agent stages.** With the deterministic provider an answer
  takes well under a second, so the box mostly shows its final state.
