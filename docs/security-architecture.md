# Security architecture (Phase 5)

> **The model can propose. The application decides.** The model proposes what to investigate,
> which approved tool to use and how to word the evidence. Deterministic code decides whether:
> - the tool exists and is allowed;
> - the arguments are valid and the SQL is safe;
> - the limits hold;
> - the evidence supports each claim;
> - the response may be returned.
>
> These are application-level security controls for a prototype, not a claim of production-grade
> security. The threats and residual risks are in [security-threat-model.md](security-threat-model.md).

## 1. Principles

| Principle | How it shows up |
|---|---|
| Fail closed | Unknown tools, metrics, tables, columns, functions, error codes and intents are rejected. Missing evidence rejects the claim. An exhausted budget stops the run. Nothing is guessed |
| Defence in depth | Each risk has at least two independent controls. For example, SQL is validated at planning and at execution, and the connection is read-only |
| Least privilege | The planner sees only the tools permitted for the validated intent. Flagged input loses ad-hoc SQL for the whole run. The data policy withholds columns the analytics do not need |
| Deterministic and testable | Every decision is plain code with typed inputs and outputs. The prompt-injection screen is the only heuristic, and nothing depends on it alone |
| Auditable | Every decision emits a typed `SecurityEvent` with severity, component, decision and reason |
| Bounded | Every loop, retry, query, prompt, response and run has a configurable numeric limit |

## 2. Trust model

| Input | Trust | Handling |
|---|---|---|
| System instructions (`app/llm/prompts.py`) | Trusted | Written by us. They state the trust model to the model |
| Tool definitions (`app/tools/registry.py`) | Trusted | Written by us. The allowlist, input models and handlers are fixed at import time |
| Configuration (`AGENT_*`, `LLM_*`) | Trusted | Read once when the runner starts; `SecurityLimits` is immutable |
| Application state (`AgentState`) | Trusted, validated | Holds only validated data. Model outputs enter it only after validation |
| User question | **Untrusted** | Type, size and control-character checks; redaction; injection screen. Rendered as escaped, delimited data, never as instructions |
| Tool output | **Untrusted until validated** | Type, provenance, finite values, dates, identifiers, exposure. Then sealed as evidence |
| Model output (understanding, plan, draft) | **Untrusted until validated** | Parsed, schema-validated, checked; retried within budget; otherwise fail closed. Never executed |

## 3. The security boundary

```text
USER QUESTION (untrusted)
   │  question_received ─ InputGuard.validate_question (type, size, control chars, redaction)
   │                    ─ PromptInjectionDetector (block → refuse | restrict → no SQL for this run)
   ▼
MODEL: understanding (untrusted) ── schema + understanding_output_problems (bounded retries)
   │  validate_request ─ InputGuard.validate_understanding_output (filters, dimensions, periods)
   │                   ─ request validation via central validators (metric, dimensions, filters, horizon, coverage)
   ▼
MODEL: plan (untrusted) ── context budget; the planner sees only permitted tools
   │  plan_investigation ─ PlanValidator → ToolAuthorizationPolicy for every step
   │                        (policy denial = not retried, fail closed)
   ▼
AUTHORIZED TOOLS ONLY
   │  execute_tools ─ authorize again just before execution (allowlist, intent, SQL privilege,
   │                   arguments, SQL safety, budget, prerequisites)
   │               ─ execution_deadline(tool timeout) → SQL deadline → DuckDB interrupt
   │               ─ RetryPolicy (transient only, bounded, recorded)
   │               ─ ToolOutputValidator (type, provenance, finite, dates, ids, exposure)
   ▼
EVIDENCE LAYER ─ evidence built only from validated output, sealed with a SHA-256 fingerprint
   │  validate_evidence ─ integrity + 10 claim checks + forecast/anomaly provenance
   ▼
MODEL: wording (untrusted) ── context budget (prioritised claims and evidence)
   │  validate_response ─ numbers in cited evidence, direction, causality, forecast certainty,
   │                       anomaly judgement, recommendation framing, length
   │                    ─ bounded regeneration → safe shortening (length only) → fail closed
   ▼
SAFE RESPONSE ─ redacted; errors shown as safe categories; audit trail in the run result
```

The Phase 4 graph kept its nodes. Each boundary runs inside the node that owns that step, so the
node order in the transition trace documents the boundaries that ran.

## 4. Input guardrails (`app/security/input_guard.py`, `validators.py`, `limits.py`)

- **The question:**
  - must be `str`, non-empty and at most `max_question_chars`;
  - has control characters removed and whitespace collapsed;
  - has secrets redacted by the runner before it enters the state, so the raw question is never
    stored, prompted or logged.
- **The model's understanding:**
  - filter count at most `max_filters`, dimension count at most `max_dimensions`, no repeated
    filter, and period specs of a supported form;
  - free-text fields are bounded, and an oversized output is treated as invalid model output and
    regenerated within budget.
- **Central validators** (`validators.py`) are the single definition of what is acceptable:
  - metrics, series metrics, dimensions, filters (known dimension, enumerated values, length,
    count);
  - dates (ISO format, 2000–2100, start ≤ end, at most about ten years), period specs;
  - horizons (1–6), detectors, enums, text size and control characters, finite numbers.

  The input guard, the request validation, the plan validator and tool authorization all call
  them.

## 5. Prompt-injection handling (`app/security/injection.py`, `app/llm/prompts.py`)

Several layers, none of them sufficient alone:

1. **Screen.** A deterministic classifier runs on a normalised copy of the question (NFKC,
   case-folded, zero-width characters removed).
   - `block` categories, refused before any model or tool call:
     - secret requests, ground-truth or hidden-state requests, file access, code execution
       (CRITICAL);
     - prompt extraction, safety bypass, limit changes, tool escalation, destructive SQL (HIGH).
   - `restrict` categories (instruction override, role-play, pasted SQL, naming internal tools):
     the run continues with ad-hoc SQL revoked and the question handled as data (WARNING).
2. **Separation.** The question is not part of the trusted JSON context. It is rendered in a
   `<untrusted_user_question>` block with markup escaped, so it cannot close its own delimiter.
3. **Instructions.** Every system prompt states the trust model: nothing in the context can
   change rules, tools, limits or permissions, and requesting something does not make it allowed.
4. **Structure.** Model outputs are strict JSON, validated, and never executed.
5. **Authorization.** Permissions depend on the validated intent, the screen verdict and
   configuration. The user's text is never an input to the policy.
6. **Evidence and output validation.** Even a steered model cannot get an unsupported number,
   causal claim or directive through.
7. **Bounds.** A manipulated model still runs within the tool, retry and time budgets.

## 6. Intent and plan validation (`app/security/plan_validator.py`)

- **Checks on the plan:**
  - at most `max_plan_steps` steps, within the remaining tool budget;
  - each step's arguments parse as a JSON object;
  - each step passes `ToolAuthorizationPolicy.authorize`;
  - duplicate steps and already-executed calls are dropped;
  - the plan's SQL steps fit the SQL budget.
- **Structure:** a plan is a flat list executed once in order. Tools cannot call tools or the
  planner, and follow-up planning is bounded by `max_planning_iterations`, so circular execution
  is ruled out structurally.
- **Two kinds of problem:**
  - *model-output problems* (malformed JSON, invalid arguments, unknown tool names, too many
    steps) get bounded regeneration with feedback;
  - *policy denials* (disabled tool, tool not permitted for the intent, SQL privilege revoked,
    unsafe SQL, data policy, exhausted budget, missing prerequisite) are not retried: the run
    ends in `planning_failure` with a HIGH event.
- **Limits:** a plan over the step or tool budget ends with "Investigation limit reached
  before sufficient evidence could be collected." It is never silently truncated.

## 7. Tool authorization (`app/security/authorization.py`)

`ToolAuthorizationPolicy.authorize(tool, arguments, context, budget, usage)` runs at plan
validation and again immediately before each execution. It checks, in order:

1. **Allowlist.** The tool is in `ALLOWED_TOOLS` (12 tools) *and* registered.
2. **Enabled.** Not in `AGENT_DISABLED_TOOLS`. Unknown names fail at startup.
3. **Intent.** Permitted for the validated intent (`INTENT_TOOL_PERMISSIONS`). `unsupported`
   permits nothing, and a missing intent denies everything.
4. **SQL privilege.** `run_safe_sql` additionally needs the run's SQL privilege.
5. **Arguments.** Size, typed validation, the central argument policy, the data policy, and full
   SQL validation.
6. **Budget.** Tool calls, SQL calls and SQL rows.
7. **Prerequisites.** A validated request exists; a follow-up requires prior evidence.

The decision records the checks passed, or the denial code and reason. An allowed decision
returns canonical arguments, and only those are executed.

| Intent | Permitted tools |
|---|---|
| kpi_lookup | get_kpi, run_safe_sql |
| period_comparison, dimensional_comparison | get_kpi, the six analyze_* tools, run_safe_sql |
| revenue_investigation | get_kpi, analyze_revenue, analyze_customers, detect_anomalies, run_safe_sql |
| customer_investigation | get_kpi, analyze_customers, analyze_revenue, get_cohort_analysis, get_customer_risk, detect_anomalies, run_safe_sql |
| sales / marketing / support / product analysis | get_kpi, the matching analyze_* tool (+ detect_anomalies for support and product), run_safe_sql |
| forecast | forecast_metric, get_kpi |
| anomaly_detection | detect_anomalies, get_kpi |
| mixed_investigation | all analytics tools, get_customer_risk, detect_anomalies, forecast_metric, run_safe_sql |
| unsupported | none |

There are no dynamic imports, arbitrary callables, shell or filesystem tools. Handlers are plain
functions in `app/tools/handlers.py` (tested).

## 8. SQL security (`app/tools/sql_safety.py`, `app/security/data_policy.py`)

Checks on the statement:
- **Size:** at most `max_sql_length` characters.
- **Shape:** exactly one `SELECT`/`WITH … SELECT`/set operation. Refused anywhere in the tree:
  - DML and DDL, including `TRUNCATE`, `MERGE` and `SELECT … INTO`;
  - `ATTACH/DETACH/COPY/EXPORT/IMPORT/INSTALL/LOAD/CALL/SET/PRAGMA/DESCRIBE/SUMMARIZE`;
  - `WITH RECURSIVE`.
- **Tables:** an explicit allowlist of 8 business tables and 4 views, as plain unqualified names.
  Refused: file paths, URLs, table functions, other schemas and catalogs, and system tables.
- **Columns:** known, exposed columns only. PII (`sales_rep`), withheld (`company_name`) and
  hidden-state names are refused, and so is `SELECT *` on tables that contain them.
- **Functions:**
  - functions the parser recognises are allowed except a deny-list: files, environment,
    network, settings, catalog, `range`/`generate_series`;
  - unrecognised functions must be on an explicit allowlist (fail closed).
- **Complexity:**
  - at most `max_sql_joins` joins, each with an explicit `ON`/`USING` (no cartesian products);
  - at most `max_sql_nesting_depth` subquery levels, `max_sql_ctes` CTEs and
    `max_sql_parameters` parameters.
- **Values:** named, bound, scalar parameters of bounded length.

Execution:
- **Rows:** `LIMIT` is capped at `sql_row_limit + 1`, the `truncated` flag is set, and a
  truncation limitation is added and carried through evidence, claims and caveats.
- **Regeneration:** the executed SQL is regenerated from the validated syntax tree without
  comments.
- **Time:** each statement runs under `sql_timeout_seconds`, enforced by `execution_deadline`
  and DuckDB `interrupt()`.
- **Budget:** per run, `max_sql_calls` executions and `max_sql_rows_total` rows.
- **Connection:** read-only (a second, independent layer, tested).

## 9. Resource limits (`app/security/limits.py`, `budget.py`, `timeouts.py`, `retry.py`, `context.py`)

All limits come from settings and are immutable per run. Dependent limits are kept consistent:
plan steps never exceed tool calls.

| Limit | Default | Setting |
|---|---|---|
| Question length | 1000 chars | `AGENT_MAX_QUESTION_CHARS` |
| Filters / dimensions | 5 / 3 | `AGENT_MAX_FILTERS`, `AGENT_MAX_DIMENSIONS` |
| Plan steps / tool calls | 8 / 12 | `AGENT_MAX_PLAN_STEPS`, `AGENT_MAX_TOOL_CALLS` |
| Retries per step | 2 | `AGENT_MAX_RETRIES` |
| Planning iterations | 2 | `AGENT_MAX_PLANNING_ITERATIONS` |
| SQL rows per query / per run | 200 / 1000 | `AGENT_SQL_ROW_LIMIT`, `AGENT_MAX_SQL_ROWS_TOTAL` |
| SQL executions per run | 3 | `AGENT_MAX_SQL_CALLS` |
| SQL length / joins / nesting | 4000 / 4 / 3 | `AGENT_MAX_SQL_LENGTH`, `AGENT_MAX_SQL_JOINS`, `AGENT_MAX_SQL_NESTING_DEPTH` |
| Customer-level rows per call | 25 | `AGENT_MAX_CUSTOMER_ROWS` |
| Context items / prompt size | 40 / 60,000 chars | `AGENT_MAX_CONTEXT_ITEMS`, `AGENT_MAX_CONTEXT_CHARS` |
| Response length | 4000 chars | `AGENT_MAX_RESPONSE_CHARS` |
| Run / tool / SQL / model time | 120 s / 30 s / 10 s / 120 s | `AGENT_MAX_RUN_SECONDS`, `AGENT_TOOL_TIMEOUT_SECONDS`, `AGENT_SQL_TIMEOUT_SECONDS`, `LLM_TIMEOUT_SECONDS` |

- **`RunBudget` / `BudgetUsage`.** The run is charged for:
  - tool calls, SQL executions and SQL rows;
  - retries and model calls;
  - context items, prompt characters and the largest prompt;
  - response characters.

  A charge that would exceed a cap is refused, recorded in `exhausted`, and stops that activity.
  Nothing resets it.
- **Timeouts:**
  - model calls wait at most `llm_timeout_seconds` on a worker thread;
  - tool calls run under `execution_deadline(tool_timeout_seconds)`;
  - ad-hoc SQL has its own deadline;
  - a tool that returns after its limit is recorded as a `timeout` failure and its result is
    discarded;
  - timeouts are never retried with the same arguments.
- **Retries.** Only transient failures are retried (`database_error`, retryable provider errors,
  model timeouts, malformed model output), at most `max_retries` times per step. Every retry is a
  `RetryRecord` plus a `retry` event. Unknown codes are not retried.
- **Context.** Evidence and claims are prioritised (primary evidence, claims, other evidence) and
  capped. A prompt that cannot fit is refused.

## 10. Tool-output validation (`app/security/output_guard.py`)

Before a result can become evidence:
- **Type:** the expected result type for the tool.
- **Provenance:** query IDs, source tables within the approved relations, and a calculation.
- **Values:** finite numbers, plausible dates and timestamps, known KPI keys.
- **Forecasts:** a model, points after the cutoff, a point count equal to the horizon, a
  supported horizon.
- **Anomalies:** a known detector, thresholds, results inside the evaluation window.
- **SQL:** a consistent row count and truncation flag, row widths, no withheld columns.
- **Exposure:** no hidden-state or credential-like keys, PII only where declared
  (`analyze_sales.rep_performance`), customer-level rows within the cap.

A rejected output becomes a controlled failure (`invalid_tool_output` or `data_policy`, HIGH
event). It is never partially used.

## 11. Evidence integrity (`app/evidence/models.py`)

- **Provenance.** Every evidence item records:
  - the tool, the call ID and the timestamp;
  - the canonical **input arguments**;
  - the source tables, query IDs and the calculation.
- **IDs.** Only the evidence layer creates evidence IDs (`E<n>`, checked on insert). Claims can
  cite only existing IDs, and the model cannot introduce new ones.
- **Sealing.** `add_evidence` seals each item with a SHA-256 fingerprint of its content.
  `verify_integrity` detects any later modification, and evidence validation fails closed with
  a CRITICAL `evidence_integrity_failed` event.

## 12. Output validation (`app/evidence/validation.py`, `app/agent/graph.py`)

| Guardrail | Rule |
|---|---|
| Claim numbers | Every number in the text must be in the cited evidence (display rounding allowed) |
| Claim direction | Increase or decrease wording must agree with the sign of cited change claims |
| Causality | "caused", "because", "due to", "led to", "resulted from", "responsible for", "as a result", "drove" and similar are rejected unless every cited claim has an explicit `causal_basis` (none today). Negated forms are allowed |
| Forecasts | Must be labelled as forecasts. Certainty wording ("will be", "is going to", "guarantee") is rejected. Forecast evidence keeps its model, cutoff, horizon and interval |
| Anomalies | Must be labelled statistically unusual. Judgement wording ("bad", "good", "crisis", "problem", …) is rejected. Anomaly evidence keeps its detector, threshold, direction, baseline and score |
| Recommendations | Must cite a recommendation claim and be framed as a suggested next step (review, investigate, check, …). Directives and promised outcomes ("immediately", "must", "fix", "will solve") are rejected |
| Facts vs inference | An inference or recommendation cannot be a key finding. Observed facts need observed evidence |
| Truncation | Text resting on truncated SQL must say so |
| Length | At most `max_response_chars`. After bounded regeneration, whole lower-priority items are dropped (recommendations, then interpretation, then trailing findings; never mid-sentence, never the answer) and a caveat says so |

A draft that still fails ends in `validation_failure`. The response names the failed checks and
never repeats the rejected content. Details stay in the redacted trace.

## 13. Secret protection and error sanitisation (`app/security/redaction.py`, `errors.py`)

- **What is redacted:**
  - known key formats (Anthropic, OpenAI, AWS, GitHub, Slack, Google, JWT, bearer,
    private-key blocks);
  - secret `name=value` or `"name": "value"` pairs;
  - registered secrets (the configured API key is registered by the LLM factory);
  - opaque secret-like environment values.
- **Where it is applied:** the question (before the state), logs, security events, error traces,
  model call records and the final response.
- **Environment access:** only the redaction utility reads `os.environ` (tested). `.env` and the
  environment are never logged.
- **Errors:**
  - users see a fixed message per error category;
  - the developer trace keeps one sanitised line: secrets and paths redacted, tracebacks withheld,
    bounded length;
  - unknown codes map to "internal error".

## 14. Data exposure and ground-truth isolation

**AGENT → APPROVED TOOLS → APPROVED DATA.** There is no path AGENT → FILESYSTEM → DATA.

- The only data interfaces are the 12 tools. They sit over the Phase 1 `Database`, which is
  read-only, and the Phase 2/3 services.
- The data-exposure policy lists approved relations and exposed columns explicitly. New tables
  are not exposed until listed. PII-tagged and withheld columns cannot be queried, and names that
  suggest hidden state are always withheld.
- Aggregated analytics are preferred:
  - per-customer and per-person breakdowns are denied;
  - customer-level tool output is capped;
  - evidence and prompts carry summaries, never raw tables.
- Hidden customer health and the injected-event ground truth are not in the database. No agent
  module imports the generator or reads files (static tests). The screen refuses requests for them.
- A regression suite checks this behaviourally, so it keeps holding as the implementation
  changes:
  - an audit hook records file, process, socket and `exec` activity during agent runs;
  - the dataset's own ground-truth text must not appear in any prompt, log line or result.

## 15. Audit events (`app/security/events.py`)

`SecurityEvent(event_id, run_id, event_type, severity, timestamp, component, action, decision, reason, details)`.
Events are carried in `AgentState.security_events` and `AgentRunResult.security_events`, and
emitted as JSON on the `agentops.security` logger. Reasons and details are redacted and bounded.
They never contain the question, prompts, rows or secrets.

| Severity | Examples |
|---|---|
| INFO | `tool_authorized`, `context_truncated` (trim), `response_truncated`, `unsupported_request` (out of scope) |
| WARNING | `suspicious_prompt` (restrict), `privileges_reduced`, `retry`, `model_output_rejected`, `input_rejected`, `timeout`, `argument_rejected` |
| HIGH | `sql_rejected`, `tool_denied`, `plan_rejected` (policy), `tool_output_rejected`, `budget_exceeded` (graph), `secret_redacted`, `output_validation_failed` (final) |
| CRITICAL | `suspicious_prompt` for secrets, hidden or ground-truth data, files, code; `evidence_integrity_failed` |

## 16. Fail-closed behaviour

| Situation | Outcome |
|---|---|
| Unknown tool, disabled tool, tool not permitted for the intent | Denied before execution; planning_failure (or a controlled tool failure at execution) |
| Unknown metric, dimension, filter, table, column, function, detector | Rejected with a typed code |
| Malformed SQL, arguments or model output | Rejected; model output is regenerated within budget, then planning_failure |
| Missing or tampered evidence | Claim removed; if nothing answers, insufficient_evidence |
| Exceeded budget or time | Stop and report the limit; nothing is truncated silently |
| Unknown error code | Not retried; shown as "internal error" |
| Prompt that cannot fit the context budget | Not sent; the step fails |

## 17. Performance

Measured on the local full dataset with the deterministic model, same machine. The Phase 4 head
(`0638bc9`) and Phase 5 were benchmarked alternately, A-B-A-B; each value is the mean of two
medians of 11 runs:

| Scenario | Phase 4 | Phase 5 | Overhead |
|---|---|---|---|
| Simple KPI ("What was revenue last month?") | 29.9 ms | 33.9 ms | +3.9 ms (+13%) |
| Multi-tool investigation (9 tool calls) | 410.6 ms | 448.1 ms | +37.5 ms (+9%) |
| Forecast (3 months) | 301.2 ms | 324.2 ms | +23.0 ms (+8%) |
| Anomaly check (2 detectors) | 283.7 ms | 313.1 ms | +29.4 ms (+10%) |

The overhead is roughly 4–40 ms per run, and run-to-run noise is about ±10 ms. Most of it comes
from:
- output validation of large results (forecast backtests, anomaly series);
- evidence fingerprints;
- per-query deadline timers;
- a second validation of every call just before execution.

Two early hot spots were removed:
- rescanning the environment on every redaction call (the scan is now cached until the
  environment changes);
- rendering each prompt twice.

## 18. Ready for Phase 6 (MCP)

The same `ToolAuthorizationPolicy`, argument policy, SQL validator, output validator, budget and
audit events can guard an MCP server. Those clients would be untrusted in exactly the way the
model is today.
