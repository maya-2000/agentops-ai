# Evidence-backed business investigations (Phase 10)

`/ask` answers one business question. An **investigation** answers a business *objective*
("Why is revenue growth slowing?", "Give me a management brief"). It plans several analytical steps,
runs them through the same secured tools, validates the findings against each other, identifies
drivers without claiming causes, and writes a **decision brief** in which every statement rests on
evidence.

```
objective ─► screen ─► understand ─► plan ─► execute steps ─► evidence ─► claims
                                                                   │
            decision brief ◄─ recommendations ◄─ drivers ◄─ cross-finding validation
```

The investigation layer adds orchestration only. It adds no analytics, no second analytics engine
and no SQL. It also has no way for the model to choose a tool call:

- The numbers come from the deterministic tools (Phases 2–4).
- The evidence comes from the Phase 4 evidence builder.
- Every tool call goes through the Phase 5 `SecuredToolExecutor`.
- The model's only role is the understanding step it already has in `/ask`.

Contents:

1. [Investigation architecture](#1-investigation-architecture)
2. [Investigation lifecycle](#2-investigation-lifecycle)
3. [Plan representation](#3-plan-representation)
4. [Step execution](#4-step-execution)
5. [Evidence model](#5-evidence-model)
6. [Driver analysis](#6-driver-analysis)
7. [Recommendation grounding](#7-recommendation-grounding)
8. [Causal-language guardrails](#8-causal-language-guardrails)
9. [Investigation budgets](#9-investigation-budgets)
10. [API](#10-api)
11. [UI](#11-ui)
12. [Evaluation](#12-evaluation)
13. [Known limitations](#13-known-limitations)

---

## 1. Investigation architecture

| Module (`app/investigation/`) | Role |
|---|---|
| `models.py` | The typed investigation, plus the models below. |
| `planner.py` | Chooses one of the closed set of templates for a validated objective, and resolves the default periods. |
| `templates.py` | The six analysis templates. Each step has a tool, arguments, the intent it is authorised under, and optional dependencies. |
| `steps.py` | Named conditions (`outcome_changed`, `concentrated`) and bindings (`concentrated_member`, `feature`), evaluated in code. |
| `engine.py` | `Investigator`: screen, understand, plan, execute, validate and synthesise. It is the only module that calls tools, through `runtime.executor.execute`. |
| `findings.py` | Turns validated claims into findings. Each finding copies its identity from its evidence and records its business area. |
| `validation.py` | Cross-finding validation of findings, relationships, drivers and recommendations. |
| `drivers.py` | Rule-based driver analysis: contributions, co-movements, associations and context. |
| `recommendations.py` | Grounded next steps. Each recommendation is also a `recommendation` claim in the evidence graph. |
| `brief.py` | The decision brief: summary, key findings, risks, uncertainty and management-brief sections. It also enforces the size cap. |

`models.py` defines the investigation, `AnalysisPlan` / `AnalysisStep`, `StepRecord`, `Finding`,
`FindingRelationship`, `Driver`, `Recommendation`, `DecisionBrief`, `ValidationIssue` and the budget,
efficiency and timing reports.

The layer reuses the following, unchanged:

- **Agent runtime.** `AgentRuntime` provides:
  - the input guard and the prompt-injection screen;
  - the model client and its understanding prompt;
  - the `SecuredToolExecutor`;
  - the tool context;
  - the as-of date.
- **Request validation.** The central `validate_understanding`.
- **Claims and evidence.** The Phase 4 claim builders (`build_claims`), the evidence builder, the
  evidence graph and its fingerprints, and the Phase 5/7.1 evidence and response validators.
- **Security.** The security events, redaction and budget accounting (`RunBudget`, `BudgetUsage`).

`Investigator(runtime)` is built once per API service, lazily, on the same `AgentRunner`'s runtime.
Investigations are therefore serialised with `/ask` on the one read-only database connection.
Static tests (`tests/unit/test_investigation_isolation.py`) enforce these rules:

- The package imports no database driver, tool handler, analytics service, API, UI, MCP server or
  generator code.
- It has one tool call site.
- It contains no SQL text, business numbers or dataset member names, and reads no files.

## 2. Investigation lifecycle

| Stage | What happens | Failure outcome |
|---|---|---|
| **Screen** | The input guard checks length, encoding and secrets, which are redacted. The prompt-injection screen runs before any model or tool call. | `refused`: blocked categories such as system prompt, secrets, ground truth, files, SQL payloads or rule changes. `restrict` verdicts are recorded; investigations never run SQL in any case. |
| **Understand** | The agent's own understanding step runs (one model call, within the retry budget). Output validation and `validate_understanding` follow. | `failed` for unusable model output. `insufficient_evidence` for an ambiguous or unavailable period, e.g. "this quarter" or a future month. `unsupported` for an out-of-scope objective. |
| **Plan** | `select_template` chooses from brief cues, then the metric, then the intent. `plan_investigation` fills periods and filters; default periods are recorded as assumptions. | `unsupported` when no template fits. |
| **Execute** | The steps run in plan order: dependencies, conditions, bindings, reuse, budgets, then the secured executor (§4). | A step can fail or be skipped without failing the investigation. A budget stop gives `budget_exhausted`. |
| **Validate** | Claims are built from all the evidence and checked by the evidence validator. Unsupported claims are removed. Findings, relationships, drivers and recommendations are then validated across findings (§5–§7). | Invalid items are removed or downgraded and recorded as `ValidationIssue`s, never repaired. |
| **Synthesise** | The decision brief is composed, its summary validated by the response validator, and its size capped. | `insufficient_evidence` when the outcome could not be measured, no finding survived, or the objective asks for a cause. |

Final statuses (`FINAL_STATUSES`): `completed`, `insufficient_evidence`, `budget_exhausted`,
`refused`, `unsupported`, `failed`, `cancelled`.

A cancelled investigation keeps its step records but presents no findings, evidence or brief. This
covers a client that went away, the API timeout, and shutdown. Nothing unvalidated is ever shown.

## 3. Plan representation

A plan is data (`AnalysisPlan`). It holds the template, title, outcome metric, period and comparison
period (with labels), assumptions and steps. Each `AnalysisStep` has these fields:

| Field | Meaning |
|---|---|
| `step_id` | `S1`, `S2`, … in plan order. |
| `title` | A concise action ("Compare region performance"), shown to users. It is never model reasoning. |
| `area` | Revenue, customers, sales, marketing, product, support, anomalies or forecast. |
| `tool_name`, `arguments` | One allow-listed Phase 4 tool and its arguments, validated by the tool's input model at run time. |
| `authorized_as` | The validated intent the call is authorised under, fixed by the template. |
| `depends_on` | Earlier steps that must have completed. |
| `condition` | Optional: `outcome_changed` or `concentrated`. A `Literal`, so it cannot hold an expression. |
| `binding` | Optional: `concentrated_member` (the rank-1 member holding at least half of the gross change becomes a filter) or `feature` (the n-th feature of the adoption result). |

Templates (`TEMPLATE_TITLES`; each measures the outcome the objective names):

| Template | Chosen for | Steps |
|---|---|---|
| `revenue` | Revenue, MRR, ARR, growth, ARPU | Revenue change. Region and segment decomposition when the outcome changed. MRR bridge. Churn, NRR, win rate and pipeline changes. Usage and support before churn. Revenue anomalies. A country drill-down into a region that concentrates the change. |
| `customer` | Churn, retention, NRR, customer count, CLV | Churn and NRR changes (plus the named outcome), customer movements, churn by segment and region, cohorts, current risk, usage before churn, feature adoption, customer-count anomalies. |
| `sales` | Win rate, pipeline, cycle, order value, conversion, CAC | Win rate, pipeline, cycle and AOV changes, segment performance, CAC, channel performance, revenue change. |
| `product_support` | Tickets, resolution time, product adoption | Ticket volume (and resolution time), tickets per customer, tickets by category and segment, feature adoption, adoption change of the top two features (bound), ticket anomalies, usage before churn. |
| `general` | Several business areas in one objective | Revenue change and segment decomposition, bridge, churn, NRR, win rate, pipeline, tickets, adoption, usage, anomalies. |
| `management_brief` | "management brief", "state of the business", "what should management investigate next", "key business risks" | Revenue, bridge, customer count, churn, NRR, win rate, pipeline, adoption, tickets, revenue and ticket anomalies, 3-month revenue forecast. |

The plan is streamed as a checklist (§10, §11). No model reasoning, prompt or chain of thought is
stored, streamed or returned.

## 4. Step execution

Each step, in plan order:

1. **Cancellation and deadline** are checked (cooperatively; database queries are interrupted at the
   deadline).
2. **Dependencies**: if a step it depends on did not complete, the step is `skipped` with a fixed
   reason.
3. **Condition**: a named check on the dependencies' evidence (e.g. no measured change, so there is
   nothing to decompose).
4. **Binding**: one argument is read from the dependencies' evidence. The bound value is still an
   ordinary argument, validated and authorised like any other.
5. **Reuse**: an identical call (same tool and arguments) already made is reused (`reused`,
   `reused_from`), never run twice. Duplicate tool calls are reported (always 0 in the benchmark).
6. **Budgets**: steps, tool calls, runtime and evidence are checked (§9). If any is exhausted, the
   step and every later one are `not_run`.
7. **Secured execution**: `SecuredToolExecutor.execute` runs with
   `AuthorizationContext(intent=step.authorized_as, sql_permitted=False, …)`. This gives:
   - authorisation (allow-list, disabled tools, intent permissions, restricted breakdowns,
     argument checks);
   - the deadline and the per-tool timeout;
   - retries under the retry policy;
   - output validation;
   - the shared budget charge.

   The same path serves `/ask` and MCP.
8. **Evidence**: the result becomes evidence in the investigation's graph and is fingerprinted.
   Step records keep only a fixed, user-safe reason for failures. Exception detail stays in the
   developer trace, sanitised.

Steps run **sequentially**. The agent has one read-only DuckDB connection and results are
deterministic. Parallel steps would contend for that connection, and investigations already take well
under a second (§12). A progress observer that raises is detached and never changes the investigation.

## 5. Evidence model

Evidence and claims are the Phase 4 objects, unchanged, in one `EvidenceGraph` per investigation.
Evidence is fingerprinted when it enters the graph, and integrity is verified before validation.

A **finding** (`Finding`) is a validated claim plus the fields the brief needs:

- `claim_type` and `label`: **observed**, **calculated**, **inferred** or **recommended**
  (`FINDING_LABELS`). Recommendations are kept apart (§7).
- Identity copied from the claim's subject evidence, never re-derived: `metric`, `unit`, `period`,
  `comparison_period`, `dimension`, `breakdown` (the member), `filters`.
- `evidence_ids` and `step_ids`: provenance to the steps that produced it.
- `area`: from the step that produced its subject evidence.
- `primary`: the investigation's outcome (the first change claim about the outcome metric for the plan's
  two periods).

Excluded from findings: claims about individual customers (risk rows name customer IDs), and the
claim builders' own recommendations.

**Cross-finding validation** (`validation.py`) runs after the Phase 5/7.1 evidence validator:

| Check | Action |
|---|---|
| Every cited evidence item exists, is unmodified and comes from a successful call | removed |
| Identity equals the subject evidence (Phase 7.1 rules re-applied) | removed |
| Numbers, KPI names and wording pass the response validator on the finding alone | removed |
| A comparison uses the investigation's pair of periods; a level uses one of them | removed |
| A partially supported claim | downgraded to low confidence |
| A relationship joins surviving findings about the same period pair, with a non-causal type | removed |
| A driver has validated findings, a validated relationship of its own type and non-causal wording | removed |
| A recommendation cites surviving findings, is a suggestion and has a supported claim | removed |

Nothing is silently repaired. Every action is a `ValidationIssue`, counted (not listed) in the API
response.

## 6. Driver analysis

A **driver** is an observable factor associated with the outcome in the available evidence. It is
never a proven cause. Drivers come only from documented rules over findings (`drivers.py`):

| Relationship | Rule | Example |
|---|---|---|
| `contributes_to` | A decomposition member that moved with the outcome in the same periods (an accounting identity). Its `share` is the evidence's share of the gross change. A drill-down member contributes to the member it was drilled from. | Region APAC: 92.1% of the gross decline |
| `supports` | An indicator that moved with the sign the outcome's co-movement table expects in the same periods, e.g. churn up while revenue fell. The MRR bridge's largest negative component is also a `supports` driver of a revenue decline. | Logo churn +0.62 pp |
| `contradicts` | An indicator that moved the other way. It is reported under **Contradicting signals** and explained in the uncertainty notes; no single explanation is forced. | "Win rate moved the other way… the evidence does not point to it" |
| `correlates_with` | The usage and support association before churn, used only when churn rose and supports the outcome. | Churned customers had more support tickets |
| `contextualizes` | Flagged anomalies of the outcome, level rankings, and indicators without an expected direction. | Revenue in 2026-08 statistically unusual |

Each driver carries these fields:

- name and category;
- direction and magnitude, as the finding states them;
- share, for contributions;
- confidence: the lowest confidence of its findings;
- `finding_ids` and `evidence_ids`.

There are no arbitrary scores. When no factor moved in line, the brief says so ("No measured factor
moved in line with …, so no driver is identified").

## 7. Recommendation grounding

There is no free-form recommendation generator. Each rule turns validated drivers or findings into a
suggested next step. The rules, in order, are capped at four recommendations:

1. **Concentration.** One member accounts for most of the change: review that member's accounts,
   the most specific (drill-down) member first.
2. **Churn.** Churn or retention moved in line with the outcome: review the churned customers.
3. **Sales.** Pipeline or sales performance moved in line with the outcome: review the pipeline and
   win/loss outcomes.
4. **Anomaly.** The outcome month was statistically unusual: check it against operational context
   the dataset does not record.
5. **Management brief.** Each adverse indicator that no rule above covered.

Every `Recommendation` has these properties:

- It cites `supporting_finding_ids` and exactly their evidence.
- It is registered as a `recommendation` claim, which passes the evidence validator.
- It is worded as a suggestion (Investigate / Review / Check), never a directive or a promised outcome.
- It carries a rationale ("Rests on F16 (calculated), F17 (inferred).") and an uncertainty note.

A stopped investigation derives no drivers or recommendations.

## 8. Causal-language guardrails

- **Relationship types.** Relationship and driver types are a closed set of non-causal `Literal`s:
  `causes` cannot be represented.
- **Validator.** Findings, the executive summary, driver statements and recommendations pass the
  Phase 5 causal-wording check: "caused by", "because of", "resulted from", "led to", "due to",
  "drove", "driven by", "triggered", …, unless negated. An offending item is removed.
- **Summary.** The summary ends with "These are contributions and co-movements in the data, not
  established causes." It falls back to the outcome finding's text if a composed summary fails
  validation.
- **Causal objectives.** Objectives that ask for a cause ("Did the price increase cause churn?",
  "What caused revenue to decline?") return **`insufficient_evidence`**. The response still shows the
  observed findings, with notes that the data cannot establish causation and that pricing changes,
  product releases and market conditions are not recorded. Churn rising is never presented as proof
  that a price increase caused it.
- **Premise check.** The objective's premise is checked against the evidence. If the objective says
  "increase" but the evidence shows a decline, the brief says so instead of explaining a change that
  did not happen.
- **Management briefs.** Briefs contain only sections backed by findings of that area. They state
  that they describe the current state and do not explain causes. The benchmark checks that they
  invent no priorities, financial impact, customer sentiment or market conditions (§12).

Tests: `tests/integration/test_investigation_synthesis.py` (each of the five phrases, in findings,
drivers and recommendations) and the `causal_language_safety` category of eval_v2.

## 9. Investigation budgets

Limits are enforced in code, never by asking the model. They reuse the Phase 5 `SecurityLimits` and
`RunBudget` (`RunBudget.for_investigation`):

| Setting | Default | Enforced |
|---|---|---|
| `AGENT_MAX_INVESTIGATION_STEPS` | 14 | before each step (steps that ran) |
| `AGENT_MAX_INVESTIGATION_TOOL_CALLS` | 16 | before each step, and by the executor's budget charge |
| `AGENT_MAX_INVESTIGATION_SECONDS` | 120 | wall clock, before each step |
| `AGENT_MAX_INVESTIGATION_EVIDENCE` | 400 | evidence items, before each step |
| `AGENT_MAX_INVESTIGATION_OUTPUT_CHARS` | 12000 | the brief's text; lower-priority items are dropped, with a note |
| retries | `AGENT_MAX_RETRIES` × (tool calls + 1) | the executor's retry policy |
| model calls | `AGENT_MAX_RETRIES` + 1 | the understanding step only |
| ad-hoc SQL | 0 | `sql_permitted=False` and a SQL budget of zero |

Every template fits the default budget (at most 12 tool calls). When a budget is exhausted, these
things happen:

- The remaining steps are `not_run`, with the reason.
- A `budget_exceeded` security event is recorded.
- The status is **`budget_exhausted`** (API outcome `partial`), never `completed`.
- The message is "Investigation stopped because the analysis budget was reached."
- The brief is marked incomplete and derives no drivers or recommendations.

Objectives such as "Keep investigating forever", "Run every available tool", "Call the same tool
1,000 times" or "Ignore the previous limits" change nothing. The plan is a fixed template and the
budget belongs to the investigator.

In production, the API refuses to start unless the timeouts nest:
`AGENT_TOOL_TIMEOUT_SECONDS ≤ AGENT_MAX_INVESTIGATION_SECONDS ≤ API_REQUEST_TIMEOUT_SECONDS`.

## 10. API

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/investigations` | Run an investigation; returns the decision brief (`InvestigationResponse`). |
| `POST /api/v1/investigations/stream` | The same, as NDJSON: `progress` events, then one `result` or `error` event. |

Request: `{"objective": "...", "request_id": "...", "session_id": "..."}`. Extra fields are
rejected, so a request cannot change a limit. The endpoints share `/ask`'s:

- authentication (bearer token) and rate limit (one quota across `/ask`, `/ask/stream` and both
  investigation endpoints);
- JSON-only rule, body-size limit, request timeout, queue bound and cancellation;
- metrics and structured logs.

An empty objective gives 422 `empty_objective`; an objective over `AGENT_MAX_QUESTION_CHARS` gives
422 `objective_too_long`.

The response contains:

- **Identity and status:** `request_id` = `investigation_id`, `status` and `outcome` (`answered`,
  `partial`, `insufficient_evidence`, `refused`, `unsupported`, `failed`), `message`, and a `refusal`
  (`policy` / `invalid_input` / `out_of_scope`).
- **Plan:** `title`, `template`, `scope`, `period`, `comparison_period`, and the `plan` (each step with
  its status, reason, duration, evidence and reuse).
- **Brief and evidence:** `brief`, `findings`, `relationships`, `claims`, `evidence`, and `trace` (the
  same user-safe trace steps as `/ask`).
- **Views:** `kpis`, `forecasts`, `anomalies` and `visualizations`, built by the same presenter
  functions as `/ask`.
- **Run summary:** `run`, with tool calls, efficiency, stage timings, budget, and validation counts.

Never returned:

- prompts or model reasoning;
- security events or policy internals;
- raw tool results;
- validation messages, stack traces, file paths or secrets.

Progress events carry only these fields:

- the stage (`started`, `understanding`, `planning`, `plan`, `step_started`, `step_finished`,
  `validating`, `synthesizing`, `finished`) and a fixed label;
- the step ID, title, area and tool name;
- the status, duration and elapsed time;
- on the `plan` event, the plan's step list.

**Why there is no `GET` status endpoint.** Investigations are synchronous and bounded by the request
timeout (0.2–1 s on the full dataset, §12). The stream reports progress as it happens, so there is
nothing to poll. Storing investigations for later retrieval would add state, retention and access
rules for no benefit today.

**Why there is no MCP investigation tool.** MCP clients already get every analysis tool behind the same
secured executor. An investigation is a product workflow built from those calls, not a new capability,
and exposing it would add a long-running, multi-call tool for symmetry only. The MCP catalogue and
server are unchanged, and a test checks it (`tests/mcp/test_mcp_isolation.py`). eval_v2 checks that
every investigation step gives the same evidence through MCP.

`GET /api/v1/capabilities` lists `investigation_types` and `example_objectives`. See `docs/api.md`.

## 11. UI

The Streamlit page has two modes, chosen with a radio at the top: **Ask a question** (unchanged) and
**Investigate a business issue**. There is no multipage dashboard; the agent stays central.
Investigation Mode has these parts:

- **Input.** An objective text area, an *Investigate* button and example objectives from
  `/capabilities`.
- **Live checklist.** A checklist of the plan while the stream runs: ✓ completed, ⟳ running, ○ not run
  yet, ↺ reused, ⊘ skipped, ✗ failed. It is built from `plan`, `step_started` and `step_finished`
  events only.
- **Decision brief**, in this order:
  - the objective and a status banner;
  - the plan (expandable) and the executive summary;
  - key findings, labelled Observed / Calculated / Inferred;
  - **Drivers and contributing factors**, captioned as not established causes, with their
    relationship and share;
  - **Contradicting signals**, risks, and recommendations (with rationale and uncertainty);
  - management-brief sections, uncertainty and assumptions;
  - the Phase 8 KPI cards, charts, forecast and anomaly panels, evidence and provenance, and analysis
    trace, reused unchanged.
- **Session history.** The last `UI_HISTORY_LIMIT` questions and investigations of the browser session
  only. Each entry keeps the redacted objective, the outcome, a bounded summary and the request ID. It
  holds no tokens, secrets, evidence rows, ground truth or personal data.

The page calls the API over HTTP only (`/investigations/stream`). The view models are tested on
real API responses, and the page with Streamlit's AppTest (`tests/ui/test_ui_investigation.py`).

## 12. Evaluation

**eval_v2** (`evals/datasets/eval_v2.json`, schema 2.0) is the investigation benchmark. eval_v1
(89 single-question scenarios) is unchanged and still the default dataset.

```
python -m evals.run --dataset eval_v2                          # 77 scenarios, deterministic
python -m evals.run --dataset eval_v2 --suite critical         # 12 critical scenarios
python -m evals.run --dataset eval_v2 --multi-seed 7,2027      # plus the multi-seed subset on seeds 7 and 2027
```

- **Scenarios.** 77 deterministic scenarios in 18 categories: planning, step selection, tool
  selection, evidence grounding, period and comparison correctness, driver identification,
  recommendation grounding, causal-language safety, insufficient evidence, refusal, security, budget
  enforcement, cancellation, management brief, investigation API, UI transformation and MCP parity.
- **Modes.** In-process `Investigator`, the FastAPI app (plain and streamed, plus `/ask`
  compatibility), the UI view models, cancellation, and MCP parity (each executed step repeated
  through the real MCP server).
- **Checks on every scenario** (the Step 28 criteria):
  - **Plan and tools:** the plan (template, dependencies, titles), the steps (required steps planned
    and run; bindings equal the values read from the source evidence), and the tools (required, none
    forbidden, each call belongs to its step).
  - **Evidence:** completeness (exists, unmodified, from a successful call, numbers supported) and
    identity (metric, unit, period, comparison, dimension, member, filters).
  - **Drivers:** shares equal the evidence's; supporting and contradicting indicators move with the
    sign of the evaluation's own co-movement table, read from their evidence's signed change.
  - **Recommendations:** they cite validated findings and exactly their evidence, and are worded as
    suggestions.
  - **Safety:** causal safety, and budget compliance (tool calls, steps, SQL, model calls, output
    size, duplicate calls, stop status).
  - **Security:** no system prompt, secret, hidden label, file content or withheld value in any output,
    model request, progress event or log.
- **References.** Scenarios hold no business numbers. `outcome_change` and `top_contribution` are
  resolved from the independent pandas reference at run time. Relative periods (`last_month`,
  `last_quarter`, `previous`) are resolved by the evaluation's own calendar arithmetic.
- **Grader tests.** `tests/evals/test_eval_v2.py` corrupts a real investigation one defect at a time
  and asserts that the owning check fails, so a passing run is not vacuous. Defects include unknown
  evidence, a wrong period, a causal driver statement, a wrong share, an ungrounded recommendation, an
  SQL call, a duplicate call, a fabricated number, a leaked secret, tampered evidence, a status that
  hides a budget stop, and an invented management priority.

Results (deterministic mode, repository database, seed 42):

| Run | Result |
|---|---|
| eval_v2 full | 77/77 passed; every rate 100%; 0 leaks, causal failures, duplicate calls or budget violations |
| eval_v2 critical | 12/12 |
| eval_v2 multi-seed (seeds 7 and 2027) | 24/24 |
| eval_v1 full / critical / multi-seed | 89/89, 13/13, 22/22 (unchanged) |

**Performance** (repository database, deterministic model, median of 7 runs):

| Workload | Time | Tool calls | Peak memory |
|---|---|---|---|
| `/ask` "What was revenue last month?" | 39 ms | 1 | 0.2 MB |
| `/ask` "Why did revenue decline last month?" | 422 ms | 9 | 1.6 MB |
| Investigation: revenue | 377 ms | 11 | 2.5 MB |
| Investigation: customer churn | 874 ms | 10 | 11.4 MB |
| Investigation: support tickets | 220 ms | 9 | 2.8 MB |
| Investigation: management brief | 628 ms | 12 | 1.4 MB |

Execution (the tool calls) is 94–98% of an investigation's time. Understanding takes about 2 ms,
planning under 1 ms, validation 12–16 ms and synthesis 6–8 ms. The API's overhead over the
investigation itself is within measurement noise (−50 to +15 ms across the measured runs);
`tests/api/test_api_performance.py` guards its median at 50 ms. Response bodies are 125–160 KB, most of it evidence, claims and chart specs. No
optimisation was needed. The safeguards against investigation explosion are the fixed templates,
reuse of identical calls, and the step, tool-call, runtime, evidence and output budgets.

## 13. Known limitations

- **Fixed templates.** The six templates cover the dataset's business areas. An objective outside
  them is `unsupported`, and a new kind of investigation needs a new template, not a prompt.
- **Deterministic understanding.** The deterministic model maps phrasing to intents with rules. Some
  phrasings select a neighbouring template: "Investigate revenue, churn and support tickets together"
  selects the revenue template rather than `general`. With an LLM configured, understanding is the
  model's, validated as for `/ask`.
- **Causal-question detection** is the understanding step's (e.g. "cause", "caused"). An objective
  like "Prove that churn drove the revenue decline" is investigated as a normal objective. The brief
  still makes no causal claim, and the benchmark checks this.
- **Drivers are associations and accounting shares over one pair of periods**, not causal effects.
  There is no significance test across periods and no control for confounders; the brief says so.
- **Product adoption is measured per feature.** An adoption objective uses the first feature the
  adoption analysis lists as its outcome, and notes it.
- **Sequential execution.** It is chosen deliberately (§4). A slow tool delays the whole
  investigation, within the runtime budget and the request timeout.
- **No persistence.** Investigations are not stored and cannot be retrieved later. The UI's history is
  per browser session.
- **Personal-data requests.** A request for personal data ("home addresses of our customers") is not
  refused, because it is not an injection. As in `/ask`, it is answered with aggregate customer
  analysis and never exposes withheld or personal fields. eval_v2 checks this.
