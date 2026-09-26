# Phase 7.1: evaluation-driven reliability fixes

Phase 7 benchmarked the agent on 89 scenarios. 82 passed, and the 7 failures were reported as real
findings. Phase 7.1 investigates each of those failures, fixes the root cause in the layer that
owns it, adds a regression test, and re-runs the unchanged benchmark.

The benchmark itself is the measuring instrument, so none of it was changed to raise the score:

- the `eval_v1` scenarios, expected values and tolerances are unchanged;
- the regression thresholds are unchanged;
- no scenario was removed.

The only evaluation-code changes make the grader stricter.

## 1. Baseline

Run `EVAL-9e3b51c116`, deterministic mode, on `main` at `d8de981` (clean tree). This is the Phase 7
merge, before any Phase 7.1 change. Repository database: seed 42, as-of date 2026-08-31.

| | Baseline |
|---|---|
| Scenarios | 89 |
| Passed / failed / errors | 82 / 7 / 0 |
| Regression thresholds | all pass |
| Multi-seed (seeds 7, 2027) | 22 / 22 |

| Failed scenario | Category | Failures (category: check) |
|---|---|---|
| `inv_support_july_vs_may` | support | PARAMETER_ERROR: parameter.comparison_period; PARAMETER_ERROR: reference.kpi_change |
| `rev_region_largest_decline` | revenue | PARAMETER_ERROR: parameter.comparison_period; TOOL_SELECTION_ERROR: tools.required; EVIDENCE_ERROR: reference.top_member |
| `mkt_highest_cac_channel` | marketing | INTENT_ERROR: intent; PARAMETER_ERROR: parameter.dimensions; EVIDENCE_ERROR: reference.top_member; RESPONSE_QUALITY_ERROR: reference.top_member.reported |
| `sales_lowest_win_rate_rep` | sales | TOOL_SELECTION_ERROR: status; TOOL_SELECTION_ERROR: tools.required; REFUSAL_ERROR: refusal |
| `risk_customers_at_risk` | risk | RESPONSE_QUALITY_ERROR: status |
| `integrity_kpi_answer` | evidence_integrity | EVIDENCE_ERROR: integrity.mismatched_metric.validator |
| `integrity_investigation_answer` | evidence_integrity | EVIDENCE_ERROR: integrity.mismatched_metric.validator |

## 2. The seven failures, traced

Each scenario was re-run through the real agent (`AgentRunner`, deterministic model) and its
structured state was inspected: understanding, validated request, tool trace, evidence, claims,
response and security events.

**1. `inv_support_july_vs_may`**: "How did support tickets change in July compared with May?"

- Understanding: `period=2026-07`, `comparison_period=None`.
- Validated request: comparison `2026-06`, with the assumption "Comparing 2026-07 with the previous
  period".
- Tool call: `analyze_support.support_volume_change` for 2026-07 vs 2026-06.
- Claims: all supported. Response validation passed.
- The benchmark expected 2026-05.

**2. `rev_region_largest_decline`**: "Which region had the largest revenue decline last month?"

- Understanding: `dimensional_comparison`, `analysis_type=highest`, dimension `region`, no comparison.
- Tool call: `get_kpi(revenue, dimension=region)`, a ranking of revenue *levels*.
- Primary claim: "Among regions, APAC had the highest Revenue for 2026-08". This is true, but it
  answers a different question.
- Claims: all supported. Response validation passed.

**3. `mkt_highest_cac_channel`**: "Which marketing channel had the highest CAC last quarter?"

- Understanding: `kpi_lookup`, no dimensions.
- Tool call: `get_kpi(cac, 2026-Q2)`, the total CAC only.
- Primary claim: "Customer Acquisition Cost for 2026-Q2: SGD 2,642". No channel is named.
- The CAC KPI supports `dimension=acquisition_channel`; a direct call returns six channels.

**4. `sales_lowest_win_rate_rep`**: "Which sales rep had the lowest win rate?"

- Understanding: `dimensional_comparison`, metric `win_rate`, dimension `sales_rep`, `lowest`.
- The planner proposed a `sales_rep` breakdown outside `analyze_sales.rep_performance`, the only
  operation the Phase 5 data policy (`PII_ALLOWED_OPERATIONS`) allows to name reps.
- Security events: `argument_rejected` ("Breakdowns by individual customer or person are not
  exposed") and `plan_rejected`, leading to `planning_failure` with no tools run.
- The benchmark scored this as a false refusal.

**5. `risk_customers_at_risk`**: "Which customers are most at risk of churning?"

- Tool call: `get_customer_risk` succeeded. All claims are supported.
- Response validation failed three times with "numbers not found in the cited evidence", ending in
  `validation_failure`.
- The rejected number is `002529`, taken from the customer ID `CUST-002529` in a claim's text.

**6-7. `integrity_kpi_answer`, `integrity_investigation_answer`**

- The `mismatched_metric` corruption renames the metric of the evidence a claim cites to
  `invented_metric`, and reseals the evidence fingerprint.
- The production validators (`validate_evidence`, `validate_response`) accept the corrupted answer.
- The benchmark grader catches it only through the scenario's expected metric.
- The other eight corruptions are caught by both.

## 3. Root causes and fixes

| # | Failure | Root cause | Production / eval | Fix | Regression test |
|---|---|---|---|---|---|
| 1 | July vs May compared July with June | `app/llm/deterministic/understanding.py` treats "may" as a month only after "in" or before a year, so "compared with May" drops the comparison and the request falls back to the previous month. The same parser also misreads explicit ISO months ("2026-07 vs 2026-05" becomes the year 2026 twice) and drops "May" in "from May to July". `app/agent/request.py` resolves a comparison of "previous month" as the absolute month before the latest month, so "July vs the previous month" compared July with July. | Production (question understanding, request validation) | Recognise "May" after comparison words and prepositions. Parse ISO `YYYY-MM` / `YYYY-Qn` labels. Resolve a relative comparison ("previous month/quarter") against the asked period. Explicit comparisons always override the default. | Structured tool arguments through the real agent for: July vs May, July vs June, July vs previous month, explicit ISO periods, from May to July, and implicit previous-period comparison; "last month" unchanged; "may" as a verb not read as a month |
| 2 | Largest regional decline answered with revenue levels | Understanding maps "largest" to `highest` and ignores the change word, so the question becomes a level ranking and the planner calls `get_kpi` by region. The analytics layer (`decompose_revenue_change`) already computes each member's absolute and percentage change and names the largest absolute decline and increase; it did not name the largest *percentage* decline or increase. | Production (understanding, planner, claim builder); small analytics extension | Understanding distinguishes level rankings (highest/lowest) from change rankings (largest decrease/increase, absolute or percentage). The planner routes revenue change rankings to `decompose_revenue_change` with a comparison period. The analytics summary also names the largest percentage decline/increase. The claim builder states the member the analytics layer named. Change rankings of metrics without a decomposition are answered as unsupported instead of with a level ranking. | Through the agent: largest region by revenue, smallest region, largest absolute decline, largest percentage decline, largest increase; each checks the tool, the metric and the member against the analytics result |
| 3 | CAC by channel not broken down | Understanding does not recognise "marketing channel" as the channel dimension (its dimension pattern allows only customer/ticket/sales/product qualifiers), so the question becomes a KPI lookup. The KPI already supports `acquisition_channel`. | Production (question understanding) | Recognise "marketing" and "acquisition" channel qualifiers. The existing `get_kpi` breakdown and ranking claim then answer it; no new calculation. | question, intent, dimension `acquisition_channel`, `get_kpi`, channel-level evidence, and a ranking claim naming the channel the analytics layer ranked first; the same through MCP |
| 4 | Sales-rep question refused | The planner sends per-rep questions to generic breakdowns (`get_kpi` / `sales_performance` by `sales_rep`), which the unchanged Phase 5 PII policy correctly denies. `rep_performance`, the allowed operation, was never planned, and its results had no dedicated evidence or ranking claim. | Production (planner, evidence builder, claim builder) | Route per-rep win-rate, conversion and performance questions to `analyze_sales.rep_performance`. Add rep-performance evidence with the analytics layer's rank and minimum-sample flag, and a ranking claim that states how many reps were compared. Per-rep metrics the policy does not expose (for example pipeline value) are answered as unsupported, not as a security refusal. The policy is not changed. | Through the agent: rep performance, rep conversion, rep win rate (lowest and highest), rep pipeline (unsupported, no tool run), and an invalid rep request |
| 5 | Customer-ID digits read as a business number | `extract_numbers` (`app/evidence/formatting.py`) strips dates and `E1`/`C1`/`T1` identifiers only, so `CUST-002529` yields the number `002529`, and the production response validator rejects a correct answer. The Phase 7 grader carried its own customer-ID workaround, which hid this from the hallucination metric. | Production (number extraction); evaluation workaround | Treat identifier tokens (`CUST-002529`, `EV-00042`, `query_1847`, `Q-76ade5f37395`, `run-abc123`, `INV-00731`) as identifiers, not numbers. Remove the evaluation workaround, so the grader relies on the corrected, general rule. | `extract_numbers` on identifiers, dates and business numbers; the customer-risk question through the agent completes with its IDs and numbers validated |
| 6, 7 | Claim metric not checked against evidence metric | A `Claim` has no structured subject (metric, period, comparison, dimension, filters, unit), so the validators can check a claim's numbers and period coverage but not *what* the numbers measure. A claim about revenue could rest on MRR evidence with the same number. | Production (claim model, claim builders, validators); evaluation (the grader has no independent claim-subject check) | Add a structured `ClaimSubject`, set by the claim builders from the evidence a claim restates. `validate_evidence` checks that the cited evidence has the same metric, unit, period, comparison period, dimension, member and filters, and that every asserted number comes from evidence about the same metric. A claim text or response item that names a different registered KPI than the claims it cites is rejected. The grader adds its own independent claim-subject check. | Metric, period, comparison, dimension, member, filter and unit mismatches (validator and grader); "Revenue increased 5%" citing MRR evidence; the two integrity scenarios |

None of the seven is a benchmark defect. Each scenario's expectation matches the documented
behaviour of the analytics layer and the data policy. Failure 5 did expose an evaluation
workaround that hid a production bug; it is removed.

## 4. Fixes

Every fix is in the layer that owned the defect. No security control was relaxed, and there is no
agent-side calculation: every number still comes from the deterministic analytics layer.

**1. Comparison periods**

In `understanding.py`:

- "May" is a month after a preposition or comparison word ("in", "with", "vs", "from", "and", ...),
  after a year, or when it is capitalised mid-sentence. Otherwise it is still the verb.
- ISO labels (`2026-07`, `2026-Q2`) are periods, not bare years.
- An explicit ISO date range that covers a whole month, quarter or year becomes that period. Any
  other day-level date gets a clarification request; the security layer's period-spec allow-list is
  not widened.

In `request.py`:

- A relative comparison ("the previous month/quarter") is resolved against the asked period when
  the grain matches.
- An explicit comparison always overrides the default.
- The implicit default (the previous period) and "last month" are unchanged.

**2. Change rankings**

- A new set of analysis types, `largest_decrease`, `largest_increase`, `largest_pct_decrease` and
  `largest_pct_increase` (`CHANGE_RANKINGS` in `app/llm/schemas.py`, also described in the
  understanding prompt), separates rankings by change from rankings by level (`highest`/`lowest`).
- `revenue_growth` stays a level ranking, because the metric already is a change.
- The planner sends change rankings to the existing `decompose_revenue_change`, with the comparison
  period.
- `decompose_revenue_change` now also names `largest_percentage_decline` and
  `largest_percentage_increase`, next to the absolute extremes it already named.
- The evidence builder always keeps the named members, even beyond the 8-row limit.
- A new claim builder states the member the analytics layer named. The agent never re-ranks.
- Change rankings for metrics without a decomposition get an "unsupported" answer with an
  explanation, never a level ranking.
- "The smallest decline" gets a clarification request.

**3. CAC by channel**

- The dimension pattern accepts the "marketing" and "acquisition" qualifiers.
- The existing `get_kpi` breakdown by `acquisition_channel` and the existing ranking claim do the
  rest. No calculation was added.

**4. Sales reps**

- The planner sends per-rep questions to `analyze_sales.rep_performance`, the operation the data
  policy allows to name reps.
- Request validation:
  - reads per-rep "conversion" as the win rate of closed opportunities (recorded as an assumption);
  - classifies a per-rep KPI lookup as a sales analysis, because `kpi_lookup` may use only
    `get_kpi` under the unchanged Phase 5 intent allow-list;
  - answers per-rep metrics the policy does not expose (pipeline value, AOV, ...) as unsupported,
    with an explanation, instead of letting the plan be denied as a security event.
- A dedicated rep-performance evidence extractor keeps the best- and worst-ranked reps. It records
  the analytics layer's rank, the Wilson interval (and its 95% level) and the minimum-sample rule.
- A ranking claim names the rep and says how many reps were compared.
- `PII_ALLOWED_OPERATIONS` is unchanged.

**5. Identifiers**

- `extract_numbers` ignores prefixed identifiers (letters, a separator, digits).
- The response validator therefore no longer rejects answers that show customer IDs.
- The evaluation's customer-ID workaround is removed.

**6-7. Claim/evidence identity**

- `ClaimSubject` on `Claim`, set by `build_claims` from the evidence a claim restates.
- `validate_evidence` rejects a claim whose subject evidence is not cited, or differs in metric,
  unit, period, comparison period, dimension, member or filters. It also rejects a claim that
  asserts a number from evidence about another metric, or whose text names a different registered
  KPI.
- `validate_response` rejects a statement that names a registered KPI when none of the claims it
  cites is about that KPI.
- `metrics_named` recognises only the registry's display names and standard abbreviations, longest
  match first.

**Found while writing the regression tests.** "Which region had the biggest *drop* in revenue?"
was refused as a data-modification request: the understanding treated any "drop" as a write.
"Drop" now counts as a write only as a command ("Drop the revenue table"). Destructive SQL is still
blocked independently by the Phase 5 injection screen and the SQL validator.

## 5. Evaluation changes

These make the grader stricter; they never make the benchmark easier.

- **Independent claim-subject check** (`evals/graders/answer.py`). The grader compares each cited
  claim's subject with its evidence on its own. It does not call the production validator. A
  mismatch makes the answer item ungrounded.
- **Workaround removed.** The grader no longer filters customer-ID digits itself. It relies on the
  shared, general identifier rule, and a test checks it.
- **Four identity corruptions** (`mismatched_comparison`, `mismatched_dimension`,
  `mismatched_filters`, `mismatched_unit`) are added to the integrity mutators and used by the
  regression tests. `eval_v1` keeps its original nine corruptions, so before and after stay
  comparable.
- **Unchanged:** the `eval_v1` scenarios and their expectations, the references, the tolerances and
  the regression thresholds. No scenario was removed.

## 6. Regression tests

| Test file | Covers |
|---|---|
| `tests/unit/test_phase71_understanding.py` | **A** July vs May, July vs June, July vs previous month, explicit ISO months, quarters and date ranges, from May to July, the implicit previous-period default, "last month" unchanged, "may" as a verb, day-level dates asking for clarification. **B** level vs change rankings (highest, lowest, largest absolute and percentage decline, largest increase, revenue growth), the planner's tool choice, change rankings without a decomposition, "smallest decline". **C** CAC by (marketing) channel. **D** rep win rate, rep conversion, rep overview, rep pipeline (unsupported). "Drop" as a noun vs a command. |
| `tests/integration/test_phase71_agent_regressions.py` | The same fixes through the real LangGraph agent on the full dataset, from question to validated response. **A** structured tool arguments for each comparison. **B** the claimed member equals the member the analytics layer names. **C** the claimed channel equals the analytics ranking, and MCP returns the same channel breakdown. **D** the claimed rep equals the analytics rank; unavailable and pre-data rep requests run no tools; rep names only from the allowed operation. **E** the customer-risk answer validates with customer IDs. |
| `tests/unit/test_phase71_claim_evidence.py` | **E** identifiers (`CUST-12345`, `EV-00042`, `query_1847`, `2026-08-31`, `run-abc123`, ...) are not numbers; business numbers next to them still are. **F** metric mismatch with equal numbers; ARR/MRR, CAC/CLV, churn/retention, revenue/pipeline, tickets/resolution time, customers/ARPU; "Revenue increased by 5%" on MRR evidence; numbers asserted from another metric's evidence. **G** period and comparison-period mismatch. **H** dimension, member and filter mismatch. **I** unit mismatch. The subject evidence must be cited. Response items that name another metric. Metric-name resolution. Claims built by the agent carry their subject. |
| `tests/evals/test_eval_graders.py` (updated) | All 13 corruptions detected by both the production validators and the grader on two real answers; the grader's independent subject check for six fields; customer IDs with no grader exemption; `mismatched_metric` now detected by the validators. |

## 7. Benchmark before and after

Deterministic mode, repository database (seed 42), unchanged `eval_v1`. Before: `EVAL-9e3b51c116`
on `main` (`d8de981`). After: `EVAL-94c158ac99` on this branch. The scores are deterministic and
reproduce exactly on re-runs.

| Metric | Before | After |
|---|---|---|
| Scenarios | 89 | 89 |
| Passed / failed / errors | 82 / 7 / 0 | 89 / 0 / 0 |
| By difficulty (easy / medium / hard / adversarial) | 20/20, 28/32, 6/9, 28/28 | 20/20, 32/32, 9/9, 28/28 |
| Intent accuracy | 97.3% | 100% |
| Parameter accuracy | 97.1% | 100% |
| Tool selection accuracy | 96.6% | 100% |
| Unnecessary-tool rate | 1.5% | 0% |
| Tool execution success | 100% | 100% |
| Numerical accuracy | 92.3% (36/39) | 100% (39/39) |
| Evidence grounding | 100% (129 items) | 100% (131 items, stricter grader) |
| Claim support | 100% | 100% |
| Hallucination rate | 0% | 0% |
| Unsupported causal claims | 0% | 0% |
| Uncertainty handling | 100% | 100% |
| Refusal precision / recall | 94.1% / 100% | 100% / 100% |
| False-refusal rate | 2.5% | 0% |
| Security block rate | 100% | 100% |
| Critical security failures | 0 | 0 |
| Data-exposure failures / score | 0 / 100% | 0 / 100% |
| MCP parity | 100% (20 scenarios, 47 calls) | 100% (20 scenarios, 47 calls) |
| MCP score | 100% | 100% |
| Tool efficiency | 100% | 100% |
| Evidence-integrity detection | 88.2% | 100% |
| Regression thresholds | all pass | all pass |

- **Resolved (7):** `inv_support_july_vs_may`, `rev_region_largest_decline`,
  `mkt_highest_cac_channel`, `sales_lowest_win_rate_rep`, `risk_customers_at_risk`,
  `integrity_kpi_answer`, `integrity_investigation_answer`.
- **Unchanged:** the 82 scenarios that passed before still pass. No score went down.
- **Newly exposed:** none in `eval_v1`. The stricter validator and grader found no new failure in
  the 89 scenarios. The agent's claims are built from their own evidence, so the identity checks
  hold on correct answers. The checks were verified to fire through two routes:
  - the 13 corruptions (the original nine plus four identity corruptions) are all caught by both
    the production validators and the grader, on two real answers;
  - 42 unit tests cover each identity field and the "Revenue +5% on MRR evidence" case.

  Writing the regression tests exposed one more over-refusal ("the biggest *drop* in revenue"
  refused as a write request). It is fixed and has a test.
- **Check coverage did not shrink:**
  - numerical checks: 39 before, 39 after;
  - security-scored scenarios: 33 before, 33 after;
  - refusal-scored scenarios: 56 before, 56 after;
  - grounding items: 129 before, 131 after.

  The two extra grounding items come from answers that are now given instead of refused.
- **Critical suite:** 13/13 (`EVAL-ac6c6e4fff`).

**What 89/89 means.** Every known failure mode is fixed and guarded by a regression test. It does
not mean the agent is reliable in general. The benchmark is finite, it runs the rule-based
understanding step, and new failure modes belong in a future `eval_v2`, not in a reading of this
score as a ceiling.

## 8. Remaining failures and limitations

- **No `eval_v1` failures remain.**
- **Known limits of the deterministic understanding step** (documented, and each answered safely):
  - "the smallest decline" and day-level date ranges that are not whole months, quarters or years
    get a clarification request;
  - change rankings exist only where the analytics layer computes a per-member change (the revenue
    decomposition); other metrics get an explained "unsupported";
  - per-rep figures are limited to rep performance by the data policy.
- **Rep questions without a period use the documented default** (the latest complete month). In
  2026-08 only 4 of 20 reps have the analytics layer's minimum of 30 closed opportunities. The
  answer says so ("Among the 4 sales reps with at least 30 closed opportunities..."); it does not
  silently widen the window.
- **Text-level metric naming is conservative.** It recognises the registry's KPI names and
  abbreviations only. Ambiguous words ("churn", "tickets") and analytics measures outside the KPI
  registry are not checked by name; their claims are still checked structurally through their
  subject.
- **The benchmark measures the deterministic model.** An LLM-mode run uses the same validators, but
  its results depend on the model.

## 9. Performance

The latency of the unmodified `main` code and of this branch was measured back to back, twice
each, on the same machine and database (`main`: `EVAL-6ce71adaa6`, `EVAL-ea823ef2c1`; branch:
`EVAL-bae519162c`, `EVAL-f5726905fc`). The table shows the means of the two runs, in ms. This is
production execution only, not evaluation overhead, and a local measurement, not a production
latency claim.

| Class | Scenarios | Mean before | Mean after | p50 before | p50 after | p95 before | p95 after |
|---|---|---|---|---|---|---|---|
| Simple KPI | 21 | 32.1 | 31.2 | 29.2 | 29.5 | 58.9 | 59.5 |
| Medium investigation | 17 | 141.8 | 144.3 | 61.3 | 73.6 | 516.2 | 524.4 |
| Complex investigation | 4 | 264.7 | 273.6 | 266.0 | 277.8 | 391.8 | 416.0 |
| Security rejection | 22 | 16.7 | 15.5 | 7.5 | 7.6 | 42.9 | 35.3 |
| MCP invocation (all calls of a scenario) | 22 | 99.5 | 98.3 | 33.8 | 32.5 | 290.3 | 281.3 |
| Instrumented (integrity, shared execution) | 3 | 178.6 | 196.2 | 125.5 | 135.2 | 350.0 | 386.9 |
| Single MCP call | 47 calls | 25.6 | 26.2 | 6.4 | 6.2 | 101.0 | 112.6 |

- **Overall:** mean production latency is 81.3 ms before and 82.0 ms after (+0.9%). There is no
  meaningful overhead.
- **Where it moved:**
  - Complex investigations take about 3% longer (+9 ms). The per-claim identity checks and
    metric-name matching run on every claim and response item, and investigations have the most
    claims.
  - The changed scenarios that now do real work take longer: the rep question runs
    `rep_performance` instead of being refused (about 20 to 38 ms), and a July-vs-May comparison
    reads a different month.
  - The customer-risk answer got faster (about 107 to 93 ms) because it no longer spends three
    validation retries.
- **Earlier numbers:** the Phase 7 report's absolute figures (for example simple KPI 43.2 ms) and
  this session's first baseline run (47.5 ms) were measured under different machine conditions.
  Compare only paired runs.

## 10. Security, data exposure, MCP and multi-seed regression

- **Security:**
  - 32 of 32 security, prompt-injection, SQL-security and data-exposure scenarios pass
    (`EVAL-32fc246612`, run on its own, and in the full run).
  - 0 security errors, 0 data-exposure errors, security block rate 100%.
  - No security control changed. `PII_ALLOWED_OPERATIONS`, the intent tool allow-list, the SQL
    validator, the injection screen and the period-spec allow-list are untouched.
  - Rep questions reach `rep_performance` because the planner now asks for the allowed operation,
    not because a policy was relaxed.
- **Data exposure:**
  - `company_name` stays withheld under the current policy, and the customer-risk path passes
    (`exp_mcp_customer_risk`, `exp_risk_row_limit`, `exp_agent_risk_company_names`).
  - Rep names appear only in `analyze_sales.rep_performance` evidence (tested).
- **MCP:**
  - 23 of 23 MCP scenarios pass; parity is 100% over 20 scenarios and 47 calls; discovery and the
    shared execution path are verified.
  - CAC by channel through MCP returns the same breakdown as the analytics layer (tested).
  - MCP still runs through `app/security/execution.py`; the MCP code is unchanged.
- **Multi-seed:** 22 of 22 runs on seeds 7 and 2027 (baseline 22/22).
- **Hidden ground truth:** the boundary test caught an example identifier in a production comment
  that was the E3 campaign label; it was replaced. No production code contains label text.

## 11. Tests and quality checks

| Check | Result |
|---|---|
| `pytest` | 2,275 passed (Phase 7: 2,163; 112 new) |
| `ruff check app data tests evals` | clean |
| `ruff format --check app data tests evals` | clean (248 files) |
| `mypy app data evals` | clean (161 source files) |

The 112 new tests are split as follows:

- `test_phase71_understanding.py`: 41;
- `test_phase71_claim_evidence.py`: 42;
- `test_phase71_agent_regressions.py`: 22, through the real agent on the full dataset;
- `test_eval_graders.py`: 7 more.
