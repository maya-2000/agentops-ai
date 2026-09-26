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

To follow: each fix, its regression tests and the before/after benchmark.
