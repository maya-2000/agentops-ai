# Engineering decisions

This document records the design decisions that shape AgentOps AI, as implemented in release 0.10.0.
Each entry gives:

- **Decision:** what was chosen;
- **Reason:** why it was chosen;
- **Trade-off:** what it costs.

Details are in the linked documents.

## 1. Deterministic analytics are the source of truth

- **Decision.** Every business number is computed by typed, deterministic tools (`app/analytics`,
  `app/forecasting`, `app/anomalies`) that run through `SecuredToolExecutor`. The language model
  interprets the question, proposes a plan and drafts the wording. It never computes a number and never
  calls a tool itself.
- **Reason.**
  - Numbers produced by a language model cannot be audited, and cannot be tested reliably.
  - A deterministic engine can be tested against an independent pandas reference. The benchmark does
    this, with 39 numerical reference checks.
  - The default model provider is also deterministic, so the whole pipeline (tests, benchmarks, CI)
    runs offline and reproducibly. Claude can be switched in through the official SDK.
- **Trade-off.**
  - The system answers only what its 12 tools can compute; a question outside them is refused as
    unsupported.
  - Each new analysis means new code and tests, not a new prompt.
- See [final-architecture.md](final-architecture.md) and [analytics.md](analytics.md).

## 2. DuckDB, read-only

- **Decision.** The data lives in one embedded DuckDB file. The application opens it read-only;
  containers mount it read-only. The only way to reach it is the 12 tools, including one guarded
  read-only `SELECT` tool.
- **Reason.**
  - It is a columnar SQL engine with no server to run. It is fast on the 2.8 million rows of the
    dataset and easy to reproduce from a seed.
  - Read-only access at three levels (connection, mount, SQL validation) means a prompt cannot change
    the data.
- **Trade-off.** One process holds one read-only connection, so runs are serialised (a single worker
  thread in `app/api/service.py`). That is right for a single-user service, not for concurrent load.
  See decision 9.

## 3. LangGraph for the agent's state machine

- **Decision.** The single-question agent is an explicit LangGraph state machine (`app/agent/graph.py`)
  with these nodes: understand, validate the request, plan, execute tools, collect evidence, validate
  evidence, generate the response, validate the response. It has five named failure states and budget
  checks between nodes.
- **Reason.**
  - Each stage is a node that can be tested on its own, and every transition is visible in the trace
    the UI shows.
  - Failure handling is explicit (`unsupported_request`, `insufficient_evidence`, `tool_error`,
    `validation_failure`, `planning_failure`) rather than hidden in a loop.
  - Only LangGraph is used; `langchain-core` comes in as a dependency of it, and no other LangChain
    package is.
- **Trade-off.**
  - It is a framework dependency, and its flow is fixed rather than open-ended.
  - The investigator reuses the agent runtime's components but runs its template steps in plain
    Python, because its flow is a fixed loop over a plan.
- See [agent-architecture.md](agent-architecture.md).

## 4. MCP as a thin adapter over the same executor

- **Decision.** An MCP server (stdio, official Python SDK) exposes the same 12 tools as
  `agentops_*`. Each MCP call goes through the same `SecuredToolExecutor` and returns the same evidence
  and provenance.
- **Reason.**
  - Other agents, such as Claude Code, can use the analytics without a second, less-guarded path to
    the data.
  - The benchmark checks that direct and MCP calls give the same result: 100% parity over 20 scenarios
    and 47 calls.
- **Trade-off.**
  - MCP clients get tools, not answers. Investigations are deliberately not exposed as an MCP tool;
    they are a workflow built from these same calls.
  - There are no file, shell or code tools.
- See [mcp-architecture.md](mcp-architecture.md).

## 5. Evidence objects with provenance and fingerprints

- **Decision.** Every tool result becomes one or more evidence items. Each carries:
  - the metric, value, period and comparison;
  - the filters and dimension;
  - the source tables, calculation and query IDs;
  - the producing tool call;
  - a SHA-256 fingerprint of its content.
- **Reason.**
  - Each number can be traced to the query that produced it, and the UI shows that trail.
  - The fingerprint lets the runtime detect evidence that changed after it was recorded.
- **Trade-off.** Tool outputs must be structured and typed, and each tool needs an evidence mapping.
  Free-form text results are not accepted as evidence.
- See [final-architecture.md](final-architecture.md).

## 6. Claim/evidence identity validation

- **Decision.** Each claim is labelled *Observed*, *Calculated*, *Inferred* or *Recommended*, cites
  its evidence IDs, and carries a structured subject: metric, period, dimension and unit. The validator
  checks that:
  - the subject matches the cited evidence;
  - the stated numbers and direction match it;
  - an *Observed* claim rests only on observed evidence.

  Claims that fail are removed.
- **Reason.** Checking numbers alone is not enough: a correct number can be attached to the wrong
  metric or period. Adding identity checks came out of the Phase 7.1 reliability work.
- **Trade-off.** Claim builders must produce structured subjects, not free text, so every new claim
  type needs its builder and validator rules updated.
- See [phase-7-1-reliability.md](phase-7-1-reliability.md).

## 7. Rule-based investigation drivers

- **Decision.** Investigations come from six fixed templates. Drivers are derived by rule from
  validated findings:
  - accounting contributions (a member's share of the gross change);
  - same-period co-movements;
  - associations (for example, usage before churn);
  - contradictions.

  Every driver cites its findings and evidence.
- **Reason.**
  - Rules are deterministic, explainable and testable.
  - The eval_v2 benchmark grades driver correctness and recommendation grounding against independent
    references.
  - A model-chosen plan could run tools outside the objective, and could not be graded as easily.
- **Trade-off.**
  - Objectives outside the six templates are unsupported.
  - Drivers compare one pair of periods, with no significance tests. They say where a change occurred
    and what moved with it, not why.
- See [investigations.md](investigations.md).

## 8. Synthetic data with hidden ground truth

- **Decision.** The data is a reproducible simulation of a B2B SaaS company (Northwind Cloud, seed 42).
  A latent customer-health variable drives behaviour but is never stored. Injected business events
  serve as evaluation labels, and they are kept outside the database and away from the agent.
- **Reason.**
  - Real business data would bring privacy obligations and could not be shared.
  - Synthetic data can be regenerated on other seeds for multi-seed checks.
  - The hidden labels let the benchmark grade behaviour against a known truth without the agent ever
    seeing it.
- **Trade-off.** The data represents one company and one generator's assumptions. Results do not
  transfer to real data without testing on it.
- See [data/README.md](../data/README.md).

## 9. Single-process deployment

- **Decision.** The API runs as one process, with one read-only database connection and one worker
  thread. Rate limits and metrics are held in memory, and nothing is stored between runs. The UI is a
  separate process that only calls the API. Both ship as hardened containers.
- **Reason.**
  - It is simple to run, reason about and secure. There is no shared state to protect and no stored
    questions or answers to leak.
  - It fits a single-team analytical service and a reproducible demo.
- **Trade-off.**
  - Runs are serialised.
  - Rate limits reset on restart and are not shared across replicas.
  - Access is one shared service token, with no user accounts; TLS and sign-in belong in a reverse
    proxy.
- See [deployment.md](deployment.md).

## 10. No unsupported causal claims

- **Decision.** Causal wording ("caused by", "due to", "led to", "resulted from", "because of") is
  rejected in claims and answers. A question that asks for a cause returns `insufficient_evidence`,
  together with the observed findings. Investigation briefs state that drivers are contributions and
  co-movements, not established causes.
- **Reason.** The analytics establish accounting shares and associations, not causation. Presenting
  correlation as cause is the most common way business analyses mislead.
- **Trade-off.** Some "why" questions get a careful non-answer instead of a confident story, and the
  user has to take the next analytical step.
- See [evaluation.md](evaluation.md) (causal-safety metrics).

## Out of scope

These are deliberately not part of this project:

- **Model-based planning.** Investigation plans chosen by the model rather than by template, and an
  LLM-mode benchmark as a release gate.
- **Persistence and scale.** Stored investigations with a status endpoint, and multi-replica
  deployment with shared rate limits.
- **Access.** User accounts and per-user attribution.
- **Data.** Connectors to real business data sources.
- **Analytics.** Statistical tests for driver significance.
