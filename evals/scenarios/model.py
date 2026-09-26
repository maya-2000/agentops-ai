"""The typed evaluation scenario: what to run and what the production system should do.

A scenario never contains an expected business number or conclusion. Numerical expectations are
*references*: named checks (``ReferenceCheck``) resolved at run time, either from the independent
Phase 2/3 reference implementations or from the hidden evaluation labels. The benchmark therefore
stays correct when the data changes (another seed), and it cannot grade the agent against itself.

Execution modes:

- ``agent``: a natural-language question to the LangGraph agent (``AgentRunner``). An optional
  ``adversarial_plan`` simulates a compromised model that proposes those tool calls instead.
- ``mcp``: calls to the real MCP server over the SDK's in-process client.
- ``parity``: the same calls through the direct secured executor (the agent's path) and through
  MCP; the outcomes and business results must match.
- ``mcp_discovery``: tool discovery and schemas over the protocol.
- ``shared_execution``: an agent question and an MCP call, instrumented to show that both go
  through ``app/security/execution.py``.
- ``evidence_integrity``: a real agent answer is mutated (fake evidence IDs, wrong period,
  fabricated numbers, causal wording, ...); every mutation must be detected.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"


class Category(StrEnum):
    KPI = "kpi"
    REVENUE = "revenue"
    INVESTIGATION = "investigation"
    CUSTOMERS = "customers"
    SALES = "sales"
    MARKETING = "marketing"
    SUPPORT = "support"
    PRODUCT = "product"
    COHORTS = "cohorts"
    RISK = "risk"
    FORECAST = "forecast"
    ANOMALY = "anomaly"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNSUPPORTED = "unsupported"
    SECURITY = "security"
    PROMPT_INJECTION = "prompt_injection"
    SQL_SECURITY = "sql_security"
    DATA_EXPOSURE = "data_exposure"
    MCP = "mcp"
    EVIDENCE_INTEGRITY = "evidence_integrity"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADVERSARIAL = "adversarial"


class Mode(StrEnum):
    AGENT = "agent"
    MCP = "mcp"
    PARITY = "parity"
    MCP_DISCOVERY = "mcp_discovery"
    SHARED_EXECUTION = "shared_execution"
    EVIDENCE_INTEGRITY = "evidence_integrity"


LatencyClass = Literal[
    "simple_kpi",
    "medium_investigation",
    "complex_investigation",
    "security_rejection",
    "mcp_invocation",
    "instrumented",
]
Tolerance = Literal["money", "rate", "count", "duration", "forecast"]
LeakKind = Literal["system_prompt", "secrets", "ground_truth", "file_contents", "withheld_fields"]
ForbiddenClaim = Literal[
    "causal_language",
    "hidden_event_labels",
    "forecast_certainty",
    "anomaly_judgement",
    "fabricated_numbers",
]
Mutation = Literal[
    "fake_evidence_id",
    "missing_evidence",
    "wrong_evidence",
    "mismatched_period",
    "mismatched_metric",
    "fabricated_number",
    "missing_provenance",
    "causal_overstatement",
    "tampered_evidence",
    "mismatched_comparison",
    "mismatched_dimension",
    "mismatched_filters",
    "mismatched_unit",
]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolCall(Strict):
    """One tool call. ``tool`` is the MCP name (``agentops_*``) in MCP modes; parity maps it to the Phase 4 tool."""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class CallExpectation(Strict):
    """The expected outcome of one call in the MCP / parity modes."""

    outcome: Literal["ok", "error"]
    error_category: str | None = None  # MCP error category, e.g. UNSAFE_QUERY
    error_codes: list[str] = Field(default_factory=list)  # acceptable internal codes, e.g. unsafe_sql


class ExpectedParameters(Strict):
    """Structured parameters the agent must choose; ``None`` means "not checked"."""

    metric: str | None = None
    period: str | None = None  # a period label, e.g. 2026-08 or 2026-Q2
    comparison_period: str | None = None
    dimensions: list[str] | None = None
    filters: dict[str, str] | None = None
    horizon: int | None = None
    detector: str | None = None  # checked in the detect_anomalies call arguments


class ReferenceCheck(Strict):
    """A check whose expected value is resolved at run time (never written into the scenario).

    kinds:
    - ``kpi_value``: a KPI for a period (and filters), from the independent pandas reference.
    - ``kpi_change``: a KPI in two periods and the change between them.
    - ``top_member``: the member of ``dimension`` that is highest / lowest / has the largest decline.
    - ``forecast``: forecast metadata; values against the independent naive/drift reference.
    - ``anomaly_event``: the month of a hidden event is flagged with the event's direction.
    - ``event_discovery``: the observable manifestation of a hidden event appears in the evidence.
    """

    kind: Literal["kpi_value", "kpi_change", "top_member", "forecast", "anomaly_event", "event_discovery"]
    metric: str | None = None
    period: str | None = None
    comparison_period: str | None = None
    dimension: str | None = None
    filters: dict[str, str] = Field(default_factory=dict)
    which: Literal["highest", "lowest", "largest_decline"] | None = None
    event: str | None = None  # hidden event ID (evaluation label), e.g. E1
    horizon: int | None = None
    min_sample: int | None = None  # the same minimum-sample argument the tool receives (e.g. min_closed)
    tolerance: Tolerance | None = None
    in_answer: bool = True  # the value or member must also appear in the delivered answer
    source: Literal["agent", "mcp"] = "agent"  # where to read the actual value
    call: int = 0  # for source "mcp": which call of the scenario carries the value


class RequiredEvidence(Strict):
    evidence_type: Literal["observed", "calculated", "forecast", "anomaly", "derived"]
    metric: str | None = None


class SecurityExpectation(Strict):
    """How the system must treat a (possibly) hostile request."""

    outcome: Literal["blocked", "restricted", "not_flagged"]
    expected_events: list[str] = Field(default_factory=list)  # SecurityEvent types that must appear
    forbidden_tools: list[str] = Field(default_factory=list)  # tools that must not execute
    must_not_leak: list[LeakKind] = Field(default_factory=list)


class ExposureExpectation(Strict):
    """Data minimisation: what a response may expose (derived from the live exposure policy)."""

    withheld_fields_masked: bool = True  # withheld / PII values never appear (policy-derived)
    max_customer_identifiers: int | None = None  # customer IDs allowed in the delivered text
    max_customer_rows: bool = True  # customer-level rows within AGENT_MAX_CUSTOMER_ROWS


class EvaluationScenario(Strict):
    scenario_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    category: Category
    difficulty: Difficulty
    mode: Mode = Mode.AGENT
    description: str = ""
    question: str | None = None
    calls: list[ToolCall] = Field(default_factory=list)
    call_expectations: list[CallExpectation] = Field(default_factory=list)
    adversarial_plan: list[ToolCall] | None = None
    limits: dict[str, Any] = Field(default_factory=dict)  # AgentConfig overrides for this scenario
    expected_status: list[str] = Field(default_factory=list)
    expected_intent: str | None = None
    expected_tools: list[str] = Field(default_factory=list)  # required (recall)
    acceptable_tools: list[str] = Field(default_factory=list)  # allowed extras (not unnecessary)
    forbidden_tools: list[str] = Field(default_factory=list)
    expected_parameters: ExpectedParameters | None = None
    reference_expectations: list[ReferenceCheck] = Field(default_factory=list)
    required_evidence: list[RequiredEvidence] = Field(default_factory=list)
    forbidden_claims: list[ForbiddenClaim] = Field(default_factory=list)
    should_refuse: bool | None = None  # None: not part of the refusal metrics
    security_expectation: SecurityExpectation | None = None
    exposure: ExposureExpectation | None = None
    mutations: list[Mutation] = Field(default_factory=list)
    max_tool_calls: int | None = None  # efficiency budget; default: required + acceptable tools
    latency_class: LatencyClass = "simple_kpi"
    suites: list[Literal["critical", "multi_seed"]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    dataset_version: str

    @model_validator(mode="after")
    def _consistent(self) -> EvaluationScenario:
        needs_question = self.mode in (Mode.AGENT, Mode.SHARED_EXECUTION, Mode.EVIDENCE_INTEGRITY)
        if needs_question and not self.question:
            raise ValueError(f"{self.scenario_id}: mode {self.mode} needs a question")
        needs_calls = self.mode in (Mode.MCP, Mode.PARITY, Mode.SHARED_EXECUTION)
        if needs_calls and not self.calls:
            raise ValueError(f"{self.scenario_id}: mode {self.mode} needs calls")
        if self.calls and self.call_expectations and len(self.calls) != len(self.call_expectations):
            raise ValueError(f"{self.scenario_id}: one call expectation per call")
        if self.mode == Mode.EVIDENCE_INTEGRITY and not self.mutations:
            raise ValueError(f"{self.scenario_id}: evidence-integrity scenarios need mutations")
        if self.category == Category.PROMPT_INJECTION and self.security_expectation is None:
            raise ValueError(f"{self.scenario_id}: prompt-injection scenarios need a security expectation")
        return self

    @property
    def efficiency_budget(self) -> int:
        if self.max_tool_calls is not None:
            return self.max_tool_calls
        return max(1, len(set(self.expected_tools)) + len(set(self.acceptable_tools)))


class DatasetManifest(Strict):
    dataset_version: str = Field(pattern=r"^eval_v\d+$")
    schema_version: str
    updated: str  # ISO date of the last material change
    description: str


class EvaluationDataset(Strict):
    manifest: DatasetManifest
    scenarios: list[EvaluationScenario]

    @model_validator(mode="after")
    def _unique(self) -> EvaluationDataset:
        ids = [s.scenario_id for s in self.scenarios]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"Duplicate scenario IDs: {', '.join(duplicates)}")
        wrong = [s.scenario_id for s in self.scenarios if s.dataset_version != self.manifest.dataset_version]
        if wrong:
            raise ValueError(f"Scenarios with another dataset version: {', '.join(wrong)}")
        if self.manifest.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Schema version {self.manifest.schema_version} is not {SCHEMA_VERSION}")
        return self
