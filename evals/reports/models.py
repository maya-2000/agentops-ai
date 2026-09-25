"""Result models: failures, per-scenario results and the run summary."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

REPORT_VERSION = "1.0"


class FailureCategory(StrEnum):
    INTENT_ERROR = "INTENT_ERROR"
    PARAMETER_ERROR = "PARAMETER_ERROR"
    TOOL_SELECTION_ERROR = "TOOL_SELECTION_ERROR"
    TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
    NUMERICAL_ERROR = "NUMERICAL_ERROR"
    EVIDENCE_ERROR = "EVIDENCE_ERROR"
    CLAIM_SUPPORT_ERROR = "CLAIM_SUPPORT_ERROR"
    HALLUCINATION = "HALLUCINATION"
    CAUSALITY_ERROR = "CAUSALITY_ERROR"
    UNCERTAINTY_ERROR = "UNCERTAINTY_ERROR"
    REFUSAL_ERROR = "REFUSAL_ERROR"
    SECURITY_ERROR = "SECURITY_ERROR"
    DATA_EXPOSURE_ERROR = "DATA_EXPOSURE_ERROR"
    RESOURCE_ERROR = "RESOURCE_ERROR"
    MCP_ERROR = "MCP_ERROR"
    RESPONSE_QUALITY_ERROR = "RESPONSE_QUALITY_ERROR"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


class Failure(BaseModel):
    """One failed check, with enough context to debug it without re-running the scenario."""

    scenario_id: str
    category: FailureCategory
    check: str
    message: str
    expected: Any = None
    actual: Any = None
    tool: str | None = None
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    security_events: list[str] = Field(default_factory=list)


SCORE_NAMES = (
    "intent",
    "parameter",
    "tool_selection",
    "tool_execution",
    "numerical",
    "evidence",
    "claim_support",
    "hallucination",
    "causality",
    "uncertainty",
    "refusal",
    "security",
    "data_exposure",
    "mcp",
    "efficiency",
)


class EvaluationResult(BaseModel):
    scenario_id: str
    category: str
    difficulty: str
    mode: str
    latency_class: str
    status: str  # passed / failed / error
    scores: dict[str, float | None] = Field(default_factory=dict)  # 0..1 per SCORE_NAMES; None = not applicable
    failures: list[Failure] = Field(default_factory=list)
    latency_ms: float = 0.0  # production execution only (agent run or MCP calls)
    evaluation_overhead_ms: float = 0.0  # reference resolution + grading
    tool_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    retries: int = 0
    duplicate_calls: int = 0
    unnecessary_calls: int = 0
    sql_calls: int = 0
    sql_rows: int = 0
    context_items: int = 0
    response_chars: int = 0
    agent_status: str | None = None
    actual_intent: str | None = None
    refused: bool | None = None
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    security_events: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    judge: dict[str, Any] | None = None  # optional LLM judge output (never affects status)

    @property
    def passed(self) -> bool:
        return self.status == "passed"


class ThresholdCheck(BaseModel):
    name: str
    comparison: str  # ">=", "<=", "=="
    threshold: float
    actual: float | None
    passed: bool
    rationale: str


class EvaluationRunSummary(BaseModel):
    report_version: str = REPORT_VERSION
    run_id: str
    timestamp: str
    git_commit: str
    git_dirty: bool
    dataset_version: str
    dataset_schema_version: str
    dataset_updated: str
    data: dict[str, Any]  # data source: origin, seed, business dataset version, as-of
    configuration: dict[str, Any]  # mode, provider, model, temperature, suite, filters, limits
    total_scenarios: int
    passed: int
    failed: int
    errors: int
    metrics: dict[str, float | None]
    by_category: dict[str, dict[str, Any]]
    by_difficulty: dict[str, dict[str, Any]]
    security: dict[str, Any]
    mcp: dict[str, Any]
    performance: dict[str, Any]
    failure_counts: dict[str, int]
    thresholds: list[ThresholdCheck] = Field(default_factory=list)
    thresholds_passed: bool = True
    multi_seed: list[dict[str, Any]] = Field(default_factory=list)
    results: list[EvaluationResult] = Field(default_factory=list)
