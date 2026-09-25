"""Retry policy: which failures may be retried, how often, and a record of every retry.

A failure is retried only if its code is explicitly classified as transient. Anything else,
including an unknown code, is not retried (fail closed). The retry count per step is bounded by
``SecurityLimits.max_retries``, and the run budget bounds the total.

- **Retryable** (transient): a database/driver error, a provider error flagged retryable
  (rate limit, overload, connection), a model call that timed out, and a malformed model output
  (a new sample may be valid).
- **Not retryable**:
  - invalid or oversized arguments, unknown/disabled/unauthorized tools, unsafe SQL, the
    data-exposure policy;
  - an unsupported request, a prompt-injection block, an exhausted budget;
  - a tool timeout (the same query would time out again), missing data, an invalid tool output.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

FailureClass = Literal["retryable", "non_retryable"]

RETRYABLE_CODES = frozenset(
    {
        "database_error",
        "provider_error_retryable",
        "llm_timeout",
        "model_output_invalid",
    }
)
NON_RETRYABLE_CODES = frozenset(
    {
        "invalid_arguments",
        "oversized_input",
        "unknown_tool",
        "unauthorized_tool",
        "tool_disabled",
        "tool_not_permitted",
        "sql_not_permitted",
        "unsafe_sql",
        "data_policy",
        "unsupported_request",
        "prompt_injection",
        "budget_exceeded",
        "prerequisite_missing",
        "timeout",
        "no_data",
        "insufficient_data",
        "insufficient_history",
        "invalid_tool_output",
        "provider_error",
        "internal_error",
    }
)


class RetryRecord(BaseModel):
    stage: str  # e.g. "execute_tools:get_kpi", "understand_question"
    attempt: int  # the attempt that failed (1-based)
    code: str
    decision: Literal["retry", "stop"]


class RetryPolicy:
    def __init__(self, max_retries: int):
        self.max_retries = max_retries

    @staticmethod
    def classify(code: str | None) -> FailureClass:
        return "retryable" if code in RETRYABLE_CODES else "non_retryable"

    def should_retry(self, code: str | None, attempt: int) -> bool:
        """Retry after failed attempt number ``attempt`` (1-based)?"""
        return self.classify(code) == "retryable" and attempt <= self.max_retries
