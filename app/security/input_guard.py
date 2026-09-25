"""Input guardrails: the first trust boundary.

Two inputs are validated here, both treated as untrusted:

1. **The user's question** (``validate_question``). Checks:
   - it must be a string, non-empty after normalisation, and no longer than ``max_question_chars``;
   - control characters are removed and whitespace collapsed;
   - secrets pasted into the question are redacted before any model, log or trace sees them;
   - the prompt-injection screen then decides ``clean`` / ``restrict`` / ``block``.
2. **The model's structured understanding** (``validate_understanding_output``), before the
   request validation resolves it. Checks:
   - filter and dimension counts are within limits, and no filter is repeated;
   - period specs have a supported format;
   - ``understanding_output_problems`` bounds free-text fields, so an oversized model output
     is rejected and regenerated within the retry budget.

   Metric names, dimensions, filter values and horizons are validated by ``app.agent.request``
   through the same central validators.

Everything is typed, and every value check calls the central validators, so the rules live in
one place.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.llm.schemas import UnderstandingOutput
from app.security.injection import InjectionScan, PromptInjectionDetector
from app.security.limits import SecurityLimits
from app.security.redaction import redact
from app.security.validators import Violation, check_period_spec, check_text

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
QuestionOutcome = Literal["accepted", "rejected", "blocked"]


class QuestionCheck(BaseModel):
    outcome: QuestionOutcome
    question: str = ""  # normalised and redacted; the only form used downstream
    reason: str | None = None
    code: str | None = None
    redacted: bool = False
    scan: InjectionScan = Field(default_factory=InjectionScan)

    @property
    def restricted(self) -> bool:
        return self.scan.verdict == "restrict"


class InputGuard:
    def __init__(self, limits: SecurityLimits, detector: PromptInjectionDetector | None = None):
        self.limits = limits
        self.detector = detector or PromptInjectionDetector()

    def validate_question(self, raw: Any) -> QuestionCheck:
        if not isinstance(raw, str):
            return QuestionCheck(outcome="rejected", code="invalid_input", reason="The question must be text.")
        cleaned = " ".join(_CONTROL.sub(" ", raw).split())
        if not cleaned:
            return QuestionCheck(outcome="rejected", code="empty_input", reason="The question is empty.")
        if len(cleaned) > self.limits.max_question_chars:
            return QuestionCheck(
                outcome="rejected",
                code="oversized_input",
                question=redact(cleaned[: self.limits.max_question_chars]),
                reason=f"The question is longer than {self.limits.max_question_chars} characters.",
            )
        redacted = redact(cleaned)
        scan = self.detector.scan(cleaned)  # screen the original: a secret-bearing question is itself suspicious
        if scan.verdict == "block":
            return QuestionCheck(
                outcome="blocked",
                code="prompt_injection",
                question=redacted,
                redacted=redacted != cleaned,
                reason="The request asks for something the agent is not permitted to do.",
                scan=scan,
            )
        return QuestionCheck(outcome="accepted", question=redacted, redacted=redacted != cleaned, scan=scan)

    def validate_understanding_output(self, u: UnderstandingOutput) -> list[Violation]:
        """Security limits on the model's structured reading of the question.

        Counts, duplicates, period-spec format and text sizes are checked here. The vocabulary
        (metric, dimensions, filter values, horizon) is checked by the request validation
        (``app.agent.request``) through the same central validators, which also map each problem
        to its user-facing outcome.
        """
        limits = self.limits
        problems: list[Violation] = []
        if len(u.filters) > limits.max_filters:
            problems.append(
                Violation(code="oversized_input", field="filters", message=f"At most {limits.max_filters} filters")
            )
        if len(u.dimensions) > limits.max_dimensions:
            problems.append(
                Violation(
                    code="oversized_input", field="dimensions", message=f"At most {limits.max_dimensions} dimensions"
                )
            )
        if len({f.dimension for f in u.filters}) != len(u.filters):
            problems.append(Violation(code="invalid_arguments", field="filters", message="A filter is repeated"))
        for field in ("period", "comparison_period"):
            value = getattr(u, field)
            if value is not None:
                problems += check_period_spec(value.strip().lower(), field)
        return problems


def understanding_output_problems(u: UnderstandingOutput, limits: SecurityLimits) -> list[str]:
    """Structural problems in model output that justify asking the model again (bounded retries)."""
    problems: list[Violation] = []
    texts = [("unsupported_reason", u.unsupported_reason or ""), *(("ambiguities", a) for a in u.ambiguities)]
    for field, text in texts:
        problems += check_text(text, field, limits.max_text_field_chars)
    if u.metric is not None:
        problems += check_text(u.metric, "metric", 64)
    for item in u.filters:
        problems += check_text(item.dimension, "filters.dimension", 64) + check_text(item.value, "filters.value", 100)
    if len(u.ambiguities) > 10 or len(u.dimensions) > 20 or len(u.filters) > 20:
        problems.append(Violation(code="oversized_input", field="understanding", message="Too many items"))
    return [f"{p.field}: {p.message}" for p in problems]
