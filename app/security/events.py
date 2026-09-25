"""Security audit events: one typed, serialisable record per security decision.

Every boundary (input guard, injection screening, plan validation, tool authorization, SQL
validation, budget, timeouts, output validation, evidence integrity) records what it decided and
why. The events travel with the run (``AgentState.security_events`` and ``AgentRunResult``) and
are emitted as JSON on the ``agentops.security`` logger. Reasons are redacted before they are
stored, and events never carry prompts, raw rows, secrets or the user's full question.

Severities:

- ``INFO``: normal decisions (a tool call authorised, context trimmed).
- ``WARNING``: suspicious or degraded (a suspicious prompt pattern, a retry, a rejected model output).
- ``HIGH``: an attempted policy violation (unsafe SQL, a denied tool, an exhausted budget).
- ``CRITICAL``: an attempt to reach secrets or hidden data, or to execute code.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.security.redaction import redact, redact_value

SECURITY_LOGGER_NAME = "agentops.security"
logger = logging.getLogger(SECURITY_LOGGER_NAME)
MAX_REASON_CHARS = 300


class Severity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


SecurityEventType = Literal[
    "input_rejected",
    "suspicious_prompt",
    "privileges_reduced",
    "secret_redacted",
    "unsupported_request",
    "model_output_rejected",
    "plan_rejected",
    "tool_authorized",
    "tool_denied",
    "argument_rejected",
    "sql_rejected",
    "budget_exceeded",
    "timeout",
    "retry",
    "tool_output_rejected",
    "evidence_validation_failed",
    "evidence_integrity_failed",
    "output_validation_failed",
    "response_truncated",
    "context_truncated",
]
Decision = Literal["allow", "deny", "restrict", "redact", "retry", "stop", "trim", "flag"]


class SecurityEvent(BaseModel):
    event_id: str
    run_id: str
    event_type: SecurityEventType
    severity: Severity
    timestamp: datetime
    component: str
    action: str
    decision: Decision
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)


def security_event(
    run_id: str,
    event_type: SecurityEventType,
    severity: Severity,
    *,
    component: str,
    action: str,
    decision: Decision,
    reason: str,
    **details: Any,
) -> SecurityEvent:
    """Create (and log) one event. Reason and details are redacted and bounded."""
    event = SecurityEvent(
        event_id=f"SE-{uuid.uuid4().hex[:12]}",
        run_id=run_id,
        event_type=event_type,
        severity=severity,
        timestamp=datetime.now(UTC),
        component=component,
        action=action,
        decision=decision,
        reason=redact(reason)[:MAX_REASON_CHARS],
        details={k: redact_value(v) for k, v in details.items()},
    )
    level = {
        Severity.INFO: logging.INFO,
        Severity.WARNING: logging.WARNING,
        Severity.HIGH: logging.WARNING,
        Severity.CRITICAL: logging.ERROR,
    }[severity]
    if logger.isEnabledFor(level):
        logger.log(level, json.dumps(event.model_dump(mode="json"), sort_keys=True, default=str))
    return event


def highest_severity(events: list[SecurityEvent]) -> Severity | None:
    order = [Severity.INFO, Severity.WARNING, Severity.HIGH, Severity.CRITICAL]
    return max((e.severity for e in events), key=order.index, default=None)
