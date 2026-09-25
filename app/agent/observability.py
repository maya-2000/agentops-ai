"""Lightweight structured logging for agent runs (one JSON object per event).

Logged: run ID, state transitions, tool calls and their success, evidence-validation results and
the final status. Never logged: API keys or other secrets, prompts, raw query rows, generator
internals or injected-event ground truth.
"""

from __future__ import annotations

import json
import logging
from typing import Any

LOGGER_NAME = "agentops.agent"
logger = logging.getLogger(LOGGER_NAME)

_ALLOWED_KEYS = {
    "node",
    "route",
    "tool_name",
    "call_id",
    "success",
    "status",
    "attempts",
    "error_code",
    "execution_time_ms",
    "query_ids",
    "evidence_count",
    "claim_count",
    "valid",
    "error_count",
    "warning_count",
    "tool_calls",
    "retries",
    "provider",
    "task",
    "iteration",
    "steps",
    "execution_time_seconds",
}


def log_event(run_id: str, event: str, **fields: Any) -> None:
    """Emit one structured event. Unknown keys are dropped so nothing sensitive is logged by accident."""
    if not logger.isEnabledFor(logging.INFO):
        return
    payload = {"run_id": run_id, "event": event}
    payload.update({k: v for k, v in fields.items() if k in _ALLOWED_KEYS})
    logger.info(json.dumps(payload, default=str, sort_keys=True))
