"""Structured request logs and in-process request metrics for the API.

One JSON line per request on the ``agentops.api`` logger, keyed by the request ID (which is also the
agent run ID, so it joins the ``agentops.agent`` and ``agentops.security`` records of the same run).
Logged: request and session IDs, method, path, HTTP status, agent status and outcome, counts and
timings. Never logged: the question, the answer, evidence values, headers, client addresses,
prompts, environment values or secrets. Unknown keys are dropped.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from typing import Any

from app.api.schemas.responses import MetricsResponse
from app.security.redaction import redact_value

API_LOGGER_NAME = "agentops.api"
logger = logging.getLogger(API_LOGGER_NAME)

_ALLOWED_KEYS = {
    "session_id",
    "method",
    "path",
    "status_code",
    "agent_status",
    "outcome",
    "error_code",
    "tool_calls",
    "evidence_count",
    "claim_count",
    "duration_ms",
    "agent_time_ms",
    "streamed",
    "error_type",  # an exception class name only, never its message
}


def log_request(request_id: str, event: str, **fields: Any) -> None:
    if not logger.isEnabledFor(logging.INFO):
        return
    payload = {"request_id": request_id, "event": event}
    payload.update({k: redact_value(v) for k, v in fields.items() if k in _ALLOWED_KEYS and v is not None})
    logger.info(json.dumps(payload, default=str, sort_keys=True))


class RequestMetrics:
    """Counters since start-up. Thread-safe; nothing is persisted."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._requests = 0
        self._in_flight = 0
        self._outcomes: Counter[str] = Counter()
        self._status_codes: Counter[str] = Counter()
        self._agent_ms: list[float] = []
        self._overhead_ms: list[float] = []

    def started(self) -> None:
        with self._lock:
            self._requests += 1
            self._in_flight += 1

    def finished(self, status_code: int) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._status_codes[str(status_code)] += 1

    def record_run(self, outcome: str, agent_ms: float, total_ms: float) -> None:
        with self._lock:
            self._outcomes[outcome] += 1
            self._agent_ms.append(agent_ms)
            self._overhead_ms.append(max(0.0, total_ms - agent_ms))
            # Bounded memory: keep the most recent 1000 samples.
            del self._agent_ms[:-1000], self._overhead_ms[:-1000]

    def snapshot(self) -> MetricsResponse:
        with self._lock:
            agent = self._agent_ms
            overhead = self._overhead_ms
            return MetricsResponse(
                uptime_seconds=round(time.monotonic() - self._started, 1),
                requests_total=self._requests,
                in_flight=self._in_flight,
                by_outcome=dict(self._outcomes),
                by_status_code=dict(self._status_codes),
                agent_time_ms_avg=round(sum(agent) / len(agent), 1) if agent else None,
                agent_time_ms_max=round(max(agent), 1) if agent else None,
                api_overhead_ms_avg=round(sum(overhead) / len(overhead), 2) if overhead else None,
            )
