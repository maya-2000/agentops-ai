"""Structured request logs and in-process request metrics for the API.

One JSON line per request on the ``agentops.api`` logger, keyed by the request ID (which is also the
agent run ID, so it joins the ``agentops.agent`` and ``agentops.security`` records of the same run).
Logged: request and session IDs, method, the matched route (never the raw path), HTTP status, agent
status and outcome, error code, counts and timings. Never logged: the question, the answer,
evidence values, headers (so never the ``Authorization`` token), client addresses, prompts,
environment values or secrets. Unknown keys are dropped. ``LOG_FORMAT=json`` adds a timestamp,
level and logger to every line (``app/logs.py``).

Metrics are counters since start-up. They are grouped so that expected outcomes stay apart from
failures:

- **Agent outcomes:** answered, partial, refused, unsupported, insufficient evidence, failed.
- **Client errors:** invalid request, unauthorized, rate limited.
- **Infrastructure errors:** timeout, busy or unavailable, internal.

Latencies keep the last 1,000 samples. There are no per-client, per-question or per-user dimensions.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import Counter, deque
from typing import Any

from app.api.schemas.responses import LatencyStats, MetricsResponse, MetricsSummary
from app.security.redaction import redact_value

API_LOGGER_NAME = "agentops.api"
logger = logging.getLogger(API_LOGGER_NAME)
SAMPLES = 1000

_ALLOWED_KEYS = {
    "session_id",
    "method",
    "endpoint",
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
_SERVICE_KEYS = {"reason", "hint", "error_type", "runs", "environment", "auth", "rate_limit", "version", "problems"}

_INFRASTRUCTURE_CODES = {"timeout", "busy", "agent_unavailable", "shutting_down", "internal_error"}
_EXPECTED_OUTCOMES = ("answered", "partial", "refused", "unsupported", "insufficient_evidence", "failed")


def log_request(request_id: str, event: str, **fields: Any) -> None:
    if not logger.isEnabledFor(logging.INFO):
        return
    payload = {"request_id": request_id, "event": event}
    payload.update({k: redact_value(v) for k, v in fields.items() if k in _ALLOWED_KEYS and v is not None})
    logger.info(json.dumps(payload, default=str, sort_keys=True))


def log_service(event: str, **fields: Any) -> None:
    """Service lifecycle events (start-up, unavailability, shutdown): reasons and counts, never paths."""
    payload = {"event": event}
    payload.update({k: redact_value(v) for k, v in fields.items() if k in _SERVICE_KEYS and v is not None})
    level = logging.ERROR if event in ("service_unavailable", "configuration_rejected") else logging.INFO
    logger.log(level, json.dumps(payload, default=str, sort_keys=True))


def _percentile(sorted_values: list[float], fraction: float) -> float:
    rank = max(1, math.ceil(fraction * len(sorted_values)))
    return sorted_values[rank - 1]


def _stats(samples: deque[float]) -> LatencyStats:
    values = sorted(samples)
    if not values:
        return LatencyStats(count=0)
    return LatencyStats(
        count=len(values),
        mean_ms=round(sum(values) / len(values), 1),
        p50_ms=round(_percentile(values, 0.5), 1),
        p95_ms=round(_percentile(values, 0.95), 1),
        max_ms=round(values[-1], 1),
    )


class RequestMetrics:
    """Counters since start-up. Thread-safe; nothing is persisted."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._requests = 0
        self._in_flight = 0
        self._outcomes: Counter[str] = Counter()
        self._status_codes: Counter[str] = Counter()
        self._error_codes: Counter[str] = Counter()
        self._agent_ms: deque[float] = deque(maxlen=SAMPLES)
        self._overhead_ms: deque[float] = deque(maxlen=SAMPLES)
        self._ask_ms: deque[float] = deque(maxlen=SAMPLES)

    def started(self) -> None:
        with self._lock:
            self._requests += 1
            self._in_flight += 1

    def finished(
        self, status_code: int, *, error_code: str | None = None, duration_ms: float | None = None, ask: bool = False
    ) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._status_codes[str(status_code)] += 1
            if error_code:
                self._error_codes[error_code] += 1
            if ask and duration_ms is not None:
                self._ask_ms.append(duration_ms)

    def record_run(self, outcome: str, agent_ms: float, total_ms: float) -> None:
        with self._lock:
            self._outcomes[outcome] += 1
            self._agent_ms.append(agent_ms)
            self._overhead_ms.append(max(0.0, total_ms - agent_ms))

    def snapshot(self, runs: dict[str, int] | None = None) -> MetricsResponse:
        with self._lock:
            agent = list(self._agent_ms)
            overhead = list(self._overhead_ms)
            errors = self._error_codes
            summary = MetricsSummary(
                **{outcome: self._outcomes[outcome] for outcome in _EXPECTED_OUTCOMES},
                client_errors=sum(
                    n
                    for code, n in errors.items()
                    if code not in _INFRASTRUCTURE_CODES and code not in ("unauthorized", "rate_limited")
                ),
                unauthorized=errors["unauthorized"],
                rate_limited=errors["rate_limited"],
                timeouts=errors["timeout"],
                unavailable=errors["busy"] + errors["agent_unavailable"] + errors["shutting_down"],
                internal_errors=errors["internal_error"],
            )
            return MetricsResponse(
                uptime_seconds=round(time.monotonic() - self._started, 1),
                requests_total=self._requests,
                in_flight=self._in_flight,
                by_outcome=dict(self._outcomes),
                by_status_code=dict(self._status_codes),
                by_error_code=dict(errors),
                summary=summary,
                request_latency=_stats(self._ask_ms),
                agent_latency=_stats(self._agent_ms),
                runs=dict(runs or {}),
                agent_time_ms_avg=round(sum(agent) / len(agent), 1) if agent else None,
                agent_time_ms_max=round(max(agent), 1) if agent else None,
                api_overhead_ms_avg=round(sum(overhead) / len(overhead), 2) if overhead else None,
            )
