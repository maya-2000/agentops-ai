"""Evaluation-side instrumentation: record what the production system did, without changing it.

- ``RecordingLLM`` wraps the configured model client and keeps every request. The leakage check
  scans those requests for hidden-label text.
- ``capture_logs`` collects the production audit and security log records (JSON lines) emitted
  on the ``agentops.*`` loggers while a scenario runs.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app.llm.base import LLMClient, LLMRequest, LLMResponse


class RecordingLLM:
    """A transparent proxy: same provider, model and outputs; every request is kept for inspection."""

    def __init__(self, inner: LLMClient):
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self.inner.generate(request)


class _Collector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = json.loads(record.getMessage())
        except (TypeError, ValueError):
            return
        if isinstance(payload, dict):
            payload["_logger"] = record.name
            self.records.append(payload)


@contextmanager
def capture_logs(*names: str) -> Iterator[list[dict[str, Any]]]:
    """Collect structured log records from the named loggers (INFO and above) for the duration."""
    collector = _Collector()
    loggers = [logging.getLogger(n) for n in names or ("agentops.security", "agentops.mcp", "agentops.agent")]
    previous = [(lg, lg.level, lg.propagate) for lg in loggers]
    for lg in loggers:
        lg.addHandler(collector)
        lg.setLevel(logging.INFO)
        lg.propagate = False  # keep benchmark output quiet; records are kept here
    try:
        yield collector.records
    finally:
        for lg, level, propagate in previous:
            lg.removeHandler(collector)
            lg.setLevel(level)
            lg.propagate = propagate
