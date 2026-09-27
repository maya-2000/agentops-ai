"""The agent service: the only path from the API to the agent.

Every question goes to ``AgentRunner.run`` (Phase 4 graph, Phase 5 secured tool execution); the
service adds nothing to the analysis. It owns:

- one ``AgentRunner`` over one read-only ``Database``, built at start-up;
- one worker thread: runs are serialised, because the DuckDB connection is never used concurrently
  (the same rule as the MCP server's lock). A lock also guards the health probe's metadata query;
- a wall-clock timeout per request (``API_REQUEST_TIMEOUT_SECONDS``, 504 after it) and a bound on
  waiting requests (``API_MAX_PENDING_REQUESTS``, 503 beyond it);
- a start-up snapshot for readiness and capabilities (dataset version, as-of date, data coverage,
  provider name);
- a ``RunTracker`` (``runs.py``) of queued and running runs, for metrics and graceful shutdown.

Bounded execution (Phase 9). Each run gets a cancel event and a deadline: the time left of the
request's budget when it starts. The agent stops at its next graph node once either fires. Database
queries are refused after the deadline, and a running query is interrupted at it. A model call waits
at most until the deadline. When the API answers 504, or a streaming client disconnects, or the
service shuts down, the run is cancelled rather than left to finish. Only the step in progress
completes, for example a model fit between queries. The worker is then free for the next request.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from typing import Literal

from app.agent.records import AgentStatus
from app.agent.runner import AgentRunner, AgentRunResult, ProgressCallback
from app.api.config import APIConfig
from app.api.errors import APIError
from app.api.observability import log_service
from app.api.runs import LiveRun, RunState, RunTracker
from app.database import Database, get_database

_RUN_STATES: dict[AgentStatus, RunState] = {
    "completed": "completed",
    "insufficient_evidence": "completed",
    "validation_failure": "completed",
    "unsupported_request": "refused",
    "tool_error": "failed",
    "planning_failure": "failed",
    "running": "failed",
}


@dataclass(frozen=True)
class ServiceRun:
    result: AgentRunResult
    stages: list[tuple[str, float]]  # (graph node, milliseconds from the run's start to the node's end)


@dataclass(frozen=True)
class ServiceSnapshot:
    """Facts fixed at start-up; the health endpoint never runs a business query."""

    agent_available: bool
    dataset_version: str | None = None
    as_of_date: date | None = None
    data_start: date | None = None
    data_end: date | None = None
    llm_provider: str | None = None
    disabled_tools: frozenset[str] = frozenset()
    max_tool_calls: int = 0


class AgentService:
    def __init__(self, runner: AgentRunner | None, db: Database | None, config: APIConfig, *, owns_db: bool = False):
        self.config = config
        self._runner = runner
        self._db = db
        self._owns_db = owns_db
        self._lock = threading.Lock()  # the one database connection: agent runs and the health probe
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agentops-agent")
        self._pending = 0  # changed only on the event loop
        self.runs = RunTracker()
        self.draining = False  # set when shutdown starts: new requests get 503, readiness reports not ready
        self.snapshot = self._take_snapshot()

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_settings(cls, config: APIConfig | None = None) -> AgentService:
        """Open the configured database read-only and build the runner. A failure leaves the service
        unavailable (readiness reports it; /ask answers 503) and is logged with a reason, never a path."""
        config = config or APIConfig.from_settings()
        try:
            db = get_database()
        except FileNotFoundError:
            log_service("service_unavailable", reason="database_missing", hint="python -m data.generator.generate")
            return cls(None, None, config)
        except Exception as exc:
            log_service("service_unavailable", reason="database_error", error_type=type(exc).__name__)
            return cls(None, None, config)
        try:
            runner = AgentRunner(db)
        except Exception as exc:
            log_service("service_unavailable", reason="agent_error", error_type=type(exc).__name__)
            db.close()
            return cls(None, None, config)
        return cls(runner, db, config, owns_db=True)

    def _take_snapshot(self) -> ServiceSnapshot:
        if self._runner is None or self._db is None:
            return ServiceSnapshot(agent_available=False)
        runtime = self._runner.runtime
        try:
            with self._lock:
                start, end = runtime.tool_context.kpi_service.coverage()
        except Exception as exc:
            log_service("service_unavailable", reason="database_error", error_type=type(exc).__name__)
            return ServiceSnapshot(agent_available=False, dataset_version=self._db.dataset_version)
        return ServiceSnapshot(
            agent_available=True,
            dataset_version=self._db.dataset_version,
            as_of_date=runtime.as_of,
            data_start=start,
            data_end=end,
            llm_provider=self._runner.llm.provider,
            disabled_tools=self._runner.config.disabled_tools,
            max_tool_calls=self._runner.config.max_tool_calls,
        )

    def close(self, grace_seconds: float | None = None) -> None:
        """Graceful shutdown: refuse new work, let live runs finish within the grace period, cancel the
        rest (they stop at their next step), then release the worker and the database connection."""
        self.draining = True
        grace = self.config.shutdown_grace_seconds if grace_seconds is None else grace_seconds
        if not self.runs.wait_idle(grace):
            stopped = self.runs.stop_all("cancelled")
            log_service("shutdown_cancelled_runs", runs=stopped)
            self.runs.wait_idle(min(max(grace, 1.0), 5.0))
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._owns_db and self._db is not None and self._lock.acquire(timeout=max(grace, 1.0)):
            try:
                self._db.close()
            finally:
                self._lock.release()

    # ------------------------------------------------------------------ requests
    @property
    def available(self) -> bool:
        return self._runner is not None and self.snapshot.agent_available

    def check_capacity(self) -> None:
        """Raise the API error a new request would get now (shutting down, unavailable, too many waiting)."""
        if self.draining:
            raise APIError("shutting_down")
        if not self.available:
            raise APIError("agent_unavailable")
        if self._pending >= self.config.max_pending_requests:
            raise APIError("busy")

    async def run(self, question: str, request_id: str, *, on_progress: ProgressCallback | None = None) -> ServiceRun:
        self.check_capacity()
        live = self.runs.queued(request_id)
        self._pending += 1
        started = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(
                self._executor, self._run_serialised, question, request_id, on_progress, live, started
            )
            result = await asyncio.wait_for(future, timeout=self.config.request_timeout_seconds)
            if result is None:  # cancelled before it started (shutdown)
                raise APIError("shutting_down")
            return result
        except TimeoutError:
            self._abandon(live, "timeout")
            raise APIError("timeout") from None
        except asyncio.CancelledError:  # the client went away (streaming) or the server is stopping
            self._abandon(live, "cancelled")
            raise
        finally:
            self._pending -= 1

    def _abandon(self, live: LiveRun, reason: Literal["timeout", "cancelled"]) -> None:
        """Stop a run nobody will read: a queued run is dropped; a running one stops at its next step."""
        self.runs.stop(live, reason)
        if live.state == "queued":
            self.runs.finished(live, reason)

    def _run_serialised(
        self,
        question: str,
        request_id: str,
        on_progress: ProgressCallback | None,
        live: LiveRun,
        request_started: float,
    ) -> ServiceRun | None:
        assert self._runner is not None
        if live.cancel.is_set():
            self.runs.finished(live, "cancelled")
            return None
        stages: list[tuple[str, float]] = []
        clock = time.perf_counter()

        def observe(node: str) -> None:  # stage timings for every run; progress events when streaming
            stages.append((node, round((time.perf_counter() - clock) * 1000, 2)))
            if on_progress is not None:
                on_progress(node)

        state: RunState = "failed"
        try:
            with self._lock:
                self.runs.running(live)
                clock = time.perf_counter()  # time the run itself, not the wait for the lock
                # The run may use what is left of the request's budget after waiting in the queue.
                remaining = self.config.request_timeout_seconds - (time.monotonic() - request_started)
                result = self._runner.run(
                    question,
                    run_id=request_id,
                    on_progress=observe,
                    cancel=live.cancel,
                    deadline_seconds=max(0.0, remaining),
                )
            state = _RUN_STATES.get(result.status, "failed")
            return ServiceRun(result=result, stages=stages)
        finally:
            self.runs.finished(live, state)

    def probe_database(self) -> bool:
        """A metadata query on the shared connection, skipped (and reported alive) while a run holds it."""
        if self._db is None:
            return False
        if not self._lock.acquire(blocking=False):
            return True  # an agent run is using the connection right now
        try:
            self._db.list_tables()
            return True
        except Exception:
            return False
        finally:
            self._lock.release()

    def readiness(self) -> dict[str, bool]:
        """What readiness checks (no business query; nothing about paths or settings)."""
        return {
            "configuration": True,  # the API does not start with an unsafe configuration
            "database": self.probe_database(),
            "agent": self.available,
            "accepting_requests": not self.draining,
        }
