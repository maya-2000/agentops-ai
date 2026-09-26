"""The agent service: the only path from the API to the agent.

Every question goes to ``AgentRunner.run`` (Phase 4 graph, Phase 5 secured tool execution); the
service adds nothing to the analysis. It owns:

- one ``AgentRunner`` over one read-only ``Database``, built at start-up;
- one worker thread: runs are serialised, because the DuckDB connection is never used concurrently
  (the same rule as the MCP server's lock). A lock also guards the health probe's metadata query;
- a wall-clock timeout per request (``API_REQUEST_TIMEOUT_SECONDS``, 504 after it) and a bound on
  waiting requests (``API_MAX_PENDING_REQUESTS``, 503 beyond it);
- a start-up snapshot for the health and capabilities endpoints (dataset version, as-of date,
  data coverage, provider name).

A run that exceeds the API timeout cannot be interrupted from outside (Python threads cannot be
killed); the agent's own limits (``AGENT_MAX_RUN_SECONDS``, per-tool and SQL timeouts, budgets) end
it, and requests queued behind it wait or time out. That is documented in docs/api.md.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date

from app.agent.runner import AgentRunner, AgentRunResult, ProgressCallback
from app.api.config import APIConfig
from app.api.errors import APIError
from app.database import Database, get_database


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
        self.snapshot = self._take_snapshot()

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_settings(cls, config: APIConfig | None = None) -> AgentService:
        """Open the configured database read-only and build the runner. A failure leaves the service
        unavailable (health reports it; /ask answers 503) instead of crashing the process."""
        config = config or APIConfig.from_settings()
        try:
            db = get_database()
        except Exception:
            return cls(None, None, config)
        try:
            runner = AgentRunner(db)
        except Exception:
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
        except Exception:
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

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._owns_db and self._db is not None:
            with self._lock:
                self._db.close()

    # ------------------------------------------------------------------ requests
    @property
    def available(self) -> bool:
        return self._runner is not None and self.snapshot.agent_available

    def check_capacity(self) -> None:
        """Raise the API error a new request would get now (unavailable or too many waiting)."""
        if not self.available:
            raise APIError("agent_unavailable")
        if self._pending >= self.config.max_pending_requests:
            raise APIError("busy")

    async def run(self, question: str, request_id: str, *, on_progress: ProgressCallback | None = None) -> ServiceRun:
        self.check_capacity()
        self._pending += 1
        try:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(self._executor, self._run_serialised, question, request_id, on_progress)
            # On timeout a queued run is cancelled before it starts; a started run finishes in the
            # background under the agent's own limits and its result is discarded.
            return await asyncio.wait_for(future, timeout=self.config.request_timeout_seconds)
        except TimeoutError:
            raise APIError("timeout") from None
        finally:
            self._pending -= 1

    def _run_serialised(self, question: str, request_id: str, on_progress: ProgressCallback | None) -> ServiceRun:
        assert self._runner is not None
        stages: list[tuple[str, float]] = []
        clock = time.perf_counter()

        def observe(node: str) -> None:  # stage timings for every run; progress events when streaming
            stages.append((node, round((time.perf_counter() - clock) * 1000, 2)))
            if on_progress is not None:
                on_progress(node)

        with self._lock:
            clock = time.perf_counter()  # time the run itself, not the wait for the lock
            result = self._runner.run(question, run_id=request_id, on_progress=observe)
        return ServiceRun(result=result, stages=stages)

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
