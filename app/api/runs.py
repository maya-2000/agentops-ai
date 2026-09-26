"""A small, in-memory record of agent runs: what is queued or running now, and how runs ended.

It is not a job system and not conversation memory: it holds no questions, answers or data, only
a run's state and timings, for metrics, readiness and graceful shutdown. Memory is bounded. At most
``max_pending + 1`` runs are live at once, and finished runs are kept only as counters.

States: ``queued`` → ``running`` → one of ``completed``, ``refused``, ``failed``, ``timeout``
(the API answered 504 and cancelled the run), ``cancelled`` (the client went away or the service
is shutting down). A run that is cancelled while it is still executing stays *live* (counted as
``stopping``) until the agent actually returns, which bounds how long it can occupy the worker.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

RunState = Literal["queued", "running", "completed", "refused", "failed", "timeout", "cancelled"]
FINAL_STATES: tuple[RunState, ...] = ("completed", "refused", "failed", "timeout", "cancelled")


@dataclass
class LiveRun:
    run_id: str
    cancel: threading.Event = field(default_factory=threading.Event)
    state: RunState = "queued"
    queued_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    stop_requested: RunState | None = None  # "timeout" or "cancelled" once asked to stop


class RunTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._live: dict[str, LiveRun] = {}
        self._finished: Counter[str] = Counter()
        self._idle = threading.Condition(self._lock)

    def queued(self, run_id: str) -> LiveRun:
        with self._lock:
            run = LiveRun(run_id=run_id)
            # Request IDs are unique per request; a duplicate (a client reusing an ID) gets a separate
            # record keyed with a suffix, so one run can never cancel another.
            key = run_id if run_id not in self._live else f"{run_id}#{id(run)}"
            self._live[key] = run
            run.run_id = key
            return run

    def running(self, run: LiveRun) -> None:
        with self._lock:
            run.state = "running"
            run.started_at = time.monotonic()

    def stop(self, run: LiveRun, reason: Literal["timeout", "cancelled"]) -> None:
        """Ask a live run to stop at its next step (idempotent; the first reason wins)."""
        with self._lock:
            if run.stop_requested is None:
                run.stop_requested = reason
        run.cancel.set()

    def finished(self, run: LiveRun, state: RunState) -> None:
        with self._lock:
            if self._live.pop(run.run_id, None) is None:
                return
            final = run.stop_requested or state
            run.state = final
            self._finished[final] += 1
            if not self._live:
                self._idle.notify_all()

    def stop_all(self, reason: Literal["timeout", "cancelled"] = "cancelled") -> int:
        with self._lock:
            runs = list(self._live.values())
        for run in runs:
            self.stop(run, reason)
        return len(runs)

    def wait_idle(self, timeout: float) -> bool:
        """Wait until no run is live (True) or the timeout passes (False)."""
        deadline = time.monotonic() + timeout
        with self._lock:
            while self._live:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(remaining)
            return True

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            live = list(self._live.values())
            return {
                "queued": sum(1 for r in live if r.state == "queued"),
                "running": sum(1 for r in live if r.state == "running" and r.stop_requested is None),
                "stopping": sum(1 for r in live if r.stop_requested is not None),
                **{state: self._finished[state] for state in FINAL_STATES},
            }
