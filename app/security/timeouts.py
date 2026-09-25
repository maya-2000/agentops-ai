"""Timeouts for the three kinds of external work in a run.

- **Model calls** run in a daemon worker thread, and the caller waits at most
  ``llm_timeout_seconds``. On timeout the call is recorded as failed with ``llm_timeout`` and the
  result, if it ever arrives, is discarded. Model clients do not touch the database connection,
  so running them on a worker thread is safe. The Anthropic client also has its own HTTP timeout.
- **Tool calls** run under ``execution_deadline(tool_timeout_seconds)``. Every query the tool
  starts is bounded by the remaining time and interrupted by the database backend when it runs
  out.
- **Ad-hoc SQL** additionally runs under ``execution_deadline(sql_timeout_seconds)``, inside the
  tool.

Tools run on the calling thread because the DuckDB connection must not be used from two threads
at once. A tool's pure-Python work between queries is therefore bounded only by the next query
and by the run's wall clock, which is checked before every tool call. The tool's elapsed time
is also compared with its limit afterwards, and an over-time result is recorded as a timeout.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class CallTimeoutError(TimeoutError):
    """The call did not finish within its time limit."""


def call_with_timeout(fn: Callable[[], T], seconds: float) -> T:
    """Run ``fn`` on a daemon thread and wait at most ``seconds`` for its result."""
    values: list[T] = []
    errors: list[BaseException] = []

    def target() -> None:
        try:
            values.append(fn())
        except BaseException as exc:  # re-raised on the calling thread below
            errors.append(exc)

    worker = threading.Thread(target=target, name="agentops-timeout", daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise CallTimeoutError(f"The call did not finish within {seconds:g} seconds")
    if errors:
        raise errors[0]
    return values[0]
