"""Execution deadlines for database queries (timeouts without threads in the caller).

``execution_deadline(seconds)`` sets a deadline in a context variable for the duration of a
block. Nested blocks keep the earliest deadline. The database backend reads the remaining time
before each query. It refuses to start a query when the deadline has passed, and otherwise
interrupts the query when the time runs out. A tool call, and ad-hoc SQL inside it, can
therefore be bounded without knowing which queries the tool runs.

Pure-Python computation between queries (for example model fitting) cannot be interrupted this
way. The deadline is checked again at the next query.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_DEADLINE: ContextVar[float | None] = ContextVar("agentops_query_deadline", default=None)


class QueryTimeoutError(TimeoutError):
    """A query was not started, or was interrupted, because its execution deadline passed."""


@contextmanager
def execution_deadline(seconds: float | None) -> Iterator[None]:
    """Bound every query started inside the block to ``seconds`` from now (``None``: no new bound)."""
    if seconds is None:
        yield
        return
    current = _DEADLINE.get()
    candidate = time.monotonic() + max(0.0, seconds)
    token = _DEADLINE.set(candidate if current is None else min(current, candidate))
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def remaining_seconds() -> float | None:
    """Seconds left before the active deadline (``None`` when no deadline is set)."""
    deadline = _DEADLINE.get()
    return None if deadline is None else deadline - time.monotonic()
