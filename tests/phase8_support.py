"""Helpers for the Phase 8 API and UI tests: an API app around a scripted or deterministic agent.

No live model, network or API key: the agent uses the deterministic model (optionally with scripted
steps, as in the Phase 5 tests), and requests go through FastAPI's in-process test client.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi.testclient import TestClient

from app.api.config import APIConfig
from app.api.main import create_app
from app.api.service import AgentService
from app.tools import ToolRegistry
from tests.phase5_support import runner

ASK = "/api/v1/ask"
STREAM = "/api/v1/ask/stream"
HEALTH = "/api/v1/health"
CAPABILITIES = "/api/v1/capabilities"
METRICS = "/api/v1/metrics"


def api_config(**overrides: Any) -> APIConfig:
    values: dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 8000,
        "request_timeout_seconds": 60.0,
        "max_request_bytes": 16384,
        "max_pending_requests": 4,
        "max_question_chars": 1000,
    }
    values.update(overrides)
    return APIConfig(**values)


def agent_service(
    db: Any,
    script: dict[Any, list[Any]] | None = None,
    registry: ToolRegistry | None = None,
    *,
    api: dict[str, Any] | None = None,
    **agent_config: Any,
) -> AgentService:
    agent, _ = runner(db, script, registry, **agent_config)
    return AgentService(agent, db, api_config(**(api or {})))


@contextmanager
def api_client(
    db: Any,
    script: dict[Any, list[Any]] | None = None,
    registry: ToolRegistry | None = None,
    *,
    api: dict[str, Any] | None = None,
    **agent_config: Any,
) -> Iterator[TestClient]:
    """A test client for an app whose service runs the (scripted) agent on ``db``. The db stays open."""
    service = agent_service(db, script, registry, api=api, **agent_config)
    try:
        with TestClient(create_app(service), raise_server_exceptions=False) as client:
            yield client
    finally:
        service.close()


class Borrowed:
    """Lends an open ``TestClient`` to the UI's ``AgentOpsClient`` (which opens and closes a client per call)."""

    def __init__(self, client: TestClient):
        self.client = client

    def __enter__(self) -> TestClient:
        return self.client

    def __exit__(self, *_: Any) -> None:
        return None


def ask(client: TestClient, question: str, **body: Any) -> dict[str, Any]:
    response = client.post(ASK, json={"question": question, **body})
    assert response.status_code == 200, response.text
    data: dict[str, Any] = response.json()
    return data
