"""The UI's HTTP client: success, the API error envelope, connection and timeout failures, and the
progress stream. The API side is either a mock transport or the real app served in-process."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.ui.client import AgentOpsClient, APIFailure
from tests.phase8_support import Borrowed, agent_service


def _client(handler: Any) -> AgentOpsClient:
    return AgentOpsClient("http://api.test/", timeout=5, transport=httpx.MockTransport(handler))


def test_success_returns_the_json_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/ask" and json.loads(request.content)["question"] == "q?"
        return httpx.Response(200, json={"outcome": "answered", "answer": "a"})

    assert _client(handler).ask("q?", session_id="s")["answer"] == "a"


def test_the_api_error_envelope_becomes_an_api_failure() -> None:
    body = {"request_id": "req-9", "error": {"code": "busy", "message": "The agent is busy.", "retryable": True}}
    with pytest.raises(APIFailure) as caught:
        _client(lambda r: httpx.Response(503, json=body)).ask("q?")
    failure = caught.value
    assert (failure.kind, failure.code, failure.status_code, failure.request_id) == ("http", "busy", 503, "req-9")
    assert failure.message == "The agent is busy." and failure.retryable


def test_an_unexpected_error_body_is_not_shown() -> None:
    html = "<html>Traceback (most recent call last): /srv/app.py</html>"
    with pytest.raises(APIFailure) as caught:
        _client(lambda r: httpx.Response(502, text=html, headers={"x-request-id": "r1"})).ask("q?")
    assert caught.value.message == "The API answered with HTTP 502." and caught.value.request_id == "r1"
    with pytest.raises(APIFailure) as caught:
        _client(lambda r: httpx.Response(200, text="not json")).ask("q?")
    assert caught.value.kind == "protocol"


def test_connection_and_timeout_failures() -> None:
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(APIFailure) as caught:
        _client(refused).health()
    assert caught.value.kind == "connection" and "http://api.test" in caught.value.message
    with pytest.raises(APIFailure) as caught:
        _client(slow).ask("q?")
    assert caught.value.kind == "timeout" and caught.value.retryable


def test_an_unavailable_health_body_is_returned_not_raised() -> None:
    body = {"status": "unavailable", "version": "0.8.0", "agent_available": False, "database_available": False}
    assert _client(lambda r: httpx.Response(503, json=body)).health()["status"] == "unavailable"


def test_stream_reports_progress_then_returns_the_result() -> None:
    lines = [
        {"type": "progress", "request_id": "r", "stage": "question_received", "label": "Screening", "elapsed_ms": 1},
        {"type": "progress", "request_id": "r", "stage": "done", "label": "Answer ready", "elapsed_ms": 2},
        {"type": "result", "request_id": "r", "data": {"answer": "a"}},
    ]
    body = "".join(json.dumps(line) + "\n" for line in lines)
    seen: list[str] = []
    client = _client(lambda r: httpx.Response(200, text=body))
    result = client.ask_stream("q?", on_progress=lambda e: seen.append(e["label"]))
    assert result == {"answer": "a"} and seen == ["Screening", "Answer ready"]


def test_stream_error_events_and_truncated_streams_fail() -> None:
    error = {"type": "error", "request_id": "r", "status_code": 504, "error": {"code": "timeout", "message": "Late."}}
    with pytest.raises(APIFailure) as caught:
        _client(lambda r: httpx.Response(200, text=json.dumps(error) + "\n")).ask_stream("q?", on_progress=print)
    assert (caught.value.code, caught.value.status_code, caught.value.request_id) == ("timeout", 504, "r")
    with pytest.raises(APIFailure) as caught:
        _client(lambda r: httpx.Response(200, text="")).ask_stream("q?", on_progress=print)
    assert caught.value.kind == "protocol"
    rejected = {"request_id": "r2", "error": {"code": "empty_question", "message": "Empty."}}
    with pytest.raises(APIFailure) as caught:
        _client(lambda r: httpx.Response(422, json=rejected)).ask_stream(" ", on_progress=print)
    assert caught.value.code == "empty_question" and caught.value.status_code == 422


def test_client_against_the_real_api(small_db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    service = agent_service(small_db)
    try:
        with TestClient(create_app(service)) as test_client:
            client = AgentOpsClient("http://testserver", timeout=30)
            monkeypatch.setattr(client, "_client", lambda timeout=None: Borrowed(test_client))
            assert client.health()["status"] == "ok"
            assert client.capabilities()["example_questions"]
            answer = client.ask("What was revenue last month?", session_id="ui-test")
            assert answer["outcome"] == "answered" and answer["session_id"] == "ui-test"
            stages: list[str] = []
            question = "What was revenue last month?"
            streamed = client.ask_stream(question, on_progress=lambda e: stages.append(e["stage"]))
            assert streamed["answer"] == answer["answer"] and stages == streamed["run"]["pipeline"]
            with pytest.raises(APIFailure) as caught:
                client.ask("   ")
            assert caught.value.code == "empty_question" and caught.value.request_id
    finally:
        service.close()
