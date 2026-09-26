"""Phase 9 UI hardening: the bearer token, readiness, bounded history and the Streamlit launcher."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from app.api.main import create_app
from app.config import PROJECT_ROOT, Settings
from app.ui import __main__ as launcher
from app.ui import view_models as vm
from app.ui.client import AgentOpsClient, APIFailure
from tests.phase8_support import TOKEN, Borrowed, agent_service, production_api

PAGE = str(PROJECT_ROOT / "app" / "ui" / "main.py")


def test_the_client_sends_the_token_and_never_shows_it() -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"status": "ok", "version": "0.9.0"})

    client = AgentOpsClient("http://api.test", timeout=5, token=TOKEN, transport=httpx.MockTransport(handler))
    client.health()
    assert seen == [f"Bearer {TOKEN}"]
    assert TOKEN not in repr(client) and "authenticated=True" in repr(client)
    anonymous = AgentOpsClient("http://api.test", timeout=5, transport=httpx.MockTransport(handler))
    anonymous.health()
    assert seen[-1] is None


def test_an_unauthorised_ui_gets_a_clear_message() -> None:
    body = {"request_id": "r", "error": {"code": "unauthorized", "message": "Missing or invalid credentials."}}
    client = AgentOpsClient(
        "http://api.test", timeout=5, transport=httpx.MockTransport(lambda r: httpx.Response(401, json=body))
    )
    with pytest.raises(APIFailure) as caught:
        client.ask("q?")
    view = vm.error_view(caught.value)
    assert view.hint and "API_AUTH_TOKEN" in view.hint and TOKEN not in view.hint
    limited = vm.error_view(APIFailure("http", "Too many requests.", code="rate_limited", status_code=429))
    assert limited.hint and "Wait" in limited.hint


def test_history_is_bounded_and_keeps_the_redacted_question() -> None:
    secret = "sk-ant-api03-" + "S" * 40
    response = {"outcome": "answered", "answer": "A" * 5000, "request_id": "r1", "question": "Revenue? [REDACTED]"}
    entry = vm.history_entry(f"Revenue? {secret}", response=response)
    assert secret not in entry.question and entry.question == "Revenue? [REDACTED]"
    assert len(entry.answer) == vm.MAX_HISTORY_TEXT_CHARS
    history: list[vm.HistoryEntry] = []
    for i in range(50):
        history = vm.bounded_history(history, vm.history_entry(f"q{i}", response=response), limit=20)
    assert len(history) == 20 and history[0].question == "Revenue? [REDACTED]"
    assert vm.bounded_history(history, entry, limit=0) == []


def test_the_launcher_hardens_streamlit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        launcher, "get_settings", lambda: Settings(app_env="production", ui_public_address="bi.example.com")
    )
    args = launcher.streamlit_args()
    assert args[:3] == ["streamlit", "run", str(PROJECT_ROOT / "app" / "ui" / "main.py")]
    for flag in (
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--server.enableXsrfProtection=true",
        "--client.showErrorDetails=none",
        "--client.toolbarMode=viewer",
        "--browser.serverAddress=bi.example.com",
        "--server.maxUploadSize=1",
    ):
        assert flag in args, flag
    monkeypatch.setattr(launcher, "get_settings", lambda: Settings())
    development = launcher.streamlit_args()
    assert "--client.showErrorDetails=full" in development and "--server.address=127.0.0.1" in development


@pytest.fixture()
def secured_ui(small_db: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    service = agent_service(small_db, api=production_api())
    with TestClient(create_app(service)) as test_client:
        monkeypatch.setattr(AgentOpsClient, "_client", lambda self, timeout=None: Borrowed(test_client))
        st.cache_data.clear()
        yield test_client
    service.close()


def test_the_page_works_against_a_secured_api(secured_ui: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # The borrowed test client carries the token, as the real client does when API_AUTH_TOKEN is set.
    secured_ui.headers["Authorization"] = f"Bearer {TOKEN}"
    app = AppTest.from_file(PAGE, default_timeout=60)
    app.run()
    assert not app.exception
    assert [s.value for s in app.sidebar.success] == ["API ready"]
    assert any("dataset" in str(c.value) for c in app.sidebar.caption)
    app.text_area(key="question").input("What was revenue last month?")
    app.button[0].click()
    app.run()
    assert not app.exception and "Answer" in [h.value for h in app.subheader]
    state = [app.session_state[key] for key in ("history", "current", "question", "session_id")]
    assert TOKEN not in str(state)


def test_the_page_explains_a_missing_token(secured_ui: TestClient) -> None:
    secured_ui.headers.pop("Authorization", None)
    app = AppTest.from_file(PAGE, default_timeout=60)
    app.run()
    assert not app.exception
    assert any("not authorised" in str(e.value) for e in app.sidebar.error)
