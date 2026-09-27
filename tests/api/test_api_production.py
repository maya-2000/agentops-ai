"""Phase 9 access boundary: authentication, rate limiting, CORS, security headers, request IDs,
content types, error envelopes, readiness and the start-up configuration checks.

The API here is configured like production: token authentication, a rate limit, docs off.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import API_VERSION
from app.api.config import APIConfig, ConfigurationError, RateLimit
from app.api.main import create_app
from app.api.security import SlidingWindowRateLimiter, bearer_token
from app.api.service import AgentService
from app.config import Settings
from tests.phase8_support import (
    ASK,
    AUTH,
    CAPABILITIES,
    HEALTH,
    METRICS,
    READINESS,
    STREAM,
    TOKEN,
    agent_service,
    api_client,
    api_config,
    ask_secured,
    production_api,
)

REVENUE = "What was revenue last month?"


@pytest.fixture(scope="module")
def secured(small_db: Any) -> Any:
    with api_client(small_db, api=production_api()) as client:
        yield client


def _error(response: Any, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status, response.text
    body: dict[str, Any] = response.json()
    assert body["error"]["code"] == code and body["request_id"] == response.headers["x-request-id"]
    assert body["error"]["request_id"] == body["request_id"]
    return body


# ---------------------------------------------------------------------------------------- authentication


def test_a_valid_token_is_accepted(secured: Any) -> None:
    data = ask_secured(secured, REVENUE)
    assert data["outcome"] == "answered"
    lower_scheme = {"Authorization": f"bearer {TOKEN}"}  # the scheme is case-insensitive (RFC 7235)
    assert secured.post(ASK, json={"question": REVENUE}, headers=lower_scheme).status_code == 200


def test_bearer_header_parsing() -> None:
    assert bearer_token(f"Bearer {TOKEN}") == TOKEN and bearer_token(f"BEARER {TOKEN}") == TOKEN
    for header in (None, "", "Bearer", "Bearer ", f"Bearer  {TOKEN}", f"Bearer {TOKEN} x", f"Token {TOKEN}"):
        assert bearer_token(header) is None, header


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": ""},
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer "},
        {"Authorization": f"Basic {TOKEN}"},
        {"Authorization": f"Bearer  {TOKEN}"},
        {"Authorization": f"Bearer {TOKEN} extra"},
        {"Authorization": f"Bearer {TOKEN}x"},
        {"Authorization": f"Bearer {TOKEN[:-1]}"},
        {"Authorization": "Bearer wrong-token-value-kkkkkkkkkkkkkkkkkkkkkkkkkkkk"},
        {"X-API-Key": TOKEN},
    ],
    ids=[
        "missing",
        "empty",
        "scheme-only",
        "scheme-space",
        "basic",
        "double-space",
        "extra-part",
        "longer",
        "shorter",
        "wrong",
        "other-header",
    ],
)
def test_missing_invalid_or_malformed_credentials_get_the_same_401(secured: Any, headers: dict[str, str]) -> None:
    response = secured.post(ASK, json={"question": REVENUE}, headers=headers)
    body = _error(response, 401, "unauthorized")
    assert body["error"]["message"] == "Missing or invalid credentials."
    assert response.headers["www-authenticate"] == "Bearer"
    assert TOKEN not in response.text


@pytest.mark.parametrize("path", [HEALTH, READINESS])
def test_liveness_and_readiness_are_public(secured: Any, path: str) -> None:
    assert secured.get(path).status_code == 200


@pytest.mark.parametrize(
    ("method", "path"),
    [("GET", CAPABILITIES), ("GET", METRICS), ("POST", STREAM), ("GET", "/api/v1/nope"), ("GET", "/api/v1/runs/x")],
)
def test_everything_else_under_the_api_needs_a_token(secured: Any, method: str, path: str) -> None:
    _error(secured.request(method, path, json={"question": REVENUE}), 401, "unauthorized")


def test_unknown_routes_answer_401_before_404(secured: Any) -> None:
    assert secured.get("/api/v1/does-not-exist").status_code == 401  # no route enumeration without a token
    assert secured.get("/api/v1/does-not-exist", headers=AUTH).status_code == 404


def test_authentication_happens_before_the_body_is_read(secured: Any) -> None:
    headers = {"content-type": "application/json"}
    _error(secured.post(ASK, content=b'{"question": ', headers=headers), 401, "unauthorized")
    _error(secured.post(ASK, content=b"x" * 50_000, headers=headers), 401, "unauthorized")


def test_the_token_never_reaches_responses_or_logs(secured: Any, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        texts = [secured.post(ASK, json={"question": REVENUE}, headers=AUTH).text]
        texts.append(secured.post(ASK, json={"question": f"What was revenue? {TOKEN}"}, headers=AUTH).text)
        texts.append(secured.post(ASK, json={"question": REVENUE}, headers={"Authorization": "Bearer nope"}).text)
        texts.append(secured.get(METRICS, headers=AUTH).text)
    blob = "\n".join(texts + [r.getMessage() for r in caplog.records])
    assert TOKEN not in blob


def test_the_token_is_not_in_reprs() -> None:
    config = APIConfig.from_settings(Settings(api_auth_token=TOKEN))
    assert TOKEN not in repr(config) and TOKEN not in repr(Settings(api_auth_token=TOKEN))


def test_disabled_authentication_needs_no_token(small_db: Any) -> None:
    with api_client(small_db, api={"auth_mode": "disabled", "environment": "development"}) as client:
        assert client.post(ASK, json={"question": REVENUE}).status_code == 200


def test_startup_refuses_token_mode_without_a_token(small_db: Any) -> None:
    service = agent_service(small_db, api={"auth_mode": "token", "auth_token": None})
    with pytest.raises(ConfigurationError, match="API_AUTH_TOKEN is not set"), TestClient(create_app(service)):
        pass
    service.close()


# ---------------------------------------------------------------------------------------- rate limiting


def test_requests_within_the_limit_pass_and_excess_gets_429(small_db: Any) -> None:
    with api_client(small_db, api=production_api(rate_limit=RateLimit(requests=3, window_seconds=60.0))) as client:
        statuses = [client.post(ASK, json={"question": "hi weather?"}, headers=AUTH).status_code for _ in range(3)]
        assert statuses == [200, 200, 200]
        response = client.post(ASK, json={"question": REVENUE}, headers=AUTH)
        body = _error(response, 429, "rate_limited")
        assert body["error"]["retryable"] and 1 <= int(response.headers["retry-after"]) <= 60
        _error(client.post(STREAM, json={"question": REVENUE}, headers=AUTH), 429, "rate_limited")
        for _ in range(5):  # liveness, readiness and other endpoints are not limited
            assert client.get(HEALTH).status_code == 200 and client.get(READINESS).status_code == 200
        assert client.get(CAPABILITIES, headers=AUTH).status_code == 200
        summary = client.get(METRICS, headers=AUTH).json()["summary"]
        assert summary["rate_limited"] == 2 and summary["answered"] + summary["unsupported"] == 3


def test_the_limit_recovers_after_the_window(small_db: Any) -> None:
    with api_client(small_db, api=production_api(rate_limit=RateLimit(requests=2, window_seconds=1.0))) as client:
        for _ in range(2):
            assert client.post(ASK, json={"question": REVENUE}, headers=AUTH).status_code == 200
        assert client.post(ASK, json={"question": REVENUE}, headers=AUTH).status_code == 429
        time.sleep(1.05)
        assert client.post(ASK, json={"question": REVENUE}, headers=AUTH).status_code == 200


def test_unauthenticated_requests_do_not_use_the_quota(small_db: Any) -> None:
    with api_client(small_db, api=production_api(rate_limit=RateLimit(requests=2, window_seconds=60.0))) as client:
        for _ in range(5):
            assert client.post(ASK, json={"question": REVENUE}).status_code == 401
        assert client.post(ASK, json={"question": REVENUE}, headers=AUTH).status_code == 200


def test_the_limiter_is_a_sliding_window_per_client_with_bounded_memory() -> None:
    now = [0.0]
    limiter = SlidingWindowRateLimiter(RateLimit(requests=2, window_seconds=10.0, max_clients=10), clock=lambda: now[0])
    assert [limiter.check("a").allowed for _ in range(3)] == [True, True, False]
    assert limiter.check("b").allowed  # clients are limited separately
    now[0] = 9.9
    denied = limiter.check("a")
    assert not denied.allowed and denied.retry_after == 1
    now[0] = 10.01
    assert limiter.check("a").allowed  # the first request left the window
    for i in range(50):
        limiter.check(f"client-{i}")
    assert limiter.tracked_clients == 10


# ---------------------------------------------------------------------------------------- CORS and headers


def test_cors_allows_only_configured_origins(small_db: Any) -> None:
    allowed = "https://agentops.example.com"
    with api_client(small_db, api=production_api(cors_origins=(allowed,))) as client:
        ok = client.get(HEALTH, headers={"Origin": allowed})
        assert ok.headers["access-control-allow-origin"] == allowed
        preflight = client.options(
            ASK,
            headers={
                "Origin": allowed,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
        assert preflight.status_code == 200 and preflight.headers["access-control-allow-origin"] == allowed
        assert "POST" in preflight.headers["access-control-allow-methods"]
        other = client.get(HEALTH, headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in other.headers
        rejected = client.options(
            ASK, headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}
        )
        assert rejected.status_code == 400 and "access-control-allow-origin" not in rejected.headers
        assert "access-control-allow-credentials" not in ok.headers


def test_without_configured_origins_there_are_no_cors_headers(secured: Any) -> None:
    response = secured.get(HEALTH, headers={"Origin": "https://agentops.example.com"})
    assert not [h for h in response.headers if h.startswith("access-control-")]


@pytest.mark.parametrize(
    ("method", "path", "headers"),
    [("GET", HEALTH, {}), ("POST", ASK, AUTH), ("POST", ASK, {}), ("GET", "/nope", {}), ("GET", "/", {})],
)
def test_security_headers_on_every_response(secured: Any, method: str, path: str, headers: dict[str, str]) -> None:
    response = secured.request(method, path, json={"question": REVENUE}, headers=headers)
    assert response.headers["content-security-policy"] == "default-src 'none'; frame-ancestors 'none'"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


def test_docs_are_off_in_production_and_on_in_development(small_db: Any, secured: Any) -> None:
    assert secured.get("/docs").status_code == 404 and secured.get("/openapi.json").status_code == 404
    with api_client(small_db, api={"auth_mode": "disabled", "docs_enabled": True}) as client:
        docs = client.get("/docs")
        assert docs.status_code == 200 and "content-security-policy" not in docs.headers  # Swagger UI loads scripts
        assert client.get("/openapi.json").status_code == 200


# ---------------------------------------------------------------------------------------- request hardening


@pytest.mark.parametrize(
    "supplied", ["x" * 65, "x" * 5000, "bad id", "<script>", "../../etc", "ok\r\nSet-Cookie: a=b", ""]
)
def test_unsafe_request_ids_are_replaced(secured: Any, supplied: str) -> None:
    response = secured.get(HEALTH, headers={"X-Request-ID": supplied} if "\r" not in supplied else {})
    request_id = response.headers["x-request-id"]
    assert request_id != supplied and request_id.startswith("R-") and len(request_id) <= 64


def test_a_valid_request_id_is_kept_and_propagated(secured: Any) -> None:
    response = secured.post(ASK, json={"question": REVENUE}, headers={**AUTH, "X-Request-ID": "deploy-check-7"})
    data = response.json()
    assert response.headers["x-request-id"] == data["request_id"] == data["run"]["run_id"] == "deploy-check-7"


@pytest.mark.parametrize(
    "content_type", ["text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x", ""]
)
def test_the_ask_endpoints_accept_json_only(secured: Any, content_type: str) -> None:
    headers = {**AUTH, "Content-Type": content_type} if content_type else AUTH
    for path in (ASK, STREAM):
        _error(secured.post(path, content=b'{"question": "hi"}', headers=headers), 415, "unsupported_media_type")
    ok = secured.post(
        ASK, content=b'{"question": "hi"}', headers={**AUTH, "Content-Type": "application/json; charset=utf-8"}
    )
    assert ok.status_code == 200


def test_error_envelopes_hold_no_internals(secured: Any) -> None:
    responses = [
        secured.post(ASK, json={"question": 5}, headers=AUTH),
        secured.post(ASK, json={"question": REVENUE, "extra": 1}, headers=AUTH),
        secured.post(ASK, content=b"{", headers={**AUTH, "content-type": "application/json"}),
        secured.post(ASK, json={"question": REVENUE}),
        secured.get("/api/v1/nope", headers=AUTH),
    ]
    for response in responses:
        body = response.json()
        assert set(body) == {"request_id", "error"} and body["error"]["request_id"] == body["request_id"]
        text = response.text
        for internal in ("Traceback", 'File "', "/home/", "site-packages", "Exception", "Error(", "SELECT "):
            assert internal not in text, (internal, text)


# ---------------------------------------------------------------------------------------- readiness and draining


def test_readiness_turns_503_while_draining(small_db: Any) -> None:
    with api_client(small_db, api=production_api()) as client:
        service: AgentService = client.app.state.service  # type: ignore[attr-defined]
        service.draining = True
        try:
            response = client.get(READINESS)
            assert response.status_code == 503 and response.json()["checks"]["accepting_requests"] is False
            _error(client.post(ASK, json={"question": REVENUE}, headers=AUTH), 503, "shutting_down")
            assert client.get(HEALTH).status_code == 200  # still alive
        finally:
            service.draining = False


def test_capabilities_carry_the_dataset_details_health_no_longer_shows(secured: Any) -> None:
    body = secured.get(CAPABILITIES, headers=AUTH).json()
    assert body["dataset_version"] and body["as_of_date"] == "2026-08-31" and body["llm_provider"]
    assert body["version"] == API_VERSION


def test_metrics_can_be_switched_off(small_db: Any) -> None:
    with api_client(small_db, api=production_api(metrics_enabled=False)) as client:
        assert client.get(METRICS, headers=AUTH).status_code == 404


# ---------------------------------------------------------------------------------------- configuration


def _problems(**settings: Any) -> list[str]:
    return APIConfig.from_settings(Settings(**settings)).startup_problems()


def test_development_defaults_need_a_token_or_an_explicit_opt_out() -> None:
    assert any("API_AUTH_TOKEN is not set" in p for p in _problems())
    assert _problems(api_auth_token=TOKEN) == []
    assert _problems(api_auth_mode="disabled") == []


def test_production_requires_explicit_secure_configuration() -> None:
    problems = _problems(app_env="production", api_auth_mode="disabled", api_rate_limit="off", api_cors_origins="*")
    text = "\n".join(problems)
    for expected in ("DATABASE_URL", "API_AUTH_MODE=disabled", "API_RATE_LIMIT=off", "'*'"):
        assert expected in text, expected
    insecure_origin = _problems(
        app_env="production", api_auth_token=TOKEN, database_url="duckdb:///x.duckdb", api_cors_origins="http://a.com"
    )
    assert insecure_origin == ["API_CORS_ORIGINS must use https:// origins when APP_ENV=production."]
    good = _problems(
        app_env="production",
        api_auth_token=TOKEN,
        database_url="duckdb:////data/db.duckdb",
        api_cors_origins="https://agentops.example.com",
    )
    assert good == []


def test_production_requires_nested_timeouts() -> None:
    problems = _problems(
        app_env="production",
        api_auth_token=TOKEN,
        database_url="duckdb:///x.duckdb",
        agent_max_run_seconds=300,
        api_request_timeout_seconds=150,
    )
    assert problems == [
        "AGENT_MAX_RUN_SECONDS must not exceed API_REQUEST_TIMEOUT_SECONDS (timeouts nest from the inside out)."
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("api_auth_token", "too-short"),
        ("api_auth_token", "has spaces " + "x" * 40),
        ("api_rate_limit", "fast"),
        ("api_rate_limit", "10/day"),
        ("api_cors_origins", "https://a.example.com/path"),
        ("api_cors_origins", "javascript:alert(1)"),
    ],
)
def test_invalid_settings_are_rejected(field: str, value: str) -> None:
    with pytest.raises(ValueError) as caught:
        Settings(**{field: value})
    if field == "api_auth_token":  # a secret is never echoed (a bad origin is named, to help the operator)
        assert value not in str(caught.value)


def test_settings_parse_rate_limits_and_origins() -> None:
    assert Settings(api_rate_limit="5/second").rate_limit == (5, 1.0)
    assert Settings(api_rate_limit="100 / hour").rate_limit == (100, 3600.0)
    assert Settings(api_rate_limit="off").rate_limit is None
    assert Settings(api_cors_origins=" https://a.com , http://localhost:8501 ").cors_origins == [
        "https://a.com",
        "http://localhost:8501",
    ]
    assert Settings(app_env="production").docs_enabled is False and Settings().docs_enabled is True


def test_the_env_example_is_safe_and_complete() -> None:
    from app.config import PROJECT_ROOT

    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    values = dict(line.split("=", 1) for line in text.splitlines() if line and not line.startswith("#") and "=" in line)
    assert values["API_AUTH_TOKEN"].strip() == "" and values["ANTHROPIC_API_KEY"].strip() == ""
    for name in ("APP_ENV", "API_AUTH_MODE", "API_RATE_LIMIT", "API_CORS_ORIGINS", "LOG_FORMAT", "UI_HISTORY_LIMIT"):
        assert name in values, name
    settings = Settings(_env_file=PROJECT_ROOT / ".env.example")  # type: ignore[call-arg]
    assert settings.api_auth_mode == "token" and settings.rate_limit == (20, 60.0)


def test_the_api_config_check_lists_every_problem() -> None:
    config = APIConfig.from_settings(Settings(app_env="production", api_auth_mode="disabled"))
    with pytest.raises(ConfigurationError) as caught:
        config.check()
    assert str(caught.value).count("\n- ") >= 2
    assert api_config().startup_problems() == []  # the Phase 8 test configuration (auth explicitly off)


def test_check_config_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from app.api import __main__ as entry
    from app.config import get_settings

    monkeypatch.setattr(entry, "configure_logging", lambda *_: None)  # keep pytest's log handlers
    monkeypatch.setattr(entry, "settings_or_exit", lambda: Settings(api_auth_token=TOKEN))
    assert entry.main(["--check-config"]) == 0
    monkeypatch.setattr(entry, "settings_or_exit", lambda: Settings(api_auth_mode="token"))
    assert entry.main(["--check-config"]) == 2
    assert "API_AUTH_TOKEN is not set" in capsys.readouterr().err
    get_settings.cache_clear()


def test_invalid_settings_are_reported_without_values_or_tracebacks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from app.api import __main__ as entry
    from app.config import get_settings

    short = "short-secret-value"
    monkeypatch.setenv("API_AUTH_TOKEN", short)
    monkeypatch.setenv("API_PORT", "not-a-port")
    get_settings.cache_clear()
    try:
        with pytest.raises(SystemExit) as stopped:
            entry.main(["--check-config"])
    finally:
        get_settings.cache_clear()
    assert stopped.value.code == 2
    err = capsys.readouterr().err
    assert "API_AUTH_TOKEN: Value error, API_AUTH_TOKEN must be at least 32 characters" in err
    assert "API_PORT:" in err and short not in err and "not-a-port" not in err and "Traceback" not in err


def test_json_is_the_only_accepted_error_body(secured: Any) -> None:
    response = secured.post(ASK, json={"question": ""}, headers=AUTH)
    assert response.headers["content-type"].startswith("application/json")
    json.loads(response.text)
