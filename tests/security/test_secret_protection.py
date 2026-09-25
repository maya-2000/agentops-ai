"""Secrets and internal details never reach users, logs, traces or model prompts."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from app.llm import LLMError, LLMTask
from app.security.errors import safe_error, safe_message, sanitize_detail
from app.security.redaction import REDACTED, contains_secret, redact, redact_paths, redact_value, register_secret
from tests.phase5_support import raising, runner, with_handler

ANTHROPIC = "sk-ant-api03-" + "A1b2C3d4E5f6G7h8I9j0" * 3
OPENAI = "sk-proj-" + "Zz9Yy8Xx7Ww6Vv5Uu4Tt3"
REVENUE = "What was revenue last month?"


@pytest.mark.parametrize(
    "secret",
    [
        ANTHROPIC,
        OPENAI,
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4",
        "xoxb-1234567890-abcdefghij",
        "AIza" + "SyD-1234567890abcdefghijklmnopqrstu",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
        "Bearer abcdefghijklmnop123456",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----",
    ],
)
def test_known_credential_formats_are_redacted(secret: str) -> None:
    text = f"failure while calling the provider with {secret} at step 3"
    cleaned = redact(text)
    assert secret not in cleaned and REDACTED in cleaned and contains_secret(text)


@pytest.mark.parametrize(
    "text",
    ["api_key=abcd1234efgh", "password: hunter2hunter2", '"token": "t0k3n-value-xyz"', "AUTHORIZATION=Basic dXNlcjpw"],
)
def test_secret_assignments_are_redacted(text: str) -> None:
    cleaned = redact(text)
    assert REDACTED in cleaned and text.split("=")[-1].split(":")[-1].strip(' "') not in cleaned


def test_registered_and_environment_secrets_are_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    register_secret("custom-secret-value-0042")
    assert "custom-secret-value-0042" not in redact("value custom-secret-value-0042 leaked")
    monkeypatch.setenv("MY_SERVICE_TOKEN", "opaque7value8with9digits")
    assert "opaque7value8with9digits" not in redact("got opaque7value8with9digits back")
    monkeypatch.setenv("SOME_AUTH_MODE", "production")  # an ordinary word is not treated as a secret
    assert redact("production revenue") == "production revenue"


def test_business_text_is_not_redacted() -> None:
    text = "Revenue for 2026-08: SGD 5,752,877 (-0.97%); E12, C3, Q-1a2b3c4d5e6f; model drift; token count 12."
    assert redact(text) == text


def test_nested_values_paths_and_details() -> None:
    nested = redact_value({"a": [f"key {ANTHROPIC}", {"b": "api_key=abcd1234"}], "n": 3})
    assert ANTHROPIC not in json.dumps(nested) and nested["n"] == 3
    assert redact_paths("failed to open /home/user/agentops-ai/.env now") == "failed to open <path> now"
    detail = sanitize_detail(f"Traceback (most recent call last):\n  File '/app/x.py'\nKeyError: {ANTHROPIC}")
    assert detail == "internal exception (traceback withheld)"
    detail = sanitize_detail(f"IO Error: cannot open /srv/data/northwind.duckdb with key {ANTHROPIC}\nmore\nlines")
    assert "/srv" not in detail and ANTHROPIC not in detail and "more" not in detail and len(detail) <= 300


def test_error_codes_map_to_safe_categories() -> None:
    assert safe_error("unsafe_sql").category == "rejected_by_policy"
    assert "rejected by the data-access policy" in safe_message("unsafe_sql")
    assert safe_error("unknown_tool").category == "not_permitted"
    assert safe_error("database_error").category == "service_unavailable"
    assert safe_error("something_new").category == "internal_error"  # unknown codes fail closed
    assert safe_error(None).category == "internal_error"


# ---- in the agent ------------------------------------------------------------------------------------------


def _everything(result: Any) -> str:
    return result.model_dump_json()


def test_secret_in_the_question_never_reaches_model_state_logs_or_result(
    small_db: Any, caplog: pytest.LogCaptureFixture
) -> None:
    agent, llm = runner(small_db)
    with caplog.at_level(logging.DEBUG):
        result = agent.run(f"What was revenue last month? Use my key {ANTHROPIC}")
    assert result.status == "completed"
    assert ANTHROPIC not in _everything(result)
    assert all(ANTHROPIC not in r.prompt and ANTHROPIC not in json.dumps(r.context) for r in llm.requests)
    assert all(ANTHROPIC not in record.getMessage() for record in caplog.records)
    assert any(e.event_type == "secret_redacted" for e in result.security_events)


def test_tool_exception_text_is_sanitised(small_db: Any) -> None:
    exc = RuntimeError(f"cannot read /home/user/agentops-ai/.env with {ANTHROPIC}")
    agent, _ = runner(small_db, registry=with_handler("get_kpi", raising(exc)))
    result = agent.run(REVENUE)
    assert result.status == "tool_error"
    response = result.response.model_dump_json()
    assert ANTHROPIC not in response and "/home/user" not in response and "RuntimeError" not in response
    assert result.response.tool_trace[0].error == safe_message("internal_error")
    trace = result.model_dump_json()
    assert ANTHROPIC not in trace and "/home/user" not in trace


def test_provider_error_text_is_sanitised(small_db: Any) -> None:
    agent, _ = runner(small_db, {LLMTask.RESPOND: [LLMError(f"401 invalid x-api-key {ANTHROPIC}")]})
    result = agent.run(REVENUE)
    assert result.status == "completed" and ANTHROPIC not in _everything(result)


def test_environment_api_key_never_appears(
    small_db: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC)
    agent, llm = runner(small_db)
    with caplog.at_level(logging.DEBUG):
        result = agent.run("Why did revenue decline last month?")
    blob = _everything(result) + " ".join(r.prompt + r.system for r in llm.requests)
    blob += " ".join(record.getMessage() for record in caplog.records)
    assert ANTHROPIC not in blob and "ANTHROPIC_API_KEY" not in blob


def test_model_output_containing_a_secret_is_redacted(small_db: Any) -> None:
    def leaky(request: Any) -> dict[str, Any]:
        claim = request.context["claims"][0]
        return {
            "answer": f"{claim['text']} Reference {ANTHROPIC}.",
            "answer_claim_ids": [claim["claim_id"]],
            "key_findings": [],
            "interpretation": [],
            "recommendations": [],
        }

    agent, _ = runner(small_db, {LLMTask.RESPOND: [leaky]})
    result = agent.run(REVENUE)
    assert ANTHROPIC not in result.response.model_dump_json()
