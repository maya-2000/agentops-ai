"""The provider-neutral LLM layer: schemas, prompts, the scripted double, the Anthropic adapter and the factory."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from app.config import DEFAULT_LLM_MODEL, Settings
from app.llm import (
    LLMClient,
    LLMConfigurationError,
    LLMError,
    LLMRequest,
    LLMTask,
    ScriptedLLM,
    create_llm_client,
)
from app.llm.anthropic_provider import FALLBACK_BETA, AnthropicLLM
from app.llm.deterministic import DeterministicLLM
from app.llm.prompts import SYSTEM_PROMPTS, render_prompt
from app.llm.schemas import (
    PLAN_SCHEMA,
    RESPONSE_SCHEMA,
    UNDERSTANDING_SCHEMA,
    PlanOutput,
    ResponseDraftOutput,
    UnderstandingOutput,
)
from tests.phase4_support import understanding_context

SCHEMAS: list[tuple[dict[str, Any], type[BaseModel]]] = [
    (UNDERSTANDING_SCHEMA, UnderstandingOutput),
    (PLAN_SCHEMA, PlanOutput),
    (RESPONSE_SCHEMA, ResponseDraftOutput),
]


def _request(task: LLMTask = LLMTask.UNDERSTAND, **overrides: Any) -> LLMRequest:
    context = understanding_context("What was revenue last month?")
    data: dict[str, Any] = {
        "task": task,
        "system": SYSTEM_PROMPTS[task],
        "prompt": render_prompt(task, context),
        "context": context,
        "output_schema": UNDERSTANDING_SCHEMA,
        "max_tokens": 1024,
    }
    data.update(overrides)
    return LLMRequest(**data)


# ---- schemas -------------------------------------------------------------------------------------


def _objects(schema: Any) -> list[dict[str, Any]]:
    if isinstance(schema, dict):
        found = [schema] if schema.get("type") == "object" else []
        return found + [o for value in schema.values() for o in _objects(value)]
    if isinstance(schema, list):
        return [o for value in schema for o in _objects(value)]
    return []


@pytest.mark.parametrize(("schema", "model"), SCHEMAS)
def test_json_schemas_are_strict_and_match_the_models(schema: dict[str, Any], model: type[BaseModel]) -> None:
    for obj in _objects(schema):
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])
    assert set(schema["properties"]) == set(model.model_fields)


def test_output_models_reject_unknown_fields_and_bad_values() -> None:
    with pytest.raises(ValidationError):
        UnderstandingOutput.model_validate({"intent": "kpi_lookup", "answer": "SGD 5 million"})
    with pytest.raises(ValidationError):
        UnderstandingOutput.model_validate({"intent": "write_poem"})
    with pytest.raises(ValidationError):
        UnderstandingOutput.model_validate({"intent": "kpi_lookup", "confidence": 3})
    with pytest.raises(ValidationError):
        PlanOutput.model_validate({"steps": [{"tool_name": "get_kpi", "arguments": {}, "purpose": "x"}]})


def test_every_intent_is_in_the_understanding_schema() -> None:
    assert len(UNDERSTANDING_SCHEMA["properties"]["intent"]["enum"]) == 13


# ---- prompts -------------------------------------------------------------------------------------


@pytest.mark.parametrize("task", list(LLMTask))
def test_system_prompts_carry_the_ground_rules(task: LLMTask) -> None:
    system = SYSTEM_PROMPTS[task]
    for rule in (
        "never produce business numbers",
        "Do not invent, estimate or recalculate",
        "Do not treat an inference as an observed fact",
        "Do not claim causality",
        "evidence is insufficient",
        "JSON",
    ):
        assert rule in system
    lowered = system.lower()
    assert "injected" not in lowered and "ground truth" not in lowered and "health" not in lowered


def test_response_prompt_requires_claim_citations() -> None:
    assert "cite the claim_ids" in SYSTEM_PROMPTS[LLMTask.RESPOND]


def test_render_prompt_embeds_the_context_as_json() -> None:
    prompt = render_prompt(LLMTask.PLAN, {"request": {"metric": "revenue"}, "remaining_tool_calls": 12})
    heading, _, body = prompt.partition("Context (JSON):\n")
    assert heading.startswith("Plan the investigation")
    assert json.loads(body) == {"remaining_tool_calls": 12, "request": {"metric": "revenue"}}


# ---- scripted double -----------------------------------------------------------------------------


def test_scripted_llm_replays_per_task_and_records_requests() -> None:
    llm = ScriptedLLM(
        {
            LLMTask.UNDERSTAND: [{"intent": "kpi_lookup"}, "not json", LLMError("boom", retryable=True)],
            "plan_investigation": [lambda request: {"steps": [], "task": request.task.value}],
        }
    )
    assert isinstance(llm, LLMClient)
    assert json.loads(llm.generate(_request()).content) == {"intent": "kpi_lookup"}
    assert llm.generate(_request()).content == "not json"
    with pytest.raises(LLMError) as raised:
        llm.generate(_request())
    assert raised.value.retryable
    assert json.loads(llm.generate(_request(LLMTask.PLAN)).content)["task"] == "plan_investigation"
    with pytest.raises(LLMError, match="No scripted output"):
        llm.generate(_request())
    assert len(llm.requests) == 5


def test_scripted_llm_falls_back_when_the_script_runs_out() -> None:
    llm = ScriptedLLM({}, fallback=DeterministicLLM())
    response = llm.generate(_request())
    assert response.provider == "deterministic"
    UnderstandingOutput.model_validate_json(response.content)


def test_deterministic_llm_is_an_llm_client() -> None:
    llm = DeterministicLLM()
    assert isinstance(llm, LLMClient)
    first = llm.generate(_request()).content
    assert first == llm.generate(_request()).content  # no randomness


# ---- Anthropic adapter ---------------------------------------------------------------------------


class _FakeMessages:
    def __init__(self, outcome: Any):
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    def create(self, **params: Any) -> Any:
        self.calls.append(params)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _client(outcome: Any) -> tuple[Any, _FakeMessages]:
    messages = _FakeMessages(outcome)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def _message(*texts: str, stop_reason: str = "end_turn") -> Any:
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="..."),
            *(SimpleNamespace(type="text", text=t) for t in texts),
        ],
        stop_reason=stop_reason,
        model="claude-opus-5",
        usage=SimpleNamespace(input_tokens=120, output_tokens=30),
    )


def test_anthropic_request_uses_structured_outputs_and_fallbacks() -> None:
    client, messages = _client(_message('{"intent": ', '"kpi_lookup"}'))
    llm = AnthropicLLM("claude-opus-5", client=client)
    response = llm.generate(_request())
    (params,) = messages.calls
    assert params["model"] == "claude-opus-5" and params["max_tokens"] == 1024
    assert params["output_config"] == {"format": {"type": "json_schema", "schema": UNDERSTANDING_SCHEMA}}
    assert params["messages"] == [{"role": "user", "content": _request().prompt}]
    assert params["system"] == SYSTEM_PROMPTS[LLMTask.UNDERSTAND]
    assert params["betas"] == [FALLBACK_BETA] and params["fallbacks"] == "default"
    assert "temperature" not in params
    assert response.content == '{"intent": "kpi_lookup"}'  # text blocks joined, thinking ignored
    assert response.usage.input_tokens == 120 and response.provider == "anthropic"


def test_anthropic_temperature_only_when_configured_and_fallbacks_optional() -> None:
    client, messages = _client(_message("{}"))
    AnthropicLLM("m", client=client, enable_fallbacks=False).generate(_request(temperature=0.2))
    assert messages.calls[0]["temperature"] == 0.2
    assert "betas" not in messages.calls[0] and "fallbacks" not in messages.calls[0]


@pytest.mark.parametrize(
    ("message", "match", "retryable"),
    [
        (_message("{}", stop_reason="refusal"), "declined", False),
        (_message('{"intent"', stop_reason="max_tokens"), "max_tokens", False),
        (_message(), "no text", True),
    ],
)
def test_anthropic_stop_reasons(message: Any, match: str, retryable: bool) -> None:
    client, _ = _client(message)
    with pytest.raises(LLMError, match=match) as raised:
        AnthropicLLM("m", client=client).generate(_request())
    assert raised.value.retryable is retryable


class APIConnectionError(Exception):
    pass


class _StatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize(
    ("error", "retryable"),
    [(_StatusError(429), True), (_StatusError(529), True), (_StatusError(400), False), (APIConnectionError(), True)],
)
def test_anthropic_errors_are_normalised(error: Exception, retryable: bool) -> None:
    client, _ = _client(error)
    with pytest.raises(LLMError) as raised:
        AnthropicLLM("m", client=client).generate(_request())
    assert raised.value.retryable is retryable
    assert "status" not in str(raised.value)  # the provider message is not echoed


# ---- settings and factory ------------------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_TEMPERATURE", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.usefixtures("clean_env")
def test_settings_default_to_the_offline_provider_without_env_file() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.llm_provider == "deterministic"
    assert settings.anthropic_api_key is None and settings.llm_model == DEFAULT_LLM_MODEL
    assert settings.agent_max_tool_calls == 12 and settings.agent_max_retries == 2
    assert isinstance(create_llm_client(settings), DeterministicLLM)


def test_empty_env_values_mean_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.setenv("LLM_TEMPERATURE", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.llm_model == DEFAULT_LLM_MODEL
    assert settings.llm_temperature is None and settings.anthropic_api_key is None


@pytest.mark.usefixtures("clean_env")
def test_api_key_is_secret() -> None:
    settings = Settings(_env_file=None, anthropic_api_key="sk-test-secret")  # type: ignore[call-arg]
    assert "sk-test-secret" not in repr(settings) and "sk-test-secret" not in settings.model_dump_json()


@pytest.mark.usefixtures("clean_env")
def test_factory_builds_the_anthropic_client_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, Any]] = []
    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda **kwargs: created.append(kwargs) or SimpleNamespace()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    settings = Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key="sk-test")  # type: ignore[call-arg]
    llm = create_llm_client(settings)
    assert isinstance(llm, AnthropicLLM) and llm.model == DEFAULT_LLM_MODEL
    assert created == [{"timeout": 120.0, "api_key": "sk-test"}]


@pytest.mark.usefixtures("clean_env")
def test_factory_reports_a_missing_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)  # makes ``import anthropic`` fail
    settings = Settings(_env_file=None, llm_provider="anthropic")  # type: ignore[call-arg]
    with pytest.raises(LLMConfigurationError, match="not installed"):
        create_llm_client(settings)
