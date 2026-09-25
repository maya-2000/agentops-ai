"""MCP server lifecycle: initialisation, capabilities, database ownership, shutdown and concurrency."""

from __future__ import annotations

import json
import logging
from typing import Any

import anyio
import pytest
from mcp import Client

import app.mcp.server as server_module
from app.agent.config import AgentConfig
from app.config import Settings
from app.mcp import MCPServerConfig, create_server
from app.mcp.__main__ import main
from app.mcp.server import INSTRUCTIONS
from tests.phase4_support import AS_OF
from tests.phase6_support import EXPECTED_TOOLS, mcp_config, server, structured, with_client


class TrackingDatabase:
    """Delegates to a real database and records whether the server closed it."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.dataset_version = inner.dataset_version
        self.closed = 0

    def query(self, *args: Any, **kwargs: Any) -> Any:
        assert not self.closed, "query after close"
        return self.inner.query(*args, **kwargs)

    def list_tables(self) -> list[str]:
        return self.inner.list_tables()

    def close(self) -> None:
        self.closed += 1


def test_server_initializes_with_its_identity_and_only_the_tools_capability(small_db: Any) -> None:
    async def fn(client: Client) -> Any:
        return client.server_info, client.server_capabilities, client.instructions

    info, capabilities, instructions = with_client(server(small_db), fn)
    assert info is not None and (info.name, info.version) == ("agentops-ai", "0.6.0")
    assert capabilities.tools is not None
    assert capabilities.resources is None and capabilities.prompts is None
    assert instructions == INSTRUCTIONS


def test_server_identity_comes_from_configuration(small_db: Any) -> None:
    config = mcp_config(server_name="agentops-dev", server_version="1.2.3")

    async def fn(client: Client) -> Any:
        return client.server_info

    info = with_client(server(small_db, config), fn)
    assert info is not None and (info.name, info.version) == ("agentops-dev", "1.2.3")


def test_server_opens_the_database_at_startup_and_closes_it_on_shutdown(
    small_db: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[TrackingDatabase] = []

    def fake_get_database(*_: Any, read_only: bool = True) -> TrackingDatabase:
        assert read_only, "the MCP server opens the database read-only"
        db = TrackingDatabase(small_db)
        opened.append(db)
        return db

    monkeypatch.setattr(server_module, "get_database", fake_get_database)
    target = create_server(mcp_config(), as_of=AS_OF)
    assert opened == [], "nothing is opened before the server runs"

    async def fn(client: Client) -> Any:
        assert len(opened) == 1 and opened[0].closed == 0
        return await client.call_tool("agentops_get_kpi", {"kpi": "customer_count", "period": "2026-08"})

    result = with_client(target, fn)
    assert not result.is_error
    assert len(opened) == 1 and opened[0].closed == 1, "closed exactly once on shutdown"


def test_injected_database_is_not_closed_by_the_server(small_db: Any) -> None:
    db = TrackingDatabase(small_db)
    result = with_client(
        server(db),
        lambda c: c.call_tool("agentops_get_kpi", {"kpi": "customer_count", "period": "2026-08"}),
    )
    assert not result.is_error and db.closed == 0


def test_lifecycle_is_audited(small_db: Any, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="agentops.mcp"):
        with_client(server(small_db), lambda c: c.list_tools())
    events = [json.loads(r.message)["event"] for r in caplog.records if r.name == "agentops.mcp"]
    assert events[0] == "server_started" and events[-1] == "server_stopped"


def test_server_can_be_restarted_cleanly(small_db: Any) -> None:
    target = server(small_db)
    for _ in range(2):
        result = with_client(target, lambda c: c.list_tools())
        assert [t.name for t in result.tools] == list(EXPECTED_TOOLS)


def test_misconfiguration_fails_at_startup_before_the_database_is_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_module, "get_database", lambda **_: pytest.fail("database opened"))
    with pytest.raises(ValueError, match="Unknown MCP tools"):
        create_server(mcp_config(enabled_tools=frozenset({"agentops_read_file"})))
    with pytest.raises(ValueError, match="Cannot disable unknown tools"):
        create_server(mcp_config(limits=AgentConfig(disabled_tools=frozenset({"read_file"}))))


def test_configuration_comes_from_settings() -> None:
    settings = Settings(
        mcp_server_name="agentops-local",
        mcp_enabled_tools="agentops_get_kpi, agentops_forecast_metric",
        mcp_max_request_bytes=4096,
        mcp_log_level="INFO",
        mcp_sql_enabled=False,
        agent_disabled_tools="forecast_metric",
    )
    config = MCPServerConfig.from_settings(settings)
    assert config.server_name == "agentops-local"
    assert config.max_request_bytes == 4096 and config.log_level == "INFO" and config.transport == "stdio"
    assert config.listed_tools == {"agentops_get_kpi"}  # forecast is disabled for the agent too
    with pytest.raises(ValueError, match="Unknown MCP tools: run_shell"):
        MCPServerConfig.from_settings(Settings(mcp_enabled_tools="agentops_get_kpi,run_shell"))


def test_cli_lists_tools_without_a_database(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list-tools"]) == 0
    names = [line.split("\t")[0] for line in capsys.readouterr().out.splitlines()]
    assert names == list(EXPECTED_TOOLS)


def test_cli_fails_fast_and_safely_without_a_database(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DATABASE_URL", "duckdb:///database/does-not-exist.duckdb")
    monkeypatch.setattr("app.database.factory.get_settings", lambda: Settings())
    assert main([]) == 1
    err = capsys.readouterr().err
    assert "could not start" in err and "Traceback" not in err
    assert "/does-not-exist" not in err, "no filesystem path in the message"


def test_concurrent_calls_are_serialised_and_request_scoped(full_db: Any) -> None:
    calls = [
        ("agentops_get_kpi", {"kpi": "revenue", "period": "2026-08"}),
        ("agentops_get_kpi", {"kpi": "mrr", "period": "2026-07"}),
        ("agentops_run_safe_sql", {"sql": "SELECT COUNT(*) AS n FROM customers"}),
        ("agentops_detect_anomalies", {"metric": "revenue"}),
        ("agentops_analyze_support", {"operation": "support_summary", "period": "2026-08"}),
        ("agentops_run_safe_sql", {"sql": "SELECT COUNT(*) AS n FROM support_tickets"}),
    ]

    async def fn(client: Client) -> list[Any]:
        results: list[Any] = [None] * len(calls)

        async def one(i: int, name: str, args: dict[str, Any]) -> None:
            results[i] = await client.call_tool(name, args)

        async with anyio.create_task_group() as tg:
            for i, (name, args) in enumerate(calls):
                tg.start_soon(one, i, name, args)
        return results

    results = with_client(server(full_db), fn)
    outputs = [structured(r) for r in results]
    assert all(not r.is_error for r in results)
    assert len({o["request_id"] for o in outputs}) == len(calls), "one run ID per request"
    for output, (name, _) in zip(outputs, calls, strict=True):
        assert output["tool_name"] == name
        # evidence is request-scoped: IDs restart at E1 and every item comes from this call
        assert output["evidence"][0]["evidence_id"] == "E1"
        assert {e["tool_call_id"] for e in output["evidence"]} == {"T1"}
        for item in output["evidence"]:
            assert set(item["query_ids"]) <= set(output["provenance"]["query_ids"])
