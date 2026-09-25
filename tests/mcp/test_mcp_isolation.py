"""Static guarantees of the MCP layer: a thin adapter with no SQL, files, network, ground truth or
business formulas of its own, one shared secured execution path, and a local transport only."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import get_args

import pytest

from app.config import PROJECT_ROOT, MCPTransport

APP = PROJECT_ROOT / "app"
MCP = sorted((APP / "mcp").rglob("*.py"))
# What the MCP layer may import from the application: configuration, the tool layer, the Phase 5
# security layer, evidence, and the Phase 2/3 *result types and vocabularies* (no services).
ALLOWED_APP_IMPORTS = {
    "app.agent.config",
    "app.agent.observability",
    "app.analytics.dimensions",
    "app.analytics.kpis",
    "app.anomalies",
    "app.anomalies.config",
    "app.config",
    "app.database.base",
    "app.database.factory",
    "app.evidence.builder",
    "app.evidence.models",
    "app.evidence.validation",
    "app.forecasting",
    "app.forecasting.evaluation",
    "app.llm.schemas",
    "app.security.authorization",
    "app.security.budget",
    "app.security.data_policy",
    "app.security.errors",
    "app.security.events",
    "app.security.execution",
    "app.security.injection",
    "app.security.limits",
    "app.security.output_guard",
    "app.security.redaction",
    "app.security.retry",
    "app.timeseries.metrics",
    "app.tools.base",
    "app.tools.registry",
    "app.tools.results",
}
ALLOWED_MCP_SDK = {"mcp", "mcp.types", "mcp.server", "mcp.server.context", "mcp.server.stdio"}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def _calls(path: Path) -> tuple[set[str], set[str]]:
    """Names of called builtins/functions and of called methods."""
    functions: set[str] = set()
    methods: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                methods.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                functions.add(node.func.id)
    return functions, methods


def _id(path: Path) -> str:
    return str(path.relative_to(APP))


@pytest.mark.parametrize("path", MCP, ids=_id)
def test_mcp_imports_only_approved_application_interfaces(path: Path) -> None:
    imports = _imports(path)
    app_imports = {m for m in imports if m.startswith("app.") and not m.startswith("app.mcp")}
    assert app_imports <= ALLOWED_APP_IMPORTS, sorted(app_imports - ALLOWED_APP_IMPORTS)
    sdk = {m for m in imports if m == "mcp" or m.startswith("mcp.")}
    assert sdk <= ALLOWED_MCP_SDK, sorted(sdk - ALLOWED_MCP_SDK)
    roots = {m.split(".")[0] for m in imports}
    forbidden = {"duckdb", "sqlglot", "sqlite3", "subprocess", "socket", "requests", "httpx", "urllib", "pickle", "os"}
    assert not roots & forbidden, sorted(roots & forbidden)
    assert not roots & {"data", "fastapi", "starlette", "uvicorn", "streamlit", "flask", "langgraph"}


@pytest.mark.parametrize("path", MCP, ids=_id)
def test_mcp_never_touches_files_sql_or_ground_truth(path: Path) -> None:
    functions, methods = _calls(path)
    assert not functions & {"open", "eval", "exec", "compile", "__import__", "getattr", "input"}, functions
    files = {"read_text", "read_bytes", "write_text", "write_bytes", "iterdir", "glob", "listdir", "walk", "unlink"}
    database_or_process = {"query", "connect", "system", "popen", "getenv", "execute_sql"}
    assert not methods & (files | database_or_process), sorted(methods & (files | database_or_process))
    code = path.read_text(encoding="utf-8")
    assert "injected_events" not in code and "ground_truth" not in code and "data/seeds" not in code
    assert "QueryRunner" not in code and ".query(" not in code
    assert not re.search(r"\bSELECT\b.*\bFROM\b", code), "no SQL text in the MCP layer"


def test_mcp_contains_no_business_formulas() -> None:
    """Numbers come only from tools: the MCP modules import no analytics, forecasting or anomaly service."""
    services = re.compile(
        r"app\.analytics\.(revenue|customers|sales|marketing|support|product|cohorts|risk|executor)|"
        r"app\.forecasting\.(service|methods|statistical|baselines|backtest|selection)|"
        r"app\.anomalies\.(service|detectors|thresholds)|app\.analytics\.kpis\.(service|sql)"
    )
    for path in MCP:
        for module in _imports(path):
            assert not services.match(module), f"{_id(path)} imports the service {module}"


def test_both_entry_points_share_one_secured_execution_path() -> None:
    graph = (APP / "agent" / "graph.py").read_text(encoding="utf-8")
    adapters = (APP / "mcp" / "adapters.py").read_text(encoding="utf-8")
    assert "SecuredToolExecutor" in graph and "SecuredToolExecutor" in adapters
    assert "registry.execute(" not in graph and "registry.execute(" not in adapters
    # one Phase 5 policy with the Phase 5 permission table: no MCP permission table of its own
    assert adapters.count("ToolAuthorizationPolicy(") == 1 and adapters.count("ToolPermissions(") == 1
    for path in MCP:
        code = path.read_text(encoding="utf-8")
        assert not re.search(r"^\w*PERMISSIONS\w*\s*[:=]", code, re.MULTILINE), _id(path)


def test_only_one_mcp_tool_catalogue_exists() -> None:
    definitions = [p for p in MCP if "types.Tool(" in p.read_text(encoding="utf-8")]
    assert [_id(p) for p in definitions] == ["mcp/registry.py"]
    names = [p for p in MCP if re.search(r'"agentops_get_kpi"', p.read_text(encoding="utf-8"))]
    assert [_id(p) for p in names] == ["mcp/registry.py"]


def test_transport_is_local_stdio_only() -> None:
    assert get_args(MCPTransport) == ("stdio",)
    imports = {m for path in MCP for m in _imports(path)}
    assert "mcp.server.stdio" in imports
    for module in imports:
        assert not re.search(r"streamable_http|\bsse\b|uvicorn|starlette|websocket", module), module
