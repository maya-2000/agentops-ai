"""Static boundaries of the Phase 8 layers (parsed source, so words in strings and docs are ignored).

    UI (app/ui) --HTTP--> API (app/api) --> AgentRunner.run --> tools --> security/execution --> data

- The API reaches the agent only through ``AgentRunner.run`` in ``service.py``. It imports result
  types and registries (names, units, descriptions), never a service, a tool handler, the secured
  executor, the database driver or the MCP server, and it runs no query of its own except the health
  probe's metadata query and the start-up coverage lookup of the agent's own KPI service.
- The UI imports only Streamlit, httpx, the standard library, ``app.config`` and itself: no agent,
  API, analytics, tool, evidence, security or database code. It reaches the API over HTTP only.
- Neither layer reads files, the environment or the hidden evaluation labels, or executes code.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT

APP = PROJECT_ROOT / "app"
API = sorted((APP / "api").rglob("*.py"))
UI = sorted((APP / "ui").rglob("*.py"))
STDLIB = set(sys.stdlib_module_names)

API_APP_IMPORTS = {
    "app.agent.observability",  # run-ID validation and generation
    "app.agent.records",  # AgentStatus
    "app.agent.request",  # ValidatedRequest (serialised as is)
    "app.agent.response",  # AgentResponse and the user-safe trace entries
    "app.agent.runner",  # AgentRunner: the only way to run the agent
    "app.analytics.dimensions",  # display names
    "app.analytics.kpis",  # KPI registry: names, units, definitions
    "app.analytics.models",  # the Scalar type
    "app.analytics.periods",  # the Period type
    "app.anomalies",  # the AnomalyReport type
    "app.anomalies.config",  # detector names
    "app.config",
    "app.database",  # get_database (opened once, read-only) and the Database protocol
    "app.evidence.models",  # Claim and Evidence (serialised as is)
    "app.forecasting",  # the ForecastResult type
    "app.security.redaction",  # log redaction
    "app.timeseries",  # series-metric registry: names and units
    "app.tools",  # ToolRegistry type
    "app.tools.registry",  # tool catalogue: names and descriptions
    "app.tools.views",  # forecast and anomaly views shared with the MCP server
}
API_THIRD_PARTY = {"fastapi", "starlette", "uvicorn", "pydantic"}
UI_THIRD_PARTY = {"streamlit", "httpx"}


def _id(path: Path) -> str:
    return str(path.relative_to(APP))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _calls(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(f".{func.attr}")
    return names


def test_both_packages_exist() -> None:
    assert (APP / "api" / "main.py").exists() and (APP / "ui" / "main.py").exists()
    assert len(API) >= 10 and len(UI) >= 4


@pytest.mark.parametrize("path", API, ids=_id)
def test_api_imports_only_approved_layers(path: Path) -> None:
    for module in _imports(path):
        root = module.split(".")[0]
        if root == "app":
            assert module in API_APP_IMPORTS or module == "app.api" or module.startswith("app.api."), module
        else:
            assert root in STDLIB or root in API_THIRD_PARTY, module
        assert not module.startswith(("app.mcp", "app.ui", "app.tools.handlers", "app.security.execution")), module


@pytest.mark.parametrize("path", API, ids=_id)
def test_api_runs_no_query_calculation_or_tool_of_its_own(path: Path) -> None:
    calls = _calls(path)
    for forbidden in (".query", ".execute", ".calculate_kpi", ".forecast", ".detect", "QueryRunner", ".handler"):
        assert forbidden not in calls, forbidden
    code = path.read_text(encoding="utf-8")
    assert not re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|DROP)\s", code), "no SQL text in the API"
    if _id(path) != "api/service.py":
        assert not re.search(r"runner\.run\(", code), "the agent runs only in service.py"
        assert "AgentRunner(" not in code and ".list_tables" not in calls and ".coverage" not in calls


def test_the_agent_is_constructed_and_run_only_by_the_service() -> None:
    service = (APP / "api" / "service.py").read_text(encoding="utf-8")
    assert "AgentRunner(db)" in service and "self._runner.run(" in service
    assert {".list_tables", ".coverage"} <= _calls(APP / "api" / "service.py")
    for path in API:
        code = path.read_text(encoding="utf-8")
        assert "build_graph" not in code and "graph.invoke" not in code and "graph.stream" not in code, path


@pytest.mark.parametrize("path", UI, ids=_id)
def test_ui_imports_only_its_client_side_dependencies(path: Path) -> None:
    for module in _imports(path):
        root = module.split(".")[0]
        if root == "app":
            assert module == "app.config" or module == "app.ui" or module.startswith("app.ui."), module
        else:
            assert root in STDLIB or root in UI_THIRD_PARTY, module
        assert root not in {"duckdb", "sqlite3", "pandas", "numpy", "fastapi", "starlette", "data"}, module


@pytest.mark.parametrize("path", API + UI, ids=_id)
def test_no_files_environment_ground_truth_or_code_execution(path: Path) -> None:
    calls = _calls(path)
    for forbidden in ("open", ".read_text", ".read_bytes", ".write_text", "eval", "exec", "compile", "__import__"):
        assert forbidden not in calls, forbidden
    code = path.read_text(encoding="utf-8")
    for marker in ("injected_events", "ground_truth", "data/seeds", "os.environ", "getenv"):
        assert marker not in code, marker
    assert not re.search(r"\b(health_score|latent|api_intensity|stage_factor)\b", code)


def test_no_hard_coded_business_numbers_in_the_api_or_ui() -> None:
    for path in API + UI:
        code = path.read_text(encoding="utf-8")
        for member in ("Singapore", "APAC", "EMEA", "Paid Search", "Enterprise"):
            assert member not in code, (path.name, member)
        large = set(re.findall(r"(?<![\w.-])\d[\d_]{3,}(?:\.\d+)?(?![\w-])", code))
        assert large <= {"1000", "10_000", "1_048_576", "65535", "8000", "8501"}, (path.name, sorted(large))


def test_the_ui_talks_to_the_api_over_http_only() -> None:
    client = (APP / "ui" / "client.py").read_text(encoding="utf-8")
    assert "httpx.Client(" in client and "/api/v1" in client
    main = (APP / "ui" / "main.py").read_text(encoding="utf-8")
    assert "ui_api_url" in main and "AgentOpsClient(" in main
