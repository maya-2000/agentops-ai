"""Static boundaries of the investigation layer (Phase 10), from parsed source.

    Investigator --> AgentRuntime (input guard, understanding) --> SecuredToolExecutor --> tools --> data

- The investigation package runs tools only through the runtime's ``SecuredToolExecutor`` (one call
  site, in ``engine.py``). It imports no database driver, tool handler, analytics service, API, UI,
  MCP server or generator code, runs no query, and has no SQL text.
- It reads no files or environment, executes no code, and never names the hidden evaluation labels.
- It holds no business numbers or dataset members: every value comes from tool evidence.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT

PACKAGE = PROJECT_ROOT / "app" / "investigation"
SOURCES = sorted(PACKAGE.rglob("*.py"))
STDLIB = set(sys.stdlib_module_names)
ALLOWED_APP_IMPORTS = (
    "app.agent.findings",  # the Phase 4 claim builders
    "app.agent.graph",  # AgentRuntime: guard, model call, secured executor
    "app.agent.observability",
    "app.agent.records",
    "app.agent.request",  # the central request validation
    "app.agent.response",  # caveats
    "app.analytics.labels",  # display names
    "app.analytics.periods",  # period arithmetic (no data access)
    "app.database.deadline",  # the cooperative deadline (no connection)
    "app.evidence.builder",
    "app.evidence.models",
    "app.evidence.validation",
    "app.investigation",
    "app.llm.base",
    "app.llm.schemas",
    "app.security.authorization",  # AuthorizationContext only
    "app.security.budget",
    "app.security.errors",
    "app.security.events",
    "app.security.input_guard",
    "app.security.redaction",
    "app.tools.base",  # result types
)


def _id(path: Path) -> str:
    return str(path.relative_to(PACKAGE))


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


def _calls(path: Path) -> list[str]:
    names: list[str] = []
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute):
                names.append(ast.unparse(func))
    return names


def _code(path: Path) -> str:
    """Source without docstrings and comments (so documentation may name what the code must not do)."""
    tree = _tree(path)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                body.pop(0)
    return ast.unparse(tree)


def test_the_package_exists() -> None:
    assert {p.name for p in SOURCES} >= {
        "engine.py",
        "models.py",
        "planner.py",
        "templates.py",
        "steps.py",
        "findings.py",
        "validation.py",
        "drivers.py",
        "recommendations.py",
        "brief.py",
    }


@pytest.mark.parametrize("path", SOURCES, ids=_id)
def test_imports_only_the_approved_layers(path: Path) -> None:
    for module in _imports(path):
        root = module.split(".")[0]
        if root == "app":
            assert module.startswith(ALLOWED_APP_IMPORTS), module
        else:
            assert root in STDLIB or root == "pydantic", module
        assert root not in {"duckdb", "sqlite3", "pandas", "numpy", "data", "fastapi", "streamlit", "httpx"}, module


@pytest.mark.parametrize("path", SOURCES, ids=_id)
def test_no_query_handler_file_environment_or_code_execution(path: Path) -> None:
    calls = _calls(path)
    for call in calls:
        name = call.rsplit(".", 1)[-1]
        assert name not in {"query", "handler", "open", "read_text", "read_bytes", "write_text", "getenv"}, call
        if "." not in call:  # builtins (``re.compile`` is a regular expression, not code)
            assert call not in {"eval", "exec", "compile", "__import__"}, call
        assert call not in {"os.system", "os.popen", "subprocess.run", "subprocess.Popen"}, call
        if name == "execute":
            assert call == "self.runtime.executor.execute" and _id(path) == "engine.py", call
    code = _code(path)
    assert not re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE)\s", code), "no SQL text"
    for marker in ("injected_events", "ground_truth", "data/seeds", "os.environ", "health_score", "latent"):
        assert marker not in code, marker


def test_every_tool_call_goes_through_the_secured_executor_once() -> None:
    calls = [c for path in SOURCES for c in _calls(path) if c.endswith(".execute")]
    assert calls == ["self.runtime.executor.execute"]
    engine = (PACKAGE / "engine.py").read_text(encoding="utf-8")
    assert "sql_permitted=False" in engine


@pytest.mark.parametrize("path", SOURCES, ids=_id)
def test_no_business_numbers_or_dataset_members(path: Path) -> None:
    code = _code(path)
    for member in ("Singapore", "APAC", "EMEA", "LATAM", "Enterprise", "SMB", "Mid-Market", "Paid Search"):
        assert member not in code, member
    large = set(re.findall(r"(?<![\w.-])\d[\d_]{3,}(?:\.\d+)?(?![\w-])", code))
    assert large <= {"1000", "20000"}, sorted(large)  # milliseconds per second; the text validator's size cap
