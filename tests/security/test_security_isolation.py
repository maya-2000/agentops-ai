"""Static guarantees: no code execution, no filesystem access and no Phase 6+ functionality in the agent path.

The checks parse the source (``ast``), so they find real calls and imports and ignore words inside
strings, docstrings and regular expressions (the injection screen legitimately names ``eval``).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT

APP = PROJECT_ROOT / "app"
ALL_SOURCES = sorted(APP.rglob("*.py"))
AGENT_PATH = sorted(p for name in ("agent", "llm", "tools", "evidence", "security") for p in (APP / name).rglob("*.py"))
SECURITY = sorted((APP / "security").rglob("*.py"))


def _id(path: Path) -> str:
    return str(path.relative_to(APP))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _imports(tree: ast.Module) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def _calls(tree: ast.Module) -> set[str]:
    """Names of called functions: ``eval`` for ``eval(...)``, ``os.system`` for ``os.system(...)``, ``.read_text``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            names.add(f".{func.attr}")
            if isinstance(func.value, ast.Name):
                names.add(f"{func.value.id}.{func.attr}")
    return names


def _keywords(tree: ast.Module) -> set[tuple[str, object]]:
    return {
        (kw.arg, kw.value.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg and isinstance(kw.value, ast.Constant)
    }


@pytest.mark.parametrize("path", ALL_SOURCES, ids=_id)
def test_no_dynamic_code_execution_anywhere_in_the_application(path: Path) -> None:
    tree = _tree(path)
    assert not _calls(tree) & {
        "eval",
        "exec",
        "compile",
        "__import__",
        "os.system",
        "os.popen",
        "os.execv",
        "os.spawnl",
        "os.startfile",
    }, path
    roots = {m.split(".")[0] for m in _imports(tree)}
    assert not roots & {"subprocess", "importlib", "pickle", "marshal", "shelve", "ctypes", "runpy", "code"}, path
    assert ("shell", True) not in _keywords(tree)


@pytest.mark.parametrize("path", AGENT_PATH, ids=_id)
def test_agent_path_has_no_filesystem_or_network_access(path: Path) -> None:
    tree = _tree(path)
    assert not _calls(tree) & {
        "open",
        ".read_text",
        ".write_text",
        ".read_bytes",
        ".write_bytes",
        ".unlink",
        "os.listdir",
        "os.walk",
        "os.scandir",
        "os.remove",
        "glob.glob",
        ".glob",
        ".rglob",
        ".iterdir",
        ".mkdir",
    }, path
    roots = {m.split(".")[0] for m in _imports(tree)}
    assert not roots & {
        "shutil",
        "socket",
        "urllib",
        "requests",
        "httpx",
        "aiohttp",
        "http",
        "ftplib",
        "smtplib",
        "glob",
        "tempfile",
        "pathlib",
    }, (path, roots)


@pytest.mark.parametrize("path", SECURITY, ids=_id)
def test_security_layer_depends_only_on_approved_packages(path: Path) -> None:
    allowed_roots = {
        "__future__",
        "app",
        "pydantic",
        "sqlglot",
        "re",
        "os",
        "json",
        "math",
        "uuid",
        "logging",
        "hashlib",
        "threading",
        "unicodedata",
        "collections",
        "dataclasses",
        "datetime",
        "enum",
        "typing",
        "time",
    }
    for module in _imports(_tree(path)):  # security sits below the agent and never reaches the generator
        assert module.split(".")[0] in allowed_roots, (path.name, module)
        assert module != "data" and not module.startswith(("app.agent", "data.")), (path.name, module)


def test_os_environ_is_read_only_by_the_redaction_utility() -> None:
    readers = []
    for path in ALL_SOURCES:
        for node in ast.walk(_tree(path)):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in ("environ", "getenv")
                and isinstance(node.value, ast.Name)
            ):
                readers.append(_id(path))
    assert sorted(set(readers)) == ["security/redaction.py"]


def test_no_phase6_or_later_packages() -> None:
    for name in ("mcp", "mcp_server", "api", "ui", "evaluation", "benchmark", "server", "web"):
        assert not (APP / name).exists(), name
    for path in ALL_SOURCES:
        roots = {m.split(".")[0] for m in _imports(_tree(path))}
        assert not roots & {"mcp", "fastapi", "starlette", "uvicorn", "streamlit", "flask"}, path


def test_limits_are_not_hard_coded_in_the_graph() -> None:
    graph = (APP / "agent" / "graph.py").read_text(encoding="utf-8")
    for literal in ("max_tool_calls=12", "12 tool", "row_limit=200", ">= 12", "timeout=30"):
        assert literal not in graph
