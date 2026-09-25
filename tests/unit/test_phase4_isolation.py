"""Architectural boundaries of the Phase 4 layers (static source checks).

- No access to injected-event ground truth, the generator or its hidden customer-health mechanism.
- No database driver and no file access: data is read through the Phase 1 ``Database`` via Phase 2/3.
- LangGraph only in the agent's graph and runner; the Anthropic SDK only in its provider module.
- No MCP, API, UI, RAG/vector or other network frameworks (later phases or out of scope).
- No hard-coded business answers: no dataset member names or business numbers in the source.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT

PACKAGES = ("agent", "llm", "tools", "evidence")
SOURCES = sorted(p for name in PACKAGES for p in (PROJECT_ROOT / "app" / name).rglob("*.py"))


def _code(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _id(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT / "app"))


def _imports(path: Path) -> list[str]:
    return re.findall(r"^\s*(?:from|import)\s+([\w.]+)", _code(path), re.MULTILINE)


def test_every_package_has_sources() -> None:
    for name in PACKAGES:
        assert (PROJECT_ROOT / "app" / name / "__init__.py").exists()
    assert len(SOURCES) >= 25


@pytest.mark.parametrize("path", SOURCES, ids=_id)
def test_no_ground_truth_or_generator_access(path: Path) -> None:
    code = _code(path)
    assert "injected_events" not in code and "ground_truth" not in code
    assert not any(module == "data" or module.startswith("data.") for module in _imports(path)), "no generator"
    assert not re.search(r"\b(health_score|customer_health|latent|api_intensity|stage_factor|calibration)\b", code)


@pytest.mark.parametrize("path", SOURCES, ids=_id)
def test_no_driver_file_access_or_out_of_scope_frameworks(path: Path) -> None:
    forbidden = {
        "duckdb",
        "sqlite3",
        "sqlalchemy",
        "psycopg",
        "langchain",
        "langchain_core",
        "langchain_community",
        "langsmith",
        "openai",
        "mcp",
        "fastapi",
        "starlette",
        "uvicorn",
        "streamlit",
        "requests",
        "httpx",
        "aiohttp",
        "urllib",
        "socket",
        "subprocess",
        "pickle",
        "random",
        "chromadb",
        "faiss",
        "pinecone",
        "sentence_transformers",
        "llama_index",
    }
    for module in _imports(path):
        assert module.split(".")[0] not in forbidden, f"{path.name} imports {module}"
    code = _code(path)
    assert not re.search(r"\bopen\(|\.read_text\(|\.read_bytes\(|\.connect\(", code), "no direct file/DB access"


@pytest.mark.parametrize("path", SOURCES, ids=_id)
def test_framework_imports_are_confined(path: Path) -> None:
    roots = {module.split(".")[0] for module in _imports(path)}
    if "langgraph" in roots:
        assert _id(path) in ("agent/graph.py", "agent/runner.py")
    if "anthropic" in roots:
        assert _id(path) == "llm/anthropic_provider.py"


@pytest.mark.parametrize("path", SOURCES, ids=_id)
def test_no_hard_coded_business_answers(path: Path) -> None:
    code = _code(path)
    for literal in ("Singapore", "Enterprise", "APAC", "EMEA", "AI Insights", "Paid Search"):
        assert literal not in code, f"{path.name} names the dataset member {literal!r}"
    # Business numbers come from tools; the only large literals allowed are configuration limits.
    large = {n for n in re.findall(r"(?<![\w.-])\d[\d_]{3,}(?:\.\d+)?(?![\w-])", code)}
    assert large <= {"1000", "4000", "5000", "10000", "16000", "20000", "64000"}, f"{path.name}: {sorted(large)}"


@pytest.mark.parametrize("path", SOURCES, ids=_id)
def test_business_dates_come_from_configuration(path: Path) -> None:
    code = _code(path)
    assert "date.today(" not in code and "datetime.today(" not in code
    assert not re.search(r"datetime\.now\(\)", code), "timestamps are timezone-aware (UTC) and never business dates"
    assert not re.search(r"\bdate\(20\d\d,", code), "no hard-coded business dates"


def test_dependencies_add_only_langgraph_and_an_optional_provider() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = {re.split(r"[<>=\[ ]", d)[0].lower() for d in project["dependencies"]}
    assert "langgraph" in names
    # "mcp" is the Phase 6 dependency; it stays out of the Phase 4 packages (import check above).
    assert not names & {"anthropic", "openai", "langchain", "langchain-community", "fastapi", "streamlit"}
    assert project["optional-dependencies"]["anthropic"] == ["anthropic>=1.0"]
