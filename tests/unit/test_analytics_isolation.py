"""Architectural boundaries of the analytics layer (static source checks).

- No access to injected-event ground truth or generator internals (hidden health mechanism).
- No database driver imports outside the executor boundary (the Phase 1 ``Database`` protocol
  is the only execution path).
- No knowledge of later phases (agent, LLM, MCP, API, UI) and no forecasting/anomaly code.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT

ANALYTICS = PROJECT_ROOT / "app" / "analytics"
SOURCES = sorted(ANALYTICS.rglob("*.py"))


def _code(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ANALYTICS)))
def test_no_ground_truth_or_generator_access(path: Path) -> None:
    code = _code(path)
    assert "injected_events" not in code
    assert "ground_truth" not in code
    assert not re.search(r"^\s*(from|import)\s+data(\.|\s)", code, re.MULTILINE), "must not import the generator"
    assert not re.search(r"\b(health_score|engagement|latent|api_intensity|stage_factor)\b", code)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ANALYTICS)))
def test_no_driver_or_later_phase_dependencies(path: Path) -> None:
    code = _code(path)
    imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", code, re.MULTILINE)
    forbidden = (
        "duckdb",
        "sqlalchemy",
        "psycopg",
        "langgraph",
        "langchain",
        "anthropic",
        "openai",
        "mcp",
        "fastapi",
        "streamlit",
        "statsmodels",
        "sklearn",
    )
    for module in imports:
        assert module.split(".")[0] not in forbidden, f"{path.name} imports {module}"


def test_phase3_modules_not_created() -> None:
    for name in ("forecasting", "anomaly", "agent", "llm", "api", "ui"):
        assert not (PROJECT_ROOT / "app" / name).exists(), f"app/{name} belongs to a later phase"
