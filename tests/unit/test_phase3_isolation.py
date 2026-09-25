"""Architectural boundaries of the Phase 3 layers (static source checks).

- No access to injected-event ground truth, the generator, or its hidden customer-health mechanism.
- No database driver: every query goes through Phase 2 (``KPIService`` / ``mrr_series``).
- No later-phase frameworks (agent, LLM, MCP, API, UI), no network clients, no ML black boxes.
- Deterministic and point-in-time: no system clock for business dates, no random numbers, no
  random train/test splitting.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT

PACKAGES = ("timeseries", "forecasting", "anomalies")
SOURCES = sorted(p for name in PACKAGES for p in (PROJECT_ROOT / "app" / name).rglob("*.py"))


def _code(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _id(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT / "app"))


def test_every_package_has_sources() -> None:
    for name in PACKAGES:
        assert (PROJECT_ROOT / "app" / name / "__init__.py").exists()


@pytest.mark.parametrize("path", SOURCES, ids=_id)
def test_no_ground_truth_or_generator_access(path: Path) -> None:
    code = _code(path)
    assert "injected_events" not in code
    assert "ground_truth" not in code
    assert not re.search(r"^\s*(from|import)\s+data(\.|\s)", code, re.MULTILINE), "must not import the generator"
    assert not re.search(r"\b(health_score|engagement|latent|api_intensity|stage_factor)\b", code)
    assert not re.search(r"\b(Singapore|Enterprise|2026-0[6-8])\b", code), "no hard-coded event details"


@pytest.mark.parametrize("path", SOURCES, ids=_id)
def test_no_driver_later_phase_or_network_dependencies(path: Path) -> None:
    imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", _code(path), re.MULTILINE)
    forbidden = {
        "duckdb",
        "sqlalchemy",
        "psycopg",
        "langgraph",
        "langchain",
        "anthropic",
        "openai",
        "mcp",
        "fastapi",
        "starlette",
        "streamlit",
        "requests",
        "httpx",
        "urllib",
        "socket",
        "sklearn",
        "torch",
        "tensorflow",
        "prophet",
        "random",
        "subprocess",
        "pickle",
    }
    for module in imports:
        assert module.split(".")[0] not in forbidden, f"{path.name} imports {module}"


@pytest.mark.parametrize("path", SOURCES, ids=_id)
def test_deterministic_and_point_in_time(path: Path) -> None:
    code = _code(path)
    assert "date.today(" not in code and "datetime.now(" not in code and "datetime.today(" not in code
    assert "np.random" not in code and "default_rng" not in code
    assert "shuffle" not in code and "train_test_split" not in code
    assert "eval(" not in code and "exec(" not in code


def test_only_the_series_preparation_reads_the_database() -> None:
    calls = r"\b(QueryRunner\(|calculate_kpi\(|mrr_series\(|\.query\(|\.execute\()"
    readers = [p for p in SOURCES if re.search(calls, _code(p))]
    assert {_id(p) for p in readers} == {"timeseries/preparation.py"}
