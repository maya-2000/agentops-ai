"""Helpers for the Phase 5 security tests: scripted agent runners, failing tools, plans and log capture."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from app.agent import AgentConfig, AgentRunner
from app.llm import ScriptedLLM
from app.llm.deterministic import DeterministicLLM
from app.tools import TOOL_DEFINITIONS, ToolRegistry
from tests.phase4_support import AS_OF

AUGUST = {"start_date": "2026-08-01", "end_date": "2026-08-31"}
JULY_COMPARISON = {"comparison_start_date": "2026-07-01", "comparison_end_date": "2026-07-31"}


def runner(
    db: Any,
    script: dict[Any, list[Any]] | None = None,
    registry: ToolRegistry | None = None,
    **config: Any,
) -> tuple[AgentRunner, ScriptedLLM]:
    """An agent whose model steps replay ``script`` and otherwise use the deterministic model."""
    llm = ScriptedLLM(script or {}, fallback=DeterministicLLM())
    return AgentRunner(db, llm=llm, config=AgentConfig(**config), registry=registry, as_of=AS_OF), llm


def plan(*steps: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "steps": [{"tool_name": t, "arguments_json": json.dumps(a), "purpose": "test"} for t, a in steps],
        "rationale": "scripted",
        "sufficient": False,
    }


def with_handler(tool: str, handler: Any) -> ToolRegistry:
    """The real registry with one tool's handler replaced."""
    return ToolRegistry(tuple(replace(d, handler=handler) if d.name == tool else d for d in TOOL_DEFINITIONS))


def raising(exc: Exception) -> Any:
    def handler(*_: Any) -> Any:
        raise exc

    return handler


def understanding(**fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "intent": "kpi_lookup",
        "metric": "revenue",
        "period": "last_month",
        "comparison_period": None,
        "dimensions": [],
        "filters": [],
        "horizon": None,
        "analysis_type": "value",
        "confidence": 0.9,
        "ambiguities": [],
        "material_ambiguity": False,
        "unsupported_reason": None,
    }
    base.update(fields)
    return base
