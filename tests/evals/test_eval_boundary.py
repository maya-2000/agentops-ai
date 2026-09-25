"""Static boundaries of the evaluation layer.

- Production (``app/``, ``data/``) never imports the evaluation framework, and ``evals`` is not
  part of the installable package.
- The evaluation never takes expected values from the system under test. It does not import the
  production calculation code (analytics, time series, forecasting, anomalies, tool handlers).
- Only ``evals/reference`` reads the hidden ground truth. The runners, the only evaluation modules
  that call production code, receive nothing from the evaluation context except the read-only
  database handle and the as-of date.
- The production code contains no hidden-label text.
- The evaluation adds no second execution pipeline: it never calls a tool registry or a tool
  handler directly.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

from app.config import PROJECT_ROOT
from evals.reference.context import REPOSITORY_GROUND_TRUTH
from evals.reference.labels import HiddenLabels

APP = PROJECT_ROOT / "app"
EVALS = PROJECT_ROOT / "evals"
PRODUCTION = [*APP.rglob("*.py"), *(PROJECT_ROOT / "data").rglob("*.py")]
EVAL_SOURCES = sorted(EVALS.rglob("*.py"))
CALCULATION_PACKAGES = ("app.analytics", "app.timeseries", "app.forecasting", "app.anomalies", "app.tools.handlers")


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def _rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def test_production_never_imports_the_evaluation_framework() -> None:
    assert PRODUCTION
    offenders = [_rel(p) for p in PRODUCTION if any(m.split(".")[0] == "evals" for m in _imports(p))]
    assert not offenders, offenders
    for path in PRODUCTION:
        text = path.read_text(encoding="utf-8")
        assert "evals." not in text and "import evals" not in text, _rel(path)


def test_evals_is_not_shipped_with_the_application() -> None:
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    include = config["tool"]["setuptools"]["packages"]["find"]["include"]
    assert not [p for p in include if p.startswith("evals")]
    assert not (APP / "evaluation").exists() and not (APP / "evals").exists()


def test_expected_values_never_come_from_the_system_under_test() -> None:
    for path in EVAL_SOURCES:
        imported = _imports(path)
        bad = sorted(m for m in imported if m.startswith(CALCULATION_PACKAGES))
        assert not bad, (_rel(path), bad)


def test_only_evals_reference_reads_the_hidden_ground_truth() -> None:
    readers, label_importers = [], []
    for path in EVAL_SOURCES:
        text = path.read_text(encoding="utf-8")
        if "HiddenLabels.load(" in text or "injected_events.json" in text:
            readers.append(_rel(path))
        if any(m.startswith("evals.reference.labels") for m in _imports(path)):
            label_importers.append(_rel(path))
    assert readers and all(r.startswith("evals/reference/") for r in readers), readers
    assert all(i.startswith("evals/reference/") for i in label_importers), label_importers
    # The file itself is opened in exactly one place.
    openers = [_rel(p) for p in EVAL_SOURCES if ".read_text(" in p.read_text(encoding="utf-8")]
    assert "evals/reference/labels.py" in openers
    assert not [o for o in openers if o.startswith(("evals/runners/", "evals/graders/"))], openers


def test_runners_hand_production_only_the_database_and_the_as_of_date() -> None:
    """The runners are the only evaluation code that calls production; labels and references never reach it."""
    runners = sorted((EVALS / "runners").glob("*.py"))
    assert runners
    for path in runners:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        used = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "ctx"
        }
        assert used <= {"db", "as_of"}, (_rel(path), used)
        text = path.read_text(encoding="utf-8")
        for forbidden in ("labels", "withheld_values", "reference_kpis", "reference_timeseries", "ground_truth"):
            assert forbidden not in text, (_rel(path), forbidden)


def test_production_holds_no_hidden_label_text() -> None:
    labels = HiddenLabels.load(REPOSITORY_GROUND_TRUTH)
    specific = [m for m in labels.leak_markers() if " " in m]  # event names, descriptions, expected signals
    answers = {labels.observable("E3").campaign_id, labels.observable("E4").sales_rep}
    assert specific and all(answers)
    for path in APP.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in specific:
            assert marker not in text, (_rel(path), marker[:60])
        for answer in answers:
            assert str(answer) not in text, (_rel(path), answer)


def test_the_evaluation_adds_no_second_execution_pipeline() -> None:
    """Tools run only through the production entry points (AgentRunner, AgentRuntime.executor, the MCP server)."""
    for path in EVAL_SOURCES:
        text = path.read_text(encoding="utf-8")
        for bypass in ("registry.execute(", ".handler(", "ToolRequest(", "calculate_kpi(", "handlers."):
            assert bypass not in text, (_rel(path), bypass)
    runners = (EVALS / "runners" / "tools.py").read_text(encoding="utf-8")
    assert "runtime.executor.execute(" in runners and "create_server(" in runners
