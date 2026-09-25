"""The command line (``python -m evals.run``): selection flags, exit codes and an end-to-end run."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals import benchmark
from evals.reference.context import DataSource
from evals.reports.models import EvaluationResult, EvaluationRunSummary
from evals.run import main
from evals.scenarios.loader import select
from evals.scenarios.model import EvaluationDataset
from tests.phase7_support import fingerprint


def test_cli_lists_the_selection(capsys: pytest.CaptureFixture[str], eval_dataset: EvaluationDataset) -> None:
    assert main(["--list", "--suite", "critical"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == len(select(eval_dataset.scenarios, suite="critical"))
    assert main(["--list", "--category", "sql_security", "--scenario", "sql_valid_select"]) == 0
    assert capsys.readouterr().out.startswith("sql_valid_select\tsql_security")


def test_cli_rejects_invalid_selections(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--category", "not_a_category"])
    assert exit_info.value.code == 2
    assert main(["--scenario", "does_not_exist", "--output", str(tmp_path)]) == 2
    assert "Unknown scenarios" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


@pytest.mark.slow
def test_cli_runs_the_critical_suite_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    eval_source: DataSource,
    critical_results: list[EvaluationResult],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(benchmark, "data_source", lambda seed: eval_source)
    assert main(["--suite", "critical", "--output", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "THRESHOLD FAILED" not in out and f"{len(critical_results)}/{len(critical_results)} passed" in out
    (report,) = tmp_path.glob("EVAL-*.json")
    written = EvaluationRunSummary.model_validate_json(report.read_text(encoding="utf-8"))
    assert fingerprint(written.results) == fingerprint(critical_results)
    assert main(["--category", "sql_security", "--output", str(tmp_path)]) == 0
