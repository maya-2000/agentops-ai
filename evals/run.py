"""Command line: ``python -m evals.run``.

Examples::

    python -m evals.run                                   # full eval_v1 benchmark, deterministic mode
    python -m evals.run --suite critical                  # the critical regression suite (fast)
    python -m evals.run --category prompt_injection       # one category
    python -m evals.run --scenario kpi_revenue_last_month # one scenario
    python -m evals.run --seed 7                          # on a dataset generated for seed 7
    python -m evals.run --multi-seed 7,2027               # plus the multi-seed subset on other seeds
    python -m evals.run --list                            # list scenarios
    python -m evals.run --dataset eval_v2                 # the investigation benchmark (Phase 10)
    python -m evals.run --dataset eval_v2 --multi-seed 7,2027

Exit codes: 0 = regression thresholds pass, 1 = a threshold failed, 2 = invalid invocation or setup.
Reports are written to ``--output`` (default ``reports/evaluation``, git-ignored) as JSON and Markdown.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.config import PROJECT_ROOT
from evals.benchmark import RunOptions, run_benchmark
from evals.reports.models import EvaluationRunSummary
from evals.reports.writer import write_reports
from evals.scenarios.investigation import (
    InvestigationCategory,
    is_investigation_dataset,
    load_investigation_dataset,
    select_investigations,
)
from evals.scenarios.loader import DEFAULT_DATASET, load_dataset, select
from evals.scenarios.model import Category

# eval_v1 and the investigation benchmark (eval_v2) have their own categories; a name may be in both.
CATEGORIES = sorted({c.value for c in Category} | {c.value for c in InvestigationCategory})


def _seeds(text: str) -> list[int]:
    return [int(s) for s in text.split(",") if s.strip()]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m evals.run", description="AgentOps evaluation and benchmark suite.")
    p.add_argument("--mode", choices=["deterministic", "llm"], default="deterministic", help="agent model mode")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="scenario dataset version (default: %(default)s)")
    p.add_argument("--suite", choices=["full", "critical", "multi_seed"], default="full")
    p.add_argument("--category", action="append", default=[], choices=CATEGORIES)
    p.add_argument("--scenario", action="append", default=[], help="scenario ID (repeatable)")
    p.add_argument(
        "--seed", type=int, help="evaluate on a dataset generated for this seed (default: the repository database)"
    )
    p.add_argument(
        "--multi-seed", type=_seeds, default=[], help="comma-separated extra seeds for the multi-seed subset"
    )
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "evaluation", help="report directory")
    p.add_argument(
        "--judge-model", help="optional LLM judge model for clarity/relevance (needs a key; never affects pass/fail)"
    )
    p.add_argument("--list", action="store_true", help="list the selected scenarios and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(message)s")
    logging.getLogger("sqlglot").setLevel(logging.ERROR)  # parser notes on the SQL attack strings
    suite = None if args.suite == "full" else args.suite
    if is_investigation_dataset(args.dataset):
        return _investigations(args, suite)
    try:
        if args.list:
            dataset = load_dataset(args.dataset)
            for s in select(dataset.scenarios, suite=suite, categories=args.category, scenario_ids=args.scenario):
                suites = ",".join(s.suites) or "-"
                print(f"{s.scenario_id}\t{s.category.value}\t{s.difficulty.value}\t{s.mode.value}\t{suites}")
            return 0
        summary = run_benchmark(_options(args, suite))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Evaluation could not run: {exc}", file=sys.stderr)
        return 2
    json_path, md_path = write_reports(summary, args.output)
    return _report(summary, json_path, md_path)


def _options(args: argparse.Namespace, suite: str | None) -> RunOptions:
    return RunOptions(
        mode=args.mode,
        dataset=args.dataset,
        suite=suite,
        categories=args.category,
        scenario_ids=args.scenario,
        seed=args.seed,
        multi_seeds=args.multi_seed,
        judge_model=args.judge_model,
    )


def _investigations(args: argparse.Namespace, suite: str | None) -> int:
    """The investigation benchmark (eval_v2): its own scenarios, graders, metrics and report."""
    from evals.investigation_benchmark import run_investigation_benchmark
    from evals.reports.investigation import write_investigation_reports

    try:
        if args.list:
            dataset = load_investigation_dataset(args.dataset)
            chosen = select_investigations(
                dataset.scenarios, suite=suite, categories=args.category, scenario_ids=args.scenario
            )
            for s in chosen:
                suites = ",".join(s.suites) or "-"
                print(f"{s.scenario_id}\t{s.category.value}\t{s.difficulty.value}\t{s.mode.value}\t{suites}")
            return 0
        summary = run_investigation_benchmark(_options(args, suite))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Evaluation could not run: {exc}", file=sys.stderr)
        return 2
    json_path, md_path = write_investigation_reports(summary, args.output)
    m = summary.metrics
    counts = f"{summary.passed}/{summary.total_scenarios} passed ({summary.failed} failed, {summary.errors} errors)"
    print(f"{summary.run_id}: {counts}")
    for name in (
        "evidence_completeness",
        "driver_correctness",
        "recommendation_grounding",
        "causal_safety",
        "budget_compliance",
    ):
        value = m.get(name)
        print(f"  {name}: {'n/a' if value is None else f'{value:.4f}'}")
    if summary.multi_seed:
        passed = sum(row["status"] == "passed" for row in summary.multi_seed)
        print(f"  multi-seed: {passed}/{len(summary.multi_seed)} passed")
    return _thresholds(summary, json_path, md_path)


def _report(summary: EvaluationRunSummary, json_path: Path, md_path: Path) -> int:
    m = summary.metrics
    counts = f"{summary.passed}/{summary.total_scenarios} passed ({summary.failed} failed, {summary.errors} errors)"
    print(f"{summary.run_id}: {counts}")
    for name in (
        "numerical_accuracy",
        "evidence_grounding_rate",
        "hallucination_rate",
        "security_block_rate",
        "mcp_parity_rate",
    ):
        value = m.get(name)
        print(f"  {name}: {'n/a' if value is None else f'{value:.4f}'}")
    return _thresholds(summary, json_path, md_path)


def _thresholds(summary: EvaluationRunSummary, json_path: Path, md_path: Path) -> int:
    for check in summary.thresholds:
        if not check.passed:
            print(f"  THRESHOLD FAILED: {check.name} {check.comparison} {check.threshold} (actual {check.actual})")
    print(f"Reports: {json_path}\n         {md_path}")
    return 0 if summary.thresholds_passed else 1


if __name__ == "__main__":
    sys.exit(main())
