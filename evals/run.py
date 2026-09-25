"""Command line: ``python -m evals.run``.

Examples::

    python -m evals.run                                   # full eval_v1 benchmark, deterministic mode
    python -m evals.run --suite critical                  # the critical regression suite (fast)
    python -m evals.run --category prompt_injection       # one category
    python -m evals.run --scenario kpi_revenue_last_month # one scenario
    python -m evals.run --seed 7                          # on a dataset generated for seed 7
    python -m evals.run --multi-seed 7,2027               # plus the multi-seed subset on other seeds
    python -m evals.run --list                            # list scenarios

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
from evals.reports.writer import write_reports
from evals.scenarios.loader import DEFAULT_DATASET, load_dataset, select
from evals.scenarios.model import Category


def _seeds(text: str) -> list[int]:
    return [int(s) for s in text.split(",") if s.strip()]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m evals.run", description="AgentOps evaluation and benchmark suite.")
    p.add_argument("--mode", choices=["deterministic", "llm"], default="deterministic", help="agent model mode")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="scenario dataset version (default: %(default)s)")
    p.add_argument("--suite", choices=["full", "critical", "multi_seed"], default="full")
    p.add_argument("--category", action="append", default=[], choices=[c.value for c in Category])
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
    if args.list:
        dataset = load_dataset(args.dataset)
        for s in select(dataset.scenarios, suite=suite, categories=args.category, scenario_ids=args.scenario):
            suites = ",".join(s.suites) or "-"
            print(f"{s.scenario_id}\t{s.category.value}\t{s.difficulty.value}\t{s.mode.value}\t{suites}")
        return 0
    options = RunOptions(
        mode=args.mode,
        dataset=args.dataset,
        suite=suite,
        categories=args.category,
        scenario_ids=args.scenario,
        seed=args.seed,
        multi_seeds=args.multi_seed,
        judge_model=args.judge_model,
    )
    try:
        summary = run_benchmark(options)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Evaluation could not run: {exc}", file=sys.stderr)
        return 2
    json_path, md_path = write_reports(summary, args.output)
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
    for check in summary.thresholds:
        if not check.passed:
            print(f"  THRESHOLD FAILED: {check.name} {check.comparison} {check.threshold} (actual {check.actual})")
    print(f"Reports: {json_path}\n         {md_path}")
    return 0 if summary.thresholds_passed else 1


if __name__ == "__main__":
    sys.exit(main())
