"""The JSON and Markdown reports of an investigation benchmark run (eval_v2).

Same safety rules as the eval_v1 reports: scenario IDs, scores, failures with expected/actual values,
tool names and arguments, never secrets, customer rows or hidden labels. The canary secret is checked
before anything is written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.graders.common import CANARY_SECRET
from evals.metrics.investigation import RATE_METRICS
from evals.reports.models import EvaluationRunSummary


def write_investigation_reports(summary: EvaluationRunSummary, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{summary.run_id}.json"
    md_path = output_dir / f"{summary.run_id}.md"
    payload = summary.model_dump_json(indent=2)
    if CANARY_SECRET in payload:
        raise RuntimeError("Refusing to write a report that contains the canary secret")
    json_path.write_text(payload + "\n", encoding="utf-8")
    md_path.write_text(render_investigation_markdown(summary), encoding="utf-8")
    return json_path, md_path


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}" if isinstance(value, float) else str(value)


def render_investigation_markdown(s: EvaluationRunSummary) -> str:
    m = s.metrics
    lines = [
        f"# AgentOps investigation benchmark: {s.run_id}",
        "",
        f"- **Dataset:** {s.dataset_version} (schema {s.dataset_schema_version}, updated {s.dataset_updated}), "
        f"{s.total_scenarios} scenarios",
        f"- **Data:** {s.data['origin']} dataset, seed {s.data['seed']}, as of {s.data['as_of']}",
        f"- **Mode:** {s.configuration['mode']} (provider {s.configuration['provider']}); "
        f"suite {s.configuration['suite']}",
        f"- **Commit:** `{s.git_commit[:12]}`{' (uncommitted changes)' if s.git_dirty else ''}; run at {s.timestamp}",
        "",
        "Local prototype benchmark, not a production latency or reliability claim.",
        "",
        "## Overall",
        "",
        f"**{s.passed} / {s.total_scenarios} scenarios passed ({_pct(m.get('pass_rate'))})**, {s.failed} failed, "
        f"{s.errors} evaluation errors. Regression thresholds: **{'PASS' if s.thresholds_passed else 'FAIL'}**.",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {name.replace('_', ' ')} | {_pct(m.get(name))} |" for name in RATE_METRICS]
    for name in (
        "security_failures",
        "causal_failures",
        "duplicate_tool_calls",
        "budget_violations",
        "mean_tool_calls",
    ):
        lines.append(f"| {name.replace('_', ' ')} | {_num(m.get(name))} |")
    lines += ["", "## Results by category", "", "| Category | Passed | Total | Pass rate |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['passed']} | {v['total']} | {_pct(v['pass_rate'])} |" for k, v in s.by_category.items()]
    perf = s.performance
    lines += [
        "",
        "## Performance (production execution only)",
        "",
        "| Mode | Count | Mean ms | p50 ms | p95 ms | Max ms |",
    ]
    lines.append("|---|---|---|---|---|---|")
    for mode, st in perf["production_latency_ms"].items():
        values = [str(st["count"]), *(_num(st[k]) for k in ("mean", "p50", "p95", "max"))]
        lines.append(f"| {mode} | {' | '.join(values)} |")
    lines += ["", "| Investigation stage | Mean ms | p95 ms |", "|---|---|---|"]
    for stage, st in perf["investigation_stage_ms"].items():
        lines.append(f"| {stage.removesuffix('_ms')} | {_num(st['mean'])} | {_num(st['p95'])} |")
    calls, overhead = perf["tool_calls"], perf["api_overhead_ms"]
    lines += [
        "",
        f"Tool calls per investigation: mean {_num(calls['mean'])}, max {_num(calls['max'])}. API overhead over the "
        f"investigation itself: mean {_num(overhead['mean'])} ms, p95 {_num(overhead['p95'])} ms.",
        "",
        "## Regression thresholds",
        "",
        "| Metric | Rule | Actual | Result |",
        "|---|---|---|---|",
    ]
    for t in s.thresholds:
        actual = "n/a" if t.actual is None else f"{t.actual:.4g}"
        lines.append(f"| {t.name} | {t.comparison} {t.threshold:g} | {actual} | {'pass' if t.passed else '**FAIL**'} |")
    if s.multi_seed:
        lines += [
            "",
            "## Multi-seed robustness",
            "",
            "| Seed | Scenario | Result | Status | Notes |",
            "|---|---|---|---|---|",
        ]
        for row in s.multi_seed:
            notes = "; ".join(row["failures"])[:160] or "-"
            lines.append(
                f"| {row['seed']} | {row['scenario_id']} | {row['status']} | {row['agent_status']} | {notes} |"
            )
    lines += ["", "## Failures", ""]
    counts = ", ".join(f"{k} {v}" for k, v in s.failure_counts.items()) or "none"
    lines += [f"By category: {counts}.", ""]
    for result in s.results:
        if result.passed:
            continue
        lines += [f"### {result.scenario_id} ({result.category}, {result.mode}): {result.status}", ""]
        for f in result.failures:
            detail = ""
            if f.expected is not None or f.actual is not None:
                detail = (
                    f" Expected `{json.dumps(f.expected, default=str)[:120]}`, "
                    f"actual `{json.dumps(f.actual, default=str)[:120]}`."
                )
            lines.append(f"- **{f.category.value}** `{f.check}`: {f.message}{detail}")
        lines.append("")
    return "\n".join(lines) + "\n"
