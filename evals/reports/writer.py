"""The machine-readable (JSON) and human-readable (Markdown) reports of a benchmark run.

Reports hold scenario IDs, scores, failures with expected/actual values, tool traces (tool names
and arguments) and security-event types. They never hold secrets, raw customer rows or hidden
labels. Answers are kept as short excerpts, and the canary secret and label text are checked
before writing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.graders.common import CANARY_SECRET
from evals.reports.models import EvaluationRunSummary


def write_reports(summary: EvaluationRunSummary, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{summary.run_id}.json"
    md_path = output_dir / f"{summary.run_id}.md"
    payload = summary.model_dump_json(indent=2)
    if CANARY_SECRET in payload:
        raise RuntimeError("Refusing to write a report that contains the canary secret")
    json_path.write_text(payload + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    return json_path, md_path


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}" if isinstance(value, float) else str(value)


def render_markdown(s: EvaluationRunSummary) -> str:
    m = s.metrics
    lines = [
        f"# AgentOps evaluation report: {s.run_id}",
        "",
        f"- **Dataset:** {s.dataset_version} (schema {s.dataset_schema_version}, updated {s.dataset_updated}), "
        f"{s.total_scenarios} scenarios",
        f"- **Data:** {s.data['origin']} dataset, seed {s.data['seed']}, business dataset {s.data['business_dataset_version']}, "
        f"as of {s.data['as_of']}",
        f"- **Mode:** {s.configuration['mode']} (provider {s.configuration['provider']}, model {s.configuration['model']}, "
        f"temperature {s.configuration['temperature']}); suite {s.configuration['suite']}",
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
    for name in (
        "intent_accuracy",
        "parameter_accuracy",
        "tool_selection_accuracy",
        "tool_execution_success",
        "numerical_accuracy",
        "evidence_grounding_rate",
        "claim_support_rate",
        "hallucination_rate",
        "unsupported_causal_claim_rate",
        "uncertainty_score",
        "refusal_precision",
        "refusal_recall",
        "false_refusal_rate",
        "security_block_rate",
        "data_exposure_score",
        "mcp_parity_rate",
        "mcp_score",
        "tool_efficiency_score",
        "unnecessary_tool_rate",
        "forbidden_tool_rate",
        "evidence_integrity_detection",
    ):
        lines.append(f"| {name.replace('_', ' ')} | {_pct(m.get(name))} |")
    lines += [
        f"| critical security failures | {_num(m.get('critical_security_failures'))} |",
        f"| data exposure failures | {_num(m.get('data_exposure_failures'))} |",
        "",
        "## Results by category",
        "",
        "| Category | Passed | Total | Pass rate |",
        "|---|---|---|---|",
    ]
    lines += [f"| {k} | {v['passed']} | {v['total']} | {_pct(v['pass_rate'])} |" for k, v in s.by_category.items()]
    lines += ["", "## Results by difficulty", "", "| Difficulty | Passed | Total | Pass rate |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['passed']} | {v['total']} | {_pct(v['pass_rate'])} |" for k, v in s.by_difficulty.items()]
    sec = s.security
    lines += [
        "",
        "## Security and data exposure",
        "",
        f"{sec['passed']} / {sec['scenarios']} security, prompt-injection, SQL-security and data-exposure scenarios passed; "
        f"{len(sec['security_errors'])} security errors, {len(sec['data_exposure_errors'])} data-exposure errors.",
        "",
    ]
    mcp = s.mcp
    lat = mcp["latency_ms"]
    lines += [
        "## MCP",
        "",
        f"{mcp['passed']} / {mcp['scenarios']} MCP scenarios passed. Direct-vs-MCP parity: {_pct(mcp['parity_rate'])} over "
        f"{mcp['parity_scenarios']} scenarios ({mcp['parity_calls']} calls). Discovery "
        f"{'passed' if mcp['discovery_passed'] else 'FAILED'}; shared execution path "
        f"{'verified' if mcp['shared_execution_passed'] else 'NOT verified'}. MCP call latency p50 {_num(lat['p50'])} ms, "
        f"p95 {_num(lat['p95'])} ms.",
        "",
        "## Latency (production execution only)",
        "",
        "| Class | Count | Mean ms | p50 ms | p95 ms | Max ms |",
        "|---|---|---|---|---|---|",
    ]
    for name, st in s.performance["production_latency_ms"].items():
        lines.append(
            f"| {name} | {st['count']} | {_num(st['mean'])} | {_num(st['p50'])} | {_num(st['p95'])} | {_num(st['max'])} |"
        )
    overhead = s.performance["evaluation_overhead_ms"]
    usage = s.performance["resource_usage"]
    lines += [
        "",
        f"Evaluation overhead (references and grading, not production): mean {_num(overhead['mean'])} ms, "
        f"p50 {_num(overhead['p50'])} ms per scenario. Resources: {usage['tool_calls']} tool calls, "
        f"{usage['failed_calls']} failed (most are expected security rejections), {usage['retries']} retries, "
        f"{usage['duplicate_calls']} duplicates, {usage['sql_calls']} executed SQL queries ({usage['sql_rows']} rows).",
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
            "| Seed | Scenario | Status | Numerical | Notes |",
            "|---|---|---|---|---|",
        ]
        for row in s.multi_seed:
            notes = "; ".join([*row["failures"], *row["not_applicable"]])[:160] or "-"
            lines.append(
                f"| {row['seed']} | {row['scenario_id']} | {row['status']} | {_pct(row['numerical'])} | {notes} |"
            )
    lines += ["", "## Failures", ""]
    counts = ", ".join(f"{k} {v}" for k, v in s.failure_counts.items()) or "none"
    lines += [f"By category: {counts}.", ""]
    for result in s.results:
        if result.passed:
            continue
        lines.append(f"### {result.scenario_id} ({result.category}, {result.difficulty}): {result.status}")
        lines.append("")
        for f in result.failures:
            detail = ""
            if f.expected is not None or f.actual is not None:
                detail = f" Expected `{json.dumps(f.expected, default=str)[:120]}`, actual `{json.dumps(f.actual, default=str)[:120]}`."
            tool = f" Tool: `{f.tool}`." if f.tool else ""
            lines.append(f"- **{f.category.value}** `{f.check}`: {f.message}{detail}{tool}")
        lines.append("")
    return "\n".join(lines) + "\n"
