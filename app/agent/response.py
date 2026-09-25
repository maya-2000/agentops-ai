"""The user-facing response and its deterministic parts (caveats, evidence references, trace, failure texts).

The language model only words the answer, findings, interpretation and recommendations (each
citing claim IDs). Caveats, assumptions, evidence references and the tool trace are attached
deterministically, so limitations can never be dropped by a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from pydantic import BaseModel, Field

from app.agent.records import AgentStatus, ToolCallRecord
from app.evidence.models import EvidenceGraph
from app.llm.schemas import DraftItemOutput, ResponseDraftOutput


def scope_text(coverage: tuple[date, date] | None) -> str:
    window = f" from {coverage[0]:%Y-%m} to {coverage[1]:%Y-%m}" if coverage else ""
    return (
        "I can only analyse the Northwind Cloud business dataset (customers, subscriptions, revenue, usage, "
        f"sales, marketing, support and product data{window}) with the supported KPIs, analytics, forecasts and "
        "anomaly checks."
    )


LIMIT_MESSAGE = "Investigation limit reached before sufficient evidence could be collected."
LIMIT_CAVEAT = "The tool-call limit was reached, so no further drill-down was performed."
MAX_CAVEATS = 6


class ResponseItem(BaseModel):
    text: str
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class EvidenceReference(BaseModel):
    evidence_id: str
    evidence_type: str
    statement: str
    tool_name: str
    query_id: str | None
    source_tables: list[str]
    period: str | None


class ToolTraceEntry(BaseModel):
    call_id: str
    tool_name: str
    success: bool
    status: str
    result_summary: str
    query_id: str | None
    evidence_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class AgentResponse(BaseModel):
    status: AgentStatus
    answer: str
    answer_claim_ids: list[str] = Field(default_factory=list)
    answer_evidence_ids: list[str] = Field(default_factory=list)
    key_findings: list[ResponseItem] = Field(default_factory=list)
    interpretation: list[ResponseItem] = Field(default_factory=list)
    recommendations: list[ResponseItem] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    tool_trace: list[ToolTraceEntry] = Field(default_factory=list)
    generated_by: str


def _item(item: DraftItemOutput, graph: EvidenceGraph) -> ResponseItem:
    evidence_ids = [e for c in item.claim_ids if c in graph.claims for e in graph.claims[c].evidence_ids]
    return ResponseItem(text=item.text, claim_ids=list(item.claim_ids), evidence_ids=list(dict.fromkeys(evidence_ids)))


def trace_entries(calls: Sequence[ToolCallRecord]) -> list[ToolTraceEntry]:
    return [
        ToolTraceEntry(
            call_id=c.call_id,
            tool_name=c.tool_name,
            success=c.success,
            status=c.status,
            result_summary=c.result_summary,
            query_id=c.query_id,
            evidence_ids=c.evidence_ids,
            error=c.error.message if c.error else None,
        )
        for c in calls
    ]


def evidence_references(graph: EvidenceGraph, evidence_ids: Sequence[str]) -> list[EvidenceReference]:
    refs = []
    for evidence_id in dict.fromkeys(evidence_ids):
        e = graph.evidence.get(evidence_id)
        if e is None:
            continue
        refs.append(
            EvidenceReference(
                evidence_id=e.evidence_id,
                evidence_type=e.evidence_type,
                statement=e.statement,
                tool_name=e.tool_name,
                query_id=e.query_id,
                source_tables=e.source_tables,
                period=e.period_label,
            )
        )
    return refs


def build_caveats(
    graph: EvidenceGraph,
    cited_claim_ids: Sequence[str],
    *,
    warnings: Sequence[str] = (),
    limit_reached: bool = False,
    causal_question: bool = False,
) -> list[str]:
    """Deterministic caveats from the cited evidence and the run itself (never dropped by the model)."""
    caveats: list[str] = []
    if causal_question:
        caveats.append("The analysis shows associations and contributions; it does not establish causes.")
    if limit_reached:
        caveats.append(LIMIT_CAVEAT)
    caveats.extend(w for w in warnings if w.startswith("Tool call failed"))
    for claim_id in cited_claim_ids:
        claim = graph.claims.get(claim_id)
        if claim is None:
            continue
        caveats.extend(claim.limitations)
        for e in graph.get_supporting_evidence(claim_id):
            if e.truncated:
                caveats.append("A query result was truncated at the row limit; its rows are not complete.")
            if e.evidence_type == "forecast":
                coverage = e.details.get("interval_coverage")
                caveats.append(
                    "Forecasts are estimates from historical patterns and cannot anticipate new events"
                    + (
                        f"; historically the prediction intervals contained {coverage:.0%} of actual values."
                        if isinstance(coverage, float)
                        else "."
                    )
                )
            if e.evidence_type == "anomaly":
                caveats.append(
                    "An anomaly flag means a movement was statistically unusual against recent months; it does not "
                    "explain the cause, and its direction is not a judgement of good or bad."
                )
            if e.details.get("association_only"):
                caveats.append("Usage and ticket comparisons before churn are associations, not causes.")
    return list(dict.fromkeys(caveats))[:MAX_CAVEATS]


def build_response(
    draft: ResponseDraftOutput,
    graph: EvidenceGraph,
    *,
    status: AgentStatus,
    calls: Sequence[ToolCallRecord],
    assumptions: Sequence[str],
    caveats: Sequence[str],
    generated_by: str,
) -> AgentResponse:
    items = [DraftItemOutput(text=draft.answer, claim_ids=draft.answer_claim_ids), *draft.key_findings]
    items += [*draft.interpretation, *draft.recommendations]
    cited = [e for item in items for c in item.claim_ids if c in graph.claims for e in graph.claims[c].evidence_ids]
    answer = _item(items[0], graph)
    return AgentResponse(
        status=status,
        answer=draft.answer,
        answer_claim_ids=answer.claim_ids,
        answer_evidence_ids=answer.evidence_ids,
        key_findings=[_item(i, graph) for i in draft.key_findings],
        interpretation=[_item(i, graph) for i in draft.interpretation],
        recommendations=[_item(i, graph) for i in draft.recommendations],
        caveats=list(caveats),
        assumptions=list(assumptions),
        evidence=evidence_references(graph, cited),
        tool_trace=trace_entries(calls),
        generated_by=generated_by,
    )


def failure_response(
    status: AgentStatus,
    message: str,
    *,
    calls: Sequence[ToolCallRecord] = (),
    assumptions: Sequence[str] = (),
    graph: EvidenceGraph | None = None,
    findings: Sequence[DraftItemOutput] = (),
    caveats: Sequence[str] = (),
    coverage: tuple[date, date] | None = None,
) -> AgentResponse:
    """A deterministic response for the failure paths. It never contains unsupported numbers."""
    answers = {
        "unsupported_request": f"{scope_text(coverage)} {message}",
        "insufficient_evidence": f"The available data is insufficient to answer this. {message}",
        "tool_error": f"The analysis could not be completed because required tools failed. {message}",
        "validation_failure": (
            "The generated explanation could not be validated against the evidence, so only the validated "
            f"findings are shown. {message}"
        ),
        "planning_failure": f"The question could not be turned into a valid analysis plan. {message}",
    }
    graph = graph or EvidenceGraph()
    items = [_item(i, graph) for i in findings]
    cited = [e for i in items for e in i.evidence_ids]
    return AgentResponse(
        status=status,
        answer=LIMIT_MESSAGE if message == LIMIT_MESSAGE else answers.get(status, message).strip(),
        key_findings=items,
        caveats=list(caveats),
        assumptions=list(assumptions),
        evidence=evidence_references(graph, cited),
        tool_trace=trace_entries(calls),
        generated_by="template",
    )
