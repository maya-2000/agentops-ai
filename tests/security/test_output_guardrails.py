"""Output guardrails: tool outputs, evidence integrity, claim integrity and the wording of the response."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest

from app.evidence import EvidenceGraph, validate_evidence, validate_response
from app.evidence.builder import build_evidence
from app.llm import LLMTask
from app.llm.schemas import DraftItemOutput, ResponseDraftOutput
from app.security.data_policy import default_exposure_policy
from app.security.limits import SecurityLimits
from app.security.output_guard import ToolOutputValidator
from app.tools import ToolContext, ToolRegistry, ToolRequest
from tests.phase4_support import assertion, claim, evidence, graph
from tests.phase5_support import runner

VALIDATOR = ToolOutputValidator(SecurityLimits(), default_exposure_policy())


@pytest.fixture(scope="module")
def outputs(full_db: Any) -> dict[str, Any]:
    """Real outputs of each tool on the full dataset (forecasts and anomalies need its 24 months)."""
    ctx = ToolContext(db=full_db, as_of=date(2026, 8, 31), sql_row_limit=20)
    registry = ToolRegistry()
    calls = {
        "kpi": ("get_kpi", {"kpi": "revenue", "period": "last_month"}),
        "forecast": ("forecast_metric", {"metric": "revenue", "horizon": 2}),
        "anomaly": ("detect_anomalies", {"metric": "revenue"}),
        "sql": ("run_safe_sql", {"sql": "SELECT segment, COUNT(*) AS n FROM customers GROUP BY segment"}),
        "risk": ("get_customer_risk", {"limit": 5}),
        "reps": ("analyze_sales", {"operation": "rep_performance"}),
        "revenue": ("analyze_revenue", {"operation": "revenue_change"}),
    }
    return {
        k: registry.execute(ToolRequest(call_id="T1", tool_name=t, arguments=a), ctx) for k, (t, a) in calls.items()
    }


def _codes(result: Any) -> list[str]:
    return [v.code for v in VALIDATOR.validate(result)]


# ---- tool outputs ----------------------------------------------------------------------------------------------


def test_real_tool_outputs_pass(outputs: dict[str, Any]) -> None:
    for name, result in outputs.items():
        assert result.success, name
        assert not VALIDATOR.validate(result), (name, VALIDATOR.validate(result))


def test_wrong_type_and_missing_provenance_are_rejected(outputs: dict[str, Any]) -> None:
    kpi = outputs["kpi"]
    assert _codes(kpi.model_copy(update={"tool_name": "forecast_metric"})) == ["invalid_tool_output"]
    assert _codes(kpi.model_copy(update={"tool_name": "shell"})) == ["invalid_tool_output"]
    assert "invalid_tool_output" in _codes(kpi.model_copy(update={"query_ids": []}))
    assert "invalid_tool_output" in _codes(kpi.model_copy(update={"calculation": None}))
    assert "data_policy" in _codes(kpi.model_copy(update={"source_tables": ["customers", "injected_events"]}))


def test_non_finite_values_bad_dates_and_unknown_identifiers_are_rejected(outputs: dict[str, Any]) -> None:
    kpi = outputs["kpi"]
    nan = kpi.model_copy(update={"result": kpi.result.model_copy(update={"value": float("nan")})})
    assert "invalid_tool_output" in _codes(nan)
    period = kpi.result.period.model_copy(update={"start": date(1500, 1, 1)})
    old = kpi.model_copy(update={"result": kpi.result.model_copy(update={"period": period})})
    assert "invalid_tool_output" in _codes(old)
    unknown = kpi.model_copy(update={"result": kpi.result.model_copy(update={"key": "stock_price"})})
    assert "invalid_tool_output" in _codes(unknown)


def test_forecast_and_anomaly_semantics(outputs: dict[str, Any]) -> None:
    forecast = outputs["forecast"]
    no_model = forecast.model_copy(update={"result": forecast.result.model_copy(update={"model": None})})
    assert "invalid_tool_output" in _codes(no_model)
    early = forecast.result.forecast_points[0].model_copy(update={"start": forecast.result.cutoff_date})
    before_cutoff = forecast.result.model_copy(
        update={"forecast_points": [early, *forecast.result.forecast_points[1:]]}
    )
    assert "invalid_tool_output" in _codes(forecast.model_copy(update={"result": before_cutoff}))
    anomaly = outputs["anomaly"]
    bad_detector = anomaly.result.model_copy(update={"detector": "magic"})
    assert "invalid_tool_output" in _codes(anomaly.model_copy(update={"result": bad_detector}))


def test_sql_output_consistency_and_exposure(outputs: dict[str, Any]) -> None:
    sql = outputs["sql"]
    lying = sql.result.model_copy(update={"row_count": 999})
    assert "invalid_tool_output" in _codes(sql.model_copy(update={"result": lying}))
    false_truncation = sql.result.model_copy(update={"truncated": True})
    assert "invalid_tool_output" in _codes(sql.model_copy(update={"result": false_truncation}))
    leaked = sql.result.model_copy(update={"columns": ["company_name", "n"]})
    assert "data_policy" in _codes(sql.model_copy(update={"result": leaked}))
    hidden = sql.result.model_copy(update={"columns": ["health_score", "n"]})
    assert "data_policy" in _codes(sql.model_copy(update={"result": hidden}))


def test_pii_and_customer_level_exposure(outputs: dict[str, Any]) -> None:
    assert not _codes(outputs["reps"])  # the one operation allowed to return rep names
    moved = outputs["reps"].model_copy(update={"tool_name": "analyze_marketing"})
    assert "data_policy" in _codes(moved)  # the same field from any other operation is refused
    risk = outputs["risk"]
    too_many = risk.result.model_copy(update={"data": risk.result.data * 10})
    assert "data_policy" in _codes(risk.model_copy(update={"result": too_many}))


def test_failed_results_carry_no_output() -> None:
    assert not VALIDATOR.validate(
        ToolRegistry().execute(
            ToolRequest(call_id="T1", tool_name="nope"),
            ToolContext(db=None, as_of=date(2026, 8, 31), sql_row_limit=5),  # type: ignore[arg-type]
        )
    )


# ---- evidence integrity ------------------------------------------------------------------------------------------


def test_evidence_is_sealed_and_tampering_is_detected(outputs: dict[str, Any]) -> None:
    g = EvidenceGraph()
    items = build_evidence(outputs["kpi"], g)
    assert items and all(e.fingerprint == e.compute_fingerprint() for e in items)
    assert items[0].input_arguments == outputs["kpi"].arguments  # which inputs produced it
    assert not g.verify_integrity()
    g.add_claim(claim("C1", items[0].statement, [items[0].evidence_id], claim_type="observed_fact"))
    items[0].value = 1.0  # tampered after creation
    assert g.verify_integrity() == [items[0].evidence_id]
    result = validate_evidence(g, successful_call_ids=["T1"])
    assert not result.valid and any("fingerprint" in e for e in result.errors)


def test_only_the_evidence_layer_assigns_ids() -> None:
    g = EvidenceGraph()
    for bad in ("X1", "E0", "E1; DROP", "evidence-1", "E1234567"):
        with pytest.raises(ValueError):
            g.add_evidence(evidence(bad))
    forged = evidence("E1")
    forged.fingerprint = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        g.add_evidence(forged)
    g.add_evidence(evidence("E1"))
    with pytest.raises(KeyError):
        g.add_claim(claim("C1", "Revenue was SGD 100.", ["E7"]))  # a claim cannot cite invented evidence


def test_fingerprints_are_deterministic() -> None:
    assert evidence("E1").compute_fingerprint() == evidence("E1").compute_fingerprint()
    assert evidence("E1").compute_fingerprint() != evidence("E1", 101.0).compute_fingerprint()


def test_forecast_and_anomaly_evidence_must_keep_their_provenance() -> None:
    forecast = evidence("E1", evidence_type="forecast", details={"model": "drift"}, period_start=date(2026, 9, 1))
    g = graph(forecast, claims=(claim("C1", "Forecast of revenue.", ["E1"], claim_type="calculated_result"),))
    assert any("forecast without" in e for e in validate_evidence(g, successful_call_ids=["T1"]).errors)
    anomaly = evidence("E1", evidence_type="anomaly", details={"detector": "iqr"})
    g = graph(
        anomaly, claims=(claim("C1", "Flagged as statistically unusual.", ["E1"], claim_type="calculated_result"),)
    )
    assert any("anomaly result without" in e for e in validate_evidence(g, successful_call_ids=["T1"]).errors)


# ---- claim integrity and wording ------------------------------------------------------------------------------------


def _change_graph() -> EvidenceGraph:
    return graph(
        evidence("E1", -0.048, evidence_type="calculated", unit="ratio", statement="Revenue changed by -4.8%."),
        evidence(
            "E2",
            5850201.0,
            evidence_type="forecast",
            unit="SGD",
            details={"model": "drift", "cutoff_date": "2026-08-31", "horizon": 1},
            period_start=date(2026, 9, 1),
            period_end=date(2026, 9, 30),
        ),
        evidence(
            "E3",
            1.0,
            evidence_type="anomaly",
            details={"detector": "iqr", "threshold": 3, "direction": "negative"},
            attributes={"expected_value": 2.0, "score": -3.5},
        ),
        claims=(
            claim(
                "C1",
                "Revenue changed by -4.8%.",
                ["E1"],
                claim_type="calculated_result",
                kind="change",
                direction="decrease",
                direction_evidence=assertion("E1", -0.048),
            ),
            claim(
                "C2",
                "The drift model forecasts SGD 5,850,201 for 2026-09.",
                ["E2"],
                claim_type="calculated_result",
                kind="forecast",
            ),
            claim(
                "C3",
                "2026-08 was flagged as statistically unusual.",
                ["E3"],
                claim_type="calculated_result",
                kind="anomaly_summary",
            ),
            claim("C4", "Review the accounts behind the decline.", ["E1"], claim_type="recommendation", primary=False),
            claim("C5", "The decline coincided with lower usage.", ["E1"], claim_type="inference", primary=False),
        ),
    )


def _errors(answer: str, ids: list[str], **sections: list[tuple[str, list[str]]]) -> list[str]:
    items = {k: [DraftItemOutput(text=t, claim_ids=c) for t, c in v] for k, v in sections.items()}
    draft = ResponseDraftOutput(answer=answer, answer_claim_ids=ids, **items)
    return validate_response(draft, _change_graph()).errors


def test_correct_wording_passes() -> None:
    assert not _errors(
        "Revenue declined by 4.8%. The model forecasts about SGD 5.85 million for 2026-09.",
        ["C1", "C2"],
        interpretation=[("The decline coincided with lower usage.", ["C5"])],
        recommendations=[("Review the accounts behind the decline.", ["C4"])],
    )


def test_wrong_number_is_rejected() -> None:
    assert any("does not appear" in e for e in _errors("Revenue decreased 12%.", ["C1"]))


def test_contradicting_direction_is_rejected() -> None:
    assert any("states an increase" in e for e in _errors("Revenue increased by 4.8%.", ["C1"]))


@pytest.mark.parametrize(
    "text",
    [
        "Singapore caused the decline.",
        "The decline was because of churn.",
        "Revenue fell as a result of churn.",
        "Churn was responsible for the drop.",
        "The drop resulted from lower usage.",
    ],
)
def test_unsupported_causal_claims_are_rejected(text: str) -> None:
    assert any("causal" in e for e in _errors("Revenue declined by 4.8%.", ["C1"], interpretation=[(text, ["C5"])]))


def test_causal_wording_needs_a_causal_basis() -> None:
    g = _change_graph()
    g.claims["C5"].causal_basis = "accounting_identity"
    draft = ResponseDraftOutput(
        answer="Revenue declined by 4.8%.",
        answer_claim_ids=["C1"],
        interpretation=[DraftItemOutput(text="The net change is due to churn exceeding new MRR.", claim_ids=["C5"])],
    )
    assert not validate_response(draft, g).errors


@pytest.mark.parametrize(
    "text",
    [
        "Revenue will be SGD 5,850,201 in 2026-09.",
        "The forecast guarantees SGD 5,850,201.",
        "Revenue is going to reach SGD 5.85m.",
    ],
)
def test_forecast_presented_as_fact_is_rejected(text: str) -> None:
    errors = _errors(text, ["C2"])
    assert any("with certainty" in e or "labelled as a forecast" in e for e in errors)


@pytest.mark.parametrize(
    "text",
    [
        "2026-08 revenue was bad.",
        "The anomaly shows a revenue crisis.",
        "Revenue was statistically unusual, a worrying problem.",
    ],
)
def test_anomaly_presented_as_business_judgement_is_rejected(text: str) -> None:
    assert any("business judgement" in e or "labelled as statistically" in e for e in _errors(text, ["C3"]))


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("Fix the accounts immediately.", "directive"),
        ("You must cut prices.", "suggested next step"),
        ("Accounts behind the decline.", "suggested next step"),
        ("Review the accounts; this will fix the decline.", "directive"),
    ],
)
def test_recommendations_must_be_suggestions(text: str, problem: str) -> None:
    errors = _errors("Revenue declined by 4.8%.", ["C1"], recommendations=[(text, ["C4"])])
    assert any(problem in e for e in errors), errors


def test_unsupported_recommendation_and_claims_without_provenance() -> None:
    errors = _errors("Revenue declined by 4.8%.", ["C1"], recommendations=[("Review the decline.", ["C1"])])
    assert any("does not cite a recommendation claim" in e for e in errors)
    g = graph(evidence("E1", source_tables=[]), claims=(claim("C1", "Revenue was SGD 100.", ["E1"]),))
    result = validate_evidence(g, successful_call_ids=["T1"])
    assert not result.valid and any("provenance" in e for e in result.errors)


# ---- in the agent -------------------------------------------------------------------------------------------------


def _draft_citing(kind: str, text: str) -> Any:
    def draft(request: Any) -> dict[str, Any]:
        claim_ = next(c for c in request.context["claims"] if c["kind"] == kind)
        return {
            "answer": text,
            "answer_claim_ids": [claim_["claim_id"]],
            "key_findings": [],
            "interpretation": [],
            "recommendations": [],
        }

    return draft


def test_agent_rejects_a_forecast_stated_as_fact(full_db: Any) -> None:
    agent, _ = runner(full_db, {LLMTask.RESPOND: [_draft_citing("forecast", "Revenue will be higher next month.")] * 3})
    result = agent.run("Forecast revenue for the next 3 months")
    assert result.status == "validation_failure" and "will be" not in result.response.answer
    assert any(e.event_type == "output_validation_failed" for e in result.security_events)


def test_agent_rejects_an_anomaly_stated_as_a_judgement(full_db: Any) -> None:
    text = "Revenue movement was statistically unusual and bad for the business."
    agent, _ = runner(full_db, {LLMTask.RESPOND: [_draft_citing("anomaly_summary", text)] * 3})
    result = agent.run("Were there any unusual movements in revenue last month?")
    assert result.status == "validation_failure" and "bad for the business" not in result.response.model_dump_json()


def test_dates_in_forecasts_stay_after_the_cutoff(outputs: dict[str, Any]) -> None:
    g = EvidenceGraph()
    forecast = [e for e in build_evidence(outputs["forecast"], g) if e.evidence_type == "forecast"]
    assert forecast and all(e.period_start and e.period_start > date(2026, 8, 31) for e in forecast)
    assert all(e.details["model"] and e.details["cutoff_date"] and e.details["horizon"] for e in forecast)
    assert all(e.period_start - date(2026, 8, 31) < timedelta(days=200) for e in forecast if e.period_start)
