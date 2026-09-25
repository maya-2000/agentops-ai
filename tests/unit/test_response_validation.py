"""The response validator: every number, claim reference and label in a draft is checked."""

from __future__ import annotations

from typing import Any

import pytest

from app.evidence import EvidenceGraph, validate_response
from app.llm.schemas import DraftItemOutput, ResponseDraftOutput
from tests.phase4_support import assertion, claim, evidence, graph


def _graph() -> EvidenceGraph:
    return graph(
        evidence("E1", 5752877.1, statement="Revenue for 2026-08: SGD 5,752,877."),
        evidence(
            "E2",
            -56286.0,
            evidence_type="calculated",
            attributes={"percentage_change": -0.00968913786},
            statement="Revenue changed by -SGD 56,286 (-0.97%).",
        ),
        evidence("E3", 5850201.0, evidence_type="forecast", statement="Forecast revenue for 2026-09."),
        evidence("E4", 1.0, evidence_type="anomaly", statement="2026-08 flagged by rolling z-score."),
        evidence("E5", 12.0, truncated=True, evidence_type="observed", tool_name="run_safe_sql"),
        claims=(
            claim("C1", "Revenue for 2026-08 was SGD 5,752,877.", ["E1"], kind="kpi_value",
                  numeric_assertions=[assertion("E1", 5752877.1)]),
            claim("C2", "Revenue decreased by SGD 56,286 (-0.97%).", ["E2"], claim_type="calculated_result",
                  kind="change"),
            claim("C3", "The drift model forecasts SGD 5,850,201.", ["E3"], claim_type="calculated_result",
                  kind="forecast"),
            claim("C4", "2026-08 was flagged as statistically unusual.", ["E4"], claim_type="calculated_result",
                  kind="anomaly_summary"),
            claim("C5", "The decline coincided with lower enterprise revenue.", ["E2"], claim_type="inference",
                  primary=False),
            claim("C6", "Review enterprise accounts.", ["E2"], claim_type="recommendation", primary=False),
            claim("C7", "Rows returned (truncated result).", ["E5"], limitations=["Truncated result."],
                  primary=False),
        ),
    )  # fmt: skip


def _draft(answer: str = "Revenue for 2026-08 was SGD 5,752,877.", ids: list[str] | None = None, **kw: Any) -> Any:
    items = {k: [DraftItemOutput(text=t, claim_ids=c) for t, c in v] for k, v in kw.items()}
    return ResponseDraftOutput(answer=answer, answer_claim_ids=["C1"] if ids is None else ids, **items)


def _errors(draft: ResponseDraftOutput, **kwargs: Any) -> list[str]:
    return validate_response(draft, _graph(), **kwargs).errors


def test_correct_response_passes() -> None:
    draft = _draft(
        key_findings=[("Revenue decreased by SGD 56,286 (-0.97%) compared with 2026-07.", ["C2"])],
        interpretation=[("The decline coincided with lower enterprise revenue.", ["C5"])],
        recommendations=[("Review enterprise accounts.", ["C6"])],
    )
    result = validate_response(draft, _graph())
    assert result.valid, result.errors


@pytest.mark.parametrize("answer", ["Revenue for 2026-08 was SGD 5.75 million.", "Revenue was about SGD 5.8m."])
def test_display_rounding_is_allowed(answer: str) -> None:
    assert not _errors(_draft(answer))


def test_wrong_number_is_rejected() -> None:
    result = validate_response(_draft("Revenue for 2026-08 was SGD 6,100,000."), _graph())
    assert not result.valid and result.unsupported_numbers == ["6,100,000"]


def test_number_from_uncited_evidence_is_rejected() -> None:
    # -0.97% is real evidence, but the answer only cites C1.
    result = validate_response(_draft("Revenue was SGD 5,752,877, down 0.97%."), _graph())
    assert result.unsupported_numbers == ["0.97%"]


def test_missing_and_unknown_claim_references() -> None:
    assert "The answer does not cite any claim." in _errors(_draft(ids=[]))
    result = validate_response(_draft(ids=["C99"]), _graph())
    assert result.invalid_claim_refs == ["C99"]
    assert any("without a claim reference" in e for e in _errors(_draft(key_findings=[("Revenue fell.", [])])))


def test_unsupported_claim_cannot_be_cited() -> None:
    g = _graph()
    g.add_evidence(evidence("E9", None, status="no_data"))
    g.add_claim(claim("C9", "Nothing.", ["E9"]))
    result = validate_response(_draft(ids=["C9"]), g)
    assert any("unsupported claim C9" in e for e in result.errors)


def test_causal_claim_is_rejected() -> None:
    draft = _draft(interpretation=[("Enterprise churn caused the decline.", ["C5"])])
    assert any("causal" in e for e in _errors(draft))
    ok = _draft(interpretation=[("The data does not show that churn caused the decline.", ["C5"])])
    assert not _errors(ok)


def test_forecast_must_be_labelled() -> None:
    assert any("forecast presented" in e for e in _errors(_draft("Revenue next month will be SGD 5,850,201.", ["C3"])))
    assert not _errors(_draft("The forecast for 2026-09 is SGD 5,850,201.", ["C3"]))


def test_anomaly_must_be_labelled() -> None:
    assert any("anomaly result" in e for e in _errors(_draft("Revenue dropped in 2026-08.", ["C4"])))
    assert not _errors(_draft("2026-08 revenue was statistically unusual.", ["C4"]))


def test_inference_cannot_be_a_key_finding() -> None:
    draft = _draft(key_findings=[("The decline coincided with lower enterprise revenue.", ["C5"])])
    assert any("inference or recommendation is presented as a finding" in e for e in _errors(draft))


def test_recommendation_must_cite_a_recommendation_claim() -> None:
    draft = _draft(recommendations=[("Review enterprise accounts.", ["C2"])])
    assert any("recommendation does not cite" in e for e in _errors(draft))


def test_truncated_result_must_be_disclosed() -> None:
    draft = _draft(key_findings=[("The query returned rows.", ["C7"])])
    assert any("truncated SQL result presented as complete" in e for e in _errors(draft))
    assert not _errors(_draft(key_findings=[("The query returned a truncated set of rows.", ["C7"])]))


def test_unanswerable_response_carries_no_fabricated_content() -> None:
    ok = _draft("This question is outside the business data available to the agent.", ids=[])
    assert not _errors(ok, answerable=False)
    fabricated = _draft("Apple stock closed at 231.40 USD.", ids=[])
    assert any("business numbers" in e for e in _errors(fabricated, answerable=False))
    padded = _draft("Not available.", ids=[], key_findings=[("Revenue fell.", ["C2"])])
    assert any("must not contain findings" in e for e in _errors(padded, answerable=False))


def test_length_limit() -> None:
    long = "Revenue for 2026-08 was SGD 5,752,877. " + "More words. " * 50
    assert any("characters (limit 100)" in e for e in _errors(_draft(long), max_chars=100))
