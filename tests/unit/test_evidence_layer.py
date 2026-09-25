"""Evidence graph, claim support, evidence validation and number handling."""

from __future__ import annotations

import pytest

from app.evidence import EvidenceGraph, validate_evidence
from app.evidence.formatting import (
    extract_numbers,
    format_percent,
    format_value,
    is_small_count,
    number_is_supported,
)
from tests.phase4_support import JULY, assertion, claim, evidence, graph

# ---- graph ---------------------------------------------------------------------------------------


def test_add_link_and_query() -> None:
    g = graph(evidence("E1", 5.0), evidence("E2", 7.0))
    c1 = claim("C1", "Revenue was SGD 5.", ["E1"])
    g.add_claim(c1)
    assert g.claims["C1"].support_status == "supported"
    g.link_claim_to_evidence("C1", "E2")
    assert [e.evidence_id for e in g.get_supporting_evidence("C1")] == ["E1", "E2"]
    g.add_claim(claim("C2", "Another claim on E2.", ["E2"]))
    assert {c.claim_id for c in g.claims_supported_by("E2")} == {"C1", "C2"}  # many-to-many
    assert g.next_evidence_id() == "E3" and g.next_claim_id() == "C3"


def test_graph_rejects_duplicates_and_unknown_evidence() -> None:
    g = graph(evidence("E1"))
    with pytest.raises(ValueError):
        g.add_evidence(evidence("E1"))
    with pytest.raises(KeyError):
        g.add_claim(claim("C1", "x", ["E9"]))
    g.add_claim(claim("C1", "x", ["E1"]))
    with pytest.raises(KeyError):
        g.link_claim_to_evidence("C1", "E9")


def test_support_status() -> None:
    g = graph(evidence("E1"), evidence("E2", None, status="no_data"), evidence("E3", query_ids=[]))
    g.add_claim(claim("C1", "a", ["E1", "E2"]))
    g.add_claim(claim("C2", "b", ["E2"]))
    g.add_claim(claim("C3", "c", []))
    g.add_claim(claim("C4", "d", ["E3"]))  # no provenance
    assert g.claims["C1"].support_status == "partially_supported"
    assert g.claims["C2"].support_status == "unsupported"
    assert g.claims["C3"].support_status == "unsupported"
    assert g.claims["C4"].support_status == "unsupported"


def test_graph_serialises() -> None:
    g = graph(evidence("E1", 5.0, attributes={"share": 0.5}), claims=(claim("C1", "x", ["E1"]),))
    restored = EvidenceGraph.model_validate_json(g.model_dump_json())
    assert restored == g
    assert restored.evidence["E1"].numbers() == [5.0, 0.5]


# ---- validation ----------------------------------------------------------------------------------


def _valid(g: EvidenceGraph, **kwargs: object) -> list[str]:
    return validate_evidence(g, successful_call_ids=["T1"], **kwargs).errors  # type: ignore[arg-type]


def test_valid_claim_passes() -> None:
    g = graph(
        evidence("E1", 5.0),
        claims=(claim("C1", "Revenue was SGD 5.", ["E1"], numeric_assertions=[assertion("E1", 5.0)]),),
    )
    result = validate_evidence(g, successful_call_ids=["T1"])
    assert result.valid and not result.errors


def test_numerical_claim_without_evidence() -> None:
    g = graph(evidence("E1"), claims=(claim("C1", "Revenue was SGD 5,000,000.", []),))
    result = validate_evidence(g, successful_call_ids=["T1"])
    assert "C1" in result.missing_evidence_claim_ids and not result.valid


def test_evidence_must_come_from_an_executed_tool_with_provenance() -> None:
    g = graph(evidence("E1", tool_call_id="T9"), claims=(claim("C1", "x", ["E1"]),))
    assert any("successful tool call" in e for e in _valid(g))
    g = graph(evidence("E1", source_tables=[]), claims=(claim("C1", "x", ["E1"]),))
    assert any("provenance" in e for e in _valid(g))


def test_contradicting_number_and_direction() -> None:
    g = graph(
        evidence("E1", -56286.0, evidence_type="calculated"),
        claims=(
            claim(
                "C1",
                "Revenue changed.",
                ["E1"],
                claim_type="calculated_result",
                numeric_assertions=[assertion("E1", -50000.0)],
            ),
        ),
    )
    assert any("contradicting" in e for e in _valid(g))
    g = graph(
        evidence("E1", -56286.0, evidence_type="calculated"),
        claims=(
            claim("C1", "Revenue increased.", ["E1"], claim_type="calculated_result", direction="increase",
                  direction_evidence=assertion("E1", -56286.0)),
        ),
    )  # fmt: skip
    assert any("direction increase contradicts" in e for e in _valid(g))


def test_observed_claim_on_calculated_evidence_must_not_pass_as_fact() -> None:
    g = graph(evidence("E1", 0.03, evidence_type="calculated"), claims=(claim("C1", "Churn was 3%.", ["E1"]),))
    assert any("marked observed" in e for e in _valid(g))


@pytest.mark.parametrize(
    "text",
    ["Singapore churn caused the decline.", "The decline was due to churn.", "Churn led to lower revenue."],
)
def test_causal_language_is_rejected(text: str) -> None:
    g = graph(evidence("E1"), claims=(claim("C1", text, ["E1"], claim_type="inference"),))
    assert any("causal" in e for e in _valid(g))


def test_negated_causal_language_is_allowed() -> None:
    text = "The analysis does not establish that support load caused churn."
    g = graph(evidence("E1"), claims=(claim("C1", text, ["E1"], claim_type="inference"),))
    assert not _valid(g)


def test_insufficient_data_and_truncation() -> None:
    g = graph(evidence("E1", None, status="insufficient_history"), claims=(claim("C1", "x", ["E1"]),))
    assert any("insufficient_history" in e for e in _valid(g))
    g = graph(evidence("E1", truncated=True), claims=(claim("C1", "Rows listed.", ["E1"]),))
    assert any("truncated" in e for e in _valid(g))
    g = graph(
        evidence("E1", truncated=True),
        claims=(claim("C1", "Rows listed.", ["E1"], limitations=["Truncated result: the rows are not complete."]),),
    )
    assert not _valid(g)


def test_stale_period() -> None:
    g = graph(
        evidence("E1", period_start=JULY[0], period_end=JULY[1], period_label="2026-07"),
        claims=(claim("C1", "August revenue.", ["E1"], about_period_start=evidence("X").period_start,
                      about_period_end=evidence("X").period_end),),
    )  # fmt: skip
    assert any("stale" in e for e in _valid(g))


def test_a_supported_primary_claim_is_required() -> None:
    g = graph(evidence("E1"), claims=(claim("C1", "x", ["E1"], primary=False),))
    assert "No supported claim answers the question." in _valid(g)
    assert validate_evidence(g, successful_call_ids=["T1"], require_primary=False).valid


def test_failed_tools_and_truncation_become_warnings() -> None:
    g = graph(evidence("E1"), claims=(claim("C1", "x", ["E1"]),))
    result = validate_evidence(g, successful_call_ids=["T1"], failed_tools=["detect_anomalies: boom"])
    assert result.valid and any("detect_anomalies" in w for w in result.warnings)


# ---- numbers -------------------------------------------------------------------------------------


def test_formatting() -> None:
    assert format_value(5752877.1, "SGD") == "SGD 5,752,877"
    assert format_value(9504.4, "SGD per customer") == "SGD 9,504"
    assert format_value(0.0331, "ratio") == "3.31%"
    assert format_value(3184, "tickets") == "3,184"
    assert format_value(None, "SGD") == "not available"
    assert format_percent(-0.0097, signed=True) == "-0.97%"
    assert format_percent(0.922) == "92.2%"


def test_number_extraction_ignores_dates_and_identifiers() -> None:
    numbers = extract_numbers("Revenue in 2026-08 (Q3 2026, E12, C3) was SGD 5,752,877 (-0.97%), 5.75 million or 12k.")
    assert [n.raw for n in numbers] == ["5,752,877", "-0.97%", "5.75 million", "12k"]
    assert numbers[2].value == pytest.approx(5_750_000) and numbers[1].percent


@pytest.mark.parametrize(
    ("text", "allowed", "ok"),
    [
        ("SGD 5,752,877", [5752877.1], True),
        ("SGD 5.75 million", [5752877.1], True),
        ("SGD 5.8 million", [5752877.1], True),
        ("SGD 5.9 million", [5752877.1], False),
        ("-0.97%", [-0.00968913786], True),
        ("0.97%", [-0.00968913786], True),
        ("1.2%", [-0.00968913786], False),
        ("95.6%", [0.9561047796], True),
        ("SGD 4,000,000", [5752877.1], False),
    ],
)
def test_number_support(text: str, allowed: list[float], ok: bool) -> None:
    (number,) = extract_numbers(text)
    assert number_is_supported(number, allowed) is ok


def test_small_counts_are_structural() -> None:
    assert all(is_small_count(n) for n in extract_numbers("the last 3 months and top 5 members"))
    assert not any(is_small_count(n) for n in extract_numbers("3.5% and 90,000"))
