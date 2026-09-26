"""The evidence layer: provenance-carrying evidence, typed claims, the claim-evidence graph and validators."""

from app.evidence.builder import build_evidence, evidence_summary
from app.evidence.models import (
    Claim,
    ClaimSubject,
    ClaimType,
    Evidence,
    EvidenceGraph,
    EvidenceType,
    NumericAssertion,
    SupportStatus,
)
from app.evidence.validation import (
    EvidenceValidationResult,
    ResponseValidationResult,
    causal_sentences,
    metrics_named,
    validate_evidence,
    validate_response,
)

__all__ = [
    "Claim",
    "ClaimSubject",
    "ClaimType",
    "Evidence",
    "EvidenceGraph",
    "EvidenceType",
    "EvidenceValidationResult",
    "NumericAssertion",
    "ResponseValidationResult",
    "SupportStatus",
    "build_evidence",
    "causal_sentences",
    "evidence_summary",
    "metrics_named",
    "validate_evidence",
    "validate_response",
]
