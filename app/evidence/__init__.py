"""The evidence layer: provenance-carrying evidence, typed claims, the claim-evidence graph and validators."""

from app.evidence.builder import build_evidence, evidence_summary
from app.evidence.models import Claim, ClaimType, Evidence, EvidenceGraph, EvidenceType, NumericAssertion, SupportStatus
from app.evidence.validation import (
    EvidenceValidationResult,
    ResponseValidationResult,
    causal_sentences,
    validate_evidence,
    validate_response,
)

__all__ = [
    "Claim",
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
    "validate_evidence",
    "validate_response",
]
