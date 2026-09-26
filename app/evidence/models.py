"""Evidence, claims and the claim-evidence graph.

- **Evidence** is a single, provenance-carrying statement produced from an executed tool result:
  a number with its unit, period, filters, tool, query IDs, source tables, calculation and
  limitations. Evidence types: ``observed`` (a direct aggregate of recorded data), ``calculated``
  (a ratio, change or decomposition computed by the analytics layer), ``forecast``, ``anomaly``,
  and ``derived`` (a structural fact about other results, such as "no month was flagged").
- **Claims** are what the answer asserts. A claim references one or more evidence items. Its type
  says what kind of statement it is: ``observed_fact``, ``calculated_result``, ``inference`` (a
  reasoned reading of evidence, never presented as observed) or ``recommendation``.
- The **EvidenceGraph** holds both and the many-to-many links between them. It is a plain,
  serialisable in-memory structure; no graph database is needed.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.analytics.models import Scalar

EvidenceType = Literal["observed", "calculated", "forecast", "anomaly", "derived"]
EvidenceStatus = Literal["ok", "no_data", "insufficient_data", "insufficient_history"]
ClaimType = Literal["observed_fact", "calculated_result", "inference", "recommendation"]
SupportStatus = Literal["supported", "partially_supported", "unsupported"]
Confidence = Literal["high", "medium", "low"]
Direction = Literal["increase", "decrease", "none"]
CausalBasis = Literal["none", "accounting_identity", "controlled_experiment"]
EVIDENCE_ID = re.compile(r"^E[1-9][0-9]{0,5}$")


class Evidence(BaseModel):
    evidence_id: str
    evidence_type: EvidenceType
    statement: str  # neutral wording of what the tool result shows
    metric: str | None = None
    value: Scalar = None
    unit: str | None = None
    display_value: str | None = None
    period_label: str | None = None
    period_start: date | None = None
    period_end: date | None = None
    comparison_label: str | None = None
    comparison_start: date | None = None
    comparison_end: date | None = None
    filters: dict[str, str] = Field(default_factory=dict)
    dimension: str | None = None
    dimension_value: str | None = None
    attributes: dict[str, Scalar] = Field(default_factory=dict)  # further numbers from the same result
    details: dict[str, Scalar] = Field(default_factory=dict)  # method provenance (model, detector, threshold, ...)
    status: EvidenceStatus = "ok"
    truncated: bool = False
    tool_name: str
    tool_call_id: str
    operation: str
    query_ids: list[str] = Field(default_factory=list)
    source_tables: list[str] = Field(default_factory=list)
    calculation: str | None = None
    execution_timestamp: datetime
    input_arguments: dict[str, Any] = Field(default_factory=dict)  # the tool call's validated arguments
    limitations: list[str] = Field(default_factory=list)
    confidence: Confidence = "high"
    fingerprint: str = ""  # SHA-256 of the content above, set when the evidence enters a graph

    def compute_fingerprint(self) -> str:
        """Deterministic hash of every field except the fingerprint itself (tamper evidence)."""
        content = self.model_dump(mode="json", exclude={"fingerprint"})
        canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def query_id(self) -> str | None:
        return self.query_ids[-1] if self.query_ids else None

    @property
    def has_provenance(self) -> bool:
        return bool(self.tool_call_id and self.query_ids and self.source_tables and self.calculation)

    def numbers(self) -> list[float]:
        """Every number this evidence item states (value, attributes and details)."""
        values = [self.value, *self.attributes.values(), *self.details.values()]
        return [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]


SUBJECT_FIELDS: tuple[str, ...] = (
    "metric",
    "unit",
    "period_label",
    "comparison_label",
    "dimension",
    "dimension_value",
    "filters",
)


class ClaimSubject(BaseModel):
    """What a claim is about, in structured form: the identity of the evidence it restates.

    Set by the claim builders from that evidence (``evidence_id``). The validator checks that the
    evidence still has the same identity: the same metric identifier (``revenue`` is not ``mrr``), unit,
    period, comparison period, dimension, member and filters. A matching number on evidence about
    something else never supports a claim.
    """

    evidence_id: str
    metric: str | None = None
    unit: str | None = None
    period_label: str | None = None
    comparison_label: str | None = None
    dimension: str | None = None
    dimension_value: str | None = None
    filters: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def of(cls, evidence: Evidence) -> ClaimSubject:
        return cls(evidence_id=evidence.evidence_id, **{f: getattr(evidence, f) for f in SUBJECT_FIELDS})

    def mismatches(self, evidence: Evidence) -> list[tuple[str, Any, Any]]:
        """(field, claimed, evidence value) for every identity field that differs."""
        return [
            (f, getattr(self, f), getattr(evidence, f))
            for f in SUBJECT_FIELDS
            if getattr(self, f) != getattr(evidence, f)
        ]


class NumericAssertion(BaseModel):
    """A number a claim states, and where it comes from (checked for contradictions)."""

    evidence_id: str
    field: str = "value"  # "value" or an attribute name
    value: float


class Claim(BaseModel):
    claim_id: str
    text: str
    claim_type: ClaimType
    evidence_ids: list[str] = Field(default_factory=list)
    support_status: SupportStatus = "unsupported"
    confidence: Confidence = "high"
    limitations: list[str] = Field(default_factory=list)
    kind: str = "fact"  # e.g. kpi_value, change, contribution, concentration, anomaly, forecast, association
    primary: bool = False  # directly answers the question
    numeric_assertions: list[NumericAssertion] = Field(default_factory=list)
    direction: Direction | None = None
    direction_evidence: NumericAssertion | None = None  # the signed number that the direction must agree with
    about_period_start: date | None = None
    about_period_end: date | None = None
    subject: ClaimSubject | None = None  # structured identity of what the claim is about (checked vs evidence)
    # Causal wording is allowed only when every cited claim has a causal basis. No builder sets one
    # today: the dataset supports associations and accounting identities, not experiments.
    causal_basis: CausalBasis = "none"


class EvidenceGraph(BaseModel):
    """Claims -> evidence (many-to-many), serialisable."""

    evidence: dict[str, Evidence] = Field(default_factory=dict)
    claims: dict[str, Claim] = Field(default_factory=dict)

    # ------------------------------------------------------------------ construction
    def next_evidence_id(self) -> str:
        return f"E{len(self.evidence) + 1}"

    def next_claim_id(self) -> str:
        return f"C{len(self.claims) + 1}"

    def add_evidence(self, evidence: Evidence) -> str:
        """Add and seal an evidence item: its fingerprint is set (or verified) here."""
        if not EVIDENCE_ID.match(evidence.evidence_id):
            raise ValueError(f"Invalid evidence id {evidence.evidence_id[:20]!r}; the evidence layer assigns E<n> ids")
        if evidence.evidence_id in self.evidence:
            raise ValueError(f"Duplicate evidence id {evidence.evidence_id}")
        expected = evidence.compute_fingerprint()
        if evidence.fingerprint and evidence.fingerprint != expected:
            raise ValueError(f"Evidence {evidence.evidence_id} does not match its fingerprint")
        evidence.fingerprint = expected
        self.evidence[evidence.evidence_id] = evidence
        return evidence.evidence_id

    def verify_integrity(self) -> list[str]:
        """Evidence IDs whose content no longer matches the fingerprint taken when they were added."""
        return [
            evidence_id
            for evidence_id, e in self.evidence.items()
            if e.evidence_id != evidence_id or not e.fingerprint or e.fingerprint != e.compute_fingerprint()
        ]

    def add_claim(self, claim: Claim) -> str:
        if claim.claim_id in self.claims:
            raise ValueError(f"Duplicate claim id {claim.claim_id}")
        for evidence_id in claim.evidence_ids:
            self._require(evidence_id)
        self.claims[claim.claim_id] = claim
        self.claims[claim.claim_id].support_status = self.validate_claim_support(claim.claim_id)
        return claim.claim_id

    def link_claim_to_evidence(self, claim_id: str, evidence_id: str) -> None:
        claim = self.claims[claim_id]
        self._require(evidence_id)
        if evidence_id not in claim.evidence_ids:
            claim.evidence_ids.append(evidence_id)
        claim.support_status = self.validate_claim_support(claim_id)

    def _require(self, evidence_id: str) -> None:
        if evidence_id not in self.evidence:
            raise KeyError(f"Unknown evidence id {evidence_id}")

    # ------------------------------------------------------------------ queries
    def get_supporting_evidence(self, claim_id: str) -> list[Evidence]:
        return [self.evidence[e] for e in self.claims[claim_id].evidence_ids if e in self.evidence]

    def claims_supported_by(self, evidence_id: str) -> list[Claim]:
        return [c for c in self.claims.values() if evidence_id in c.evidence_ids]

    def validate_claim_support(self, claim_id: str) -> SupportStatus:
        """``supported`` when every linked evidence item is usable, ``partially_supported`` when some are."""
        evidence = self.get_supporting_evidence(claim_id)
        if not evidence:
            return "unsupported"
        usable = [e for e in evidence if e.status == "ok" and e.has_provenance]
        if len(usable) == len(evidence):
            return "supported"
        return "partially_supported" if usable else "unsupported"

    def claims_of_type(self, *claim_types: ClaimType) -> list[Claim]:
        return [c for c in self.claims.values() if c.claim_type in claim_types]
