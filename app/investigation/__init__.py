"""Evidence-backed business investigations (Phase 10): plan -> execute -> validate -> synthesize.

An investigation extends the agent rather than replacing it: the same runtime, input screening,
understanding, secured tool execution, evidence builder, claim builders and validators, with a
template-based analysis plan, bounded multi-step execution, cross-finding validation, rule-based
driver analysis, grounded recommendations and a decision brief. See ``docs/investigations.md``.
"""

from app.investigation.engine import Investigator, ProgressCallback, new_investigation_id
from app.investigation.models import (
    FINAL_STATUSES,
    AnalysisPlan,
    AnalysisStep,
    DecisionBrief,
    Driver,
    Finding,
    FindingRelationship,
    Investigation,
    InvestigationStatus,
    Recommendation,
    StepRecord,
)

__all__ = [
    "FINAL_STATUSES",
    "AnalysisPlan",
    "AnalysisStep",
    "DecisionBrief",
    "Driver",
    "Finding",
    "FindingRelationship",
    "Investigation",
    "InvestigationStatus",
    "Investigator",
    "ProgressCallback",
    "Recommendation",
    "StepRecord",
    "new_investigation_id",
]
