"""Deliberately injected business events and their ground-truth record.

The events are applied *inside* the simulation (they change the behaviour of customers,
campaigns and reps) so they surface only through ordinary business data. Their definitions
and the affected entities are written to a separate ground-truth JSON file for the
evaluation framework; nothing in this module is ever loaded into the business database.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SingaporeEnterpriseChurnWave:  # E1
    month: date = date(2026, 8, 1)
    country: str = "Singapore"
    segment: str = "Enterprise"
    churn_share: float = 0.20
    contraction_share: float = 0.30
    contraction_factor: tuple[float, float] = (0.55, 0.80)
    last_service_day_window: tuple[date, date] = (date(2026, 8, 1), date(2026, 8, 12))
    mid_market_churn_multiplier: float = 2.0  # mild spill-over into SG Mid-Market
    billing_ticket_rate: dict[int, float] = field(default_factory=lambda: {7: 0.6, 8: 0.4})


@dataclass(frozen=True)
class SupportTicketSpike:  # E2
    start: date = date(2026, 6, 8)
    end: date = date(2026, 7, 24)
    category_multipliers: dict[str, float] = field(
        default_factory=lambda: {"Bug": 2.6, "Integration": 2.0, "Performance": 1.4}
    )
    resolution_time_factor: float = 1.7
    engagement_shock: float = 0.40  # applied the month after a customer hits the faulty release


@dataclass(frozen=True)
class InefficientPaidSocialCampaign:  # E3
    channel: str = "Paid Social"
    quarter_start: date = date(2026, 4, 1)
    theme: str = "Brand Awareness Video"
    spend_multiplier: float = 1.6
    lead_quality: float = 0.38  # lead-to-customer conversion relative to channel norm


@dataclass(frozen=True)
class LowConversionSalesRep:  # E4
    stage_conversion_factor: float = 0.85  # applied to each of the four funnel transitions
    excluded_regions: tuple[str, ...] = ("LATAM",)  # keep the rep in a well-populated territory


@dataclass(frozen=True)
class AIInsightsLaunch:  # E5
    launch_date: date = date(2026, 3, 2)
    feature_id: str = "F12"
    feature_name: str = "AI Insights"
    eligible_plans: dict[str, float] = field(
        default_factory=lambda: {"Growth": 0.35, "Professional": 1.0, "Enterprise": 1.0}
    )
    monthly_adoption_hazard: float = 0.16


@dataclass(frozen=True)
class SmbChurnPattern:  # E6 (structural; implemented through segment churn intercepts)
    segment: str = "SMB"


@dataclass(frozen=True)
class PreChurnUsageDecline:  # E7
    selection_date: date = date(2026, 3, 31)
    drift_start: date = date(2026, 4, 1)
    cohort_share: float = 0.03
    min_tenure_months: int = 6
    monthly_engagement_drift: float = 0.30
    late_churn_logit_boost: float = 1.0
    late_churn_months: tuple[date, ...] = (date(2026, 7, 1), date(2026, 8, 1))


@dataclass(frozen=True)
class EventSettings:
    e1_sg_enterprise_churn: SingaporeEnterpriseChurnWave = field(default_factory=SingaporeEnterpriseChurnWave)
    e2_support_spike: SupportTicketSpike = field(default_factory=SupportTicketSpike)
    e3_paid_social: InefficientPaidSocialCampaign = field(default_factory=InefficientPaidSocialCampaign)
    e4_low_conversion_rep: LowConversionSalesRep = field(default_factory=LowConversionSalesRep)
    e5_ai_insights: AIInsightsLaunch = field(default_factory=AIInsightsLaunch)
    e6_smb_churn: SmbChurnPattern = field(default_factory=SmbChurnPattern)
    e7_pre_churn_decline: PreChurnUsageDecline = field(default_factory=PreChurnUsageDecline)


EVENTS = EventSettings()

EVENT_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "E1": {
        "name": "Singapore Enterprise churn wave",
        "description": (
            "In August 2026 a cluster of Singapore Enterprise accounts churned at renewal and "
            "several others contracted, preceded by Billing complaints in July-August."
        ),
        "expected_signals": (
            "August 2026 revenue/MRR decline; Singapore and Enterprise are the largest negative "
            "contributors; SG Enterprise logo churn far above its baseline; Billing tickets with "
            "negative sentiment from SG Enterprise accounts."
        ),
    },
    "E2": {
        "name": "Support-ticket spike after a faulty release",
        "description": (
            "Between 8 June and 24 July 2026 Bug, Integration and Performance tickets surged and "
            "resolution times lengthened; affected accounts became less engaged afterwards."
        ),
        "expected_signals": (
            "Ticket volume per active customer in June-July well above March-May; higher Bug and "
            "Integration share; longer resolution times; more tickets among later churners."
        ),
    },
    "E3": {
        "name": "Inefficient Paid Social campaign",
        "description": "A Q2 2026 Paid Social campaign spent heavily but converted poorly.",
        "expected_signals": "Highest CAC of all campaigns; CAC far above the Paid Social median.",
    },
    "E4": {
        "name": "Sales representative with consistently low conversion",
        "description": "One rep's deals progress through each funnel stage less often than peers'.",
        "expected_signals": "Lowest observed win rate of all reps, clearly below the team median.",
    },
    "E5": {
        "name": "AI Insights feature launch",
        "description": "The AI Insights feature launched on 2 March 2026 and adoption ramped up.",
        "expected_signals": (
            "product_features rows for AI Insights start on the launch date; adoption rate rises "
            "steadily; broader feature usage among adopting accounts."
        ),
    },
    "E6": {
        "name": "Structurally higher SMB churn",
        "description": "SMB customers churn at a consistently higher rate than larger segments.",
        "expected_signals": "SMB has the highest logo churn rate in every period.",
    },
    "E7": {
        "name": "Usage decline before churn",
        "description": (
            "From April 2026 a cohort of established accounts steadily disengaged; many churned "
            "in July-August 2026 and the rest remain at risk."
        ),
        "expected_signals": (
            "Weekly active users fall well below earlier levels in the weeks before churn; "
            "rising support tickets for the same accounts."
        ),
    },
}


class GroundTruthLog:
    """Collects ground-truth details while the simulation runs."""

    def __init__(self) -> None:
        self._events: dict[str, dict[str, Any]] = {}

    def record(self, event_id: str, *, injected: bool, period: tuple[date, date] | None, **details: Any) -> None:
        entry: dict[str, Any] = dict(EVENT_DESCRIPTIONS[event_id])
        entry["event_id"] = event_id
        entry["injected"] = injected
        if period is not None:
            entry["period_start"], entry["period_end"] = period[0].isoformat(), period[1].isoformat()
        existing = self._events.get(event_id, {})
        existing.update(entry)
        existing.setdefault("details", {}).update(details)
        self._events[event_id] = existing

    def get(self, event_id: str) -> dict[str, Any]:
        return self._events[event_id]

    def to_dict(self, *, seed: int, dataset_version: str) -> dict[str, Any]:
        """JSON-native representation (dates as ISO strings), identical to the file on disk."""
        return _jsonable(
            {
                "purpose": (
                    "Ground truth for evaluation only. NOT part of the agent-facing database and must "
                    "never be exposed to the agent."
                ),
                "dataset_version": dataset_version,
                "random_seed": seed,
                "parameters": asdict(EVENTS),
                "events": [self._events[k] for k in sorted(self._events)],
            }
        )

    def write(self, path: Path, *, seed: int, dataset_version: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(seed=seed, dataset_version=dataset_version), fh, indent=2)
            fh.write("\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return value


def load_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    """Load the ground-truth file keyed by event id (evaluation/validation use only)."""
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    return {event["event_id"]: event for event in payload["events"]}
