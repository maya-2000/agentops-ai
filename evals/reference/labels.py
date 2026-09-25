"""Hidden evaluation labels: the generator's injected-event ground truth. EVALUATION ONLY.

This is the only module in the repository (outside the generator and its tests) that reads the
ground truth (``data/seeds/injected_events.json``, or the copy written next to a generated test
dataset). The labels are used to decide what the benchmark should expect. They are never passed
to the production system: not to the agent, its model, the MCP server, the tools, prompts or
evidence. ``tests/evals/test_eval_boundary.py`` enforces this statically. At run time, every
production output (model requests, tool results, evidence, responses) is scanned for the label
text (``leak_markers``).

The benchmark never rewards naming an event ("E1 occurred"). Each event's labels are turned into
*observable* expectations (``ObservableEvent``): the month, country, segment, campaign, rep or
feature a correct analysis of the business data would surface.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class HiddenEvent(BaseModel):
    event_id: str
    name: str
    description: str
    expected_signals: str
    injected: bool
    period_start: date | None = None
    period_end: date | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ObservableEvent(BaseModel):
    """What a correct analysis of the observable data should surface for one hidden event."""

    event_id: str
    month: str | None = None  # YYYY-MM of the main effect
    months: list[str] = Field(default_factory=list)  # all affected months, where relevant
    direction: str | None = None  # the observable direction of the main effect: increase / decrease
    country: str | None = None
    segment: str | None = None
    channel: str | None = None
    campaign_id: str | None = None
    sales_rep: str | None = None
    feature: str | None = None
    ticket_categories: list[str] = Field(default_factory=list)


class HiddenLabels(BaseModel):
    dataset_version: str
    random_seed: int
    events: dict[str, HiddenEvent]
    parameters: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> HiddenLabels:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            dataset_version=str(raw.get("dataset_version", "unknown")),
            random_seed=int(raw.get("random_seed", -1)),
            events={e["event_id"]: HiddenEvent.model_validate(e) for e in raw["events"]},
            parameters=raw.get("parameters", {}),
        )

    def observable(self, event_id: str) -> ObservableEvent:
        """Observable expectations for one event, from the structured labels (never the event name)."""
        event = self.events[event_id]
        p = self._params(event_id)
        month = event.period_start.strftime("%Y-%m") if event.period_start else None
        if event_id == "E1":
            return ObservableEvent(
                event_id=event_id,
                month=str(p["month"])[:7],
                direction="decrease",
                country=p["country"],
                segment=p["segment"],
            )
        if event_id == "E2":
            start, end = date.fromisoformat(p["start"]), date.fromisoformat(p["end"])
            months = [f"{y}-{m:02d}" for y, m in _months(start, end)]
            return ObservableEvent(
                event_id=event_id,
                month=months[0],
                months=months,
                direction="increase",
                ticket_categories=sorted(p.get("category_multipliers", {})),
            )
        if event_id == "E3":
            return ObservableEvent(
                event_id=event_id,
                month=str(p["quarter_start"])[:7],
                channel=p["channel"],
                campaign_id=event.details.get("campaign_id"),
                direction="increase",  # cost per conversion is higher than its peers
            )
        if event_id == "E4":
            return ObservableEvent(event_id=event_id, sales_rep=event.details.get("sales_rep"), direction="decrease")
        if event_id == "E5":
            return ObservableEvent(
                event_id=event_id, month=str(p["launch_date"])[:7], feature=p["feature_name"], direction="increase"
            )
        if event_id == "E6":
            return ObservableEvent(event_id=event_id, segment=p["segment"], direction="increase")
        if event_id == "E7":
            return ObservableEvent(event_id=event_id, month=month, direction="decrease")
        raise KeyError(f"No observable mapping for hidden event {event_id!r}")

    def leak_markers(self) -> list[str]:
        """Label text that must never appear in any production output (names, descriptions, signals)."""
        markers = {"injected_events", "ground truth for evaluation", "ground_truth"}
        for event in self.events.values():
            markers.update({event.name, event.description, event.expected_signals})
        return sorted(m for m in markers if len(m) >= 8)

    def _params(self, event_id: str) -> dict[str, Any]:
        prefix = event_id.lower() + "_"
        for key, value in self.parameters.items():
            if key.startswith(prefix):
                return value
        return {}


def _months(start: date, end: date) -> list[tuple[int, int]]:
    months = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months
