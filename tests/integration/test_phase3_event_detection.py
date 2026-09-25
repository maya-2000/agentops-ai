"""Evaluation-only check: do the detectors find the dataset's observable events from business data alone?

This is the **only** Phase 3 test that reads the injected-event ground truth, and only to learn
*when* each event happened, so detector output can be compared with it. The detectors never see
the ground truth (``test_phase3_isolation.py`` enforces that statically). The ground truth read
here is the copy the generator wrote for this test session's own dataset.

Scope, stated honestly:

- E1 (August 2026 churn wave) and E2 (June-July 2026 ticket spike) are month-level movements in
  supported series and are checked here.
- E3 (a campaign's cost efficiency) and E4/E6/E7 are cross-sectional or customer-level patterns,
  not monthly series, and are covered by the Phase 2 analytics.
- E5 (feature launch in March 2026) leaves only 6 monthly points in the feature's series, below
  the minimum history, so it is reported as ``insufficient_history``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.anomalies import AnomalyService
from app.database.base import Database
from data.generator.generate import GenerationResult

pytestmark = pytest.mark.slow

DETECTORS = ["rolling_zscore", "iqr", "forecast_residual"]


@pytest.fixture(scope="module")
def events(full_dataset: GenerationResult) -> dict[str, dict[str, Any]]:
    truth = json.loads(full_dataset.config.ground_truth_path.read_text(encoding="utf-8"))
    return {event["event_id"]: event for event in truth["events"]}


@pytest.fixture(scope="module")
def service(full_db: Database) -> AnomalyService:
    return AnomalyService(full_db)


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize("metric", ["revenue", "mrr"])
def test_e1_churn_wave_is_a_negative_anomaly(
    service: AnomalyService, events: dict[str, dict[str, Any]], detector: str, metric: str
) -> None:
    month = events["E1"]["period_start"][:7]
    total = service.detect(metric, detector=detector).result(month)
    assert total.is_anomaly and total.direction == "negative", total.explanation
    # The affected cell (Singapore Enterprise, named in the event description) shows it much more strongly.
    cell = service.detect(metric, detector=detector, filters={"country": "Singapore", "segment": "Enterprise"})
    flagged = cell.result(month)
    assert flagged.is_anomaly and flagged.direction == "negative" and flagged.severity == "extreme"
    assert cell.anomalies[0].period == month  # the top-ranked anomaly in that cell


@pytest.mark.parametrize("detector", DETECTORS)
def test_e2_ticket_spike_is_a_positive_anomaly_that_starts_in_the_event_month(
    service: AnomalyService, events: dict[str, dict[str, Any]], detector: str
) -> None:
    month = events["E2"]["period_start"][:7]
    report = service.detect("support_ticket_volume", detector=detector)
    result = report.result(month)
    assert result.is_anomaly and result.direction == "positive" and result.severity == "extreme"
    assert result.anomaly_start == month
    assert report.anomalies[0].period == month


def test_e5_launch_series_is_too_short(service: AnomalyService, events: dict[str, dict[str, Any]]) -> None:
    assert events["E5"]["period_start"].startswith("2026-03")
    report = service.detect("product_adoption", filters={"product_feature": "AI Insights"})
    assert report.status == "insufficient_history"
