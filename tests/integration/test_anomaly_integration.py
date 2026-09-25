"""Anomaly detection against the generated dataset: every detector and metric, dimensions, scans and determinism.

These tests check structure, determinism and point-in-time behaviour. They do not use the
injected-event ground truth (see ``test_phase3_event_detection.py`` for the separate evaluation).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from app.analytics.errors import InvalidPeriodError, UnsupportedDimensionError
from app.anomalies import AnomalyReport, AnomalyService, detect_anomalies
from app.database.base import Database
from app.timeseries import UnsupportedMethodError

pytestmark = pytest.mark.slow

DETECTORS = ["rolling_zscore", "iqr", "forecast_residual"]
METRICS: list[tuple[str, dict[str, str]]] = [
    ("revenue", {}),
    ("mrr", {}),
    ("customer_count", {}),
    ("support_ticket_volume", {}),
    ("product_adoption", {"product_feature": "Dashboards"}),
]


@pytest.fixture(scope="module")
def service(full_db: Database) -> AnomalyService:
    return AnomalyService(full_db)


def comparable(report: AnomalyReport) -> list[dict[str, Any]]:
    rows = []
    for result in report.results:
        data = result.model_dump(mode="json")
        for key in ("operation_id", "execution_timestamp", "query_id"):
            data.pop(key)
        rows.append(data)
    return rows


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize(("metric", "filters"), METRICS)
def test_every_detector_on_every_metric(
    service: AnomalyService, detector: str, metric: str, filters: dict[str, str]
) -> None:
    report = service.detect(metric, detector=detector, filters=filters)
    assert report.status == "ok", report.message
    assert report.evaluation_start == date(2025, 9, 1) and report.evaluation_end == date(2026, 8, 31)
    assert [r.period for r in report.results] == [p.period for p in report.series.points[-12:]]
    for r in report.results:
        assert r.period_end <= date(2026, 8, 31)
        assert r.historical_window.end < r.period and r.historical_window.observations >= 6
        assert r.observed_value >= 0 and r.detector == detector
        assert r.source_tables and r.query_id == report.series.query_id
        assert r.threshold == report.method.threshold
    ranked = report.anomalies
    assert [a.rank_key for a in ranked] == sorted(a.rank_key for a in ranked)


def test_detection_is_deterministic(full_db: Database) -> None:
    for detector in DETECTORS:
        first = detect_anomalies(full_db, "support_ticket_volume", detector=detector)
        second = detect_anomalies(full_db, "support_ticket_volume", detector=detector)
        assert comparable(first) == comparable(second)


@pytest.mark.parametrize(
    "filters",
    [{"segment": "Enterprise"}, {"region": "EMEA"}, {"country": "Japan"}, {"plan": "Starter"}],
)
def test_explicit_dimensional_detection(service: AnomalyService, filters: dict[str, str]) -> None:
    report = service.detect("mrr", detector="rolling_zscore", filters=filters)
    assert report.status == "ok" and report.filters == filters
    assert all(r.filters == filters for r in report.results)


def test_custom_range_and_window(service: AnomalyService) -> None:
    report = service.detect(
        "revenue", start_date=date(2026, 1, 15), end_date=date(2026, 6, 30), detector="iqr", window=8
    )
    assert [r.period for r in report.results] == ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
    assert report.method.window == 8 and all(r.historical_window.window_months == 8 for r in report.results)
    assert report.series.end == date(2026, 6, 30)  # nothing after the evaluation end is read


def test_scan_ranks_across_metrics(service: AnomalyService) -> None:
    scan = service.scan(detector="forecast_residual")
    assert [r.metric for r in scan.reports] == ["revenue", "mrr", "customer_count", "support_ticket_volume"]
    keys = [(*a.rank_key, a.metric) for a in scan.anomalies]
    assert keys == sorted(keys)


def test_invalid_requests(service: AnomalyService) -> None:
    with pytest.raises(UnsupportedMethodError):
        service.detect("revenue", detector="autoencoder")
    with pytest.raises(InvalidPeriodError):
        service.detect("revenue", end_date=date(2026, 12, 31))
    with pytest.raises(InvalidPeriodError):
        service.detect("revenue", start_date=date(2026, 8, 1), end_date=date(2026, 7, 31))
    with pytest.raises(UnsupportedDimensionError):
        service.detect("mrr", filters={"ticket_category": "Bug"})


def test_short_series_report_insufficient_history(service: AnomalyService) -> None:
    report = service.detect("product_adoption", filters={"product_feature": "AI Insights"})
    assert report.status == "insufficient_history" and not report.results and report.skipped
