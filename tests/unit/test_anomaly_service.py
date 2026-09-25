"""The anomaly service core (``detect_in_series``) on synthetic series: result contract, ranking and runs."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pytest

from app.analytics.errors import InvalidRequestError
from app.anomalies import AnomalyConfig, AnomalyReport, detect_in_series
from app.anomalies.service import _configure
from app.timeseries import UnsupportedMethodError
from tests.phase3_support import noisy, synthetic_series

START, END = date(2025, 9, 1), date(2026, 8, 31)  # the last 12 of 24 synthetic months


def detect(
    values: Sequence[float | None], config: AnomalyConfig | None = None, metric: str = "revenue"
) -> AnomalyReport:
    return detect_in_series(synthetic_series(values, metric=metric), config or AnomalyConfig(), START, END)


def spiky() -> list[float | None]:
    values: list[float | None] = list(noisy(24, slope=0.0, sd=10.0))
    values[18] = (values[18] or 0.0) + 300.0  # 2026-03
    values[19] = (values[19] or 0.0) + 320.0  # 2026-04 (same direction: one run)
    values[22] = (values[22] or 0.0) - 150.0  # 2026-07
    return values


@pytest.mark.parametrize("detector", ["rolling_zscore", "iqr", "forecast_residual"])
def test_report_contract(detector: str) -> None:
    config = AnomalyConfig(detector=detector, transform="level")  # type: ignore[arg-type]
    report = detect(spiky(), config)
    assert report.status == "ok" and report.detector == detector
    assert [r.period for r in report.results] == synthetic_series(noisy(24)).labels()[12:]
    for r in report.results:
        assert r.historical_window.end < r.period  # the scored month is never in its own window
        assert r.historical_window.observations >= config.min_history
        assert r.operation_id == report.operation_id and r.execution_timestamp == report.execution_timestamp
        assert r.detector == detector and r.explanation.startswith(r.period)
        assert f"flag threshold {r.threshold:g}" in r.explanation
        assert r.direction == ("positive" if r.deviation > 0 else "negative" if r.deviation < 0 else "none")
        assert r.is_anomaly == (r.severity in ("significant", "extreme"))
    flagged = report.anomalies
    assert {"2026-03"} <= {a.period for a in flagged}
    keys = [a.rank_key for a in flagged]
    assert keys == sorted(keys)  # ranked: unbounded first, then |score| descending
    assert sum(report.severity_counts.values()) == len(report.results)
    assert report.method.detector == detector and report.method.window == 12
    assert any("does not establish a cause" in note for note in report.limitations)
    assert any("Direction is statistical" in note for note in report.limitations)


def test_consecutive_same_direction_anomalies_share_a_start() -> None:
    report = detect(spiky(), AnomalyConfig(detector="forecast_residual"))
    march, april = report.result("2026-03"), report.result("2026-04")
    assert march.is_anomaly and march.anomaly_start == "2026-03"
    if april.is_anomaly and april.direction == march.direction:
        assert april.anomaly_start == "2026-03"
    for r in report.results:
        if not r.is_anomaly:
            assert r.anomaly_start is None


def test_default_transform_per_metric() -> None:
    assert detect(noisy(24)).method.transform == "pct_change"
    adoption = detect([0.5 + 0.001 * i for i in range(24)], metric="product_adoption")
    assert adoption.method.transform == "difference"
    residual = detect(noisy(24), AnomalyConfig(detector="forecast_residual"))
    assert residual.method.transform is None and residual.method.expectation_model == "drift"


def test_missing_and_undefined_months_are_skipped_with_reasons() -> None:
    values: list[float | None] = list(noisy(24))
    values[20] = None  # 2026-05
    report = detect(values)
    reasons = {s.period: s.reason for s in report.skipped}
    assert reasons["2026-05"] == "missing observation"
    assert "undefined" in reasons["2026-06"]  # change from a missing month
    assert "2026-05" not in {r.period for r in report.results}


def test_insufficient_history_and_empty_ranges() -> None:
    short = detect_in_series(synthetic_series(noisy(5)), AnomalyConfig(), date(2024, 9, 1), date(2025, 1, 31))
    assert short.status == "insufficient_history" and not short.results and short.skipped
    outside = detect_in_series(synthetic_series(noisy(5)), AnomalyConfig(), date(2025, 6, 1), date(2025, 8, 31))
    assert outside.status == "no_data"
    empty = detect([None, None])
    assert empty.status == "no_data"


def test_series_beyond_the_evaluation_end_is_refused() -> None:
    with pytest.raises(InvalidRequestError):
        detect_in_series(synthetic_series(noisy(24)), AnomalyConfig(), START, date(2026, 5, 31))


def test_flag_severity_watch_flags_more() -> None:
    values = spiky()
    strict = detect(values, AnomalyConfig(transform="level"))
    loose = detect(values, AnomalyConfig(transform="level", flag_severity="watch"))
    assert len(loose.anomalies) >= len(strict.anomalies)
    assert all(a.severity != "normal" for a in loose.anomalies)


def test_configuration_overrides() -> None:
    base = AnomalyConfig(min_history=6)
    assert _configure(base, "iqr", None, None).detector == "iqr"
    narrowed = _configure(base, None, 4, None)
    assert narrowed.window == 4 and narrowed.min_history == 4
    assert _configure(base, None, None, "level").transform == "level"
    assert _configure(base, None, None, None) is base
    with pytest.raises(UnsupportedMethodError):
        _configure(base, "black_box", None, None)
