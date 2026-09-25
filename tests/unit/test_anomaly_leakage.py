"""Leakage regression tests for anomaly detection.

1. The current observation cannot contaminate its own baseline: changing x_t changes only the
   observed value, deviation and score, never the baseline, dispersion, quartiles, forecast or bounds.
2. Later observations cannot change an earlier assessment.
"""

from __future__ import annotations

import pytest

from app.anomalies import AnomalyConfig
from app.anomalies.detectors import Assessment, DetectorOutput, forecast_residual, iqr_detector, rolling_zscore
from app.timeseries.metrics import Transform
from tests.phase3_support import noisy, synthetic_series

T = 20
BASE = noisy(30, level=1000.0, slope=10.0, sd=15.0)
LABELS = synthetic_series(BASE).labels()
CONFIG = AnomalyConfig(window=12, min_history=6)
BASELINE_FIELDS = {
    "rolling_zscore": ("baseline_mean", "baseline_std"),
    "iqr": ("q1", "median", "q3", "iqr", "lower_fence", "upper_fence", "outer_lower_fence", "outer_upper_fence"),
    "forecast_residual": ("one_step_forecast", "residual_mean", "residual_std", "training_start", "training_end"),
}


def run(detector: str, values: list[float], indices: list[int], transform: Transform = "level") -> DetectorOutput:
    if detector == "rolling_zscore":
        return rolling_zscore(values, indices, CONFIG, transform)
    if detector == "iqr":
        return iqr_detector(values, indices, CONFIG, transform)
    return forecast_residual(values, indices, CONFIG, LABELS)


def at(output: DetectorOutput, index: int) -> Assessment:
    return next(a for a in output.assessments if a.index == index)


@pytest.mark.parametrize("detector", ["rolling_zscore", "iqr", "forecast_residual"])
@pytest.mark.parametrize("transform", ["level", "pct_change"])
def test_current_observation_does_not_define_its_own_threshold(detector: str, transform: Transform) -> None:
    changed = list(BASE)
    changed[T] += 5000.0
    before, after = at(run(detector, BASE, [T], transform), T), at(run(detector, changed, [T], transform), T)
    for field in BASELINE_FIELDS[detector]:
        assert getattr(before.details, field) == getattr(after.details, field), field
    assert (before.expected, before.lower_bound, before.upper_bound, before.threshold) == (
        after.expected,
        after.lower_bound,
        after.upper_bound,
        after.threshold,
    )
    assert (before.window_start, before.window_end) == (after.window_start, after.window_end) == (T - 12, T - 1)
    assert after.observed != before.observed and after.score != before.score


@pytest.mark.parametrize("detector", ["rolling_zscore", "iqr", "forecast_residual"])
def test_future_observations_do_not_change_past_assessments(detector: str) -> None:
    changed = BASE[: T + 1] + [v * 10.0 for v in BASE[T + 1 :]]
    indices = list(range(12, 30))
    before, after = run(detector, BASE, indices), run(detector, changed, indices)
    for index in range(12, T + 1):
        assert at(before, index) == at(after, index)
    assert at(before, T + 1) != at(after, T + 1)  # control: the changed months themselves do change


def test_window_moves_with_the_scored_month() -> None:
    output = rolling_zscore(BASE, [15, 25], CONFIG, "level")
    assert [(a.window_start, a.window_end) for a in output.assessments] == [(3, 14), (13, 24)]
