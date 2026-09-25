"""Anomaly detectors and the severity policy on synthetic series with known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from app.analytics.errors import InvalidRequestError
from app.anomalies import AnomalyConfig, IQRDetails, RollingZScoreDetails, StandardizedThresholds
from app.anomalies.config import IQRFences
from app.anomalies.detectors import (
    forecast_residual,
    implied_level,
    iqr_detector,
    rolling_zscore,
    transform_series,
)
from app.anomalies.models import ForecastResidualDetails
from app.anomalies.thresholds import classify_iqr, classify_standardized, flag_threshold, is_flagged
from tests.phase3_support import constant, linear, noisy, synthetic_series

LEVEL = AnomalyConfig(window=12, min_history=6)
LABELS = synthetic_series(noisy(30)).labels()


# ------------------------------------------------------------------------------------------ transforms
def test_transforms() -> None:
    y = [100.0, 110.0, 0.0, 50.0]
    assert list(transform_series(y, "level")) == y
    diff = transform_series(y, "difference")
    assert math.isnan(diff[0]) and list(diff[1:]) == [10.0, -110.0, 50.0]
    pct = transform_series(y, "pct_change")
    assert math.isnan(pct[0]) and pct[1] == pytest.approx(0.1) and pct[2] == pytest.approx(-1.0)
    assert math.isnan(pct[3])  # previous month is zero: undefined, never infinite
    assert implied_level(200.0, 0.05, "pct_change") == pytest.approx(210.0)
    assert implied_level(200.0, 5.0, "difference") == 205.0
    assert implied_level(float("nan"), 7.0, "level") == 7.0


# ------------------------------------------------------------------------------------------ z-score
def test_rolling_zscore_matches_a_hand_computation() -> None:
    y = [10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 30.0]
    out = rolling_zscore(y, [6], AnomalyConfig(window=6, min_history=6), "level")
    (a,) = out.assessments
    history = np.array(y[:6])
    mean, std = history.mean(), history.std(ddof=1)
    assert isinstance(a.details, RollingZScoreDetails)
    assert a.details.baseline_mean == pytest.approx(mean) and a.details.baseline_std == pytest.approx(std)
    assert a.score == pytest.approx((30.0 - mean) / std)
    assert a.expected == pytest.approx(mean) and a.deviation == pytest.approx(30.0 - mean)
    assert a.severity == "extreme"
    assert a.lower_bound == pytest.approx(mean - 3 * std) and a.upper_bound == pytest.approx(mean + 3 * std)
    t = (30.0 - mean) / (std * math.sqrt(1 + 1 / 6))
    assert a.details.tail_probability == pytest.approx(2 * stats.t.sf(abs(t), df=5))
    assert (a.window_start, a.window_end, a.window_observations) == (0, 5, 6)


def test_rolling_zscore_directions_and_normal_points() -> None:
    base = noisy(30, slope=0.0, sd=10.0)
    spike, drop = list(base), list(base)
    spike[25] += 200.0
    drop[25] -= 200.0
    up = rolling_zscore(spike, [25], LEVEL, "level").assessments[0]
    down = rolling_zscore(drop, [25], LEVEL, "level").assessments[0]
    assert up.score is not None and up.score > 4 and up.deviation > 0
    assert down.score is not None and down.score < -4 and down.deviation < 0
    normal = rolling_zscore(base, list(range(12, 30)), LEVEL, "level").assessments
    assert all(a.severity in ("normal", "watch") for a in normal)


def test_pct_change_transform_reports_expected_in_metric_units() -> None:
    y = [100.0 * 1.02**i for i in range(14)]
    y[13] = y[12] * 0.97  # -3% after a steady +2%
    (a,) = rolling_zscore(y, [13], LEVEL, "pct_change").assessments
    assert a.expected == pytest.approx(y[12] * 1.02)
    assert a.deviation < 0 and a.score is None  # identical prior growth: zero dispersion, unbounded score
    assert a.severity == "extreme"


def test_constant_series_is_normal_and_a_break_is_extreme() -> None:
    flat = rolling_zscore(constant(20), list(range(6, 20)), LEVEL, "level").assessments
    assert all(a.score == 0.0 and a.severity == "normal" for a in flat)
    broken = constant(20)
    broken[15] = 101.0
    a = next(a for a in rolling_zscore(broken, [15], LEVEL, "level").assessments)
    assert a.score is None and a.severity == "extreme"


def test_minimum_history_and_missing_values() -> None:
    y: list[float] = noisy(20)
    y[10] = float("nan")
    out = rolling_zscore(y, list(range(20)), LEVEL, "level")
    scored = {a.index for a in out.assessments}
    skipped = {s.index: s.reason for s in out.skipped}
    assert scored.isdisjoint(skipped) and scored | set(skipped) == set(range(20))
    assert all("prior observation" in skipped[i] for i in range(6))  # fewer than 6 prior months
    assert skipped[10] == "missing observation"
    pct = rolling_zscore(y, [11], LEVEL, "pct_change")
    assert "undefined" in pct.skipped[0].reason  # previous month missing


# ------------------------------------------------------------------------------------------ IQR
def test_iqr_matches_numpy_quartiles_and_tukey_fences() -> None:
    y = [float(v) for v in [5, 7, 6, 9, 8, 10, 7, 6, 8, 9, 7, 8, 25]]
    (a,) = iqr_detector(y, [12], LEVEL, "level").assessments
    q1, q3 = np.percentile(y[:12], [25, 75])
    assert isinstance(a.details, IQRDetails)
    assert a.details.q1 == pytest.approx(q1) and a.details.q3 == pytest.approx(q3)
    assert a.details.upper_fence == pytest.approx(q3 + 1.5 * (q3 - q1))
    assert a.details.outer_upper_fence == pytest.approx(q3 + 3.0 * (q3 - q1))
    assert a.score == pytest.approx((25 - q3) / (q3 - q1))
    assert a.upper_bound == pytest.approx(q3 + 1.5 * (q3 - q1)) and a.threshold == 1.5
    assert a.severity == "extreme"


def test_iqr_score_is_zero_inside_the_box_and_never_watch() -> None:
    y = noisy(40, slope=0.0)
    out = iqr_detector(y, list(range(12, 40)), LEVEL, "level").assessments
    assert all(a.severity != "watch" for a in out)
    inside = [
        a
        for a in out
        if isinstance(a.details, IQRDetails) and a.details.q1 <= a.details.transformed_value <= a.details.q3
    ]
    assert inside and all(a.score == 0.0 for a in inside)


def test_iqr_zero_spread() -> None:
    y = constant(15)
    y[14] = 120.0
    a = iqr_detector(y, [14], LEVEL, "level").assessments[0]
    assert a.score is None and a.severity == "extreme"


# ------------------------------------------------------------------------------------ forecast residual
def test_forecast_residual_on_a_trend_with_a_spike() -> None:
    y = noisy(30, level=1000.0, slope=20.0, sd=5.0)
    y[26] += 150.0
    out = forecast_residual(y, [26], LEVEL, LABELS)
    (a,) = out.assessments
    assert isinstance(a.details, ForecastResidualDetails)
    d = a.details
    assert d.expectation_model == "drift"
    assert (d.training_start, d.training_end) == (LABELS[14], LABELS[25])  # the 12 months before the spike
    window = y[14:26]
    assert d.one_step_forecast == pytest.approx(window[-1] + (window[-1] - window[0]) / 11)
    assert d.residual == pytest.approx(y[26] - d.one_step_forecast)
    assert a.expected == pytest.approx(d.one_step_forecast + d.residual_mean)
    assert a.score == pytest.approx((d.residual - d.residual_mean) / d.residual_std)
    assert a.score is not None and a.score > 4 and a.severity == "extreme"


def test_forecast_residual_linear_series_has_zero_residuals() -> None:
    out = forecast_residual(linear(30), list(range(20, 30)), LEVEL, LABELS)
    assert out.assessments and all(a.score == 0.0 and a.severity == "normal" for a in out.assessments)


@pytest.mark.parametrize("model", ["naive", "moving_average", "ets_damped_trend", "seasonal_naive"])
def test_forecast_residual_supports_other_expectation_models(model: str) -> None:
    config = AnomalyConfig(window=12, min_history=12 if model in ("ets_damped_trend", "seasonal_naive") else 6)
    config = config.model_copy(update={"expectation_model": model})
    out = forecast_residual(noisy(30), list(range(24, 30)), config, LABELS)
    assert out.assessments
    assert all(
        isinstance(a.details, ForecastResidualDetails) and a.details.expectation_model == model for a in out.assessments
    )


def test_forecast_residual_skips_without_history() -> None:
    out = forecast_residual(noisy(10), list(range(10)), LEVEL, LABELS)
    assert not out.assessments and len(out.skipped) == 10


# ------------------------------------------------------------------------------------------ severity policy
@pytest.mark.parametrize(
    ("score", "severity"),
    [(0.0, "normal"), (1.99, "normal"), (-2.0, "watch"), (2.99, "watch"), (3.0, "significant"),
     (-3.99, "significant"), (4.0, "extreme"), (-12.0, "extreme"), (None, "extreme")],
)  # fmt: skip
def test_standardized_severity(score: float | None, severity: str) -> None:
    assert classify_standardized(score, StandardizedThresholds()) == severity


@pytest.mark.parametrize(
    ("score", "severity"),
    [
        (0.0, "normal"),
        (1.49, "normal"),
        (1.5, "significant"),
        (-2.99, "significant"),
        (3.0, "extreme"),
        (None, "extreme"),
    ],
)
def test_iqr_severity(score: float | None, severity: str) -> None:
    assert classify_iqr(score, IQRFences()) == severity


def test_flagging_rules() -> None:
    assert is_flagged("significant", "significant") and is_flagged("extreme", "significant")
    assert not is_flagged("watch", "significant") and is_flagged("watch", "watch")
    assert flag_threshold("rolling_zscore", AnomalyConfig()) == 3.0
    assert flag_threshold("forecast_residual", AnomalyConfig(flag_severity="watch")) == 2.0
    assert flag_threshold("iqr", AnomalyConfig()) == 1.5
    assert flag_threshold("iqr", AnomalyConfig(flag_severity="extreme")) == 3.0


def test_config_validation() -> None:
    with pytest.raises(InvalidRequestError):
        StandardizedThresholds(watch=3.0, significant=2.0)
    with pytest.raises(InvalidRequestError):
        IQRFences(inner=3.0, outer=1.5)
    with pytest.raises(InvalidRequestError):
        AnomalyConfig(window=6, min_history=8)
    with pytest.raises(ValueError):
        AnomalyConfig(detector="isolation_forest")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AnomalyConfig(expectation_model="lstm")  # type: ignore[arg-type]
