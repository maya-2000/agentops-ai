"""Forecasting methods on synthetic series with known answers."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from app.forecasting import MODEL_NAMES, ForecastConfig, build_method
from app.forecasting.base import MethodUnavailableError, normal_quantile
from app.forecasting.baselines import DriftMethod, MovingAverageMethod, NaiveMethod, SeasonalNaiveMethod
from app.forecasting.statistical import ETSDampedTrendMethod
from app.timeseries import UnsupportedMethodError
from tests.phase3_support import constant, linear, noisy, seasonal

Z95 = 1.959963984540054


def test_normal_quantile() -> None:
    assert normal_quantile(0.95) == pytest.approx(Z95)
    assert normal_quantile(0.8) == pytest.approx(1.2815515655)


def test_registry_builds_every_model() -> None:
    config = ForecastConfig()
    for name in MODEL_NAMES:
        assert build_method(name, config).name == name
    with pytest.raises(UnsupportedMethodError):
        build_method("prophet", config)
    with pytest.raises(UnsupportedMethodError):
        build_method("__import__('os')", config)


# ------------------------------------------------------------------------------------------ naive
def test_naive_repeats_last_value_with_sqrt_h_interval() -> None:
    y = np.array([10.0, 12.0, 11.0, 15.0])
    out = NaiveMethod().forecast(y, 3, 0.95)
    assert out.mean == (15.0, 15.0, 15.0)
    sigma = math.sqrt((2**2 + 1**2 + 4**2) / 3)  # residuals 2, -1, 4
    assert out.parameters["residual_sigma"] == pytest.approx(sigma)
    assert out.upper is not None and out.lower is not None
    for h in range(1, 4):
        assert out.upper[h - 1] - 15.0 == pytest.approx(Z95 * sigma * math.sqrt(h))
        assert 15.0 - out.lower[h - 1] == pytest.approx(Z95 * sigma * math.sqrt(h))


def test_naive_on_constant_series_has_zero_width_interval() -> None:
    out = NaiveMethod().forecast(np.array(constant(12)), 2, 0.95)
    assert out.mean == (100.0, 100.0) and out.lower == (100.0, 100.0) and out.upper == (100.0, 100.0)


def test_naive_without_enough_residuals_reports_no_interval() -> None:
    out = NaiveMethod().forecast(np.array([5.0, 6.0]), 2, 0.95)
    assert out.mean == (6.0, 6.0) and out.lower is None and out.interval_note


# --------------------------------------------------------------------------------- seasonal naive
def test_seasonal_naive_reproduces_a_pure_seasonal_pattern() -> None:
    y = np.array(seasonal(36))
    out = SeasonalNaiveMethod(12).forecast(y, 6, 0.95)
    assert out.mean == pytest.approx(seasonal(42)[36:])
    assert out.lower == pytest.approx(out.mean) and out.upper == pytest.approx(out.mean)  # zero residuals


def test_seasonal_naive_wraps_beyond_one_season_and_widens_by_cycle() -> None:
    y = np.arange(1.0, 15.0)  # 14 points, season 4
    out = SeasonalNaiveMethod(4).forecast(y, 6, 0.95)
    assert out.mean == (11.0, 12.0, 13.0, 14.0, 11.0, 12.0)
    sigma = 4.0  # every seasonal difference equals 4
    widths = [u - m for u, m in zip(out.upper or (), out.mean, strict=True)]
    assert widths[:4] == pytest.approx([Z95 * sigma] * 4)
    assert widths[4:] == pytest.approx([Z95 * sigma * math.sqrt(2)] * 2)


def test_seasonal_naive_needs_one_full_season() -> None:
    with pytest.raises(MethodUnavailableError):
        SeasonalNaiveMethod(12).forecast(np.arange(11.0), 1, 0.95)
    assert SeasonalNaiveMethod(12).forecast(np.arange(12.0), 1, 0.95).lower is None  # no residuals yet


# -------------------------------------------------------------------------------- moving average
def test_moving_average_uses_the_last_window() -> None:
    y = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    assert MovingAverageMethod(3).forecast(y, 2, 0.95).mean == (20.0, 20.0)
    out = MovingAverageMethod(3).forecast(y, 1, 0.95)
    # one-step errors of the method on its own history: t=3..5 (only 2 two-step errors exist, so h=1 here)
    errors = [10 - 2, 20 - 5, 30 - 11]
    sigma = math.sqrt(sum(e * e for e in errors) / 3)
    assert out.upper is not None and out.upper[0] - 20.0 == pytest.approx(Z95 * sigma)


def test_moving_average_without_enough_h_step_errors_has_no_interval() -> None:
    out = MovingAverageMethod(3).forecast(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 2, 0.95)
    assert out.lower is None and out.interval_note


# ---------------------------------------------------------------------------------------- drift
def test_drift_continues_a_linear_trend_exactly() -> None:
    out = DriftMethod().forecast(np.array(linear(10, 100.0, 10.0)), 3, 0.95)
    assert out.mean == pytest.approx((200.0, 210.0, 220.0))
    assert out.parameters["slope_per_month"] == pytest.approx(10.0)
    assert out.lower == pytest.approx(out.mean)  # zero residual variance


def test_drift_interval_formula() -> None:
    y = np.array([10.0, 13.0, 14.0, 19.0, 20.0])
    out = DriftMethod().forecast(y, 2, 0.95)
    slope = 10.0 / 4
    residuals = [3 - slope, 1 - slope, 5 - slope, 1 - slope]
    sigma = math.sqrt(sum(r * r for r in residuals) / (len(residuals) - 1))
    for h in (1, 2):
        expected = Z95 * sigma * math.sqrt(h * (1 + h / 4))
        assert out.upper is not None and out.upper[h - 1] - out.mean[h - 1] == pytest.approx(expected)


# ---------------------------------------------------------------------------------------- ETS
def test_ets_continues_a_noisy_trend_and_is_deterministic() -> None:
    y = np.array(noisy(24, level=1000.0, slope=20.0, sd=5.0))
    first = ETSDampedTrendMethod().forecast(y, 3, 0.95)
    second = ETSDampedTrendMethod().forecast(y, 3, 0.95)
    assert first == second
    assert first.mean[0] > y[-1] - 30  # follows the upward trend (damped, so not faster than it)
    assert first.mean[0] < first.mean[1] < first.mean[2]
    assert {"smoothing_level", "smoothing_trend", "damping_trend"} <= set(first.parameters)
    assert 0 < float(first.parameters["damping_trend"]) <= 1


def test_ets_needs_ten_observations() -> None:
    with pytest.raises(MethodUnavailableError):
        ETSDampedTrendMethod().forecast(np.arange(9.0), 1, 0.95)


# ------------------------------------------------------------------------------------ all methods
@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("horizon", [1, 3, 6])
def test_every_method_returns_ordered_bounds_and_widening_intervals(name: str, horizon: int) -> None:
    y = np.array(noisy(24))
    out = build_method(name).forecast(y, horizon, 0.95)
    assert len(out.mean) == horizon
    assert out.lower is not None and out.upper is not None
    widths = []
    for low, mean, high in zip(out.lower, out.mean, out.upper, strict=True):
        assert low <= mean <= high
        widths.append(high - low)
    if name != "moving_average":  # empirical per-step errors need not be monotone
        assert all(b >= a - 1e-9 for a, b in pairwise(widths))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_method_rejects_missing_values(name: str) -> None:
    y = np.array(noisy(24))
    y[5] = np.nan
    with pytest.raises(MethodUnavailableError):
        build_method(name).forecast(y, 1, 0.95)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_method_handles_zero_values(name: str) -> None:
    y = np.array([0.0, 5.0, 0.0, 7.0, 3.0, 0.0, 6.0, 4.0, 5.0, 0.0, 8.0, 6.0, 7.0, 5.0])
    out = build_method(name).forecast(y, 2, 0.95)
    assert all(math.isfinite(v) for v in out.mean)


def test_higher_confidence_gives_wider_intervals() -> None:
    y = np.array(noisy(24))
    narrow = NaiveMethod().forecast(y, 1, 0.8)
    wide = NaiveMethod().forecast(y, 1, 0.99)
    assert narrow.upper is not None and wide.upper is not None
    assert wide.upper[0] - wide.mean[0] > narrow.upper[0] - narrow.mean[0]
