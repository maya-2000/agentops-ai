"""Interpretable anomaly detectors (pure functions of a numeric monthly series).

Every statistic used to judge month ``t`` comes from the months **before** ``t``: the window
``[t - window, t - 1]``. The observation being judged never contributes to its own baseline,
dispersion, quartiles, forecast or threshold. Later months are never read.

Transforms (rolling z-score and IQR only):

- ``level``: the value itself.
- ``difference``: value(t) - value(t-1).
- ``pct_change``: value(t) / value(t-1) - 1. It is undefined when value(t-1) is 0 or missing.

The business series trend upward, so a level-based baseline would flag ordinary growth. The
change-based transforms compare this month's movement with recent movements instead. Expected
values are always reported in the metric's own unit: for ``pct_change`` the expected level is
``value(t-1) * (1 + baseline change)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from app.anomalies.config import AnomalyConfig, Severity
from app.anomalies.models import AnomalyDetails, ForecastResidualDetails, IQRDetails, RollingZScoreDetails
from app.anomalies.thresholds import classify_iqr, classify_standardized, flag_threshold
from app.forecasting.base import ForecastMethod, MethodUnavailableError
from app.forecasting.config import ForecastConfig
from app.forecasting.methods import build_method
from app.timeseries.metrics import Transform

_RELATIVE_ZERO = 1e-12  # dispersion at or below this (relative to the centre's scale) is treated as zero


@dataclass(frozen=True)
class Assessment:
    index: int
    observed: float
    expected: float
    deviation: float
    score: float | None
    threshold: float
    severity: Severity
    lower_bound: float | None
    upper_bound: float | None
    window_start: int
    window_end: int
    window_observations: int
    details: AnomalyDetails


@dataclass(frozen=True)
class Skip:
    index: int
    reason: str


@dataclass(frozen=True)
class DetectorOutput:
    assessments: list[Assessment]
    skipped: list[Skip]


# ---------------------------------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------------------------------


def transform_series(values: Sequence[float] | np.ndarray, kind: Transform) -> np.ndarray:
    """Transformed series aligned with the input (index 0 is ``nan`` for change transforms)."""
    y = np.asarray(values, dtype=float)
    if kind == "level":
        return y.copy()
    out = np.full(len(y), np.nan)
    previous, current = y[:-1], y[1:]
    if kind == "difference":
        out[1:] = current - previous
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(previous != 0, current / previous - 1.0, np.nan)
    out[1:] = np.where(np.isfinite(ratio), ratio, np.nan)
    return out


def implied_level(previous: float, centre: float, kind: Transform) -> float:
    """The metric level that a transformed-space value corresponds to, given last month's level."""
    if kind == "level":
        return centre
    if kind == "difference":
        return previous + centre
    return previous * (1.0 + centre)


# ---------------------------------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------------------------------


def rolling_zscore(
    values: Sequence[float] | np.ndarray, indices: Sequence[int], config: AnomalyConfig, transform: Transform
) -> DetectorOutput:
    """z = (x_t - mean(prior window)) / std(prior window, ddof=1), on the transformed series."""
    y = np.asarray(values, dtype=float)
    z = transform_series(y, transform)
    threshold = flag_threshold("rolling_zscore", config)
    assessments: list[Assessment] = []
    skipped: list[Skip] = []
    for i in indices:
        prior = _prior(z, i, config, transform, y, skipped)
        if prior is None:
            continue
        history, start = prior
        n = len(history)
        mean, std = float(np.mean(history)), float(np.std(history, ddof=1))
        previous = float(y[i - 1]) if transform != "level" else math.nan
        expected = implied_level(previous, mean, transform)
        score = _standardise(float(z[i]), mean, std)
        tail = _t_tail(score, n)
        assessments.append(
            Assessment(
                index=i,
                observed=float(y[i]),
                expected=expected,
                deviation=float(y[i]) - expected,
                score=score,
                threshold=threshold,
                severity=classify_standardized(score, config.thresholds),
                lower_bound=implied_level(previous, mean - threshold * std, transform),
                upper_bound=implied_level(previous, mean + threshold * std, transform),
                window_start=start,
                window_end=i - 1,
                window_observations=n,
                details=RollingZScoreDetails(
                    transform=transform,
                    transformed_value=float(z[i]),
                    baseline_mean=mean,
                    baseline_std=std,
                    tail_probability=tail,
                ),
            )
        )
    return DetectorOutput(assessments, skipped)


def iqr_detector(
    values: Sequence[float] | np.ndarray, indices: Sequence[int], config: AnomalyConfig, transform: Transform
) -> DetectorOutput:
    """Tukey fences from the prior window's quartiles (linear interpolation, Hyndman-Fan type 7)."""
    y = np.asarray(values, dtype=float)
    z = transform_series(y, transform)
    fences = config.iqr_fences
    multiplier = flag_threshold("iqr", config)
    assessments: list[Assessment] = []
    skipped: list[Skip] = []
    for i in indices:
        prior = _prior(z, i, config, transform, y, skipped)
        if prior is None:
            continue
        history, start = prior
        q1, median, q3 = (float(q) for q in np.percentile(history, [25, 50, 75]))
        iqr = q3 - q1
        x = float(z[i])
        if _is_zero(iqr, max(abs(q1), abs(q3))):
            score: float | None = 0.0 if q1 <= x <= q3 else None
        elif x > q3:
            score = (x - q3) / iqr
        elif x < q1:
            score = (x - q1) / iqr
        else:
            score = 0.0
        previous = float(y[i - 1]) if transform != "level" else math.nan
        expected = implied_level(previous, median, transform)
        assessments.append(
            Assessment(
                index=i,
                observed=float(y[i]),
                expected=expected,
                deviation=float(y[i]) - expected,
                score=score,
                threshold=multiplier,
                severity=classify_iqr(score, fences),
                lower_bound=implied_level(previous, q1 - multiplier * iqr, transform),
                upper_bound=implied_level(previous, q3 + multiplier * iqr, transform),
                window_start=start,
                window_end=i - 1,
                window_observations=len(history),
                details=IQRDetails(
                    transform=transform,
                    transformed_value=x,
                    q1=q1,
                    median=median,
                    q3=q3,
                    iqr=iqr,
                    lower_fence=q1 - fences.inner * iqr,
                    upper_fence=q3 + fences.inner * iqr,
                    outer_lower_fence=q1 - fences.outer * iqr,
                    outer_upper_fence=q3 + fences.outer * iqr,
                    inner_multiplier=fences.inner,
                    outer_multiplier=fences.outer,
                ),
            )
        )
    return DetectorOutput(assessments, skipped)


def forecast_residual(
    values: Sequence[float] | np.ndarray,
    indices: Sequence[int],
    config: AnomalyConfig,
    labels: Sequence[str],
    forecast_config: ForecastConfig | None = None,
) -> DetectorOutput:
    """Standardised one-step forecast residual, with dispersion from prior one-step residuals.

    For month t the expectation model is fitted on the ``window`` months before t and forecasts t
    (one step). The prior residuals come from the same procedure at each earlier month, so none of
    them uses x_t. The expected value is the forecast plus the mean prior residual (bias
    correction), and ``score = (x_t - expected) / std(prior residuals)``.
    """
    y = np.asarray(values, dtype=float)
    method = build_method(config.expectation_model, forecast_config or ForecastConfig())
    threshold = flag_threshold("forecast_residual", config)
    cache: dict[int, tuple[float, int] | None] = {}

    def one_step(j: int) -> tuple[float, int] | None:
        if j not in cache:
            cache[j] = _one_step_forecast(y, j, config, method)
        return cache[j]

    assessments: list[Assessment] = []
    skipped: list[Skip] = []
    for i in indices:
        if not math.isfinite(y[i]):
            skipped.append(Skip(i, "missing observation"))
            continue
        current = one_step(i)
        if current is None:
            skipped.append(
                Skip(
                    i,
                    f"the {config.expectation_model} expectation could not be fitted: fewer than "
                    f"{config.min_history} consecutive observed months before this month",
                )
            )
            continue
        forecast, training_start = current
        start = max(0, i - config.window)
        prior = []
        for j in range(start, i):
            fitted = one_step(j) if math.isfinite(y[j]) else None
            if fitted is not None:
                prior.append(float(y[j]) - fitted[0])
        if len(prior) < config.min_history:
            skipped.append(
                Skip(
                    i,
                    f"only {len(prior)} prior one-step residual(s) in the {config.window}-month window; "
                    f"{config.min_history} required",
                )
            )
            continue
        residuals = np.array(prior)
        mean, std = float(np.mean(residuals)), float(np.std(residuals, ddof=1))
        residual = float(y[i]) - forecast
        expected = forecast + mean
        score = _standardise(residual, mean, std)
        assessments.append(
            Assessment(
                index=i,
                observed=float(y[i]),
                expected=expected,
                deviation=float(y[i]) - expected,
                score=score,
                threshold=threshold,
                severity=classify_standardized(score, config.thresholds),
                lower_bound=expected - threshold * std,
                upper_bound=expected + threshold * std,
                window_start=start,
                window_end=i - 1,
                window_observations=len(prior),
                details=ForecastResidualDetails(
                    expectation_model=config.expectation_model,
                    training_start=labels[training_start],
                    training_end=labels[i - 1],
                    one_step_forecast=forecast,
                    residual=residual,
                    residual_mean=mean,
                    residual_std=std,
                    standardized_residual=score,
                    tail_probability=_t_tail(score, len(prior)),
                ),
            )
        )
    return DetectorOutput(assessments, skipped)


# ---------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------


def _prior(
    z: np.ndarray, i: int, config: AnomalyConfig, transform: Transform, y: np.ndarray, skipped: list[Skip]
) -> tuple[np.ndarray, int] | None:
    """Finite transformed values in the window before month ``i`` (never month ``i`` itself)."""
    if not math.isfinite(y[i]):
        skipped.append(Skip(i, "missing observation"))
        return None
    if not math.isfinite(z[i]):
        reason = (
            "the month-over-month change is undefined (previous month missing or zero)"
            if transform != "level"
            else "missing observation"
        )
        skipped.append(Skip(i, reason))
        return None
    start = max(0, i - config.window)
    window = z[start:i]
    history = window[np.isfinite(window)]
    if len(history) < config.min_history:
        skipped.append(
            Skip(
                i,
                f"only {len(history)} prior observation(s) in the {config.window}-month window; "
                f"{config.min_history} required",
            )
        )
        return None
    return history, start


def _one_step_forecast(
    y: np.ndarray, j: int, config: AnomalyConfig, method: ForecastMethod
) -> tuple[float, int] | None:
    """One-step forecast of month ``j`` from the contiguous observed months in the window before it."""
    start = max(0, j - config.window)
    window = y[start:j]
    finite = np.isfinite(window)
    if not finite.all():
        last_gap = int(np.where(~finite)[0][-1])
        start, window = start + last_gap + 1, window[last_gap + 1 :]
    if len(window) < config.min_history:
        return None
    try:
        output = method.forecast(window, 1, 0.95)
    except MethodUnavailableError:
        return None
    return float(output.mean[0]), start


def _is_zero(dispersion: float, centre_scale: float) -> bool:
    return dispersion <= _RELATIVE_ZERO * max(1.0, centre_scale)


def _standardise(value: float, mean: float, std: float) -> float | None:
    if _is_zero(std, abs(mean)):
        return 0.0 if math.isclose(value, mean, rel_tol=1e-12, abs_tol=1e-12) else None
    return (value - mean) / std


def _t_tail(score: float | None, n: int) -> float | None:
    """Two-sided tail probability of a new observation vs a sample of n (Student t, n-1 df).

    ``(x - mean) / (std * sqrt(1 + 1/n))`` follows t(n-1) for i.i.d. normal data. ``None`` when
    the score is undefined.
    """
    if score is None or n < 2:
        return None
    t = score / math.sqrt(1.0 + 1.0 / n)
    return float(2.0 * stats.t.sf(abs(t), df=n - 1))
