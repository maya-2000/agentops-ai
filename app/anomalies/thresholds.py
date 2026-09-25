"""The severity policy: deterministic, documented cut-offs per detector family.

Standardised detectors (rolling z-score, forecast residual), with ``s`` the standardised score:

    |s| < 2          normal
    2 <= |s| < 3     watch
    3 <= |s| < 4     significant
    |s| >= 4         extreme

IQR detector (Tukey's fences, ``s`` = distance beyond the nearer quartile in IQR units):

    |s| < 1.5        normal        (inside the inner fences)
    1.5 <= |s| < 3   significant   (Tukey "outlier")
    |s| >= 3         extreme       (Tukey "far out")

Tukey defines no band below the inner fence, so the IQR detector never reports ``watch``.
For a normal distribution the inner fence lies about 2.7 standard deviations from the median and
the outer fence about 4.7, broadly in line with the standardised cut-offs.

A score is undefined when the historical window has zero dispersion (all values equal) and the
observation differs from it. That observation lies outside everything previously seen, so it is
classed ``extreme`` with ``score=None``.
"""

from __future__ import annotations

from app.anomalies.config import (
    SEVERITY_ORDER,
    AnomalyConfig,
    DetectorName,
    IQRFences,
    Severity,
    StandardizedThresholds,
)


def classify_standardized(score: float | None, thresholds: StandardizedThresholds) -> Severity:
    if score is None:
        return "extreme"
    magnitude = abs(score)
    if magnitude >= thresholds.extreme:
        return "extreme"
    if magnitude >= thresholds.significant:
        return "significant"
    if magnitude >= thresholds.watch:
        return "watch"
    return "normal"


def classify_iqr(score: float | None, fences: IQRFences) -> Severity:
    if score is None:
        return "extreme"
    magnitude = abs(score)
    if magnitude >= fences.outer:
        return "extreme"
    if magnitude >= fences.inner:
        return "significant"
    return "normal"


def flag_threshold(detector: DetectorName, config: AnomalyConfig) -> float:
    """The score at which a month is flagged (``is_anomaly``) for this detector and configuration."""
    if detector == "iqr":
        return config.iqr_fences.outer if config.flag_severity == "extreme" else config.iqr_fences.inner
    return float(getattr(config.thresholds, config.flag_severity))


def is_flagged(severity: Severity, flag_severity: Severity) -> bool:
    return SEVERITY_ORDER.index(severity) >= SEVERITY_ORDER.index(flag_severity)


def severity_policy_text(detector: DetectorName, config: AnomalyConfig) -> str:
    if detector == "iqr":
        f = config.iqr_fences
        return (
            f"Tukey fences on the prior window: normal inside Q1-{f.inner:g}*IQR..Q3+{f.inner:g}*IQR; significant "
            f"beyond the inner fence; extreme beyond Q1-{f.outer:g}*IQR..Q3+{f.outer:g}*IQR. Flagged from "
            f"{'extreme' if config.flag_severity == 'extreme' else 'significant'}."
        )
    t = config.thresholds
    return (
        f"|score| < {t.watch:g} normal; {t.watch:g}-{t.significant:g} watch; {t.significant:g}-{t.extreme:g} "
        f"significant; >= {t.extreme:g} extreme. Flagged from {config.flag_severity}."
    )
