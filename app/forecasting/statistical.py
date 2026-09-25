"""The statistical forecasting model: additive damped-trend exponential smoothing, ETS(A,Ad,N).

Why this model:

- Damped-trend smoothing (Gardner & McKenzie, 1985) follows a local level and trend but flattens
  the trend over the horizon. It is a strong, well-studied default for short business series.
- It has five interpretable parameters: level and trend smoothing (alpha, beta), damping (phi),
  and the initial level and trend.
- As a linear state-space model (Hyndman et al., 2008) it has **analytic prediction intervals**.
  They are computed by statsmodels' ``ETSModel`` with ``method="exact"``, which is deterministic
  (no simulation).

Seasonal ETS / Holt-Winters is deliberately **not** offered. The data window has 24 monthly
points, and backtest folds train on 12 to 23 months, so fewer than two full seasonal cycles are
available when a seasonal model would have to be validated.

The parameters are estimated by maximum likelihood with a deterministic optimiser. Optimiser
warnings are recorded as notes on the forecast rather than hidden.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from app.forecasting.base import ForecastMethod, MethodOutput, MethodUnavailableError

_MAX_ITERATIONS = 1000
_INDEX_ORIGIN = "2000-01"  # statsmodels needs a date index; only its monthly frequency matters


class ETSDampedTrendMethod(ForecastMethod):
    name = "ets_damped_trend"
    display_name = "ETS(A,Ad,N) damped-trend exponential smoothing"
    description = (
        "Additive-error exponential smoothing with a damped additive trend and no seasonality, "
        "fitted by maximum likelihood (statsmodels ETSModel)."
    )
    interval_method = "model-derived: analytic ETS(A,Ad,N) prediction variance (statsmodels, method='exact')"
    min_observations = 10

    def _forecast(self, y: np.ndarray, horizon: int, level: float) -> MethodOutput:
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel  # heavy import, loaded on first use

        # statsmodels 0.15 requires a pandas index for prediction frames (a bare ndarray fails).
        index = pd.period_range(_INDEX_ORIGIN, periods=len(y), freq="M")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model = ETSModel(
                    pd.Series(y, index=index),
                    error="add",
                    trend="add",
                    damped_trend=True,
                    seasonal=None,
                    initialization_method="estimated",
                )
                fitted = model.fit(disp=False, maxiter=_MAX_ITERATIONS)
                frame = fitted.get_prediction(start=len(y), end=len(y) + horizon - 1, method="exact").summary_frame(
                    alpha=1.0 - level
                )
            except Exception as exc:  # any fitting failure makes the candidate unavailable, never a crash
                raise MethodUnavailableError(f"ETS fit failed: {exc}") from exc
        mean = frame["mean"].to_numpy(dtype=float)
        lower = frame["pi_lower"].to_numpy(dtype=float)
        upper = frame["pi_upper"].to_numpy(dtype=float)
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
            raise MethodUnavailableError("ETS produced non-finite forecasts")
        parameters: dict[str, float | int | str] = {
            name: float(value) for name, value in zip(fitted.param_names, fitted.params, strict=True)
        }
        parameters["in_sample_mse"] = float(fitted.mse)
        parameters["aicc"] = float(fitted.aicc) if np.isfinite(fitted.aicc) else "undefined"
        notes = tuple(sorted({f"{w.category.__name__}: {w.message}" for w in caught}))
        return MethodOutput(tuple(mean.tolist()), tuple(lower.tolist()), tuple(upper.tolist()), parameters, notes=notes)
