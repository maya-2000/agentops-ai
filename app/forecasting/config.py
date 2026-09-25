"""Typed forecasting configuration. Every tunable number lives here (no magic numbers in the models)."""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.analytics.errors import InvalidRequestError

ModelName = Literal["naive", "seasonal_naive", "moving_average", "drift", "ets_damped_trend"]
MODEL_NAMES: tuple[ModelName, ...] = get_args(ModelName)
BASELINE_MODEL: ModelName = "naive"
SelectionMetric = Literal["mae", "rmse"]

SUPPORTED_HORIZONS = range(1, 7)  # 1 to 6 months; 1, 3 and 6 are the documented standard horizons
MIN_RESIDUALS_FOR_INTERVAL = 3  # fewer residuals than this and no interval is reported


class ForecastConfig(BaseModel):
    """Configuration of backtesting, model selection and intervals.

    - ``minimum_history``: months in the first backtest training window. Every fold trains on at
      least this many months (expanding window).
    - ``min_backtest_folds``: a forecast needs at least this many rolling-origin folds, so the
      required history is ``minimum_history + (min_backtest_folds - 1) * backtest_step + horizon``.
    - ``max_zero_share``: series with a larger share of zero months are intermittent. The additive
      models here assume a continuous level, so such series return ``insufficient_data``.
    - ``mape_near_zero_fraction``: MAPE is withheld when any actual is zero or smaller than this
      fraction of the mean absolute actual (single tiny denominators would dominate it).
    - ``random_seed``: recorded in provenance. Every current method is deterministic (closed-form or
      a deterministic optimiser, analytic intervals), so no step draws random numbers.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    minimum_history: int = Field(default=12, ge=6, le=36)
    min_backtest_folds: int = Field(default=3, ge=1, le=24)
    backtest_step: int = Field(default=1, ge=1, le=6)
    candidates: tuple[ModelName, ...] = MODEL_NAMES
    selection_metric: SelectionMetric = "mae"
    season_length: int = Field(default=12, ge=2, le=12)
    moving_average_window: int = Field(default=3, ge=2, le=12)
    max_zero_share: float = Field(default=0.2, ge=0.0, le=1.0)
    mape_near_zero_fraction: float = Field(default=0.01, ge=0.0, le=0.5)
    random_seed: int = 42

    @model_validator(mode="after")
    def _baseline_required(self) -> ForecastConfig:
        if BASELINE_MODEL not in self.candidates:
            raise InvalidRequestError(f"The {BASELINE_MODEL!r} baseline must always be a candidate")
        if len(set(self.candidates)) != len(self.candidates):
            raise InvalidRequestError("Model candidates must be unique")
        return self

    def required_history(self, horizon: int) -> int:
        """Months needed for ``min_backtest_folds`` rolling-origin folds at this horizon."""
        return self.minimum_history + (self.min_backtest_folds - 1) * self.backtest_step + horizon
