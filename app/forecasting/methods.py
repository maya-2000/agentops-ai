"""The allow-list of forecasting methods. Callers name a method; they never supply code or formulas."""

from __future__ import annotations

from app.forecasting.base import ForecastMethod
from app.forecasting.baselines import DriftMethod, MovingAverageMethod, NaiveMethod, SeasonalNaiveMethod
from app.forecasting.config import MODEL_NAMES, ForecastConfig
from app.forecasting.statistical import ETSDampedTrendMethod
from app.timeseries.errors import UnsupportedMethodError


def build_method(name: str, config: ForecastConfig | None = None) -> ForecastMethod:
    """Instantiate an allow-listed method with the configured parameters."""
    config = config or ForecastConfig()
    if name == "naive":
        return NaiveMethod()
    if name == "seasonal_naive":
        return SeasonalNaiveMethod(config.season_length)
    if name == "moving_average":
        return MovingAverageMethod(config.moving_average_window)
    if name == "drift":
        return DriftMethod()
    if name == "ets_damped_trend":
        return ETSDampedTrendMethod()
    raise UnsupportedMethodError(f"Unknown forecasting model {name!r}. Supported models: {', '.join(MODEL_NAMES)}")
