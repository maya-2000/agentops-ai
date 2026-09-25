"""Deterministic monthly forecasting with rolling-origin backtests and baseline comparison."""

from app.forecasting.backtest import BacktestFold, BacktestResult, fold_origins, rolling_origin_backtest
from app.forecasting.config import BASELINE_MODEL, MODEL_NAMES, SUPPORTED_HORIZONS, ForecastConfig, ModelName
from app.forecasting.evaluation import ErrorMetrics, error_metrics
from app.forecasting.methods import build_method
from app.forecasting.models import (
    CandidateSummary,
    ForecastPoint,
    ForecastRequest,
    ForecastResult,
    ModelSpecification,
    PredictionInterval,
    SeasonalityAssessment,
)
from app.forecasting.selection import SelectionEvaluation, evaluate_selection, select_model
from app.forecasting.service import ForecastService, forecast_metric, forecast_series

__all__ = [
    "BASELINE_MODEL",
    "MODEL_NAMES",
    "SUPPORTED_HORIZONS",
    "BacktestFold",
    "BacktestResult",
    "CandidateSummary",
    "ErrorMetrics",
    "ForecastConfig",
    "ForecastPoint",
    "ForecastRequest",
    "ForecastResult",
    "ForecastService",
    "ModelName",
    "ModelSpecification",
    "PredictionInterval",
    "SeasonalityAssessment",
    "SelectionEvaluation",
    "build_method",
    "error_metrics",
    "evaluate_selection",
    "fold_origins",
    "forecast_metric",
    "forecast_series",
    "rolling_origin_backtest",
    "select_model",
]
