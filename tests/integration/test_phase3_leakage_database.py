"""End-to-end leakage regression test on the real database.

A copy of the generated database is altered **after** a cutoff: revenue after the cutoff is
multiplied by 5, subscriptions starting after it get 5x the MRR, tickets created after it are
deleted, and feature adoption after it is set to 0.99. Forecasts made at the cutoff, and
anomaly detections ending at it, must be identical on the original and the altered database.
A control confirms the alterations are visible to requests made after the cutoff.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pytest

from app.anomalies import AnomalyReport, AnomalyService
from app.database.base import Database
from app.database.duckdb_backend import DuckDBDatabase
from app.forecasting import ForecastRequest, ForecastResult, ForecastService
from data.generator.generate import GenerationResult

pytestmark = pytest.mark.slow

CUTOFF = date(2026, 5, 31)
CASES: list[tuple[str, dict[str, str]]] = [
    ("revenue", {}),
    ("mrr", {}),
    ("customer_count", {"segment": "SMB"}),
    ("support_ticket_volume", {}),
    ("product_adoption", {"product_feature": "Reports"}),
]


@pytest.fixture(scope="module")
def altered_db(full_dataset: GenerationResult, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Database]:
    path = Path(tmp_path_factory.mktemp("altered")) / "altered.duckdb"
    shutil.copyfile(full_dataset.config.db_path, path)
    con = duckdb.connect(str(path))
    try:
        cutoff = CUTOFF.isoformat()
        con.execute(f"UPDATE daily_revenue SET revenue = revenue * 5 WHERE date > DATE '{cutoff}'")
        con.execute(
            "UPDATE subscriptions SET monthly_recurring_revenue = monthly_recurring_revenue * 5 "
            f"WHERE start_date > DATE '{cutoff}'"
        )
        con.execute(f"DELETE FROM support_tickets WHERE created_at >= DATE '{cutoff}' + INTERVAL 1 DAY")
        con.execute(f"UPDATE product_features SET adoption_rate = 0.99 WHERE date > DATE '{cutoff}'")
    finally:
        con.close()
    db = DuckDBDatabase(path, dataset_version=full_dataset.manifest["dataset_version"])
    yield db
    db.close()


def forecast_content(result: ForecastResult) -> dict[str, Any]:
    data = result.model_dump(mode="json", exclude={"provenance"})
    data["history"].pop("provenance")
    for key in ("operation_id", "execution_timestamp", "query_ids", "query_id"):
        data.pop(key)
        data["history"].pop(key, None)
    return data


def anomaly_content(report: AnomalyReport) -> list[dict[str, Any]]:
    return [
        r.model_dump(mode="json", exclude={"operation_id", "execution_timestamp", "query_id"}) for r in report.results
    ]


@pytest.mark.parametrize(("metric", "filters"), CASES)
@pytest.mark.parametrize("horizon", [1, 3])
def test_forecast_at_cutoff_ignores_altered_future(
    full_db: Database, altered_db: Database, metric: str, filters: dict[str, str], horizon: int
) -> None:
    request = ForecastRequest(metric=metric, horizon=horizon, filters=filters, cutoff_date=CUTOFF)
    original = ForecastService(full_db).forecast(request)
    altered = ForecastService(altered_db).forecast(request)
    assert original.status == "ok"
    assert forecast_content(original) == forecast_content(altered)


@pytest.mark.parametrize(("metric", "filters"), CASES)
@pytest.mark.parametrize("detector", ["rolling_zscore", "iqr", "forecast_residual"])
def test_anomalies_up_to_cutoff_ignore_altered_future(
    full_db: Database, altered_db: Database, metric: str, filters: dict[str, str], detector: str
) -> None:
    original = AnomalyService(full_db).detect(metric, end_date=CUTOFF, detector=detector, filters=filters)
    altered = AnomalyService(altered_db).detect(metric, end_date=CUTOFF, detector=detector, filters=filters)
    assert original.status == "ok" and original.results
    assert anomaly_content(original) == anomaly_content(altered)


def test_control_alterations_are_visible_after_the_cutoff(full_db: Database, altered_db: Database) -> None:
    original = ForecastService(full_db).forecast(metric="revenue", horizon=1)
    altered = ForecastService(altered_db).forecast(metric="revenue", horizon=1)
    assert original.forecast_points[0].predicted_value != altered.forecast_points[0].predicted_value
    tickets = AnomalyService(altered_db).detect("support_ticket_volume", detector="rolling_zscore")
    assert tickets.result("2026-06").observed_value == 0.0
