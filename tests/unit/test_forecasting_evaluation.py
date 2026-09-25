"""Error metrics, including the zero-denominator policy for MAPE and WAPE."""

from __future__ import annotations

import math

import pytest

from app.forecasting import error_metrics


def test_hand_computed_metrics() -> None:
    m = error_metrics("x", 2, 2, [110.0, 90.0, 100.0], [100.0, 100.0, 80.0])
    # errors (forecast - actual): +10, -10, +20
    assert m.sample_count == 3 and m.fold_count == 2 and m.horizon == 2
    assert m.mae == pytest.approx(40 / 3)
    assert m.rmse == pytest.approx(math.sqrt((100 + 100 + 400) / 3))
    assert m.bias == pytest.approx(20 / 3)
    assert m.wape == pytest.approx(40 / 280)
    assert m.mape == pytest.approx((0.1 + 0.1 + 0.25) / 3)
    assert m.mape_note is None


def test_zero_actual_withholds_mape_but_keeps_wape() -> None:
    m = error_metrics("x", 1, 3, [1.0, 2.0, 1.0], [0.0, 2.0, 2.0])
    assert m.mape is None and m.mape_note and "zero" in m.mape_note
    assert m.wape == pytest.approx(2 / 4)
    assert m.mae == pytest.approx(2 / 3)


def test_near_zero_actual_withholds_mape() -> None:
    m = error_metrics("x", 1, 3, [1.0, 100.0, 100.0], [0.5, 100.0, 100.0], near_zero_fraction=0.01)
    assert m.mape is None and m.mape_note and "below" in m.mape_note
    assert error_metrics("x", 1, 3, [1.0, 100.0, 100.0], [0.5, 100.0, 100.0], near_zero_fraction=0.0).mape is not None


def test_all_zero_actuals() -> None:
    m = error_metrics("x", 1, 2, [1.0, 1.0], [0.0, 0.0])
    assert m.wape is None and m.mape is None and m.mae == pytest.approx(1.0)


def test_no_predictions() -> None:
    m = error_metrics("x", 3, 0, [], [])
    assert m.sample_count == 0 and m.mae is None and m.rmse is None and m.bias is None


def test_interval_coverage_counts_only_points_with_bounds() -> None:
    m = error_metrics(
        "x", 1, 4, [10.0, 10.0, 10.0, 10.0], [9.0, 12.0, 10.0, 50.0], [8.0, 8.0, None, 8.0], [11.0, 11.0, None, 11.0]
    )
    assert m.interval_sample_count == 3
    assert m.interval_coverage == pytest.approx(1 / 3)
    assert error_metrics("x", 1, 1, [1.0], [1.0]).interval_coverage is None


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError):
        error_metrics("x", 1, 1, [1.0, 2.0], [1.0])
