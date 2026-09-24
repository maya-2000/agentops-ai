"""Same seed + configuration => identical dataset; different seed => different dataset."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from data.generator.generate import GenerationResult, generate_dataset

SMALL = {
    "customer_count": 100,
    "start_date": date(2026, 3, 1),
    "end_date": date(2026, 8, 31),
    "strict_event_validation": False,
}


def _without_timestamp(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text())
    manifest.pop("generation_timestamp")
    return manifest


def test_same_seed_reproduces_identical_data(small_dataset: GenerationResult, config_factory) -> None:  # type: ignore[no-untyped-def]
    rerun = generate_dataset(config_factory(**SMALL))
    assert rerun.checksums == small_dataset.checksums
    assert rerun.fingerprint == small_dataset.fingerprint
    assert json.loads(rerun.config.ground_truth_path.read_text()) == json.loads(
        small_dataset.config.ground_truth_path.read_text()
    )
    assert _without_timestamp(rerun.config.metadata_dir / "dataset_manifest.json") == _without_timestamp(
        small_dataset.config.metadata_dir / "dataset_manifest.json"
    )


@pytest.mark.parametrize("seed", [7])
def test_different_seed_changes_data(small_dataset: GenerationResult, config_factory, seed: int) -> None:  # type: ignore[no-untyped-def]
    other = generate_dataset(config_factory(seed=seed, **SMALL))
    assert other.fingerprint != small_dataset.fingerprint
    assert other.row_counts["customers"] == small_dataset.row_counts["customers"] == 100
