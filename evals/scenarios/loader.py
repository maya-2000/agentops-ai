"""Load and select scenarios from a versioned dataset (``evals/datasets/<version>.json``)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from evals.scenarios.model import Category, EvaluationDataset, EvaluationScenario

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
DEFAULT_DATASET = "eval_v1"


def dataset_path(version: str = DEFAULT_DATASET) -> Path:
    path = DATASETS_DIR / f"{version}.json"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in DATASETS_DIR.glob("eval_v*.json")))
        raise FileNotFoundError(f"Unknown dataset {version!r}; available: {available}")
    return path


def load_dataset(version: str = DEFAULT_DATASET, path: Path | None = None) -> EvaluationDataset:
    raw = json.loads((path or dataset_path(version)).read_text(encoding="utf-8"))
    dataset = EvaluationDataset.model_validate(raw)
    if path is None and dataset.manifest.dataset_version != version:
        raise ValueError(f"{version}.json declares {dataset.manifest.dataset_version}")
    return dataset


def select(
    scenarios: Iterable[EvaluationScenario],
    *,
    suite: str | None = None,
    categories: Iterable[str] = (),
    scenario_ids: Iterable[str] = (),
) -> list[EvaluationScenario]:
    """Filter by suite (``critical``, ``multi_seed``; ``None`` or ``full`` for all), category and ID."""
    chosen = list(scenarios)
    wanted_ids = set(scenario_ids)
    if wanted_ids:
        known = {s.scenario_id for s in chosen}
        missing = sorted(wanted_ids - known)
        if missing:
            raise ValueError(f"Unknown scenarios: {', '.join(missing)}")
        chosen = [s for s in chosen if s.scenario_id in wanted_ids]
    wanted_categories = {Category(c) for c in categories}
    if wanted_categories:
        chosen = [s for s in chosen if s.category in wanted_categories]
    if suite not in (None, "full"):
        chosen = [s for s in chosen if suite in s.suites]
    return chosen
