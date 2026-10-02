"""Persist and load the leakage-safe case-level split for the 233-sample experiment."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit

from .common import RESULTS

SPLIT_PATH = RESULTS / "fixed_split_assignments.csv"
SEED = 42
SPLITS = ("train", "validation", "test")


def create_fixed_split(metadata_path: Path, output_path: Path = SPLIT_PATH) -> Path:
    """Create the one permitted split; refuses to overwrite any existing assignment."""
    if output_path.exists():
        raise FileExistsError(f"Fixed split already exists: {output_path}. It will not be regenerated.")
    with metadata_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Metadata is empty.")
    cases: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        cases[row["case_id"]].append(row)
    # Stratify case groups by their label composition: Normal-only, Tumor-only, or paired.
    case_ids = sorted(cases)
    strata = []
    for case_id in case_ids:
        labels = {row["label"] for row in cases[case_id]}
        strata.append("paired" if len(labels) == 2 else next(iter(labels)))
    # First reserve 15% test; then split the remaining 85% into 70/15 overall.
    first = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_val_idx, test_idx = next(first.split(case_ids, strata))
    remaining_ids = [case_ids[i] for i in train_val_idx]
    remaining_strata = [strata[i] for i in train_val_idx]
    second = StratifiedShuffleSplit(n_splits=1, test_size=(0.15 / 0.85), random_state=SEED)
    train_idx, validation_idx = next(second.split(remaining_ids, remaining_strata))
    case_split = {case_ids[i]: "test" for i in test_idx}
    case_split.update({remaining_ids[i]: "train" for i in train_idx})
    case_split.update({remaining_ids[i]: "validation" for i in validation_idx})
    assigned = [{"case_id": row["case_id"], "sample_id": row["sample_id"], "label": row["label"], "split": case_split[row["case_id"]]} for row in rows]
    validate_assignments(assigned, expected_samples=len(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "sample_id", "label", "split"])
        writer.writeheader(); writer.writerows(assigned)
    return output_path


def validate_assignments(rows: list[dict[str, str]], expected_samples: int | None = None) -> None:
    if expected_samples is not None and len(rows) != expected_samples:
        raise ValueError(f"Expected {expected_samples} assignments, found {len(rows)}.")
    samples = [row["sample_id"] for row in rows]
    if len(samples) != len(set(samples)):
        raise ValueError("A sample occurs more than once in the fixed split.")
    case_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["split"] not in SPLITS:
            raise ValueError(f"Invalid split: {row['split']}")
        case_splits[row["case_id"]].add(row["split"])
    leaking = [case_id for case_id, split_set in case_splits.items() if len(split_set) != 1]
    if leaking:
        raise ValueError(f"Patient leakage: {len(leaking)} Case IDs occur in multiple splits.")
    for split in SPLITS:
        labels = {row["label"] for row in rows if row["split"] == split}
        if labels != {"0", "1"}:
            raise ValueError(f"Split {split} lacks one class.")


def load_fixed_split(metadata_path: Path, split_path: Path = SPLIT_PATH) -> list[dict[str, str]]:
    if not split_path.exists():
        raise FileNotFoundError(f"Fixed split is required but missing: {split_path}. Refusing to regenerate it.")
    with split_path.open(encoding="utf-8", newline="") as handle:
        assignments = list(csv.DictReader(handle))
    with metadata_path.open(encoding="utf-8", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    validate_assignments(assignments, expected_samples=len(metadata))
    expected = {(row["case_id"], row["sample_id"], row["label"]) for row in metadata}
    actual = {(row["case_id"], row["sample_id"], row["label"]) for row in assignments}
    if actual != expected:
        raise ValueError("Fixed split assignments do not exactly match the active metadata dataset.")
    return assignments


def split_summary(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    return {
        split: {
            "samples": sum(row["split"] == split for row in rows),
            "tumor": sum(row["split"] == split and row["label"] == "1" for row in rows),
            "normal": sum(row["split"] == split and row["label"] == "0" for row in rows),
            "unique_cases": len({row["case_id"] for row in rows if row["split"] == split}),
        }
        for split in SPLITS
    }
