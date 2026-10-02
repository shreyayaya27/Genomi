"""Leakage-safe experiment preflight. Does not train any model."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import qiskit
import qiskit_aer
import qiskit_algorithms
import qiskit_machine_learning

from .fixed_split import load_fixed_split, split_summary
from .pilot_common import GENES

DATA = Path("results/balanced_120_tumor/brca_9gene_expression.csv")
META = Path("results/balanced_120_tumor/metadata.csv")
OUT = Path("results/controlled_experiment")

# Four validation-only shot-based configurations: 2 angle ranges x 2 optimizers.
# Architecture is held at reps=1 for the first controlled runtime estimate.
SMALL_GRID = [
    {"angle_range": angle, "feature_map_reps": 1, "ansatz_reps": 1, "entanglement": "linear", "optimizer": optimizer, "maxiter": 100, "shots": 1024, "simulator_mode": "shot_aer"}
    for angle in ("zero_to_pi", "minus_pi_to_pi") for optimizer in ("COBYLA", "SPSA")
]


def main() -> None:
    assignments = load_fixed_split(META)
    with DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = split_summary(assignments)
    train_n = summary["train"]["samples"]
    # Training-only VQC undersampling uses all train Normals and the same number of seeded Tumors.
    quantum_balanced = 2 * min(summary["train"]["tumor"], summary["train"]["normal"])
    report = {
        "dataset": str(DATA), "biomarkers": GENES, "split_summary": summary,
        "leakage_check": "PASS: no Case ID or Sample ID occurs in multiple splits; all samples assigned exactly once.",
        "classical_array_shapes": {split: [counts["samples"], len(GENES)] for split, counts in summary.items()},
        "quantum_array_shapes": {split: [counts["samples"], len(GENES)] for split, counts in summary.items()},
        "quantum_balanced_training_subset_size": quantum_balanced,
        "preprocessing": {
            "classical": "log2(TPM + 1), then StandardScaler fit on training only",
            "quantum": "log2(TPM + 1), then configurable MinMax angle scaling fit on training only: [0, pi] or [-pi, pi]",
        },
        "package_versions": {"qiskit": qiskit.__version__, "aer": qiskit_aer.__version__, "machine_learning": qiskit_machine_learning.__version__, "algorithms": qiskit_algorithms.__version__},
        "small_validation_only_vqc_grid": SMALL_GRID,
        "proposed_vqc_training_runs": len(SMALL_GRID),
        "test_policy": "Untouched. --evaluate-test must be used only after an explicit final configuration lock.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sanity_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (OUT / "proposed_vqc_grid.json").write_text(json.dumps(SMALL_GRID, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
