"""Create a reproducible 120-Tumor + all-Normal working subset; source data is untouched."""
from __future__ import annotations

import csv
import random

from .common import DATA, RESULTS
from .pilot_common import GENES

SEED = 42
TARGET_TUMOR = 120
OUT_DIR = RESULTS / "balanced_120_tumor"


def main() -> None:
    with (RESULTS / "metadata.csv").open(encoding="utf-8", newline="") as handle:
        metadata = [row for row in csv.DictReader(handle) if row["status"] == "usable"]
    with (DATA / "brca_9gene_expression.csv").open(encoding="utf-8", newline="") as handle:
        expression = {row["Sample_ID"]: row for row in csv.DictReader(handle)}
    tumors = [row for row in metadata if row["label"] == "1"]
    normals = [row for row in metadata if row["label"] == "0"]
    # One representative per case prevents duplicate-case selection; fixed seed is reproducible.
    representative = {}
    for row in sorted(tumors, key=lambda item: item["sample_id"]):
        representative.setdefault(row["case_id"], row)
    candidates = list(representative.values())
    if len(candidates) < TARGET_TUMOR:
        raise RuntimeError(f"Need {TARGET_TUMOR} unique Tumor cases; found {len(candidates)}.")
    selected_tumors = random.Random(SEED).sample(candidates, TARGET_TUMOR)
    selected = sorted([*selected_tumors, *normals], key=lambda row: (row["label"], row["sample_id"]))
    if OUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing {OUT_DIR}")
    OUT_DIR.mkdir(parents=True)
    meta_fields = list(metadata[0])
    with (OUT_DIR / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=meta_fields); writer.writeheader(); writer.writerows(selected)
    fields = ["Sample_ID", *GENES, "Label"]
    with (OUT_DIR / "brca_9gene_expression.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in selected:
            writer.writerow(expression[row["sample_id"]])
    report = [
        "TEMPORARY BALANCED WORKING SUBSET", f"Seed: {SEED}", f"Tumor: {len(selected_tumors)} (one per Case ID)",
        f"Normal: {len(normals)} (all available local Normal samples)", f"Total: {len(selected)}",
        "Source GDC files and full derived data were not modified or deleted.",
        "Tumor cases were selected randomly with a fixed seed; selection is for model-training/evaluation experiments only.",
    ]
    (OUT_DIR / "subset_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Created {OUT_DIR}: {len(selected_tumors)} Tumor + {len(normals)} Normal")


if __name__ == "__main__":
    main()
