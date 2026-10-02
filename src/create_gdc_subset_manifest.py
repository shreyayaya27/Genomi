"""Create a deterministic, metadata-matched GDC manifest subset without changing source files."""
from __future__ import annotations

import csv
from pathlib import Path

from .common import DATA, expression_files, file_id_from_expression

TARGET_PER_CLASS = 50
OUT_MANIFEST = DATA / "gdc_manifest_additional_100_balanced.txt"
OUT_SHEET = DATA / "gdc_sample_sheet_additional_100_balanced.tsv"


def main() -> None:
    manifests = [
        path for path in DATA.glob("gdc_manifest*.txt")
        if "additional_100" not in path.name and sum(1 for _ in path.open(encoding="utf-8")) == 836
    ]
    sheets = [path for path in DATA.glob("gdc_sample_sheet*.tsv") if sum(1 for _ in path.open(encoding="utf-8")) == 836]
    if len(manifests) != 1 or len(sheets) != 1:
        raise RuntimeError("Expected exactly one 835-file manifest and one matching 835-row Sample Sheet in data/.")
    with manifests[0].open(encoding="utf-8", newline="") as handle:
        manifest = {row["id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    with sheets[0].open(encoding="utf-8", newline="") as handle:
        sample_sheet = {row["File ID"]: row for row in csv.DictReader(handle, delimiter="\t")}
    existing_ids = {file_id_from_expression(path) for path in expression_files()}
    existing_rows = [sample_sheet[file_id] for file_id in existing_ids if file_id in sample_sheet]
    seen_samples = {row["Sample ID"] for row in existing_rows}
    seen_cases = {row["Case ID"] for row in existing_rows}
    selected: list[dict[str, str]] = []
    for tissue in ("Tumor", "Normal"):
        picked = 0
        # File-ID order makes this selection reproducible; it is not random sampling.
        for file_id in sorted(manifest):
            row = sample_sheet.get(file_id)
            if (file_id in existing_ids or not row or row["Tissue Type"] != tissue
                    or row["Sample ID"] in seen_samples or row["Case ID"] in seen_cases):
                continue
            selected.append({"id": file_id, **manifest[file_id]})
            seen_samples.add(row["Sample ID"]); seen_cases.add(row["Case ID"])
            picked += 1
            if picked == TARGET_PER_CLASS:
                break
        if picked != TARGET_PER_CLASS:
            raise RuntimeError(f"Only {picked} unique {tissue} rows could be selected.")
    if OUT_MANIFEST.exists() or OUT_SHEET.exists():
        raise FileExistsError("Subset files already exist; refusing to overwrite them.")
    fields = ["id", "filename", "md5", "size", "state"]
    with OUT_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader(); writer.writerows(selected)
    sheet_fields = list(next(iter(sample_sheet.values())).keys())
    with OUT_SHEET.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sheet_fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader(); writer.writerows([sample_sheet[row["id"]] for row in selected])
    print(f"Wrote {OUT_MANIFEST} and {OUT_SHEET}")
    print("Selected: Tumor=50, Normal=50; excludes existing UUIDs and duplicate Sample/Case IDs.")


if __name__ == "__main__":
    main()
