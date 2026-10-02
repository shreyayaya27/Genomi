"""Create a GDC manifest for Normal files not yet present locally."""
from __future__ import annotations

import csv

from .common import DATA, expression_files, file_id_from_expression

OUT = DATA / "gdc_manifest_remaining_normal.txt"


def main() -> None:
    manifests = [
        path for path in DATA.glob("gdc_manifest*.txt")
        if "additional_" not in path.name and "remaining_" not in path.name
        and sum(1 for _ in path.open(encoding="utf-8")) == 836
    ]
    sheets = [path for path in DATA.glob("gdc_sample_sheet*.tsv") if sum(1 for _ in path.open(encoding="utf-8")) == 836]
    if len(manifests) != 1 or len(sheets) != 1:
        raise RuntimeError("Expected exactly one full 835-file manifest and matching 835-row Sample Sheet.")
    with manifests[0].open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))
    with sheets[0].open(encoding="utf-8", newline="") as handle:
        tissue = {row["File ID"]: row["Tissue Type"] for row in csv.DictReader(handle, delimiter="\t")}
    existing = {file_id_from_expression(path) for path in expression_files()}
    remaining = [row for row in manifest if tissue.get(row["id"]) == "Normal" and row["id"] not in existing]
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing {OUT}")
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "filename", "md5", "size", "state"], delimiter="\t")
        writer.writeheader(); writer.writerows(remaining)
    print(f"Wrote {len(remaining)} remaining Normal files to {OUT}")


if __name__ == "__main__":
    main()
