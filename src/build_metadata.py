from __future__ import annotations

import csv
from collections import Counter

from .common import LABELS, RESULTS, expression_files, file_id_from_expression, read_manifest, read_sample_sheet, write_report


FIELDS = ["sample_id", "case_id", "file_id", "file_name", "tissue_type", "tumor_descriptor", "specimen_type", "label", "status"]


def main() -> None:
    sheet, manifest = read_sample_sheet(), read_manifest()
    files = expression_files()
    rows = []
    for path in files:
        file_id = file_id_from_expression(path)
        meta = sheet.get(file_id, {})
        tissue = meta.get("Tissue Type", "")
        status = "usable" if file_id in manifest and tissue in LABELS and meta.get("Sample ID") else "unusable"
        rows.append({
            "sample_id": meta.get("Sample ID", ""), "case_id": meta.get("Case ID", ""),
            "file_id": file_id, "file_name": path.name, "tissue_type": tissue,
            "tumor_descriptor": meta.get("Tumor Descriptor", ""), "specimen_type": meta.get("Specimen Type", ""),
            "label": LABELS.get(tissue, ""), "status": status,
        })
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    duplicate_samples = [key for key, n in Counter(r["sample_id"] for r in rows if r["sample_id"]).items() if n > 1]
    duplicate_cases = [key for key, n in Counter(r["case_id"] for r in rows if r["case_id"]).items() if n > 1]
    report = [
        "METADATA BUILD REPORT", f"Expression files discovered: {len(files)}", f"Usable rows: {sum(r['status'] == 'usable' for r in rows)}",
        f"Tumor: {sum(r['tissue_type'] == 'Tumor' for r in rows)}", f"Normal: {sum(r['tissue_type'] == 'Normal' for r in rows)}",
        f"Missing tissue type: {sum(not r['tissue_type'] for r in rows)}", f"Otherwise unusable: {sum(r['status'] != 'usable' for r in rows)}",
        f"Duplicate Sample IDs: {len(duplicate_samples)}", f"Duplicate Case IDs: {len(duplicate_cases)}",
    ]
    write_report(RESULTS / "metadata_report.txt", report)


if __name__ == "__main__":
    main()
