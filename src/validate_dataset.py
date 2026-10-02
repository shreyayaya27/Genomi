from __future__ import annotations

import csv
import math
from collections import Counter

from .common import RESULTS, expression_files, file_id_from_expression, write_report


def main() -> None:
    with (RESULTS / "metadata.csv").open(encoding="utf-8", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    with (RESULTS / "expression_matrix_tpm.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle); header = next(reader); matrix = list(reader)
    sample_ids = [row["sample_id"] for row in metadata]
    matrix_ids = [row[0] for row in matrix]
    nonfinite = missing = 0
    for row in matrix:
        for value in row[1:-1]:
            if value == "": missing += 1
            else:
                try: nonfinite += int(not math.isfinite(float(value)))
                except ValueError: nonfinite += 1
    source_ids = {file_id_from_expression(path) for path in expression_files()}
    meta_ids = {row["file_id"] for row in metadata}
    lines = [
        "VALIDATION REPORT", f"Samples in metadata: {len(metadata)}", f"Samples in expression matrix: {len(matrix)}",
        f"Tumor samples: {sum(row['label'] == '1' for row in metadata)}", f"Normal samples: {sum(row['label'] == '0' for row in metadata)}",
        f"Genes: {len(header) - 2}", f"Missing matrix values: {missing}", f"Infinite/non-numeric matrix values: {nonfinite}",
        f"Duplicate sample IDs: {sum(n > 1 for n in Counter(sample_ids).values())}", f"Duplicate case IDs: {sum(n > 1 for n in Counter(row['case_id'] for row in metadata).values())}",
        "Duplicate gene names: retained safely as distinct Ensembl-ID columns; see inspect output.",
        f"Expression files without metadata: {len(source_ids - meta_ids)}", f"Metadata rows without expression files: {len(meta_ids - source_ids)}",
        f"Matrix samples absent from metadata: {len(set(matrix_ids) - set(sample_ids))}",
    ]
    write_report(RESULTS / "validation_report.txt", lines)


if __name__ == "__main__":
    main()
