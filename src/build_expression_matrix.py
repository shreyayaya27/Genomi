from __future__ import annotations

import csv
import math

from .common import RESULTS, TPM_COLUMN, iter_gene_rows


def main() -> None:
    metadata_path = RESULTS / "metadata.csv"
    with metadata_path.open(encoding="utf-8", newline="") as handle:
        metadata = [row for row in csv.DictReader(handle) if row["status"] == "usable"]
    if not metadata:
        raise RuntimeError("No usable metadata rows. Run build_metadata.py and resolve its report first.")
    reference = list(iter_gene_rows(next_path(metadata[0]["file_id"])))
    # Retain the exact GDC/GENCODE identifier, including version and _PAR_Y
    # suffixes. Removing either would collapse 44 distinct PAR-Y rows.
    gene_ids = [row["gene_id"] for row in reference]
    if len(gene_ids) != len(set(gene_ids)):
        raise RuntimeError("Duplicate exact Ensembl IDs; refusing to overwrite columns.")
    out = RESULTS / "expression_matrix_tpm.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Sample_ID", *gene_ids, "Label"])
        for meta in metadata:
            rows = list(iter_gene_rows(next_path(meta["file_id"])))
            ids = [row["gene_id"] for row in rows]
            if ids != gene_ids:
                raise ValueError(f"Gene order/schema differs for {meta['file_id']}")
            values = []
            for row in rows:
                value = float(row[TPM_COLUMN])
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite TPM in {meta['file_id']}")
                values.append(value)
            writer.writerow([meta["sample_id"], *values, meta["label"]])


def next_path(file_id: str):
    from .common import expression_files, file_id_from_expression
    matches = [path for path in expression_files() if file_id_from_expression(path) == file_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one expression file for {file_id}; found {len(matches)}")
    return matches[0]


if __name__ == "__main__":
    main()
