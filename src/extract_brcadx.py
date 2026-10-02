from __future__ import annotations

import csv

from .build_expression_matrix import next_path
from .common import DATA, RESULTS, TPM_COLUMN, iter_gene_rows, write_report

GENES = ["NEK2", "PKMYT1", "MMP11", "CPA1", "COL10A1", "HSD17B13", "CA4", "MYOC", "LYVE1"]


def main() -> None:
    with (RESULTS / "metadata.csv").open(encoding="utf-8", newline="") as handle:
        metadata = [row for row in csv.DictReader(handle) if row["status"] == "usable"]
    mapping = {gene: [] for gene in GENES}
    first_rows = list(iter_gene_rows(next_path(metadata[0]["file_id"])))
    for row in first_rows:
        if row["gene_name"] in mapping:
            mapping[row["gene_name"]].append(row["gene_id"])
    output = DATA / "brca_9gene_expression.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["Sample_ID", *GENES, "Label"])
        for meta in metadata:
            values = {row["gene_name"]: row[TPM_COLUMN] for row in iter_gene_rows(next_path(meta["file_id"])) if row["gene_name"] in mapping and len(mapping[row["gene_name"]]) == 1}
            writer.writerow([meta["sample_id"], *[values.get(gene, "") for gene in GENES], meta["label"]])
    lines = ["BRCA DX 9-GENE EXTRACTION", "Expression: TPM (tpm_unstranded)"]
    for gene in GENES:
        ids = mapping[gene]
        lines.append(f"{gene}: " + (", ".join(ids) if ids else "MISSING") + (" [AMBIGUOUS]" if len(ids) > 1 else ""))
    write_report(RESULTS / "brcadx_report.txt", lines)


if __name__ == "__main__":
    main()
