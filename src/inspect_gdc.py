from __future__ import annotations

from collections import Counter

from .common import expression_files, iter_gene_rows, manifest_path, read_manifest, read_sample_sheet, sample_sheet_path


def main() -> None:
    files = expression_files()
    if not files:
        raise RuntimeError("No STAR Counts TSV files found under data/")
    first = files[0]
    rows = list(iter_gene_rows(first))
    symbols = Counter(row["gene_name"] for row in rows if row["gene_name"])
    with first.open(encoding="utf-8") as handle:
        comment = handle.readline().rstrip()
        columns = handle.readline().rstrip().split("\t")
    sheet = read_sample_sheet()
    print(f"Expression example: {first}")
    print(f"Metadata row: {comment}")
    print("Columns: " + ", ".join(columns))
    print(f"Biological gene rows: {len(rows)}; summary rows excluded: 4")
    print(f"Duplicate gene-symbol groups: {sum(n > 1 for n in symbols.values())}")
    print(f"Sample Sheet: {sample_sheet_path()} ({len(sheet)} rows)")
    print(f"Manifest: {manifest_path()} ({len(read_manifest())} rows)")


if __name__ == "__main__":
    main()
