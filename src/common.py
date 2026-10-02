from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TPM_COLUMN = "tpm_unstranded"
LABELS = {"Normal": "0", "Tumor": "1"}


def one_file(pattern: str) -> Path:
    matches = list(DATA.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {pattern} under {DATA}; found {len(matches)}: {matches}")
    return matches[0]


def sample_sheet_path() -> Path:
    # A project can retain a subset Sample Sheet beside the parent cohort sheet.
    # Use the largest Sheet, which is the full cohort metadata (835 rows here).
    matches = list(DATA.rglob("gdc_sample_sheet*.tsv"))
    if not matches:
        raise RuntimeError(f"No GDC Sample Sheet found under {DATA}")
    return max(matches, key=lambda path: sum(1 for _ in path.open(encoding="utf-8")))


def manifest_path() -> Path:
    # Prefer the full cohort manifest over intentionally smaller download subsets.
    matches = [path for path in DATA.rglob("gdc_manifest*.txt") if "additional_" not in path.name]
    if not matches:
        raise RuntimeError(f"No GDC manifest found under {DATA}")
    return max(matches, key=lambda path: sum(1 for _ in path.open(encoding="utf-8")))


def expression_files() -> list[Path]:
    return sorted(DATA.rglob("*.rna_seq.augmented_star_gene_counts.tsv"))


def file_id_from_expression(path: Path) -> str:
    # GDC archives store each expression file inside a directory named by File ID.
    return path.parent.name


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_sample_sheet() -> dict[str, dict[str, str]]:
    rows = read_tsv(sample_sheet_path())
    return {row["File ID"]: row for row in rows}


def read_manifest() -> dict[str, dict[str, str]]:
    rows = read_tsv(manifest_path())
    return {row["id"]: row for row in rows}


def iter_gene_rows(path: Path):
    """Yield biological gene rows, excluding STAR summary rows such as N_unmapped."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if not line.startswith("#"):
                header = line.rstrip("\r\n").split("\t")
                break
        else:
            raise ValueError(f"No table header found in {path}")
        reader = csv.DictReader(handle, fieldnames=header, delimiter="\t")
        required = {"gene_id", "gene_name", TPM_COLUMN}
        if not required.issubset(header):
            raise ValueError(f"Missing required columns in {path}: {required - set(header)}")
        for row in reader:
            if row["gene_id"].startswith("ENSG"):
                yield row


def write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
