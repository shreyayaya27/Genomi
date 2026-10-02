"""Shared input validation and preprocessing for pilot classical/quantum workflows."""
from __future__ import annotations

import csv

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .common import DATA, RESULTS

GENES = ["NEK2", "PKMYT1", "MMP11", "CPA1", "COL10A1", "HSD17B13", "CA4", "MYOC", "LYVE1"]
SEED = 42


def load_inputs():
    with (DATA / "brca_9gene_expression.csv").open(encoding="utf-8", newline="") as handle:
        expression = list(csv.DictReader(handle))
    with (RESULTS / "metadata.csv").open(encoding="utf-8", newline="") as handle:
        metadata = {row["sample_id"]: row for row in csv.DictReader(handle) if row["status"] == "usable"}
    if not expression or list(expression[0]) != ["Sample_ID", *GENES, "Label"]:
        raise ValueError("BrcaDx CSV columns do not exactly match the expected 9-gene signature.")
    sample_ids = [row["Sample_ID"] for row in expression]
    if len(sample_ids) != len(set(sample_ids)) or set(sample_ids) != set(metadata):
        raise ValueError("BrcaDx CSV and usable metadata must contain the same unique samples.")
    labels = np.array([int(row["Label"]) for row in expression])
    expected = np.array([int(metadata[sample_id]["label"]) for sample_id in sample_ids])
    if set(labels) != {0, 1} or not np.array_equal(labels, expected):
        raise ValueError("Labels must be Normal=0/Tumor=1 and match metadata.")
    X = np.array([[float(row[gene]) for gene in GENES] for row in expression], dtype=float)
    if not np.isfinite(X).all() or (X < 0).any():
        raise ValueError("TPM feature matrix has non-finite or negative values.")
    return sample_ids, X, labels


def log2_tpm_plus_one(values):
    return np.log2(np.asarray(values, dtype=float) + 1.0)


def preprocessing():
    return Pipeline([
        ("log2_tpm_plus_one", FunctionTransformer(log2_tpm_plus_one, validate=False)),
        ("standard_scaler", StandardScaler()),
    ])
