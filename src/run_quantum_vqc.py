"""Simulator-only, pilot-scale 9-qubit VQC. No feature selection or hardware use."""
from __future__ import annotations

import csv
import json
import time
import argparse
from pathlib import Path

import joblib
import numpy as np
import qiskit
import qiskit_aer
import qiskit_machine_learning
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import SamplerV2
from qiskit_machine_learning.algorithms.classifiers import VQC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .pilot_common import GENES, SEED, preprocessing
from .common import RESULTS

OUT = RESULTS / "quantum_vqc"
N_QUBITS = 9
N_SPLITS = 5
SHOTS = 256
# The 18-parameter ansatz requires COBYLA to use at least 20 evaluations.
MAXITER = 20


def build_vqc(seed: int) -> tuple[VQC, dict[str, int]]:
    # Aer 0.17 cannot execute the opaque library-instruction wrappers directly.
    # Decomposition preserves the requested ZZFeatureMap / RealAmplitudes design
    # while providing Aer-native gates to VQC's sampler.
    feature_map = zz_feature_map(feature_dimension=N_QUBITS, reps=1, entanglement="linear")
    ansatz = real_amplitudes(num_qubits=N_QUBITS, reps=1, entanglement="linear")
    sampler = SamplerV2(default_shots=SHOTS, seed=seed)
    model = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=COBYLA(maxiter=MAXITER), sampler=sampler)
    circuit = feature_map.compose(ansatz)
    return model, {
        "num_qubits": circuit.num_qubits,
        "circuit_depth": circuit.depth(),
        "feature_map_parameters": feature_map.num_parameters,
        "ansatz_parameters": ansatz.num_parameters,
    }


def load_subset_inputs(expression_path: Path, metadata_path: Path):
    with expression_path.open(encoding="utf-8", newline="") as handle:
        expression = list(csv.DictReader(handle))
    with metadata_path.open(encoding="utf-8", newline="") as handle:
        metadata = {row["sample_id"]: row for row in csv.DictReader(handle) if row["status"] == "usable"}
    if not expression or list(expression[0]) != ["Sample_ID", *GENES, "Label"]:
        raise ValueError("BrcaDx CSV columns do not exactly match the expected nine genes.")
    sample_ids = [row["Sample_ID"] for row in expression]
    if len(sample_ids) != len(set(sample_ids)) or set(sample_ids) != set(metadata):
        raise ValueError("Expression and metadata must contain the same unique samples.")
    y = np.array([int(row["Label"]) for row in expression])
    expected = np.array([int(metadata[sample_id]["label"]) for sample_id in sample_ids])
    if set(y) != {0, 1} or not np.array_equal(y, expected):
        raise ValueError("Labels must be Normal=0/Tumor=1 and match metadata.")
    X = np.array([[float(row[gene]) for gene in GENES] for row in expression], dtype=float)
    groups = np.array([metadata[sample_id]["case_id"] for sample_id in sample_ids])
    if not np.isfinite(X).all() or (X < 0).any():
        raise ValueError("TPM features have non-finite or negative values.")
    return sample_ids, X, y, groups


def main(expression_path: Path | None = None, metadata_path: Path | None = None, output_dir: Path | None = None) -> None:
    output_dir = output_dir or OUT
    expression_path = expression_path or (RESULTS.parent / "data" / "brca_9gene_expression.csv")
    metadata_path = metadata_path or (RESULTS / "metadata.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_ids, raw_tpm, y, groups = load_subset_inputs(expression_path, metadata_path)
    if raw_tpm.shape[1] != N_QUBITS or len(GENES) != N_QUBITS:
        raise ValueError("The VQC requires exactly nine BrcaDx feature columns.")
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    first_train, first_test = next(cv.split(raw_tpm, y, groups))
    fold_tumor = int(y[first_test].sum())
    fold_normal = int((y[first_test] == 0).sum())
    predicted = np.empty(len(y), dtype=int)
    probability = np.empty(len(y), dtype=float)
    fold_ids = np.empty(len(y), dtype=int)
    fold_runtimes = []
    _, details = build_vqc(SEED)
    total_start = time.perf_counter()
    for fold, (train, test) in enumerate(cv.split(raw_tpm, y, groups), start=1):
        # The scaler is fit only on the fold's training TPM data.
        scaler = preprocessing().fit(raw_tpm[train])
        X_train, X_test = scaler.transform(raw_tpm[train]), scaler.transform(raw_tpm[test])
        model, _ = build_vqc(SEED + fold)
        start = time.perf_counter()
        model.fit(X_train, y[train])
        fold_runtimes.append(time.perf_counter() - start)
        predicted[test] = model.predict(X_test).astype(int)
        probability[test] = model.predict_proba(X_test)[:, 1]
        fold_ids[test] = fold
    total_runtime = time.perf_counter() - total_start
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y, predicted), "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "roc_auc": roc_auc_score(y, probability), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "total_runtime_seconds": total_runtime, "mean_fold_train_seconds": float(np.mean(fold_runtimes)), **details,
    }
    with (output_dir / "vqc_oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "case_id", "fold", "true_label", "true_class", "predicted_label", "predicted_class", "tumor_probability"])
        writer.writeheader()
        for sample_id, case_id, fold, actual, pred, prob in zip(sample_ids, groups, fold_ids, y, predicted, probability):
            writer.writerow({"sample_id": sample_id, "case_id": case_id, "fold": fold, "true_label": actual, "true_class": "Tumor" if actual else "Normal",
                             "predicted_label": pred, "predicted_class": "Tumor" if pred else "Normal", "tumor_probability": prob})
    with (output_dir / "vqc_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2); handle.write("\n")
    # A final model and preprocessing pipeline are fitted on all 25 pilot samples solely as reusable artifacts.
    final_preprocessing = preprocessing().fit(raw_tpm)
    final_model, _ = build_vqc(SEED)
    final_model.fit(final_preprocessing.transform(raw_tpm), y)
    joblib.dump(final_preprocessing, output_dir / "vqc_preprocessing_pipeline.joblib")
    final_model.to_dill(str(output_dir / "vqc_final_model.dill"))
    report = [
        "Qura-Q 9-qubit VQC pilot (local Aer simulator only)", "",
        "Scope: execution/pipeline verification only. These results are not meaningful estimates of clinical or generalization performance.",
        f"Samples: {len(y)} (Tumor={int(y.sum())}, Normal={int((y == 0).sum())}); target Normal=0, Tumor=1.",
        "Features: " + ", ".join(GENES), "Preprocessing: raw TPM -> log2(TPM + 1) -> StandardScaler.",
        "StandardScaler was fit inside each training fold; no feature selection or dimensionality reduction was used.",
        f"Cross-validation: StratifiedGroupKFold(n_splits={N_SPLITS}, shuffle=True, random_state={SEED}), grouped by Case ID; first test fold has {fold_tumor} Tumor and {fold_normal} Normal samples.",
        f"Quantum circuit: ZZFeatureMap(9 qubits, reps=1, linear entanglement) + RealAmplitudes(9 qubits, reps=1, linear entanglement).",
        f"Aer SamplerV2: {SHOTS} shots; COBYLA maxiter={MAXITER}; no real hardware.",
        f"Circuit depth: {details['circuit_depth']}; qubits: {details['num_qubits']}; feature-map parameters: {details['feature_map_parameters']}; ansatz parameters: {details['ansatz_parameters']}.", "",
        f"Out-of-fold accuracy: {metrics['accuracy']:.3f}", f"Out-of-fold balanced accuracy: {metrics['balanced_accuracy']:.3f}",
        f"Out-of-fold ROC-AUC: {metrics['roc_auc']:.3f} (defined because both classes appear across out-of-fold predictions)",
        f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}", f"Total CV runtime: {total_runtime:.2f} seconds; mean fold training runtime: {np.mean(fold_runtimes):.2f} seconds.", "",
        "Limitation: this is still a small, non-independent pilot with a fixed optimizer budget; metrics must not be presented as clinical or generalization performance claims.",
    ]
    (output_dir / "vqc_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    versions = {"qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__, "qiskit_machine_learning": qiskit_machine_learning.__version__,
                "random_seed": SEED, "shots": SHOTS, "cobyla_maxiter": MAXITER}
    (output_dir / "run_metadata.json").write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    main(args.expression, args.metadata, args.output)
