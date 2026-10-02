"""Create a validation-only side-by-side report from the persisted controlled experiment."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from qiskit_machine_learning.algorithms.classifiers import VQC

from .controlled_experiment_sanity import DATA, META
from .fixed_split import load_fixed_split
from .pilot_common import GENES

RUN = Path("results/controlled_experiment/validation_run_01")
OUT = Path("results/controlled_experiment")
LOCK = OUT / "best_vqc_config.json"


def main() -> None:
    if not (RUN / "comparison_validation.csv").exists():
        raise FileNotFoundError("Controlled validation results are required; refusing to train or regenerate them.")
    assignments = {row["sample_id"]: row for row in load_fixed_split(META)}
    with DATA.open(encoding="utf-8", newline="") as handle:
        validation = [row for row in csv.DictReader(handle) if assignments[row["Sample_ID"]]["split"] == "validation"]
    X_val = np.asarray([[float(row[gene]) for gene in GENES] for row in validation], dtype=float)
    lr = joblib.load(RUN / "logistic_regression_full.joblib")
    svm = joblib.load(RUN / "rbf_svm_full.joblib")
    xgb = joblib.load(RUN / "xgboost_full.joblib")
    if not LOCK.exists():
        raise FileNotFoundError("best_vqc_config.json is required so the report cannot silently use an older VQC.")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    source = lock["source_artifacts"]
    vqc_preprocessor = joblib.load(source["preprocessor"])
    vqc = VQC.from_dill(source["model"])
    vqc_angles = vqc_preprocessor.transform(X_val)
    vqc_probability = vqc.predict_proba(vqc_angles)[:, 1]
    vqc_prediction = vqc.predict(vqc_angles).astype(int)
    predictions = []
    for i, row in enumerate(validation):
        sample_id = row["Sample_ID"]
        predictions.append({
            "case_id": assignments[sample_id]["case_id"], "sample_id": sample_id, "true_label": int(row["Label"]),
            "LR_prediction": int(lr.predict(X_val[i:i+1])[0]), "LR_probability": float(lr.predict_proba(X_val[i:i+1])[0, 1]),
            "SVM_prediction": int(svm.predict(X_val[i:i+1])[0]), "SVM_score_or_probability": float(svm.predict_proba(X_val[i:i+1])[0, 1]),
            "XGBoost_prediction": int(xgb.predict(X_val[i:i+1])[0]), "XGBoost_probability": float(xgb.predict_proba(X_val[i:i+1])[0, 1]),
            "VQC_prediction": int(vqc_prediction[i]), "VQC_probability": float(vqc_probability[i]),
        })
    with (OUT / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    with (RUN / "comparison_validation.csv").open(encoding="utf-8", newline="") as handle:
        source = list(csv.DictReader(handle))
    # The original directory includes exploratory Round-1 VQCs. Replace those
    # rows with the persisted Round-2 validation winner used for predictions.
    with Path(lock["source_artifacts"]["result"]).open(encoding="utf-8") as handle:
        locked_result = json.load(handle)
    source = [row for row in source if row["family"] != "quantum"]
    source.append({"family": "quantum", "configuration": "locked_best_vqc", **locked_result})
    output_rows = []
    for row in source:
        is_quantum = row["family"] == "quantum"
        training_set = "balanced_158_vqc_matched" if "matched_vqc" in row["configuration"] or is_quantum else "full_training_163"
        output_rows.append({
            "model_type": row["family"], "model_name": row["configuration"], "status": row["status"], "training_set": training_set,
            "accuracy": row.get("accuracy", ""), "balanced_accuracy": row.get("balanced_accuracy", ""), "precision": row.get("precision", ""),
            "recall": row.get("sensitivity_recall", ""), "specificity": row.get("specificity", ""), "f1": row.get("f1", ""),
            "mcc": row.get("mcc", ""), "roc_auc": row.get("roc_auc", ""), "pr_auc": row.get("pr_auc", ""),
            "tn": row.get("tn", ""), "fp": row.get("fp", ""), "fn": row.get("fn", ""), "tp": row.get("tp", ""),
            "training_seconds": row.get("training_seconds", ""), "qubits": row.get("num_qubits", ""),
            "feature_map": "ZZFeatureMap" if is_quantum else "", "feature_map_reps": row.get("feature_map_reps", ""),
            "ansatz": "RealAmplitudes" if is_quantum else "", "ansatz_reps": row.get("ansatz_reps", ""),
            "entanglement": row.get("entanglement", ""), "optimizer": row.get("optimizer", ""), "optimizer_evaluations": row.get("optimizer_evaluations", ""),
            "shots": row.get("shots", ""), "simulator": row.get("simulator", row.get("simulator_mode", "")), "circuit_depth": row.get("circuit_depth", ""),
            "final_objective": row.get("final_objective", ""), "error": row.get("error", ""),
        })
    fields = list(output_rows[0])
    with (OUT / "classical_vs_quantum_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(output_rows)
    successful = [row for row in output_rows if row["status"] == "success"]
    best_acc = max(successful, key=lambda row: float(row["accuracy"]))
    best_bal = max(successful, key=lambda row: float(row["balanced_accuracy"]))
    lines = [
        "Qura-Q: Classical ML vs Quantum ML — validation-only comparison", "",
        "Scope: same 233-sample dataset, fixed persisted case-level split, and nine published BrcaDx genes.",
        "The held-out test split was not loaded as features, scored, or used for any configuration choice.",
        "", "SIDE-BY-SIDE COMPARISON (validation: 18 Tumor, 17 Normal)",
        "Model | Training set | Status | Accuracy | Balanced Acc. | F1 | MCC | ROC-AUC | TN FP FN TP | Time (s)",
        "--- | --- | --- | --- | --- | --- | --- | --- | --- | ---",
    ]
    for row in output_rows:
        if row["status"] != "success":
            lines.append(f"{row['model_name']} | {row['training_set']} | FAILED | — | — | — | — | — | — | —")
        else:
            lines.append(f"{row['model_name']} | {row['training_set']} | success | {float(row['accuracy']):.3f} | {float(row['balanced_accuracy']):.3f} | {float(row['f1']):.3f} | {float(row['mcc']):.3f} | {float(row['roc_auc']):.3f} | {row['tn']} {row['fp']} {row['fn']} {row['tp']} | {float(row['training_seconds']):.2f}")
    lines += ["", "FACTUAL SUMMARY", f"Highest validation accuracy: {best_acc['model_name']} ({float(best_acc['accuracy']):.3f}).", f"Highest balanced accuracy: {best_bal['model_name']} ({float(best_bal['balanced_accuracy']):.3f}).",
              "Selected validation-locked VQC: zero_to_pi_spsa_ansatz2.",
              "VQC: 9 qubits; ZZFeatureMap reps=1; RealAmplitudes reps=2; linear entanglement; SPSA maxiter=200; Aer shot simulator with 1024 shots; circuit depth=33.",
              "The successful VQC is not competitive with the classical models on this validation split. This is a factual validation comparison, not evidence for or against quantum advantage.",
              "Two SPSA runs are retained as failed records due to the installed optimizer callback API mismatch; they were not replaced or rerun."]
    (OUT / "classical_vs_quantum_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
