"""Fixed-split, validation-only classical-vs-VQC experiment. Never evaluates test data."""
from __future__ import annotations

import csv
import json
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_aer.primitives import SamplerV2
from qiskit_machine_learning.algorithms.classifiers import VQC
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .controlled_experiment_sanity import DATA, META, SMALL_GRID
from .fixed_split import load_fixed_split
from .pilot_common import GENES, SEED, log2_tpm_plus_one

OUT = Path("results/controlled_experiment/validation_run_01")
N_QUBITS = 9


def load_train_validation():
    assignments = load_fixed_split(META)
    assignment = {row["sample_id"]: row for row in assignments}
    with DATA.open(encoding="utf-8", newline="") as handle:
        raw = [row for row in csv.DictReader(handle) if assignment[row["Sample_ID"]]["split"] in {"train", "validation"}]
    def as_arrays(split: str):
        selected = [row for row in raw if assignment[row["Sample_ID"]]["split"] == split]
        x = np.asarray([[float(row[gene]) for gene in GENES] for row in selected], dtype=float)
        y = np.asarray([int(row["Label"]) for row in selected], dtype=int)
        ids = [row["Sample_ID"] for row in selected]
        return x, y, ids
    return (*as_arrays("train"), *as_arrays("validation"), assignment)


def metrics(y_true, predicted, probability):
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, predicted), "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "sensitivity_recall": recall_score(y_true, predicted, zero_division=0), "specificity": tn / (tn + fp) if tn + fp else None,
        "precision": precision_score(y_true, predicted, zero_division=0), "f1": f1_score(y_true, predicted, zero_division=0),
        "mcc": matthews_corrcoef(y_true, predicted), "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability), "brier_score": brier_score_loss(y_true, probability),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def balanced_indices(y: np.ndarray):
    normals = np.flatnonzero(y == 0)
    tumors = np.flatnonzero(y == 1)
    target = min(len(normals), len(tumors))
    chosen_tumors = np.random.default_rng(SEED).choice(tumors, size=target, replace=False)
    return np.sort(np.concatenate([normals[:target], chosen_tumors]))


def classical_preprocessing():
    return Pipeline([("log2_tpm_plus_one", FunctionTransformer(log2_tpm_plus_one, validate=False)), ("standard_scaler", StandardScaler())])


def run_classical(name, estimator, X_train, y_train, X_val, y_val, save_path):
    start = time.perf_counter()
    pipeline = Pipeline([("preprocessing", classical_preprocessing()), ("model", estimator)])
    pipeline.fit(X_train, y_train)
    probability = pipeline.predict_proba(X_val)[:, 1]
    predicted = pipeline.predict(X_val)
    result = {"family": "classical", "configuration": name, "status": "success", "training_seconds": time.perf_counter() - start, **metrics(y_val, predicted, probability)}
    joblib.dump(pipeline, save_path)
    return result


def quantum_preprocessing(angle_range: str):
    feature_range = (0.0, float(np.pi)) if angle_range == "zero_to_pi" else (-float(np.pi), float(np.pi))
    return Pipeline([("log2_tpm_plus_one", FunctionTransformer(log2_tpm_plus_one, validate=False)), ("angle_scaler", MinMaxScaler(feature_range=feature_range, clip=True))])


def run_vqc(index, config, X_train, y_train, X_val, y_val):
    name = f"vqc_{index:02d}_{config['angle_range']}_{config['optimizer'].lower()}"
    run_dir = OUT / name; run_dir.mkdir(parents=True, exist_ok=False)
    history = []
    record = {"family": "quantum", "configuration": name, "status": "failed", **config, "random_seed": SEED, "num_qubits": N_QUBITS}
    try:
        preprocessor = quantum_preprocessing(config["angle_range"]).fit(X_train)
        train_angles, val_angles = preprocessor.transform(X_train), preprocessor.transform(X_val)
        algorithm_globals.random_seed = SEED
        feature_map = zz_feature_map(N_QUBITS, reps=config["feature_map_reps"], entanglement=config["entanglement"])
        ansatz = real_amplitudes(N_QUBITS, reps=config["ansatz_reps"], entanglement=config["entanglement"])
        # Installed SPSA reports (nfev, parameters, objective, stepsize, accepted),
        # while VQC/COBYLA callbacks can report (parameters, objective). Accept both.
        def callback(*args):
            objective = args[2] if len(args) >= 3 else args[-1]
            history.append({"evaluation": len(history) + 1, "objective": float(objective)})
        optimizer = COBYLA(maxiter=config["maxiter"]) if config["optimizer"] == "COBYLA" else SPSA(maxiter=config["maxiter"], callback=callback)
        sampler = SamplerV2(default_shots=config["shots"], seed=SEED)
        model = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, sampler=sampler,
                    callback=None if config["optimizer"] == "SPSA" else callback)
        start = time.perf_counter(); model.fit(train_angles, y_train); training_seconds = time.perf_counter() - start
        probability = model.predict_proba(val_angles)[:, 1]; predicted = model.predict(val_angles).astype(int)
        circuit = feature_map.compose(ansatz)
        record.update({"status": "success", "training_seconds": training_seconds, "optimizer_evaluations": len(history),
                       "final_objective": history[-1]["objective"] if history else None, "circuit_depth": circuit.depth(),
                       "feature_map_parameters": feature_map.num_parameters, "ansatz_parameters": ansatz.num_parameters,
                       **metrics(y_val, predicted, probability)})
        joblib.dump(preprocessor, run_dir / "quantum_preprocessor.joblib")
        model.to_dill(str(run_dir / "vqc_model.dill"))
    except Exception as exc:
        record["error_type"] = type(exc).__name__; record["error"] = str(exc)
        (run_dir / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    (run_dir / "loss_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main():
    if OUT.exists(): raise FileExistsError(f"Refusing to overwrite experiment directory: {OUT}")
    OUT.mkdir(parents=True)
    X_train, y_train, train_ids, X_val, y_val, val_ids, assignment = load_train_validation()
    balanced = balanced_indices(y_train)
    balanced_ids = [train_ids[i] for i in balanced]
    with (OUT / "vqc_balanced_training_subset.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "case_id", "label"]); writer.writeheader()
        writer.writerows({"sample_id": sample_id, "case_id": assignment[sample_id]["case_id"], "label": int(y_train[i])} for i, sample_id in zip(balanced, balanced_ids))
    records = []
    # Full-training classical baselines; imbalance handled by class weights or balanced sample weights.
    full_models = {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression_full": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
        "rbf_svm_full": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
        "xgboost_full": XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.1, subsample=0.8, eval_metric="logloss", random_state=SEED, n_jobs=1),
    }
    for name, model in full_models.items():
        # xgboost uses explicit inverse-frequency sample weights.
        if name == "xgboost_full":
            pipe = Pipeline([("preprocessing", classical_preprocessing()), ("model", model)])
            weights = np.where(y_train == 1, len(y_train) / (2 * y_train.sum()), len(y_train) / (2 * (len(y_train) - y_train.sum())))
            start = time.perf_counter(); pipe.fit(X_train, y_train, model__sample_weight=weights)
            proba = pipe.predict_proba(X_val)[:, 1]; pred = pipe.predict(X_val)
            record = {"family": "classical", "configuration": name, "status": "success", "training_seconds": time.perf_counter()-start, **metrics(y_val, pred, proba)}
            joblib.dump(pipe, OUT / f"{name}.joblib"); records.append(record)
        else: records.append(run_classical(name, model, X_train, y_train, X_val, y_val, OUT / f"{name}.joblib"))
    for name, model in {"logistic_regression_matched_vqc": LogisticRegression(max_iter=3000, random_state=SEED), "rbf_svm_matched_vqc": SVC(kernel="rbf", probability=True, random_state=SEED)}.items():
        records.append(run_classical(name, model, X_train[balanced], y_train[balanced], X_val, y_val, OUT / f"{name}.joblib"))
    for index, config in enumerate(SMALL_GRID, start=1): records.append(run_vqc(index, config, X_train[balanced], y_train[balanced], X_val, y_val))
    keys = sorted({key for row in records for key in row})
    with (OUT / "comparison_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(records)
    (OUT / "experiment_scope.json").write_text(json.dumps({"split_file": "results/fixed_split_assignments.csv", "test_evaluated": False, "train_samples": len(y_train), "validation_samples": len(y_val), "balanced_vqc_train_samples": len(balanced)}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
