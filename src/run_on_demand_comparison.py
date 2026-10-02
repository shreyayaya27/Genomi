"""Reproducible final-test and external-scoring commands for Qura-Q.

The test command deliberately has no tuning options: the VQC configuration is
locked from validation before the held-out test set is read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from .controlled_experiment_sanity import DATA as CURRENT_DATA, META as CURRENT_META
from .fixed_split import create_fixed_split, load_fixed_split
from .pilot_common import GENES, SEED, log2_tpm_plus_one

CONTROLLED = Path("results/controlled_experiment")
LOCK_PATH = CONTROLLED / "best_vqc_config.json"
ROUND2_RESULT = CONTROLLED / "validation_round_02" / "zero_to_pi_spsa_ansatz2" / "result.json"
ROUND2_DIR = ROUND2_RESULT.parent
FINAL_ROOT = Path("results/final_test")


def classical_preprocessing() -> Pipeline:
    return Pipeline([("log2_tpm_plus_one", FunctionTransformer(log2_tpm_plus_one, validate=False)),
                     ("standard_scaler", StandardScaler())])


def quantum_preprocessing(angle_range: str) -> Pipeline:
    feature_range = (0.0, float(np.pi)) if angle_range == "zero_to_pi" else (-float(np.pi), float(np.pi))
    return Pipeline([("log2_tpm_plus_one", FunctionTransformer(log2_tpm_plus_one, validate=False)),
                     ("angle_scaler", MinMaxScaler(feature_range=feature_range, clip=True))])


def fingerprint(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.resolve().as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_or_validate_lock() -> dict:
    """Create the immutable configuration lock from the completed validation run."""
    source = json.loads(ROUND2_RESULT.read_text(encoding="utf-8"))
    if source.get("status") != "success":
        raise RuntimeError("The selected validation VQC did not complete successfully.")
    expected = {"angle_range": "zero_to_pi", "feature_map": "ZZFeatureMap", "feature_map_reps": 1,
                "ansatz": "RealAmplitudes", "ansatz_reps": 2, "entanglement": "linear",
                "optimizer": "SPSA", "maxiter": 200, "shots": 1024, "simulator": "shot_aer",
                "seed": SEED, "qubits": 9}
    for key, value in expected.items():
        if source.get(key) != value:
            raise RuntimeError(f"Validation artifact does not match the approved VQC lock: {key}.")
    locked = {
        "lock_version": 1,
        "selection_policy": "Highest validation balanced accuracy among the predeclared Round-2 VQC configurations; no test data accessed for selection.",
        "configuration": expected,
        "validation_metrics": {key: source.get(key) for key in ("accuracy", "balanced_accuracy", "f1", "mcc", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp")},
        "source_artifacts": {"run_directory": str(ROUND2_DIR), "result": str(ROUND2_RESULT),
                             "model": str(ROUND2_DIR / "model.dill"), "preprocessor": str(ROUND2_DIR / "preprocessor.joblib")},
        "genes": GENES,
        "dataset_split_fingerprint": fingerprint(CURRENT_DATA, CURRENT_META, Path("results/fixed_split_assignments.csv")),
    }
    if LOCK_PATH.exists():
        existing = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        if existing != locked:
            raise RuntimeError("best_vqc_config.json already exists but differs from the approved lock; refusing to replace it.")
        return existing
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps(locked, indent=2) + "\n", encoding="utf-8")
    return locked


def read_dataset(expression: Path, metadata: Path, labels_required: bool) -> tuple[list[dict], dict[str, dict], np.ndarray, np.ndarray | None]:
    with expression.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with metadata.open(encoding="utf-8", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))
    required_expression = ["Sample_ID", *GENES]
    if not rows or any(column not in rows[0] for column in required_expression):
        raise ValueError("Expression CSV must contain Sample_ID and the exact nine BrcaDx TPM gene columns.")
    ids = [row["Sample_ID"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Expression CSV contains duplicate Sample_ID values.")
    if not metadata_rows or not {"sample_id", "case_id"}.issubset(metadata_rows[0]):
        raise ValueError("Metadata CSV must contain sample_id and case_id.")
    metadata_by_id = {row["sample_id"]: row for row in metadata_rows}
    if len(metadata_by_id) != len(metadata_rows) or set(ids) != set(metadata_by_id):
        raise ValueError("Expression and metadata must contain exactly the same unique samples.")
    try:
        X = np.asarray([[float(row[gene]) for gene in GENES] for row in rows], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("All nine gene values must be numeric TPM values.") from exc
    if not np.isfinite(X).all() or (X < 0).any():
        raise ValueError("TPM values must be finite and non-negative.")
    expression_has_labels = "Label" in rows[0]
    metadata_has_labels = "label" in metadata_rows[0]
    if labels_required and not (expression_has_labels or metadata_has_labels):
        raise ValueError("Labels are required: provide Label in expression or label in metadata (Normal=0, Tumor=1).")
    y = None
    if expression_has_labels or metadata_has_labels:
        raw_labels = [row.get("Label", metadata_by_id[row["Sample_ID"]].get("label")) for row in rows]
        if any(label not in {"0", "1", 0, 1} for label in raw_labels):
            raise ValueError("Labels must be Normal=0 or Tumor=1.")
        y = np.asarray([int(label) for label in raw_labels], dtype=int)
        if metadata_has_labels and expression_has_labels:
            meta_y = np.asarray([int(metadata_by_id[row["Sample_ID"]]["label"]) for row in rows], dtype=int)
            if not np.array_equal(y, meta_y):
                raise ValueError("Expression Label and metadata label disagree.")
    return rows, metadata_by_id, X, y


def metric_record(y: np.ndarray, pred: np.ndarray, probability: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else None, "f1": float(f1_score(y, pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y, pred)), "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
            "pr_auc": float(average_precision_score(y, probability)) if len(np.unique(y)) == 2 else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def balanced_indices(y: np.ndarray) -> np.ndarray:
    normal, tumor = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    n = min(len(normal), len(tumor))
    chosen_tumor = np.random.default_rng(SEED).choice(tumor, size=n, replace=False)
    return np.sort(np.concatenate((normal[:n], chosen_tumor)))


def fit_classical(name: str, model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, output: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    pipeline = Pipeline([("preprocessing", classical_preprocessing()), ("model", model)])
    start = time.perf_counter()
    if name == "xgboost_full":
        weights = np.where(y_train == 1, len(y_train) / (2 * y_train.sum()), len(y_train) / (2 * (len(y_train) - y_train.sum())))
        pipeline.fit(X_train, y_train, model__sample_weight=weights)
    else:
        pipeline.fit(X_train, y_train)
    seconds = time.perf_counter() - start
    probability = pipeline.predict_proba(X_test)[:, 1]
    prediction = pipeline.predict(X_test).astype(int)
    joblib.dump(pipeline, output / f"{name}.joblib")
    return ({"model_type": "classical", "model_name": name, "status": "success", "training_seconds": seconds,
             **metric_record(y_test, prediction, probability)}, prediction, probability)


def fit_vqc(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, output: Path, lock: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    from qiskit.circuit.library import real_amplitudes, zz_feature_map
    from qiskit_algorithms.optimizers import SPSA
    from qiskit_algorithms.utils import algorithm_globals
    from qiskit_aer.primitives import SamplerV2
    from qiskit_machine_learning.algorithms.classifiers import VQC
    cfg = lock["configuration"]
    history: list[dict] = []
    prep = quantum_preprocessing(cfg["angle_range"]).fit(X_train)
    train_angles, test_angles = prep.transform(X_train), prep.transform(X_test)
    algorithm_globals.random_seed = SEED
    feature_map = zz_feature_map(9, reps=cfg["feature_map_reps"], entanglement=cfg["entanglement"])
    ansatz = real_amplitudes(9, reps=cfg["ansatz_reps"], entanglement=cfg["entanglement"])
    def callback(*args):
        objective = args[2] if len(args) >= 3 else args[-1]
        history.append({"evaluation": len(history) + 1, "objective": float(objective)})
    optimizer = SPSA(maxiter=cfg["maxiter"], callback=callback)
    model = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer,
                sampler=SamplerV2(default_shots=cfg["shots"], seed=SEED), callback=None)
    start = time.perf_counter(); model.fit(train_angles, y_train); seconds = time.perf_counter() - start
    probability = model.predict_proba(test_angles)[:, 1]
    prediction = model.predict(test_angles).astype(int)
    final_objective = history[-1]["objective"] if history else getattr(getattr(model, "fit_result", None), "fun", None)
    record = {"model_type": "quantum", "model_name": "locked_best_vqc", "status": "success", "training_seconds": seconds,
              "qubits": 9, "feature_map": cfg["feature_map"], "feature_map_reps": cfg["feature_map_reps"],
              "ansatz": cfg["ansatz"], "ansatz_reps": cfg["ansatz_reps"], "entanglement": cfg["entanglement"],
              "optimizer": cfg["optimizer"], "optimizer_evaluations": getattr(getattr(model, "fit_result", None), "nfev", len(history)),
              "shots": cfg["shots"], "simulator": cfg["simulator"], "circuit_depth": feature_map.compose(ansatz).depth(),
              "final_objective": float(final_objective) if final_objective is not None else None,
              **metric_record(y_test, prediction, probability)}
    joblib.dump(prep, output / "locked_best_vqc_preprocessor.joblib")
    model.to_dill(str(output / "locked_best_vqc_model.dill"))
    (output / "locked_best_vqc_loss_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return record, prediction, probability


def write_outputs(output: Path, rows: list[dict], prediction_rows: list[dict], metadata: dict) -> None:
    comparison_fields = sorted({key for row in rows for key in row})
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields); writer.writeheader(); writer.writerows(rows)
    fields = list(prediction_rows[0])
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(prediction_rows)
    lines = ["Qura-Q final held-out test comparison", "", "Models were selected on validation and refit on train + validation. The test set was used once for this report.", "",
             "Model | Accuracy | Balanced Acc. | F1 | MCC | ROC-AUC | TN FP FN TP | Train seconds", "--- | --- | --- | --- | --- | --- | --- | ---"]
    for row in rows:
        lines.append(f"{row['model_name']} | {row['accuracy']:.3f} | {row['balanced_accuracy']:.3f} | {row['f1']:.3f} | {row['mcc']:.3f} | {row['roc_auc']:.3f} | {row['tn']} {row['fp']} {row['fn']} {row['tp']} | {row['training_seconds']:.2f}")
    lines += ["", "Test samples: " + str(len(prediction_rows)), "Biomarkers: " + ", ".join(GENES),
              "VQC uses 9 qubits; ZZFeatureMap reps=1; RealAmplitudes reps=2; linear entanglement; SPSA maxiter=200; Aer 1,024 shots."]
    (output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_current(run_name: str) -> None:
    # Keep the presentation and external-scoring commands usable in a Python
    # environment that does not have XGBoost installed. Only final retraining
    # requires this optional training dependency.
    from xgboost import XGBClassifier
    lock = make_or_validate_lock()
    output = FINAL_ROOT / run_name
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing final-test run: {output}")
    rows, metadata, X, y = read_dataset(CURRENT_DATA, CURRENT_META, labels_required=True)
    assignments = {row["sample_id"]: row for row in load_fixed_split(CURRENT_META)}
    if lock["dataset_split_fingerprint"] != fingerprint(CURRENT_DATA, CURRENT_META, Path("results/fixed_split_assignments.csv")):
        raise RuntimeError("Dataset or fixed split differs from the locked validation experiment.")
    dev_idx = np.asarray([i for i, row in enumerate(rows) if assignments[row["Sample_ID"]]["split"] in {"train", "validation"}])
    test_idx = np.asarray([i for i, row in enumerate(rows) if assignments[row["Sample_ID"]]["split"] == "test"])
    X_dev, y_dev, X_test, y_test = X[dev_idx], y[dev_idx], X[test_idx], y[test_idx]
    if (len(X_dev), len(X_test), int(y_dev.sum()), int(y_test.sum())) != (198, 35, 102, 18):
        raise RuntimeError("Current fixed-split counts are not the approved 198 development / 35 test contract.")
    output.mkdir(parents=True)
    records: list[dict] = []
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    specs = [("dummy_majority", DummyClassifier(strategy="most_frequent"), X_dev, y_dev),
             ("logistic_regression_full", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED), X_dev, y_dev),
             ("rbf_svm_full", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED), X_dev, y_dev),
             ("xgboost_full", XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.1, subsample=0.8, eval_metric="logloss", random_state=SEED, n_jobs=1), X_dev, y_dev)]
    matched = balanced_indices(y_dev)
    specs += [("logistic_regression_matched_vqc", LogisticRegression(max_iter=3000, random_state=SEED), X_dev[matched], y_dev[matched]),
              ("rbf_svm_matched_vqc", SVC(kernel="rbf", probability=True, random_state=SEED), X_dev[matched], y_dev[matched])]
    for name, model, train_X, train_y in specs:
        record, pred, probability = fit_classical(name, model, train_X, train_y, X_test, y_test, output)
        record["training_set"] = "balanced_192_vqc_matched" if "matched" in name else "full_development_198"
        records.append(record); predictions[name] = (pred, probability)
    record, pred, probability = fit_vqc(X_dev[matched], y_dev[matched], X_test, y_test, output, lock)
    record["training_set"] = "balanced_192_vqc_matched"; records.append(record); predictions["VQC"] = (pred, probability)
    prediction_rows = []
    for position, source_index in enumerate(test_idx):
        sample_id = rows[source_index]["Sample_ID"]
        entry = {"case_id": metadata[sample_id]["case_id"], "sample_id": sample_id, "true_label": int(y_test[position])}
        for name, (predicted, probabilities) in predictions.items():
            prefix = "VQC" if name == "VQC" else name
            entry[f"{prefix}_prediction"] = int(predicted[position]); entry[f"{prefix}_probability"] = float(probabilities[position])
        prediction_rows.append(entry)
    (output / "best_vqc_config.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    (output / "run_metadata.json").write_text(json.dumps({"run_name": run_name, "development_samples": 198, "test_samples": 35,
        "development_counts": {"normal": 96, "tumor": 102}, "test_counts": {"normal": 17, "tumor": 18},
        "balanced_vqc_training_samples": 192, "test_accessed_after_lock": True}, indent=2) + "\n", encoding="utf-8")
    write_outputs(output, records, prediction_rows, metadata)
    print(f"Completed immutable final test: {output}")


def score_external(expression: Path, metadata_path: Path, model_run: str, output_name: str, vqc_source: str) -> None:
    from qiskit_machine_learning.algorithms.classifiers import VQC
    model_dir = FINAL_ROOT / model_run
    if not model_dir.exists(): raise FileNotFoundError(f"Final model run not found: {model_dir}")
    lock_path = model_dir / "best_vqc_config.json"
    if not lock_path.exists():
        raise FileNotFoundError("Saved final-model feature signature is missing.")
    saved_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if saved_lock.get("genes") != GENES:
        raise ValueError("Saved model feature signature is incompatible with the required nine BrcaDx genes.")
    output = FINAL_ROOT / output_name
    if output.exists(): raise FileExistsError(f"Refusing to overwrite external scoring output: {output}")
    rows, metadata, X, y = read_dataset(expression, metadata_path, labels_required=False)
    output.mkdir(parents=True)
    model_files = ["dummy_majority", "logistic_regression_full", "rbf_svm_full", "xgboost_full", "logistic_regression_matched_vqc", "rbf_svm_matched_vqc"]
    results = {}
    for name in model_files:
        model = joblib.load(model_dir / f"{name}.joblib")
        results[name] = (model.predict(X).astype(int), model.predict_proba(X)[:, 1])
    if vqc_source == "validation_locked":
        lock = make_or_validate_lock()
        source = lock["source_artifacts"]
        vqc = VQC.from_dill(source["model"])
        prep = joblib.load(source["preprocessor"])
    else:
        vqc = VQC.from_dill(str(model_dir / "locked_best_vqc_model.dill"))
        prep = joblib.load(model_dir / "locked_best_vqc_preprocessor.joblib")
    angles = prep.transform(X); results["VQC"] = (vqc.predict(angles).astype(int), vqc.predict_proba(angles)[:, 1])
    prediction_rows = []
    for i, row in enumerate(rows):
        entry = {"case_id": metadata[row["Sample_ID"]]["case_id"], "sample_id": row["Sample_ID"]}
        if y is not None: entry["true_label"] = int(y[i])
        for name, (pred, probability) in results.items():
            entry[f"{name}_prediction"] = int(pred[i]); entry[f"{name}_probability"] = float(probability[i])
        prediction_rows.append(entry)
    fields = list(prediction_rows[0])
    with (output / "external_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(prediction_rows)
    if y is not None:
        metrics = {name: metric_record(y, pred, probability) for name, (pred, probability) in results.items()}
        (output / "external_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output / "scoring_configuration.json").write_text(json.dumps({"classical_model_run": str(model_dir), "vqc_source": vqc_source}, indent=2) + "\n", encoding="utf-8")
    print(f"External scoring complete: {output}")


def evaluate_locked_validation_vqc() -> None:
    """Score the exact saved validation model on test without retraining it.

    This is retained separately from the final-refit test.  Since the held-out
    test set was already evaluated by the final protocol, this is explicitly a
    post-hoc diagnostic, not a new pristine final test.
    """
    from qiskit_machine_learning.algorithms.classifiers import VQC
    lock = make_or_validate_lock()
    output = CONTROLLED / "locked_validation_vqc_test"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite locked-validation test diagnostic: {output}")
    rows, metadata, X, y = read_dataset(CURRENT_DATA, CURRENT_META, labels_required=True)
    assignments = {row["sample_id"]: row for row in load_fixed_split(CURRENT_META)}
    test_idx = np.asarray([i for i, row in enumerate(rows) if assignments[row["Sample_ID"]]["split"] == "test"])
    X_test, y_test = X[test_idx], y[test_idx]
    source = lock["source_artifacts"]
    preprocessor = joblib.load(source["preprocessor"])
    model = VQC.from_dill(source["model"])
    angles = preprocessor.transform(X_test)
    probability = model.predict_proba(angles)[:, 1]
    prediction = model.predict(angles).astype(int)
    source_result = json.loads(Path(source["result"]).read_text(encoding="utf-8"))
    validation = {"split": "validation", "evaluation_type": "original model-selection validation", "model_name": "zero_to_pi_spsa_ansatz2",
                  **{key: source_result.get(key) for key in ("accuracy", "balanced_accuracy", "precision", "sensitivity_recall", "specificity", "f1", "mcc", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp", "training_seconds")}}
    test = {"split": "test", "evaluation_type": "post_hoc: exact validation-trained artifact, no retraining", "model_name": "zero_to_pi_spsa_ansatz2",
            **metric_record(y_test, prediction, probability), "training_seconds": 0.0}
    output.mkdir(parents=True)
    fields = sorted(set(validation) | set(test))
    with (output / "vqc_validation_vs_test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows([validation, test])
    prediction_rows = [{"case_id": metadata[rows[index]["Sample_ID"]]["case_id"], "sample_id": rows[index]["Sample_ID"], "true_label": int(y_test[pos]),
                        "VQC_prediction": int(prediction[pos]), "VQC_probability": float(probability[pos])} for pos, index in enumerate(test_idx)]
    with (output / "locked_validation_vqc_test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0])); writer.writeheader(); writer.writerows(prediction_rows)
    (output / "README.txt").write_text("This compares the saved validation-winning VQC on its original validation split and the same saved artifact on test.\nThe test score is post-hoc because the held-out test set was previously evaluated by the final-refit protocol.\n", encoding="utf-8")
    print(f"Saved validation-vs-test VQC diagnostic: {output}")


def demo_validation() -> None:
    """Print the persisted validation comparison for a fast live demonstration.

    This intentionally reads audited saved results rather than retraining a
    stochastic VQC in front of an audience.  It never loads the test set.
    """
    lock = make_or_validate_lock()
    comparison_path = CONTROLLED / "classical_vs_quantum_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError("Validation comparison is missing; run the controlled validation report first.")
    with comparison_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = next((row for row in rows if row["model_type"] == "quantum" and row["model_name"] == "zero_to_pi_spsa_ansatz2"), None)
    if selected is None:
        raise RuntimeError("The validation comparison does not contain the locked Round-2 VQC.")
    expected = lock["validation_metrics"]
    if any(abs(float(selected[key]) - float(expected[key])) > 1e-12 for key in ("accuracy", "balanced_accuracy", "roc_auc")):
        raise RuntimeError("Saved comparison metrics do not match best_vqc_config.json; refusing to present them.")
    print("\nQura-Q - Classical ML vs Quantum ML")
    print("Validation demonstration only: 35 patients (18 Tumor, 17 Normal). Test data is not loaded.\n")
    presentation_rows = [
        next((row for row in rows if row["model_name"] == "logistic_regression_matched_vqc"), None),
        selected,
    ]
    if any(row is None for row in presentation_rows):
        raise RuntimeError("The required matched Logistic Regression or locked VQC result is missing.")
    print(f"{'Branch':14} {'Model':34} {'Training set':27} {'Accuracy':>9} {'Balanced':>9} {'F1':>8} {'ROC-AUC':>9}")
    print("-" * 116)
    for branch, row in zip(("Classical ML", "Quantum ML"), presentation_rows):
        print(f"{branch:14} {row['model_name']:34} {row['training_set']:27} {float(row['accuracy']):9.3f} {float(row['balanced_accuracy']):9.3f} {float(row['f1']):8.3f} {float(row['roc_auc']):9.3f}")
    print("\nLocked VQC: 9 qubits | ZZFeatureMap reps=1 | RealAmplitudes reps=2 | SPSA | Aer, 1,024 shots")
    print("Interpretation: this is a reproducible validation comparison, not a held-out-test claim or quantum advantage.")


def prepare_experiment(expression: Path, metadata: Path, experiment_name: str) -> None:
    rows, _, _, y = read_dataset(expression, metadata, labels_required=True)
    if len(np.unique(y)) != 2: raise ValueError("A new controlled experiment needs both Normal and Tumor samples.")
    output = Path("results/experiments") / experiment_name
    if output.exists(): raise FileExistsError(f"Experiment directory already exists: {output}")
    output.mkdir(parents=True)
    split = create_fixed_split(metadata, output / "fixed_split_assignments.csv")
    (output / "dataset_contract.json").write_text(json.dumps({"expression": str(expression), "metadata": str(metadata), "samples": len(rows),
        "genes": GENES, "seed": SEED, "fixed_split": str(split), "next_step": "Run validation-only model selection before any final test evaluation."}, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared new persisted experiment: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qura-Q on-demand controlled comparison")
    sub = parser.add_subparsers(dest="command", required=True)
    p_test = sub.add_parser("test-current"); p_test.add_argument("--run-name", default="final_fit_42_complete")
    p_score = sub.add_parser("score-external"); p_score.add_argument("--expression", type=Path, required=True); p_score.add_argument("--metadata", type=Path, required=True); p_score.add_argument("--model-run", default="final_fit_42_complete"); p_score.add_argument("--output-name", default="external_scoring"); p_score.add_argument("--vqc-source", choices=("validation_locked", "final_refit"), default="validation_locked")
    p_prep = sub.add_parser("prepare-experiment"); p_prep.add_argument("--expression", type=Path, required=True); p_prep.add_argument("--metadata", type=Path, required=True); p_prep.add_argument("--experiment-name", required=True)
    sub.add_parser("evaluate-locked-validation-vqc", help="Post-hoc test score for the exact saved validation VQC; does not retrain it.")
    sub.add_parser("demo-validation", help="Instantly display the audited classical-vs-QML validation comparison; never accesses test data.")
    args = parser.parse_args()
    if args.command == "test-current": test_current(args.run_name)
    elif args.command == "score-external": score_external(args.expression, args.metadata, args.model_run, args.output_name, args.vqc_source)
    elif args.command == "prepare-experiment": prepare_experiment(args.expression, args.metadata, args.experiment_name)
    elif args.command == "evaluate-locked-validation-vqc": evaluate_locked_validation_vqc()
    else: demo_validation()


if __name__ == "__main__": main()
