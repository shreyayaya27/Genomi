"""Pilot-only BrcaDx baseline workflow. No feature selection or quantum modeling."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .common import DATA, RESULTS
from .pilot_common import GENES, SEED, load_inputs, preprocessing
OUT = RESULTS / "pilot_baseline"


def models():
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "rbf_svm": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
        # Fixed conservative settings; no hyperparameter search is performed on this pilot.
        "xgboost": XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.1, subsample=0.8,
                                 colsample_bytree=1.0, eval_metric="logloss", random_state=SEED,
                                 n_jobs=1, scale_pos_weight=0.25),
    }


def metric_row(name, y, predicted, probability):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "model": name, "accuracy": accuracy_score(y, predicted), "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "precision_tumor": precision_score(y, predicted, zero_division=0), "recall_tumor": recall_score(y, predicted, zero_division=0),
        "f1_tumor": f1_score(y, predicted, zero_division=0), "roc_auc": roc_auc_score(y, probability),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sample_ids, X, y = load_inputs()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    first_train, first_test = next(cv.split(X, y))
    fold_tumor = int(y[first_test].sum())
    fold_normal = int((y[first_test] == 0).sum())
    fitted_preprocess = preprocessing().fit(X)
    joblib.dump(fitted_preprocess, OUT / "preprocessing_pipeline.joblib")
    all_predictions, metrics = [], []
    for name, estimator in models().items():
        pipeline = Pipeline([("preprocessing", preprocessing()), ("model", estimator)])
        prediction = np.empty(len(y), dtype=int)
        probability = np.empty(len(y), dtype=float)
        folds = np.empty(len(y), dtype=int)
        for fold, (train, test) in enumerate(cv.split(X, y), start=1):
            model = clone(pipeline).fit(X[train], y[train])
            prediction[test] = model.predict(X[test])
            probability[test] = model.predict_proba(X[test])[:, 1]
            folds[test] = fold
        metrics.append(metric_row(name, y, prediction, probability))
        for sample_id, actual, pred, prob, fold in zip(sample_ids, y, prediction, probability, folds):
            all_predictions.append({"model": name, "sample_id": sample_id, "fold": fold, "true_label": actual,
                                    "true_class": "Tumor" if actual else "Normal", "predicted_label": pred,
                                    "predicted_class": "Tumor" if pred else "Normal", "tumor_probability": prob})
        # Final artifact is fitted on all pilot data; it is not a performance estimate.
        joblib.dump(pipeline.fit(X, y), OUT / f"{name}_pipeline.joblib")
    with (OUT / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_predictions[0])); writer.writeheader(); writer.writerows(all_predictions)
    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    report = [
        "Qura-Q BrcaDx pilot classical baselines", "", "Scope: pipeline verification only; do not interpret these metrics as meaningful model performance.",
        f"Samples: {len(y)} (Tumor={int(y.sum())}, Normal={int((y == 0).sum())})", "Target: Normal=0, Tumor=1 (verified against metadata.csv)",
        "Features: " + ", ".join(GENES), "", "Preprocessing: raw TPM -> log2(TPM + 1) -> StandardScaler.",
        "The scaler is fit separately within each training fold. No feature selection, hyperparameter search, Qiskit, or VQC was used.",
        f"Cross-validation: StratifiedKFold(n_splits=5, shuffle=True, random_state=42). Each test fold has {fold_tumor} Tumor and {fold_normal} Normal samples.",
        "Limitation: this remains a small, non-independent pilot cohort; use results for workflow verification, not clinical or generalization claims.", "",
        "Out-of-fold metrics:",
    ]
    for row in metrics:
        report.append(f"{row['model']}: accuracy={row['accuracy']:.3f}, balanced_accuracy={row['balanced_accuracy']:.3f}, "
                      f"precision_tumor={row['precision_tumor']:.3f}, recall_tumor={row['recall_tumor']:.3f}, "
                      f"f1_tumor={row['f1_tumor']:.3f}, roc_auc={row['roc_auc']:.3f}, "
                      f"TN={row['tn']}, FP={row['fp']}, FN={row['fn']}, TP={row['tp']}")
    report += ["", "Quantum readiness: nine numerical features can technically encode a 9-qubit model (one feature per qubit), but that is not advisable for this 25-sample pilot.",
               "Exact next step for a 6-8 feature Qura-Q experiment: after expanding and locking a cohort, fit a feature-reduction method only on training folds, lock the selected 6-8 genes, then apply that fixed transformation to validation/test data.",
               "Do not select or reduce features on the current 25-sample dataset."]
    (OUT / "pilot_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUT / "run_metadata.json").write_text(json.dumps({"random_seed": SEED, "n_splits": 5, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
