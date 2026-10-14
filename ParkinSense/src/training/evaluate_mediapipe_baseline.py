"""
ParkinSense — MediaPipe Classifier Baseline & Model Comparison Experiment
========================================================================
Runs group-aware (subject-wise) cross-validation to compare candidate ML models
and feature sets for the MediaPipe-based Parkinson's gait detection pipeline.

Ensures no participant's recordings are split across train and validation folds.
Evaluates accuracy, precision, sensitivity, specificity, F1 score, ROC-AUC, Brier score,
and confusion matrix.

Results are saved to `data/processed/mediapipe_model_comparison.csv`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)

# Resolve project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DATA_PATH = _PROJECT_ROOT / "data" / "processed" / "pd_dataset.csv"
OUTPUT_CSV = _PROJECT_ROOT / "data" / "processed" / "mediapipe_model_comparison.csv"


def load_dataset() -> pd.DataFrame:
    """Load the processed PD gait dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df


def get_candidates() -> Dict[str, Any]:
    """Define candidate classification models and pipelines for comparison."""
    candidates = {
        # 1. Production Baseline RF (300 trees, unconstrained)
        "RandomForest_Current_Baseline (300 trees, leaf=1)": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=1,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        # 2. Regularized RF (min_samples_leaf=2)
        "RandomForest_Regularized (leaf=2)": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_depth=15,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        # 3. Regularized RF (min_samples_leaf=3)
        "RandomForest_Regularized (leaf=3)": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            max_depth=12,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        # 4. Regularized RF (min_samples_leaf=5)
        "RandomForest_Regularized (leaf=5)": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            max_depth=10,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        # 5. Standardized RBF SVM
        "StandardScaler + RBF_SVM (C=1.0)": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                C=1.0,
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        # 6. Standardized RBF SVM (C=0.5, stronger regularization)
        "StandardScaler + RBF_SVM (C=0.5)": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                C=0.5,
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        # 7. HistGradientBoostingClassifier
        "HistGradientBoosting (balanced)": HistGradientBoostingClassifier(
            max_iter=100,
            class_weight="balanced",
            random_state=42,
        ),
    }
    return candidates


def evaluate_model_group_cv(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
) -> Dict[str, float]:
    """Run GroupKFold CV and compute comprehensive metrics."""
    gkf = GroupKFold(n_splits=n_splits)

    accuracies = []
    precisions = []
    sensitivities = []
    specificities = []
    f1_scores = []
    roc_aucs = []
    brier_scores = []

    total_cm = np.zeros((2, 2), dtype=int)

    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit model
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        sens = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)

        cm = confusion_matrix(y_val, preds, labels=[0, 1])
        total_cm += cm
        tn, fp, fn, tp = cm.ravel()

        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        auc = roc_auc_score(y_val, probs) if y_val.nunique() == 2 else np.nan
        brier = brier_score_loss(y_val, probs)

        accuracies.append(acc)
        precisions.append(prec)
        sensitivities.append(sens)
        specificities.append(spec)
        f1_scores.append(f1)
        if not np.isnan(auc):
            roc_aucs.append(auc)
        brier_scores.append(brier)

    return {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "precision_mean": float(np.mean(precisions)),
        "precision_std": float(np.std(precisions)),
        "sensitivity_mean": float(np.mean(sensitivities)),
        "sensitivity_std": float(np.std(sensitivities)),
        "specificity_mean": float(np.mean(specificities)),
        "specificity_std": float(np.std(specificities)),
        "f1_mean": float(np.mean(f1_scores)),
        "f1_std": float(np.std(f1_scores)),
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_std": float(np.std(roc_aucs)),
        "brier_score_mean": float(np.mean(brier_scores)),
        "brier_score_std": float(np.std(brier_scores)),
        "total_tn": int(total_cm[0, 0]),
        "total_fp": int(total_cm[0, 1]),
        "total_fn": int(total_cm[1, 0]),
        "total_tp": int(total_cm[1, 1]),
    }


def run_experiments() -> pd.DataFrame:
    """Execute evaluation across all candidate models and feature configurations."""
    df = load_dataset()

    target_col = "diagnosis"
    ignore_cols = ["subject", "trial", "diagnosis", "diagnosis_name"]

    y = df[target_col]
    groups = df["subject"]

    # Feature sets:
    # 1. Full 290 features (includes fps, num_frames)
    X_full = df.drop(columns=ignore_cols)

    # 2. Pure 288 gait features (excludes fps, num_frames)
    X_pure = df.drop(columns=ignore_cols + ["fps", "num_frames"])

    candidates = get_candidates()
    results: List[Dict[str, Any]] = []

    print("=" * 80)
    print("PARKINSENSE MEDIAPIPE CLASSIFIER BENCHMARK (SUBJECT-WISE 5-FOLD CV)")
    print("=" * 80)
    print(f"Dataset: {len(df)} samples | {groups.nunique()} unique subjects")
    print(f"Class Distribution: {dict(y.value_counts())} (0=Healthy, 1=Parkinson's)")
    print("-" * 80)

    for feature_set_name, X_data in [
        ("290_features (with fps/num_frames)", X_full),
        ("288_features (gait only, no fps/frames)", X_pure),
    ]:
        print(f"\n>>> FEATURE SET: {feature_set_name} ({X_data.shape[1]} features)")

        for model_name, model in candidates.items():
            metrics = evaluate_model_group_cv(model, X_data, y, groups, n_splits=5)

            entry = {
                "feature_set": feature_set_name,
                "num_features": X_data.shape[1],
                "model_name": model_name,
                "accuracy": round(metrics["accuracy_mean"], 4),
                "accuracy_std": round(metrics["accuracy_std"], 4),
                "sensitivity": round(metrics["sensitivity_mean"], 4),
                "sensitivity_std": round(metrics["sensitivity_std"], 4),
                "specificity": round(metrics["specificity_mean"], 4),
                "specificity_std": round(metrics["specificity_std"], 4),
                "precision": round(metrics["precision_mean"], 4),
                "precision_std": round(metrics["precision_std"], 4),
                "f1_score": round(metrics["f1_mean"], 4),
                "f1_std": round(metrics["f1_std"], 4),
                "roc_auc": round(metrics["roc_auc_mean"], 4),
                "roc_auc_std": round(metrics["roc_auc_std"], 4),
                "brier_score": round(metrics["brier_score_mean"], 4),
                "brier_score_std": round(metrics["brier_score_std"], 4),
                "total_tn": metrics["total_tn"],
                "total_fp": metrics["total_fp"],
                "total_fn": metrics["total_fn"],
                "total_tp": metrics["total_tp"],
            }
            results.append(entry)

            print(
                f"  {model_name[:45]:<45s} | Acc: {entry['accuracy']:.4f} ± {entry['accuracy_std']:.4f} | "
                f"Sens: {entry['sensitivity']:.4f} | Spec: {entry['specificity']:.4f} | "
                f"AUC: {entry['roc_auc']:.4f} | Brier: {entry['brier_score']:.4f}"
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "=" * 80)
    print(f"Results successfully saved to: {OUTPUT_CSV}")
    print("=" * 80)
    return results_df


if __name__ == "__main__":
    run_experiments()
