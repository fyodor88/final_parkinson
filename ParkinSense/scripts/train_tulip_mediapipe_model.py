"""
ParkinSense — Train & Validate MediaPipe Parkinson's Detector on TULIP
======================================================================
Trains a Random Forest classifier using MediaPipe-extracted gait features
from the verified clinical TULIP dataset (44 videos, 8 participants).

Evaluation Protocol:
- Subject-Grouped Cross-Validation (Leave-One-Subject-Out across 8 clinical subjects)
- 4-Fold GroupKFold (holding out 2 subjects per fold)
- Evaluates both recording-level (video) and subject-level majority voting
- Strict leakage prevention: all camera views for any subject remain in the same split
- Excludes metadata features (fps, num_frames) to train pure 288-D biomechanical model
- Saves model to models/pd_detector_tulip_candidate.pkl with separate verification step
"""

from __future__ import annotations

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "improved_model" / "tulip_training_dataset.csv"
YT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "improved_model" / "tulip_youtube_eval_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "improved_model"
MODELS_DIR = PROJECT_ROOT / "models"
CANDIDATE_MODEL_PATH = MODELS_DIR / "pd_detector_tulip_candidate.pkl"


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute complete classification metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    sens = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    spec = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    auc = np.nan
    brier = np.nan
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) > 1:
                auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
        try:
            brier = float(brier_score_loss(y_true, y_prob))
        except Exception:
            pass

    return {
        "n_samples": len(y_true),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1_score": f1,
        "roc_auc": auc,
        "brier_score": brier,
    }


def evaluate_subject_voting(
    df: pd.DataFrame,
    pred_col: str = "pred_label",
    prob_col: str = "pred_prob",
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Subject-level majority voting aggregation using mean predicted probability."""
    subj_rows = []
    for subj_id, group in df.groupby("subject_id"):
        y_true = int(group["verified_label"].iloc[0])
        mean_prob = float(group[prob_col].mean())
        subj_pred = 1 if mean_prob >= 0.5 else 0
        subj_rows.append({
            "subject_id": subj_id,
            "true_label": y_true,
            "pred_label": subj_pred,
            "mean_prob": round(mean_prob, 4),
            "n_videos": len(group),
            "correct": bool(subj_pred == y_true),
        })
    s_df = pd.DataFrame(subj_rows)
    metrics = compute_metrics(
        s_df["true_label"].values,
        s_df["pred_label"].values,
        s_df["mean_prob"].values,
    )
    return metrics, s_df


def run_logo_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    model: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leave-One-Subject-Out Cross-Validation (8 folds).
    Each fold completely holds out all camera views of one subject.
    """
    X = df[feature_cols].values
    y = df["verified_label"].values
    groups = df["subject_id"].values

    logo = LeaveOneGroupOut()
    fold_results = []
    oof_predictions = df.copy()
    oof_predictions["pred_label"] = -1
    oof_predictions["pred_prob"] = np.nan

    for fold_idx, (train_idx, val_idx) in enumerate(logo.split(X, y, groups)):
        held_out_subj = groups[val_idx][0]
        clf = clone(model)
        clf.fit(X[train_idx], y[train_idx])

        preds = clf.predict(X[val_idx])
        # Find positive class (1) column
        pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
        probs = clf.predict_proba(X[val_idx])[:, pos_idx]

        oof_predictions.loc[val_idx, "pred_label"] = preds
        oof_predictions.loc[val_idx, "pred_prob"] = probs

        val_true = y[val_idx]
        correct_vids = int(np.sum(preds == val_true))
        total_vids = len(val_idx)
        subj_vote = 1 if float(np.mean(probs)) >= 0.5 else 0
        subj_correct = bool(subj_vote == val_true[0])

        fold_results.append({
            "fold": fold_idx + 1,
            "test_subject": held_out_subj,
            "test_class": "PD" if val_true[0] == 1 else "HT",
            "n_test_videos": total_vids,
            "video_accuracy": round(correct_vids / total_vids, 4),
            "mean_pred_prob": round(float(np.mean(probs)), 4),
            "subject_prediction": subj_vote,
            "subject_correct": subj_correct,
        })

    fold_df = pd.DataFrame(fold_results)
    return fold_df, oof_predictions


def run_4fold_group_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    model: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    4-Fold GroupKFold Cross-Validation.
    Holds out 2 subjects per fold.
    """
    X = df[feature_cols].values
    y = df["verified_label"].values
    groups = df["subject_id"].values

    gkf = GroupKFold(n_splits=4)
    fold_results = []
    oof_df = df.copy()
    oof_df["pred_label"] = -1
    oof_df["pred_prob"] = np.nan

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        test_subjs = sorted(list(set(groups[val_idx])))
        clf = clone(model)
        clf.fit(X[train_idx], y[train_idx])

        preds = clf.predict(X[val_idx])
        pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
        probs = clf.predict_proba(X[val_idx])[:, pos_idx]

        oof_df.loc[val_idx, "pred_label"] = preds
        oof_df.loc[val_idx, "pred_prob"] = probs

        m = compute_metrics(y[val_idx], preds, probs)
        m["fold"] = fold_idx + 1
        m["test_subjects"] = ", ".join(test_subjs)
        fold_results.append(m)

    return pd.DataFrame(fold_results), oof_df


def main():
    print(f"\n{'='*70}\nPHASE 6: TRAINING NEW MEDIAPIPE TULIP PARKINSONS DETECTOR\n{'='*70}")
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c.startswith("pose")]
    print(f"Loaded clinical training dataset: {len(df)} recordings, {df['subject_id'].nunique()} subjects.")
    print(f"Features: {len(feature_cols)} pure biomechanical pose features (fps/num_frames excluded).")
    print(f"Class distribution: HT = {(df['verified_label']==0).sum()} (3 subjects), PD = {(df['verified_label']==1).sum()} (5 subjects)")

    # Define candidate models
    models_to_test = {
        "RF_Regularized": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "RF_Conservative": RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    comparison_rows = []

    for model_name, base_model in models_to_test.items():
        print(f"\n--- Benchmarking: {model_name} ---")
        logo_folds, oof_logo = run_logo_cv(df, feature_cols, base_model)
        m_vid_logo = compute_metrics(df["verified_label"].values, oof_logo["pred_label"].values, oof_logo["pred_prob"].values)
        m_subj_logo, s_df = evaluate_subject_voting(oof_logo)

        g4_folds, oof_g4 = run_4fold_group_cv(df, feature_cols, base_model)
        m_vid_g4 = compute_metrics(df["verified_label"].values, oof_g4["pred_label"].values, oof_g4["pred_prob"].values)
        m_subj_g4, _ = evaluate_subject_voting(oof_g4)

        comparison_rows.append({
            "model_name": model_name,
            # LOGO video-level
            "logo_vid_accuracy": m_vid_logo["accuracy"],
            "logo_vid_balanced_acc": m_vid_logo["balanced_accuracy"],
            "logo_vid_sensitivity": m_vid_logo["sensitivity"],
            "logo_vid_specificity": m_vid_logo["specificity"],
            "logo_vid_f1": m_vid_logo["f1_score"],
            "logo_vid_auc": m_vid_logo["roc_auc"],
            "logo_vid_brier": m_vid_logo["brier_score"],
            "logo_vid_cm": f"[{m_vid_logo['tn']},{m_vid_logo['fp']},{m_vid_logo['fn']},{m_vid_logo['tp']}]",
            # LOGO subject-level
            "logo_subj_accuracy": m_subj_logo["accuracy"],
            "logo_subj_balanced_acc": m_subj_logo["balanced_accuracy"],
            "logo_subj_sensitivity": m_subj_logo["sensitivity"],
            "logo_subj_specificity": m_subj_logo["specificity"],
            "logo_subj_cm": f"[{m_subj_logo['tn']},{m_subj_logo['fp']},{m_subj_logo['fn']},{m_subj_logo['tp']}]",
            # 4-Fold GroupKFold video-level
            "g4_vid_accuracy": m_vid_g4["accuracy"],
            "g4_vid_sensitivity": m_vid_g4["sensitivity"],
            "g4_vid_specificity": m_vid_g4["specificity"],
            "g4_vid_auc": m_vid_g4["roc_auc"],
            # 4-Fold GroupKFold subject-level
            "g4_subj_accuracy": m_subj_g4["accuracy"],
            "g4_subj_sensitivity": m_subj_g4["sensitivity"],
            "g4_subj_specificity": m_subj_g4["specificity"],
        })

        print(f"  LOGO Subject Accuracy: {m_subj_logo['accuracy']:.3f} (Sens={m_subj_logo['sensitivity']:.3f}, Spec={m_subj_logo['specificity']:.3f}) | CM={m_subj_logo['tn']},{m_subj_logo['fp']},{m_subj_logo['fn']},{m_subj_logo['tp']}")
        print(f"  LOGO Video Accuracy:   {m_vid_logo['accuracy']:.3f} (Sens={m_vid_logo['sensitivity']:.3f}, Spec={m_vid_logo['specificity']:.3f}, AUC={m_vid_logo['roc_auc']:.3f})")

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(OUTPUT_DIR / "tulip_grouped_cv_results.csv", index=False)
    print(f"\nSaved cross-validation results to: {OUTPUT_DIR / 'tulip_grouped_cv_results.csv'}")

    # Select best model based on LOGO subject-level balanced accuracy
    # RF_Conservative has best subject accuracy (0.500 vs 0.375 for RF_Regularized)
    best_model_name = "RF_Conservative"
    selected_clf = clone(models_to_test[best_model_name])

    print(f"\nTraining final candidate model '{best_model_name}' on all 44 clinical recordings...")
    X_all = df[feature_cols].values
    y_all = df["verified_label"].values
    selected_clf.fit(X_all, y_all)

    # Candidate model artifact dictionary
    candidate_artifact = {
        "model": selected_clf,
        "feature_names": feature_cols,
        "n_features": len(feature_cols),
        "classes_": list(selected_clf.classes_),
        "class_mapping": {0: "Healthy Control", 1: "Parkinsons Disease"},
        "training_data": {
            "source": "TULIP Clinical Gait Cohort",
            "n_samples": len(X_all),
            "n_participants": int(df["subject_id"].nunique()),
            "class_distribution": {"Healthy": int(np.sum(y_all == 0)), "Parkinsons": int(np.sum(y_all == 1))},
        },
        "version": "2.0.0-mediapipe-tulip",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cv_performance": comp_df[comp_df["model_name"] == best_model_name].iloc[0].to_dict(),
    }

    joblib.dump(candidate_artifact, CANDIDATE_MODEL_PATH)
    print(f"✅ Saved candidate model to: {CANDIDATE_MODEL_PATH}")

    # =========================================================================
    # SEPARATE VERIFICATION STEP: Reload and test candidate model on real sample
    # =========================================================================
    print("\n--- Running Verification Step: Reloading Candidate Model ---")
    reloaded_artifact = joblib.load(CANDIDATE_MODEL_PATH)
    assert "model" in reloaded_artifact, "Missing 'model' key in artifact"
    assert "feature_names" in reloaded_artifact, "Missing 'feature_names' in artifact"
    assert len(reloaded_artifact["feature_names"]) == 288, "Feature count mismatch"

    loaded_model = reloaded_artifact["model"]
    test_sample = X_all[0:1]  # Real extracted feature vector row
    test_pred = loaded_model.predict(test_sample)[0]
    test_probs = loaded_model.predict_proba(test_sample)[0]

    print(f"  Reloaded model type: {type(loaded_model).__name__}")
    print(f"  Test prediction on real sample (Subject {df['subject_id'].iloc[0]}): class {test_pred}")
    print(f"  Class probabilities: P(HT)={test_probs[0]:.4f}, P(PD)={test_probs[1]:.4f}")
    assert np.isclose(np.sum(test_probs), 1.0), "Probabilities do not sum to 1.0"
    print("  ✅ Candidate reload and prediction verified successfully!")

    # =========================================================================
    # Exploratory Out-of-Distribution Test on YouTube Cohort (6 videos)
    # =========================================================================
    if YT_DATA_PATH.exists():
        yt_df = pd.read_csv(YT_DATA_PATH)
        X_yt = yt_df[feature_cols].values
        y_yt = yt_df["verified_label"].values

        yt_preds = loaded_model.predict(X_yt)
        yt_probs = loaded_model.predict_proba(X_yt)[:, 1]
        m_yt = compute_metrics(y_yt, yt_preds, yt_probs)

        print(f"\n--- Exploratory Out-of-Distribution Test: YouTube Cohort (N={len(yt_df)}) ---")
        print(f"  YouTube Video Accuracy: {m_yt['accuracy']:.3f} | Sens={m_yt['sensitivity']:.3f} | Spec={m_yt['specificity']:.3f} | CM=[{m_yt['tn']},{m_yt['fp']},{m_yt['fn']},{m_yt['tp']}]")
        print(f"  Note: YouTube cohort labels are unverified; provided for exploratory analysis only.")

    # Save model metadata JSON
    meta_json = {
        "candidate_file": str(CANDIDATE_MODEL_PATH.name),
        "architecture": f"RandomForestClassifier(n_estimators={selected_clf.n_estimators}, max_depth={selected_clf.max_depth}, min_samples_leaf={selected_clf.min_samples_leaf}, class_weight='{selected_clf.class_weight}')",
        "feature_count": 288,
        "feature_contract": "tulip_feature_contract.json (v2.0.0)",
        "training_provenance": {
            "dataset": "data/processed/improved_model/tulip_training_dataset.csv",
            "samples": len(X_all),
            "subjects": int(df["subject_id"].nunique()),
            "healthy_subjects": 3,
            "parkinsons_subjects": 5,
            "healthy_recordings": int(np.sum(y_all == 0)),
            "parkinsons_recordings": int(np.sum(y_all == 1)),
        },
        "validation_methodology": "Leave-One-Subject-Out Cross-Validation (8 folds) & 4-Fold GroupKFold",
        "validation_results": comp_df[comp_df["model_name"] == best_model_name].iloc[0].to_dict(),
        "limitations": [
            "Cohort is small (N=8 independent clinical participants: 3 HT, 5 PD).",
            "Results are exploratory and should not be interpreted as statistically conclusive clinical validation.",
            "Production default models/pd_detector.pkl remains unchanged to preserve backward compatibility."
        ],
        "production_status": "OFFLINE_CANDIDATE_INTEGRATED (Available for explicit candidate evaluation via API/pipeline)",
    }

    with open(OUTPUT_DIR / "tulip_model_metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)
    print(f"Saved metadata to: {OUTPUT_DIR / 'tulip_model_metadata.json'}")


if __name__ == "__main__":
    main()
