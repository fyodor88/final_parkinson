"""
ParkinSense — Independent External TULIP Evaluation Script
==========================================================
Performs a strictly leakage-free evaluation of models trained ONLY on DNE
against the independent TULIP external benchmark.

Methodology Guarantees:
1. Training data: 100% DNE data only (322 samples, 58 subjects).
2. TULIP held out completely: Zero TULIP samples used in training, feature selection,
   scaler fitting, or hyperparameter tuning.
3. Preprocessing (StandardScaler) fitted strictly on DNE training data.
4. Participant grouping: All camera views for a subject are grouped together.
5. Cohort stratification: Clinical TULIP (8 subjects, 44 videos) and YouTube TULIP
   (6 subjects, 6 videos) are evaluated and reported separately.
6. Multi-level reporting: Video-level and subject-level majority voting metrics.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
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

DNE_PATH = PROJECT_ROOT / "data" / "processed" / "pd_dataset.csv"
TULIP_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "improved_model" / "tulip_extracted_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "improved_model"
ensure_dir(OUTPUT_DIR)


def load_and_prepare_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Load DNE and TULIP datasets, curate subject IDs, and identify feature sets.
    """
    dne_df = pd.read_csv(DNE_PATH)
    tulip_df = pd.read_csv(TULIP_CACHE_PATH)

    # 288 pure biomechanical features
    pose_288_features = [
        c for c in dne_df.columns
        if c.startswith("pose") and any(s in c for s in ["_mean", "_std", "_min", "_max"])
    ]
    # 290 baseline features (including fps, num_frames)
    pose_290_features = ["fps", "num_frames"] + pose_288_features

    # Fix TULIP subject IDs so YouTube HT and PD are distinct subjects
    corrected_tulip = tulip_df.copy()
    corrected_subjects = []
    cohort_types = []

    for _, row in corrected_tulip.iterrows():
        subj = str(row["subject"])
        cat = str(row.get("category", ""))
        diag = int(row["diagnosis"])
        trial = str(row.get("trial", ""))

        if "yt" in subj.lower() or "videoplayback" in trial.lower() or "vidssave" in trial.lower():
            cohort_types.append("youtube")
            if cat == "PD" or diag == 1:
                # Disambiguate youtube PD subjects
                m_num = "1"
                if "720p" in trial.lower():
                    m_num = "2" if "subject_yt_2" in subj else "3"
                corrected_subjects.append(f"tulip_yt_pd_{m_num}")
            else:
                m_num = "1"
                if "(1)" in trial:
                    m_num = "2"
                elif "(2)" in trial:
                    m_num = "3"
                corrected_subjects.append(f"tulip_yt_ht_{m_num}")
        else:
            cohort_types.append("clinical")
            corrected_subjects.append(subj)

    corrected_tulip["subject"] = corrected_subjects
    corrected_tulip["cohort"] = cohort_types

    return dne_df, corrected_tulip, pose_288_features, pose_290_features


def get_model_definitions():
    """Returns candidate model specifications."""
    return {
        "RF_Baseline_290": {
            "name": "RandomForest Baseline (290 features)",
            "features": "290",
            "model": RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "needs_scaling": False,
        },
        "RF_Regularized_288": {
            "name": "RandomForest Regularized (288 features)",
            "features": "288",
            "model": RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                max_depth=15,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "needs_scaling": False,
        },
        "RF_Conservative_288": {
            "name": "RandomForest Conservative (288 features)",
            "features": "288",
            "model": RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                max_depth=10,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "needs_scaling": False,
        },
        "HistGBM_288": {
            "name": "HistGradientBoosting (288 features)",
            "features": "288",
            "model": HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=6,
                min_samples_leaf=5,
                learning_rate=0.05,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=42,
            ),
            "needs_scaling": False,
        },
        "SVM_RBF_288": {
            "name": "SVM RBF + StandardScaler (288 features)",
            "features": "288",
            "model": SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=42,
            ),
            "needs_scaling": True,
        },
        "SVM_Linear_288": {
            "name": "SVM Linear + StandardScaler (288 features)",
            "features": "288",
            "model": SVC(
                kernel="linear",
                C=0.1,
                class_weight="balanced",
                probability=True,
                random_state=42,
            ),
            "needs_scaling": True,
        },
        "LogReg_L2_288": {
            "name": "Logistic Regression + StandardScaler (288 features)",
            "features": "288",
            "model": LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
            "needs_scaling": True,
        },
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute comprehensive diagnostic metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    auc = np.nan
    brier = np.nan
    p_mean = np.nan
    p_std = np.nan
    p_min = np.nan
    p_max = np.nan

    if y_prob is not None:
        p_mean = float(np.mean(y_prob))
        p_std = float(np.std(y_prob))
        p_min = float(np.min(y_prob))
        p_max = float(np.max(y_prob))
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
        "total_samples": len(y_true),
        "true_ht": int(np.sum(y_true == 0)),
        "true_pd": int(np.sum(y_true == 1)),
        "pred_ht": int(np.sum(y_pred == 0)),
        "pred_pd": int(np.sum(y_pred == 1)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1_score": float(f1),
        "roc_auc": auc,
        "brier_score": brier,
        "prob_mean": p_mean,
        "prob_std": p_std,
        "prob_min": p_min,
        "prob_max": p_max,
    }


def evaluate_subject_majority_voting(
    df: pd.DataFrame,
    pred_col: str = "pred_label",
    prob_col: str = "pred_prob",
) -> Dict[str, Any]:
    """Aggregate predictions at subject level via mean predicted probability."""
    subj_rows = []
    for subj_id, group in df.groupby("subject"):
        y_true = int(group["diagnosis"].iloc[0])
        mean_p = float(group[prob_col].mean()) if prob_col in group else float(group[pred_col].mean())
        subj_pred = 1 if mean_p >= 0.5 else 0
        cohort = group["cohort"].iloc[0] if "cohort" in group else "unknown"
        subj_rows.append({
            "subject": subj_id,
            "cohort": cohort,
            "num_videos": len(group),
            "true_label": y_true,
            "pred_label": subj_pred,
            "mean_prob": mean_p,
            "correct": bool(subj_pred == y_true),
        })

    subj_df = pd.DataFrame(subj_rows)
    metrics = compute_metrics(
        subj_df["true_label"].values,
        subj_df["pred_label"].values,
        subj_df["mean_prob"].values,
    )
    return metrics, subj_df


def run_cross_validation_dne(
    dne_df: pd.DataFrame,
    model_defs: dict,
    feat_288: list[str],
    feat_290: list[str],
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Run 5-fold StratifiedGroupKFold on DNE training dataset.
    """
    print(f"\n{'='*70}\nPART A: DNE 5-Fold Subject-Grouped Cross-Validation\n{'='*70}")
    groups = dne_df["subject"].values
    y = dne_df["diagnosis"].values

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(cv.split(dne_df, y, groups))

    cv_results = []

    for model_key, mdef in model_defs.items():
        feature_cols = feat_290 if mdef["features"] == "290" else feat_288
        X = dne_df[feature_cols].values
        needs_scaling = mdef["needs_scaling"]

        fold_metrics = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_val = X[val_idx], y[val_idx]

            if needs_scaling:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_va = scaler.transform(X_va)

            clf = clone(mdef["model"])
            clf.fit(X_tr, y_tr)

            y_pred = clf.predict(X_va)
            y_prob = clf.predict_proba(X_va)[:, 1] if hasattr(clf, "predict_proba") else None

            m = compute_metrics(y_val, y_pred, y_prob)
            m["model_key"] = model_key
            m["fold"] = fold_idx + 1
            fold_metrics.append(m)

        df_m = pd.DataFrame(fold_metrics)
        row = {
            "model_key": model_key,
            "model_name": mdef["name"],
            "features": mdef["features"],
            "cv_acc_mean": df_m["accuracy"].mean(),
            "cv_acc_std": df_m["accuracy"].std(),
            "cv_bal_acc_mean": df_m["balanced_accuracy"].mean(),
            "cv_sens_mean": df_m["sensitivity"].mean(),
            "cv_sens_std": df_m["sensitivity"].std(),
            "cv_spec_mean": df_m["specificity"].mean(),
            "cv_spec_std": df_m["specificity"].std(),
            "cv_f1_mean": df_m["f1_score"].mean(),
            "cv_auc_mean": df_m["roc_auc"].mean(),
            "cv_auc_std": df_m["roc_auc"].std(),
            "cv_brier_mean": df_m["brier_score"].mean(),
        }
        cv_results.append(row)
        print(f"  {mdef['name']:<48} | Acc={row['cv_acc_mean']:.3f}±{row['cv_acc_std']:.3f} | Sens={row['cv_sens_mean']:.3f} | Spec={row['cv_spec_mean']:.3f} | AUC={row['cv_auc_mean']:.3f}")

    return pd.DataFrame(cv_results)


def run_independent_external_evaluation(
    dne_df: pd.DataFrame,
    tulip_df: pd.DataFrame,
    model_defs: dict,
    feat_288: list[str],
    feat_290: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train strictly on 100% DNE data and evaluate on TULIP cohorts.
    """
    print(f"\n{'='*70}\nPART B: Genuinely Independent External TULIP Evaluation\n{'='*70}")

    video_eval_rows = []
    subject_details_list = []

    y_dne = dne_df["diagnosis"].values

    for model_key, mdef in model_defs.items():
        feat_cols = feat_290 if mdef["features"] == "290" else feat_288
        X_dne = dne_df[feat_cols].values
        X_tulip = tulip_df[feat_cols].values
        needs_scaling = mdef["needs_scaling"]

        # Train on DNE ONLY
        if needs_scaling:
            scaler = StandardScaler()
            X_dne_fit = scaler.fit_transform(X_dne)
            X_tulip_trans = scaler.transform(X_tulip)
        else:
            scaler = None
            X_dne_fit = X_dne
            X_tulip_trans = X_tulip

        clf = clone(mdef["model"])
        clf.fit(X_dne_fit, y_dne)

        # Predict on TULIP
        preds = clf.predict(X_tulip_trans)
        probs = clf.predict_proba(X_tulip_trans)[:, 1] if hasattr(clf, "predict_proba") else None

        tulip_res = tulip_df.copy()
        tulip_res["pred_label"] = preds
        tulip_res["pred_prob"] = probs if probs is not None else preds.astype(float)

        # 1. Overall TULIP (50 videos)
        m_overall = compute_metrics(tulip_res["diagnosis"].values, preds, probs)
        m_subj_overall, subj_df_overall = evaluate_subject_majority_voting(tulip_res)

        # 2. Clinical Cohort Only (44 videos, 8 subjects)
        clin_mask = tulip_res["cohort"] == "clinical"
        tulip_clin = tulip_res[clin_mask]
        m_clin_vid = compute_metrics(
            tulip_clin["diagnosis"].values,
            tulip_clin["pred_label"].values,
            tulip_clin["pred_prob"].values,
        )
        m_clin_subj, subj_df_clin = evaluate_subject_majority_voting(tulip_clin)

        # 3. YouTube Cohort Only (6 videos, 6 subjects)
        yt_mask = tulip_res["cohort"] == "youtube"
        tulip_yt = tulip_res[yt_mask]
        m_yt_vid = compute_metrics(
            tulip_yt["diagnosis"].values,
            tulip_yt["pred_label"].values,
            tulip_yt["pred_prob"].values,
        )
        m_yt_subj, subj_df_yt = evaluate_subject_majority_voting(tulip_yt)

        # Store for summary
        video_eval_rows.append({
            "model_key": model_key,
            "model_name": mdef["name"],
            # Clinical Subject-level
            "clin_subj_acc": m_clin_subj["accuracy"],
            "clin_subj_sens": m_clin_subj["sensitivity"],
            "clin_subj_spec": m_clin_subj["specificity"],
            "clin_subj_tp": m_clin_subj["tp"],
            "clin_subj_fp": m_clin_subj["fp"],
            "clin_subj_tn": m_clin_subj["tn"],
            "clin_subj_fn": m_clin_subj["fn"],
            "clin_subj_auc": m_clin_subj["roc_auc"],
            # Clinical Video-level
            "clin_vid_acc": m_clin_vid["accuracy"],
            "clin_vid_sens": m_clin_vid["sensitivity"],
            "clin_vid_spec": m_clin_vid["specificity"],
            "clin_vid_cm": f"[{m_clin_vid['tn']},{m_clin_vid['fp']},{m_clin_vid['fn']},{m_clin_vid['tp']}]",
            "clin_pred_pd_count": m_clin_vid["pred_pd"],
            "clin_prob_mean": m_clin_vid["prob_mean"],
            "clin_prob_std": m_clin_vid["prob_std"],
            # YouTube Video-level
            "yt_vid_acc": m_yt_vid["accuracy"],
            "yt_vid_sens": m_yt_vid["sensitivity"],
            "yt_vid_spec": m_yt_vid["specificity"],
            "yt_vid_cm": f"[{m_yt_vid['tn']},{m_yt_vid['fp']},{m_yt_vid['fn']},{m_yt_vid['tp']}]",
            "yt_pred_pd_count": m_yt_vid["pred_pd"],
            # Overall Video-level
            "all_vid_acc": m_overall["accuracy"],
            "all_vid_sens": m_overall["sensitivity"],
            "all_vid_spec": m_overall["specificity"],
            "all_vid_f1": m_overall["f1_score"],
            "all_vid_auc": m_overall["roc_auc"],
            "all_pred_pd_count": m_overall["pred_pd"],
            "all_vid_cm": f"[{m_overall['tn']},{m_overall['fp']},{m_overall['fn']},{m_overall['tp']}]",
        })

        # Save subject breakdown for clinical
        for _, srow in subj_df_clin.iterrows():
            subject_details_list.append({
                "model_key": model_key,
                "cohort": "clinical",
                "subject": srow["subject"],
                "true_label": srow["true_label"],
                "pred_label": srow["pred_label"],
                "mean_prob": round(srow["mean_prob"], 4),
                "correct": srow["correct"],
                "num_videos": srow["num_videos"],
            })
        for _, srow in subj_df_yt.iterrows():
            subject_details_list.append({
                "model_key": model_key,
                "cohort": "youtube",
                "subject": srow["subject"],
                "true_label": srow["true_label"],
                "pred_label": srow["pred_label"],
                "mean_prob": round(srow["mean_prob"], 4),
                "correct": srow["correct"],
                "num_videos": srow["num_videos"],
            })

        print(f"  {mdef['name']:<48}")
        print(f"    Clinical Subjects (N=8): Acc={m_clin_subj['accuracy']:.3f} | Sens={m_clin_subj['sensitivity']:.3f} | Spec={m_clin_subj['specificity']:.3f} | CM=[TN={m_clin_subj['tn']}, FP={m_clin_subj['fp']}, FN={m_clin_subj['fn']}, TP={m_clin_subj['tp']}]")
        print(f"    Clinical Videos   (N=44): Acc={m_clin_vid['accuracy']:.3f} | Sens={m_clin_vid['sensitivity']:.3f} | Spec={m_clin_vid['specificity']:.3f} | Pred PD={m_clin_vid['pred_pd']}/44 | Mean P(PD)={m_clin_vid['prob_mean']:.4f}±{m_clin_vid['prob_std']:.4f}")
        print(f"    YouTube Videos    (N=6):  Acc={m_yt_vid['accuracy']:.3f} | Sens={m_yt_vid['sensitivity']:.3f} | Spec={m_yt_vid['specificity']:.3f} | Pred PD={m_yt_vid['pred_pd']}/6")

    summary_df = pd.DataFrame(video_eval_rows)
    subj_df = pd.DataFrame(subject_details_list)

    summary_df.to_csv(OUTPUT_DIR / "tulip_independent_evaluation_summary.csv", index=False)
    subj_df.to_csv(OUTPUT_DIR / "tulip_independent_subject_details.csv", index=False)

    return summary_df, subj_df


def main():
    dne_df, tulip_df, feat_288, feat_290 = load_and_prepare_datasets()
    model_defs = get_model_definitions()

    print(f"Loaded DNE dataset:   {len(dne_df)} samples, {dne_df['subject'].nunique()} subjects (HT: {(dne_df['diagnosis']==0).sum()}, PD: {(dne_df['diagnosis']==1).sum()})")
    print(f"Loaded TULIP dataset: {len(tulip_df)} samples, {tulip_df['subject'].nunique()} subjects")
    print(f"  - Clinical cohort:  {(tulip_df['cohort']=='clinical').sum()} videos, {tulip_df[tulip_df['cohort']=='clinical']['subject'].nunique()} subjects (3 HT, 5 PD)")
    print(f"  - YouTube cohort:   {(tulip_df['cohort']=='youtube').sum()} videos, {tulip_df[tulip_df['cohort']=='youtube']['subject'].nunique()} subjects (3 HT, 3 PD)")

    # Part A: DNE Cross-Validation
    cv_df = run_cross_validation_dne(dne_df, model_defs, feat_288, feat_290, n_splits=5)
    cv_df.to_csv(OUTPUT_DIR / "dne_cv_results_independent.csv", index=False)

    # Part B: Independent External Evaluation
    summary_df, subj_df = run_independent_external_evaluation(dne_df, tulip_df, model_defs, feat_288, feat_290)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE — SUMMARY SAVED TO data/processed/improved_model/")
    print("="*70)


if __name__ == "__main__":
    main()
