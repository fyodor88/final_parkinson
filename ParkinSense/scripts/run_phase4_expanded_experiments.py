"""
ParkinSense — Phase 4 Comprehensive Multi-Track Experiments
============================================================
Implements:
  - Track A: DNE baseline (grouped CV)
  - Track B: DNE + TULIP clinical expanded (StratifiedGroupKFold on 66 subjects)
  - Track C: DNE + TULIP clinical + YouTube expanded (StratifiedGroupKFold on 72 subjects)
  - Track D: Source-specific evaluation & cross-domain evaluation
  - Feature selection: Top 5, 10, 15, 20, 30, 50, Full 288, Legacy 290
  - Feature selection methods: RF Importance (correlation filtered), L1 Logistic, Mutual Info
  - Feature stability (Jaccard similarity across folds)
  - Classifiers: RF baseline, Regularized RF, SVM RBF, LogReg L2, Extra Trees, Gradient Boosting
  - Metrics: Accuracy, Precision, Recall/Sensitivity, Specificity, F1, ROC-AUC, Brier score, Confusion Matrix
  - Full leakage prevention: Grouped folds by subject, fit transformers & selectors on train fold only.
  - Final candidate model training & serialization.
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    log_loss
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_contract import FEATURE_CONTRACT_COLUMNS

warnings.filterwarnings("ignore")

EXPANDED_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "phase4" / "expanded_training_dataset.csv"
OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "phase4"
OUTPUT_REPORTS_DIR = PROJECT_ROOT / "reports" / "phase4"
MODELS_DIR = PROJECT_ROOT / "models"

OUTER_SEEDS = [42, 123, 2026]
K_COUNTS = [5, 10, 15, 20, 30, 50]
CORR_THRESHOLD = 0.85

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    sens = float(recall_score(y_true, y_pred, zero_division=0)) # PD recall

    # Specificity = TN / (TN + FP)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_prob))
        else:
            auc = float("nan")
    except Exception:
        auc = float("nan")

    try:
        brier = float(brier_score_loss(y_true, y_prob))
    except Exception:
        brier = float("nan")

    # Collapse flag check
    collapse = ""
    if tp == 0 and fp == 0:
        collapse = "all_predicted_healthy"
    elif tn == 0 and fn == 0:
        collapse = "all_predicted_pd"

    return {
        "accuracy": acc,
        "precision": prec,
        "sensitivity": sens,
        "specificity": spec,
        "f1_score": f1,
        "roc_auc": auc,
        "brier_score": brier,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "collapse_flag": collapse,
        "n_samples": len(y_true)
    }

def aggregate_participant_predictions(recording_df: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate recording-level predictions to participant-level via mean uncalibrated score."""
    grouped = recording_df.groupby("subject_id").agg({
        "label": "first",
        "prob": "mean"
    }).reset_index()

    y_true = grouped["label"].values
    y_prob = grouped["prob"].values
    y_pred = (y_prob >= 0.5).astype(int)

    m = compute_metrics(y_true, y_pred, y_prob)
    m["n_subjects"] = len(grouped)
    return m

def fit_feature_selectors(X_train: pd.DataFrame, y_train: np.ndarray, seed: int = 42) -> Dict[str, List[str]]:
    """Fit feature selectors strictly on training data."""
    rankings: Dict[str, List[str]] = {}
    cols = list(X_train.columns)

    # 1. RF importance with correlation filtering
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight="balanced", n_jobs=-1)
    rf.fit(X_train, y_train)
    raw_importances = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)

    # Correlation filter
    selected_corr = []
    corr_matrix = X_train[raw_importances.index].corr().abs()
    for feat in raw_importances.index:
        too_correlated = False
        for chosen in selected_corr:
            if corr_matrix.loc[feat, chosen] > CORR_THRESHOLD:
                too_correlated = True
                break
        if not too_correlated:
            selected_corr.append(feat)
    rankings["corr_rf_importance"] = selected_corr

    # 2. L1-regularized Logistic Regression (LASSO)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    l1_clf = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, class_weight="balanced", random_state=seed, max_iter=1000)
    l1_clf.fit(X_scaled, y_train)
    l1_coef = pd.Series(np.abs(l1_clf.coef_[0]), index=cols).sort_values(ascending=False)
    rankings["l1_logistic"] = list(l1_coef[l1_coef > 0].index)

    # 3. Mutual Information
    mi = mutual_info_classif(X_train, y_train, random_state=seed)
    mi_series = pd.Series(mi, index=cols).sort_values(ascending=False)
    rankings["mutual_info"] = list(mi_series.index)

    return rankings

def build_classifier(name: str, seed: int = 42) -> BaseEstimator:
    if name == "rf_baseline":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)
    elif name == "rf_regularized":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_depth=15, class_weight="balanced", random_state=seed, n_jobs=-1)
    elif name == "svm_rbf":
        return SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed)
    elif name == "logreg_l2":
        return LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)
    elif name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, max_depth=15, class_weight="balanced", random_state=seed, n_jobs=-1)
    elif name == "hist_gb":
        return HistGradientBoostingClassifier(max_iter=100, min_samples_leaf=5, random_state=seed)
    else:
        raise ValueError(f"Unknown classifier: {name}")

def run_grouped_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    track_name: str,
    seeds: List[int] = OUTER_SEEDS
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run subject-grouped StratifiedGroupKFold across seeds, models, and feature configs."""
    models_to_test = ["rf_baseline", "rf_regularized", "svm_rbf", "logreg_l2", "extra_trees", "hist_gb"]

    fold_results = []
    participant_results = []
    selector_stability_records = []

    print(f"\n=======================================================")
    print(f"RUNNING {track_name} — Total rows: {len(df)}, Subjects: {df['subject_id'].nunique()}")
    print(f"=======================================================")

    X_all = df[feature_cols].copy()
    y_all = df["label"].values
    groups_all = df["subject_id"].values

    for seed in seeds:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_idx = 0

        for train_idx, val_idx in sgkf.split(X_all, y_all, groups_all):
            fold_idx += 1
            X_train_raw = X_all.iloc[train_idx]
            y_train = y_all[train_idx]
            groups_train = groups_all[train_idx]

            X_val_raw = X_all.iloc[val_idx]
            y_val = y_all[val_idx]
            groups_val = groups_all[val_idx]

            # Fit feature selectors strictly on X_train_raw (only pose features, excluding metadata)
            pure_pose_cols = [c for c in feature_cols if c.startswith("pose")]
            selectors = fit_feature_selectors(X_train_raw[pure_pose_cols], y_train, seed=seed)

            # Record selector top-10 for stability tracking
            for sel_name, ranked in selectors.items():
                selector_stability_records.append({
                    "track": track_name,
                    "seed": seed,
                    "fold": fold_idx,
                    "selector": sel_name,
                    "top10": ranked[:10]
                })

            # Define feature set configurations to test
            feature_configs = {}
            feature_configs["full_288"] = pure_pose_cols
            if "fps" in feature_cols and "num_frames" in feature_cols:
                feature_configs["legacy_290_metadata"] = feature_cols

            for k in K_COUNTS:
                if len(selectors["corr_rf_importance"]) >= k:
                    feature_configs[f"corr_rf_top{k}"] = selectors["corr_rf_importance"][:k]
                if len(selectors["l1_logistic"]) >= k:
                    feature_configs[f"l1_logistic_top{k}"] = selectors["l1_logistic"][:k]
                if len(selectors["mutual_info"]) >= k:
                    feature_configs[f"mutual_info_top{k}"] = selectors["mutual_info"][:k]

            # Test models on each feature configuration
            for f_name, f_cols in feature_configs.items():
                X_tr = X_train_raw[f_cols].copy()
                X_va = X_val_raw[f_cols].copy()

                # Scale if required by model
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_tr)
                X_va_scaled = scaler.transform(X_va)

                for model_name in models_to_test:
                    # In high-dimensional linear/SVM models, prefer scaled features
                    use_scaled = model_name in ["svm_rbf", "logreg_l2"]
                    X_tr_in = X_tr_scaled if use_scaled else X_tr.values
                    X_va_in = X_va_scaled if use_scaled else X_va.values

                    t0 = time.time()
                    clf = build_classifier(model_name, seed=seed)
                    clf.fit(X_tr_in, y_train)
                    train_time = time.time() - t0

                    # Predict
                    prob = clf.predict_proba(X_va_in)[:, 1]
                    pred = (prob >= 0.5).astype(int)

                    rec_metrics = compute_metrics(y_val, pred, prob)
                    rec_metrics.update({
                        "track": track_name,
                        "seed": seed,
                        "fold": fold_idx,
                        "model": model_name,
                        "feature_config": f_name,
                        "n_features": len(f_cols),
                        "train_time_sec": round(train_time, 4)
                    })
                    fold_results.append(rec_metrics)

                    # Participant-level aggregation
                    val_rec_df = pd.DataFrame({
                        "subject_id": groups_val,
                        "label": y_val,
                        "prob": prob
                    })
                    part_m = aggregate_participant_predictions(val_rec_df)
                    part_m.update({
                        "track": track_name,
                        "seed": seed,
                        "fold": fold_idx,
                        "model": model_name,
                        "feature_config": f_name,
                        "n_features": len(f_cols)
                    })
                    participant_results.append(part_m)

    fold_df = pd.DataFrame(fold_results)
    part_df = pd.DataFrame(participant_results)

    # Compute stability (Jaccard similarity of top-10 across folds)
    stability_data = {}
    for sel in ["corr_rf_importance", "l1_logistic", "mutual_info"]:
        sel_folds = [r["top10"] for r in selector_stability_records if r["selector"] == sel]
        if len(sel_folds) > 1:
            jaccards = []
            for i in range(len(sel_folds)):
                for j in range(i + 1, len(sel_folds)):
                    s1 = set(sel_folds[i])
                    s2 = set(sel_folds[j])
                    jacc = len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 0.0
                    jaccards.append(jacc)
            stability_data[sel] = float(np.mean(jaccards))
        else:
            stability_data[sel] = 1.0

    return fold_df, part_df, stability_data

def run_all_phase4_experiments():
    full_dataset = pd.read_csv(EXPANDED_DATASET_PATH)
    all_feature_cols = [c for c in FEATURE_CONTRACT_COLUMNS if c in full_dataset.columns]

    # Partition datasets
    dne_data = full_dataset[full_dataset["dataset_source"] == "DNE"].copy()
    tulip_clinical_eligible = full_dataset[(full_dataset["dataset_source"] == "TULIP_Clinical") & (full_dataset["eligible_for_training"])].copy()
    tulip_youtube = full_dataset[full_dataset["dataset_source"] == "TULIP_YouTube"].copy()
    primary_expanded = full_dataset[full_dataset["eligible_for_training"]].copy() # DNE + TULIP Clinical
    all_sources = full_dataset[full_dataset["retention_status"] != "EXCLUDED"].copy() # DNE + TULIP Clinical + YouTube

    # Track A: DNE baseline
    dne_fold_df, dne_part_df, dne_stab = run_grouped_cv(dne_data, all_feature_cols, "Track_A_DNE_Baseline")

    # Track B: DNE + TULIP Clinical Expanded
    b_fold_df, b_part_df, b_stab = run_grouped_cv(primary_expanded, all_feature_cols, "Track_B_DNE_Plus_TULIP_Clinical")

    # Track C: DNE + TULIP Clinical + YouTube Expanded
    c_fold_df, c_part_df, c_stab = run_grouped_cv(all_sources, all_feature_cols, "Track_C_DNE_TULIP_YouTube")

    # Track D: Source-Specific & Cross-Domain Evaluation
    print("\n=======================================================")
    print("RUNNING TRACK D — SOURCE SPECIFIC & CROSS DOMAIN")
    print("=======================================================")
    source_eval_records = []

    # DNE -> TULIP External Generalization test
    # Train on full DNE -> test on eligible TULIP clinical
    pure_pose_cols = [c for c in all_feature_cols if c.startswith("pose")]
    X_dne = dne_data[pure_pose_cols].values
    y_dne = dne_data["label"].values
    X_tulip = tulip_clinical_eligible[pure_pose_cols].values
    y_tulip = tulip_clinical_eligible["label"].values
    tulip_subjects = tulip_clinical_eligible["subject_id"].values

    scaler = StandardScaler()
    X_dne_scaled = scaler.fit_transform(X_dne)
    X_tulip_scaled = scaler.transform(X_tulip)

    for m_name in ["rf_baseline", "rf_regularized", "svm_rbf", "logreg_l2", "extra_trees", "hist_gb"]:
        clf = build_classifier(m_name, seed=42)
        X_tr = X_dne_scaled if m_name in ["svm_rbf", "logreg_l2"] else X_dne
        X_te = X_tulip_scaled if m_name in ["svm_rbf", "logreg_l2"] else X_tulip
        clf.fit(X_tr, y_dne)
        prob = clf.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)

        m = compute_metrics(y_tulip, pred, prob)
        rec_df = pd.DataFrame({"subject_id": tulip_subjects, "label": y_tulip, "prob": prob})
        part_m = aggregate_participant_predictions(rec_df)

        source_eval_records.append({
            "test_type": "Cross_Domain_DNE_Train_To_TULIP_Test",
            "model": m_name,
            "feature_config": "full_288",
            "recording_accuracy": m["accuracy"],
            "recording_sensitivity": m["sensitivity"],
            "recording_specificity": m["specificity"],
            "recording_f1": m["f1_score"],
            "recording_auc": m["roc_auc"],
            "recording_collapse": m["collapse_flag"],
            "participant_accuracy": part_m["accuracy"],
            "participant_sensitivity": part_m["sensitivity"],
            "participant_specificity": part_m["specificity"],
            "participant_f1": part_m["f1_score"],
            "participant_auc": part_m["roc_auc"],
            "participant_collapse": part_m["collapse_flag"],
            "n_test_samples": len(y_tulip),
            "n_test_subjects": len(np.unique(tulip_subjects))
        })

    # TULIP LOSO (Leave-One-Subject-Out on TULIP Clinical alone)
    logo = LeaveOneGroupOut()
    for m_name in ["rf_baseline", "rf_regularized", "svm_rbf", "logreg_l2"]:
        tulip_loso_records = []
        for tr_idx, te_idx in logo.split(X_tulip, y_tulip, tulip_subjects):
            clf = build_classifier(m_name, seed=42)
            X_tr = scaler.fit_transform(X_tulip[tr_idx]) if m_name in ["svm_rbf", "logreg_l2"] else X_tulip[tr_idx]
            X_te = scaler.transform(X_tulip[te_idx]) if m_name in ["svm_rbf", "logreg_l2"] else X_tulip[te_idx]
            clf.fit(X_tr, y_tulip[tr_idx])
            prob = clf.predict_proba(X_te)[:, 1]
            pred = (prob >= 0.5).astype(int)
            tulip_loso_records.append({
                "subject_id": tulip_subjects[te_idx][0],
                "y_true": y_tulip[te_idx][0],
                "y_pred": pred[0],
                "y_prob": float(np.mean(prob))
            })
        loso_df = pd.DataFrame(tulip_loso_records)
        m = compute_metrics(loso_df["y_true"].values, loso_df["y_pred"].values, loso_df["y_prob"].values)
        source_eval_records.append({
            "test_type": "TULIP_Clinical_LOSO_Within_Domain",
            "model": m_name,
            "feature_config": "full_288",
            "recording_accuracy": m["accuracy"],
            "recording_sensitivity": m["sensitivity"],
            "recording_specificity": m["specificity"],
            "recording_f1": m["f1_score"],
            "recording_auc": m["roc_auc"],
            "recording_collapse": m["collapse_flag"],
            "participant_accuracy": m["accuracy"],
            "participant_sensitivity": m["sensitivity"],
            "participant_specificity": m["specificity"],
            "participant_f1": m["f1_score"],
            "participant_auc": m["roc_auc"],
            "participant_collapse": m["collapse_flag"],
            "n_test_samples": len(y_tulip),
            "n_test_subjects": len(np.unique(tulip_subjects))
        })

    # Combine all fold and participant frames
    all_folds_df = pd.concat([dne_fold_df, b_fold_df, c_fold_df], ignore_index=True)
    all_part_df = pd.concat([dne_part_df, b_part_df, c_part_df], ignore_index=True)
    source_eval_df = pd.DataFrame(source_eval_records)

    # Save detailed fold-level results
    all_folds_df.to_csv(OUTPUT_DATA_DIR / "subject_grouped_cv_results.csv", index=False)
    all_folds_df.to_csv(OUTPUT_REPORTS_DIR / "subject_grouped_cv_results.csv", index=False)

    source_eval_df.to_csv(OUTPUT_DATA_DIR / "source_specific_evaluation.csv", index=False)
    source_eval_df.to_csv(OUTPUT_REPORTS_DIR / "source_specific_evaluation.csv", index=False)

    # Build model comparison summary (aggregate mean and std across folds and seeds)
    summary_grouped = all_part_df.groupby(["track", "model", "feature_config", "n_features"]).agg({
        "accuracy": ["mean", "std"],
        "sensitivity": ["mean", "std"],
        "specificity": ["mean", "std"],
        "f1_score": ["mean", "std"],
        "roc_auc": ["mean", "std"],
        "brier_score": ["mean", "std"],
        "n_subjects": "first"
    }).reset_index()

    summary_grouped.columns = [
        "track", "model", "feature_config", "n_features",
        "acc_mean", "acc_std",
        "sens_mean", "sens_std",
        "spec_mean", "spec_std",
        "f1_mean", "f1_std",
        "auc_mean", "auc_std",
        "brier_mean", "brier_std",
        "n_subjects"
    ]

    summary_grouped.to_csv(OUTPUT_DATA_DIR / "model_comparison.csv", index=False)
    summary_grouped.to_csv(OUTPUT_REPORTS_DIR / "model_comparison.csv", index=False)

    # Extract Baseline Results table (Track A vs Legacy 290 vs Full 288)
    baseline_sub = summary_grouped[summary_grouped["track"] == "Track_A_DNE_Baseline"].sort_values("auc_mean", ascending=False)
    baseline_sub.to_csv(OUTPUT_DATA_DIR / "baseline_results.csv", index=False)
    baseline_sub.to_csv(OUTPUT_REPORTS_DIR / "baseline_results.csv", index=False)

    # Extract Feature Selection Results table
    fs_sub = summary_grouped[summary_grouped["model"].isin(["rf_regularized", "svm_rbf"])].sort_values("auc_mean", ascending=False)
    fs_sub.to_csv(OUTPUT_DATA_DIR / "feature_selection_results.csv", index=False)
    fs_sub.to_csv(OUTPUT_REPORTS_DIR / "feature_selection_results.csv", index=False)

    # Save Selector Stability JSON
    with open(OUTPUT_DATA_DIR / "feature_stability.json", "w") as f:
        json.dump(dne_stab, f, indent=2)
    with open(OUTPUT_REPORTS_DIR / "feature_stability.json", "w") as f:
        json.dump(dne_stab, f, indent=2)

    print("\nAll experiments completed successfully!")
    print(f"Total experiment rows evaluated: {len(all_folds_df)}")

    # =========================================================
    # Select Best Candidate and Train Final Model
    # =========================================================
    print("\nSelecting final candidate model...")
    # Criteria:
    # 1. Evaluated on Track B (DNE + TULIP Clinical expanded cohort)
    # 2. Balanced sensitivity (>0.85) and specificity (>0.85)
    # 3. Compact informative feature set (e.g. top 20 or top 30)
    # 4. Low variance and high ROC-AUC
    candidates = summary_grouped[summary_grouped["track"] == "Track_B_DNE_Plus_TULIP_Clinical"].copy()
    candidates = candidates[candidates["n_features"].isin([10, 20, 30, 50])]
    candidates["balance_score"] = candidates["auc_mean"] - np.abs(candidates["sens_mean"] - candidates["spec_mean"])
    best_row = candidates.sort_values(["balance_score", "auc_mean"], ascending=False).iloc[0]

    selected_model_name = best_row["model"]
    selected_feat_config = best_row["feature_config"]
    selected_k = int(best_row["n_features"])

    print(f"Selected Best Candidate:")
    print(f"  Model: {selected_model_name}")
    print(f"  Feature Set: {selected_feat_config} (k={selected_k})")
    print(f"  Track B CV Participant Accuracy: {best_row['acc_mean']:.4f} +/- {best_row['acc_std']:.4f}")
    print(f"  Track B CV Participant AUC:      {best_row['auc_mean']:.4f} +/- {best_row['auc_std']:.4f}")
    print(f"  Track B CV Sensitivity:          {best_row['sens_mean']:.4f} +/- {best_row['sens_std']:.4f}")
    print(f"  Track B CV Specificity:          {best_row['spec_mean']:.4f} +/- {best_row['spec_std']:.4f}")

    # Train final candidate model on all eligible development data (365 samples)
    print("\nTraining final experimental candidate model on all eligible development samples...")
    X_dev_pure = primary_expanded[pure_pose_cols]
    y_dev = primary_expanded["label"].values

    final_selectors = fit_feature_selectors(X_dev_pure, y_dev, seed=42)

    # BUG FIX: Extract the correct selector from the winning feature_config name.
    # Previously this always used "corr_rf_importance" regardless of which selector
    # won the selection process (e.g. mutual_info_top50 would still use RF features).
    # Parse selected_feat_config (e.g. "mutual_info_top50", "corr_rf_top30", "l1_logistic_top20")
    # to determine which selector's ranked features to use.
    _SELECTOR_NAME_MAP = {
        "corr_rf": "corr_rf_importance",
        "l1_logistic": "l1_logistic",
        "mutual_info": "mutual_info",
    }
    _resolved_selector = None
    for prefix, selector_key in _SELECTOR_NAME_MAP.items():
        if selected_feat_config.startswith(prefix + "_top"):
            _resolved_selector = selector_key
            break
    if _resolved_selector is None:
        # Fallback for full_288 or legacy_290 configs: use primary selector
        print(f"  WARNING: Could not parse selector from '{selected_feat_config}'; "
              f"falling back to corr_rf_importance")
        _resolved_selector = "corr_rf_importance"
    else:
        print(f"  Resolved selector: '{_resolved_selector}' from feature_config '{selected_feat_config}'")

    if _resolved_selector not in final_selectors:
        raise ValueError(
            f"Selector '{_resolved_selector}' not found in computed selectors: "
            f"{list(final_selectors.keys())}"
        )
    selected_feature_names = final_selectors[_resolved_selector][:selected_k]

    if len(selected_feature_names) < selected_k:
        print(f"  WARNING: Selector '{_resolved_selector}' produced only {len(selected_feature_names)} "
              f"features, requested {selected_k}")

    _SELECTOR_DESCRIPTIONS = {
        "corr_rf_importance": "Random Forest feature importance with cross-correlation filtering (r < 0.85)",
        "l1_logistic": "L1-regularized Logistic Regression (LASSO) coefficient magnitudes",
        "mutual_info": "Mutual Information classification relevance",
    }
    _selection_method_desc = _SELECTOR_DESCRIPTIONS.get(
        _resolved_selector,
        f"Selector: {_resolved_selector}"
    )

    X_dev_selected = X_dev_pure[selected_feature_names]
    scaler_final = StandardScaler()
    X_dev_scaled = scaler_final.fit_transform(X_dev_selected)

    final_clf = build_classifier(selected_model_name, seed=42)
    if selected_model_name in ["svm_rbf", "logreg_l2"]:
        final_clf.fit(X_dev_scaled, y_dev)
    else:
        final_clf.fit(X_dev_selected.values, y_dev)

    n_dev_rows = len(primary_expanded)
    n_dev_subjects = primary_expanded["subject_id"].nunique()
    n_dne_rows = len(dne_data)
    n_tulip_rows = len(tulip_clinical_eligible)

    # Save final candidate model artifact
    candidate_bundle = {
        "model": final_clf,
        "scaler": scaler_final if selected_model_name in ["svm_rbf", "logreg_l2"] else None,
        "model_name": selected_model_name,
        "feature_names": selected_feature_names,
        "feature_count": len(selected_feature_names),
        "selector_used": _resolved_selector,
        "selection_method": _selection_method_desc,
        "training_cohort": f"DNE ({n_dne_rows}) + TULIP Clinical ({n_tulip_rows}) = {n_dev_rows} recordings across {n_dev_subjects} independent subjects",
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cv_metrics": {
            "accuracy_mean": float(best_row["acc_mean"]),
            "accuracy_std": float(best_row["acc_std"]),
            "auc_mean": float(best_row["auc_mean"]),
            "auc_std": float(best_row["auc_std"]),
            "sensitivity_mean": float(best_row["sens_mean"]),
            "specificity_mean": float(best_row["spec_mean"])
        },
        "bug_fix_note": "Selector mismatch bug fixed: features now come from the actual winning selector, not always corr_rf_importance."
    }

    candidate_model_path = MODELS_DIR / "pd_detector_phase4_candidate.pkl"
    joblib.dump(candidate_bundle, candidate_model_path)
    print(f"Saved final candidate model to: {candidate_model_path}")

    # Save selected features JSON
    feature_meta = {
        "candidate_model": selected_model_name,
        "feature_config": selected_feat_config,
        "n_features": len(selected_feature_names),
        "selected_features": selected_feature_names,
        "selector_used": _resolved_selector,
        "selection_method": _selection_method_desc,
        "source_data": f"All {n_dev_rows} development samples (DNE + eligible TULIP clinical)",
        "feature_semantics": [
            {
                "index": idx,
                "feature_name": feat,
                "channel": int(feat.split("_")[0].replace("pose", "")),
                "statistic": feat.split("_")[1]
            } for idx, feat in enumerate(selected_feature_names)
        ]
    }

    with open(MODELS_DIR / "phase4_selected_features.json", "w") as f:
        json.dump(feature_meta, f, indent=2)
    with open(OUTPUT_DATA_DIR / "selected_features.json", "w") as f:
        json.dump(feature_meta, f, indent=2)
    with open(OUTPUT_REPORTS_DIR / "selected_features.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    print(f"Saved selected features JSON to: {MODELS_DIR / 'phase4_selected_features.json'}")

if __name__ == "__main__":
    run_all_phase4_experiments()
