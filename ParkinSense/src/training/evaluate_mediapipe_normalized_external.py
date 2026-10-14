"""
ParkinSense — Final Locked TULIP External Evaluation of Normalized Random Forest
================================================================================
Evaluates the NEW normalized Random Forest model (`models/pd_detector_mediapipe_normalized.pkl`)
on the locked TULIP benchmark dataset (44 videos across 8 subjects).

Generates:
  - `data/processed/TULIP/final_normalized_external_subject_results.csv`
  - `data/processed/TULIP/final_normalized_external_video_results.csv`
  - `docs/final_normalized_external_evaluation.md`
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "pd_detector_mediapipe_normalized.pkl"
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "pd_detector_baseline.pkl"

SPATIAL_FEATS_PATH = PROJECT_ROOT / "data" / "processed" / "TULIP" / "tulip_normalized_spatial_features.csv"
SPATIAL_META_PATH = PROJECT_ROOT / "data" / "processed" / "TULIP" / "tulip_normalized_spatial_metadata.csv"

OUT_SUBJ_CSV = PROJECT_ROOT / "data" / "processed" / "TULIP" / "final_normalized_external_subject_results.csv"
OUT_VID_CSV = PROJECT_ROOT / "data" / "processed" / "TULIP" / "final_normalized_external_video_results.csv"
OUT_REPORT_MD = PROJECT_ROOT / "docs" / "final_normalized_external_evaluation.md"

def evaluate_external():
    print(f"Loading new normalized model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    baseline_model = joblib.load(BASELINE_MODEL_PATH)

    df_feats = pd.read_csv(SPATIAL_FEATS_PATH)
    df_meta = pd.read_csv(SPATIAL_META_PATH)

    feature_cols = list(model.feature_names_in_)
    df_feats = df_feats[feature_cols]

    # Video-level Predictions with NEW Normalized RF
    new_probs = model.predict_proba(df_feats)[:, 1]
    new_preds = (new_probs >= 0.50).astype(int)

    df_vid = df_meta.copy()
    df_vid["new_pd_probability"] = np.round(new_probs, 4)
    df_vid["new_pred_label"] = new_preds
    df_vid["new_correct_pred"] = (new_preds == df_vid["true_label"])

    df_vid.to_csv(OUT_VID_CSV, index=False)
    print(f"Saved video-level evaluation to: {OUT_VID_CSV}")

    # Subject-level Aggregation (Mean P(PD) across available cameras)
    subj_rows = []
    for subject_id, group in df_vid.groupby("subject_id"):
        true_label = group["true_label"].iloc[0]
        category = group["category"].iloc[0]
        mean_prob = float(group["new_pd_probability"].mean())
        subj_pred = 1 if mean_prob >= 0.50 else 0

        subj_rows.append({
            "subject_id": subject_id,
            "category": category,
            "true_label": true_label,
            "num_cameras": len(group),
            "mean_pd_probability": round(mean_prob, 4),
            "predicted_label": subj_pred,
            "correct_pred": (subj_pred == true_label),
        })

    df_subj = pd.DataFrame(subj_rows).sort_values(["category", "subject_id"])
    df_subj.to_csv(OUT_SUBJ_CSV, index=False)
    print(f"Saved subject-level evaluation to: {OUT_SUBJ_CSV}")

    # Compute Subject Metrics for NEW Normalized RF
    y_true_sub = df_subj["true_label"].values
    y_pred_sub = df_subj["predicted_label"].values
    y_prob_sub = df_subj["mean_pd_probability"].values

    acc_new = accuracy_score(y_true_sub, y_pred_sub)
    prec_new = precision_score(y_true_sub, y_pred_sub, zero_division=0)
    rec_new = recall_score(y_true_sub, y_pred_sub, zero_division=0)
    spec_new = recall_score(y_true_sub == 0, y_pred_sub == 0, zero_division=0)
    f1_new = f1_score(y_true_sub, y_pred_sub, zero_division=0)
    try:
        auc_new = roc_auc_score(y_true_sub, y_prob_sub)
    except Exception:
        auc_new = np.nan
    cm_new = confusion_matrix(y_true_sub, y_pred_sub, labels=[0, 1])

    # Compute Video Metrics for NEW Normalized RF
    y_true_vid = df_vid["true_label"].values
    y_pred_vid = df_vid["new_pred_label"].values
    y_prob_vid = df_vid["new_pd_probability"].values

    acc_vid = accuracy_score(y_true_vid, y_pred_vid)
    prec_vid = precision_score(y_true_vid, y_pred_vid, zero_division=0)
    rec_vid = recall_score(y_true_vid, y_pred_vid, zero_division=0)
    spec_vid = recall_score(y_true_vid == 0, y_pred_vid == 0, zero_division=0)
    f1_vid = f1_score(y_true_vid, y_pred_vid, zero_division=0)
    try:
        auc_vid = roc_auc_score(y_true_vid, y_prob_vid)
    except Exception:
        auc_vid = np.nan

    # Generate Report MD
    report_md = f"""# Final Normalized MediaPipe Model External TULIP Evaluation

## 1. Executive Summary
This document presents the final locked external evaluation of the newly trained normalized Random Forest model (`pd_detector_mediapipe_normalized.pkl`) on the 8-subject TULIP benchmark dataset (44 videos).

---

## 2. Subject-Level External Evaluation Metrics

* **Accuracy**: **{acc_new * 100:.2f}%**
* **Precision**: **{prec_new * 100:.2f}%**
* **Sensitivity / Recall**: **{rec_new * 100:.2f}%**
* **Specificity**: **{spec_new * 100:.2f}%**
* **F1-Score**: **{f1_new:.4f}**
* **ROC-AUC**: **{auc_new:.4f}**
* **Confusion Matrix [TN, FP, FN, TP]**: [{cm_new[0,0]}, {cm_new[0,1]}, {cm_new[1,0]}, {cm_new[1,1]}]

---

## 3. Video-Level External Evaluation Metrics

* **Video Accuracy**: **{acc_vid * 100:.2f}%**
* **Video Precision**: **{prec_vid * 100:.2f}%**
* **Video Sensitivity**: **{rec_vid * 100:.2f}%**
* **Video Specificity**: **{spec_vid * 100:.2f}%**
* **Video F1-Score**: **{f1_vid:.4f}**
* **Video ROC-AUC**: **{auc_vid:.4f}**

---

## 4. Subject-by-Subject Prediction Breakdown

| Subject ID | Category | Ground Truth | Num Cameras | Mean $P(\\text{{PD}})$ | Predicted Label | Correct? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for _, row in df_subj.iterrows():
        report_md += f"| **{row['subject_id']}** | {row['category']} | {row['true_label']} | {row['num_cameras']} | {row['mean_pd_probability']:.4f} | {row['predicted_label']} ({'PD' if row['predicted_label']==1 else 'Healthy'}) | {'✅ True' if row['correct_pred'] else '❌ False'} |\n"

    report_md += f"""
---

## 5. Controlled Three-Way Model Architecture Comparison

| Experiment | Accuracy | Sensitivity | Specificity | Precision | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Baseline (Old RF + Unnormalized)** | **37.50%** | **0.00%** | **100.00%** | **0.00%** | **0.0000** | **0.8333** |
| **Old RF + Normalized Features** | **37.50%** | **0.00%** | **100.00%** | **0.00%** | **0.0000** | **0.5333** |
| **New Normalized RF + Normalized Features** | **{acc_new * 100:.2f}%** | **{rec_new * 100:.2f}%** | **{spec_new * 100:.2f}%** | **{prec_new * 100:.2f}%** | **{f1_new:.4f}** | **{auc_new:.4f}** |
"""

    OUT_REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT_MD, "w") as f:
        f.write(report_md)
    print(f"Saved external evaluation report to: {OUT_REPORT_MD}")

if __name__ == "__main__":
    evaluate_external()
