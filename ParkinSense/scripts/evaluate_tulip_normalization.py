"""
ParkinSense — Controlled Normalization Experiments & TULIP Evaluation
======================================================================
Runs Experiments A (Baseline), B (Spatial Normalization), and C (Spatiotemporal Normalization)
across all 44 TULIP videos for the 8 reserved subjects using the baseline model `pd_detector_baseline.pkl`.

Optimized with single-pass MediaPipe landmark caching for high execution speed.
"""

from __future__ import annotations

import os
import re
import sys
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.mediapipe_predict import (
    extract_landmarks_from_video,
    build_smpl_dataframe,
)
from src.features.smpl_feature_extractor import SMPLFeatureExtractor
from src.pose.mediapipe_normalization import (
    build_normalized_smpl_dataframe,
    extract_normalized_features,
    DEFAULT_NORMALIZATION_CONFIG,
)
from src.utils import get_logger, ensure_dir
from scripts.audit_feature_domain import compute_domain_shift_metrics

logger = get_logger(__name__)


def extract_camera_id(filename: str) -> str:
    """Extract camera identifier from filename."""
    match = re.search(r"cam(?:era|rea)?\s*(\d+)", filename, re.IGNORECASE)
    if match:
        return f"Camera {match.group(1)}"
    return "Camera Unknown"


def extract_video_data(rec: dict) -> dict:
    """Worker function for single-pass MediaPipe video landmark extraction."""
    vpath = Path(rec["video_path"])
    t0 = time.time()
    try:
        (world_coords, img_coords, vis, f_idx, fps, total_video_frames) = extract_landmarks_from_video(vpath)
        elapsed = time.time() - t0
        return {
            "rec": rec,
            "world_coords": world_coords,
            "visibility": vis,
            "fps": fps,
            "total_video_frames": total_video_frames,
            "status": "SUCCESS",
            "extraction_sec": round(elapsed, 2),
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "rec": rec,
            "status": "FAILED",
            "error": str(e),
            "extraction_sec": round(elapsed, 2),
        }


def evaluate_extracted_video(extracted_data: dict, exp_mode: str, model_path_str: str) -> dict:
    """Evaluate pre-extracted video landmark data under experiment configuration."""
    rec = extracted_data["rec"]
    if extracted_data["status"] != "SUCCESS":
        return {
            "exp_mode": exp_mode,
            "subject_id": rec["subject_id"],
            "category": rec["category"],
            "true_label": rec["true_label"],
            "camera_id": rec["camera_id"],
            "video_name": rec["video_name"],
            "video_path": rec["video_path"],
            "pred_label": np.nan,
            "pd_probability": np.nan,
            "status": "FAILED",
            "error": extracted_data.get("error", "Extraction failed"),
            "features": None,
        }

    model = joblib.load(model_path_str)
    feature_cols = list(model.feature_names_in_)
    world_coords = extracted_data["world_coords"]
    vis = extracted_data["visibility"]
    fps = extracted_data["fps"]

    try:
        if exp_mode == "BASELINE":
            # Baseline pipeline (unnormalized)
            smpl_df = build_smpl_dataframe(world_coords, vis)
            extractor = SMPLFeatureExtractor()
            feat_df = extractor.extract_features(smpl_df, fps=fps)

        elif exp_mode == "SPATIAL":
            # Spatial Normalization pipeline (Pelvis centering + scale + canonical frame)
            config = {**DEFAULT_NORMALIZATION_CONFIG, "temporal_normalization": False, "gait_cycle_segmentation": False}
            smpl_df, norm_stats = build_normalized_smpl_dataframe(world_coords, vis, config=config)
            feat_df = extract_normalized_features(smpl_df, fps=fps, config=config)

        elif exp_mode == "SPATIOTEMPORAL":
            # Spatiotemporal Normalization pipeline
            config = {**DEFAULT_NORMALIZATION_CONFIG, "temporal_normalization": True, "gait_cycle_segmentation": True}
            smpl_df, norm_stats = build_normalized_smpl_dataframe(world_coords, vis, config=config)
            feat_df = extract_normalized_features(smpl_df, fps=fps, config=config)
        else:
            raise ValueError(f"Unknown experiment mode: {exp_mode}")

        feat_df = feat_df[feature_cols]

        pred_label = int(model.predict(feat_df)[0])
        probs = model.predict_proba(feat_df)[0]
        pd_probability = float(probs[1]) if len(probs) > 1 else float(pred_label)

        return {
            "exp_mode": exp_mode,
            "subject_id": rec["subject_id"],
            "category": rec["category"],
            "true_label": rec["true_label"],
            "camera_id": rec["camera_id"],
            "video_name": rec["video_name"],
            "video_path": rec["video_path"],
            "pred_label": pred_label,
            "pd_probability": round(pd_probability, 4),
            "status": "SUCCESS",
            "features": feat_df.iloc[0].to_dict(),
        }

    except Exception as e:
        return {
            "exp_mode": exp_mode,
            "subject_id": rec["subject_id"],
            "category": rec["category"],
            "true_label": rec["true_label"],
            "camera_id": rec["camera_id"],
            "video_name": rec["video_name"],
            "video_path": rec["video_path"],
            "pred_label": np.nan,
            "pd_probability": np.nan,
            "status": "FAILED",
            "error": str(e),
            "features": None,
        }


def run_experiment_on_cached(exp_mode: str, cached_data: list[dict], model_path_str: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """Execute evaluation instantly across all cached video landmarks."""
    results = [evaluate_extracted_video(item, exp_mode, model_path_str) for item in cached_data]
    results.sort(key=lambda x: (x["category"], x["subject_id"], x["camera_id"]))

    video_rows = []
    feature_rows = []
    for r in results:
        v_dict = {k: v for k, v in r.items() if k != "features"}
        video_rows.append(v_dict)
        if r["features"] is not None:
            feature_rows.append(r["features"])

    df_video = pd.DataFrame(video_rows)
    df_features = pd.DataFrame(feature_rows)

    df_valid = df_video[df_video["status"] == "SUCCESS"].copy()
    subject_rows = []

    for subject_id, group in df_valid.groupby("subject_id"):
        true_label = group["true_label"].iloc[0]
        category = group["category"].iloc[0]
        mean_prob = float(group["pd_probability"].mean())
        subj_pred = 1 if mean_prob >= 0.50 else 0

        subject_rows.append({
            "exp_mode": exp_mode,
            "subject_id": subject_id,
            "category": category,
            "true_label": true_label,
            "num_cameras": len(group),
            "mean_pd_probability": round(mean_prob, 4),
            "subject_pred_label": subj_pred,
            "correct_pred": (subj_pred == true_label),
        })

    df_subject = pd.DataFrame(subject_rows).sort_values(["category", "subject_id"])

    sub_y_true = df_subject["true_label"].values
    sub_y_pred = df_subject["subject_pred_label"].values
    sub_y_prob = df_subject["mean_pd_probability"].values

    acc = accuracy_score(sub_y_true, sub_y_pred)
    prec = precision_score(sub_y_true, sub_y_pred, zero_division=0)
    rec = recall_score(sub_y_true, sub_y_pred, zero_division=0)
    spec = recall_score(sub_y_true == 0, sub_y_pred == 0, zero_division=0)
    f1 = f1_score(sub_y_true, sub_y_pred, zero_division=0)
    try:
        auc = roc_auc_score(sub_y_true, sub_y_prob)
    except Exception:
        auc = np.nan
    cm = confusion_matrix(sub_y_true, sub_y_pred, labels=[0, 1])

    ht_mean_prob = float(df_subject[df_subject["true_label"] == 0]["mean_pd_probability"].mean())
    pd_mean_prob = float(df_subject[df_subject["true_label"] == 1]["mean_pd_probability"].mean())

    summary = {
        "exp_mode": exp_mode,
        "total_videos": len(df_video),
        "successful_videos": len(df_valid),
        "total_subjects": len(df_subject),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4) if not np.isnan(auc) else np.nan,
        "ht_mean_pd_prob": round(ht_mean_prob, 4),
        "pd_mean_pd_prob": round(pd_mean_prob, 4),
        "prob_delta": round(pd_mean_prob - ht_mean_prob, 4),
        "cm_tn": int(cm[0, 0]),
        "cm_fp": int(cm[0, 1]),
        "cm_fn": int(cm[1, 0]),
        "cm_tp": int(cm[1, 1]),
    }

    return df_video, df_subject, summary, df_features


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join([header, separator] + rows)


def main():
    tulip_dir = PROJECT_ROOT / "data" / "raw" / "TULIP" / "gait"
    output_dir = PROJECT_ROOT / "data" / "processed" / "TULIP"
    ensure_dir(output_dir)

    model_path = PROJECT_ROOT / "models" / "pd_detector_baseline.pkl"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "models" / "pd_detector.pkl"

    video_records = []
    for category in ["PD", "HT"]:
        cat_dir = tulip_dir / category
        if not cat_dir.exists():
            continue
        ground_truth = 1 if category == "PD" else 0
        for subject_dir in sorted(cat_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            subject_id = subject_dir.name
            for video_file in sorted(subject_dir.glob("*.mp4")):
                cam_id = extract_camera_id(video_file.name)
                video_records.append({
                    "subject_id": subject_id,
                    "category": category,
                    "true_label": ground_truth,
                    "camera_id": cam_id,
                    "video_name": video_file.name,
                    "video_path": str(video_file),
                })

    print(f"Loaded {len(video_records)} TULIP video records across 8 subjects.")
    max_workers = min(4, os.cpu_count() or 4)

    # Single-pass landmark extraction
    print(f"Extracting MediaPipe Pose landmarks for {len(video_records)} videos using {max_workers} processes...")
    t_start = time.time()
    cached_data = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_video_data, rec): rec for rec in video_records}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            cached_data.append(res)
            print(f"[{completed}/{len(video_records)}] Extracted landmarks: {res['rec']['category']}/{res['rec']['subject_id']}/{res['rec']['video_name']} ({res['status']})")

    extract_time = time.time() - t_start
    print(f"Single-pass landmark extraction finished in {round(extract_time, 2)} seconds.")

    # Load Training Dataset for Feature Domain Shift Audit
    train_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "pd_dataset.csv")
    model = joblib.load(model_path)
    feature_cols = list(model.feature_names_in_)
    df_train_feats = train_df[feature_cols]

    # Save Locked Test Manifest
    manifest_df = pd.DataFrame(video_records)[["subject_id", "category", "camera_id", "video_name", "video_path"]]
    manifest_df.to_csv(output_dir / "tulip_locked_test_manifest.csv", index=False)
    print(f"Saved locked test manifest to: {output_dir / 'tulip_locked_test_manifest.csv'}")

    # Run EXPERIMENT A (Baseline)
    df_vid_a, df_sub_a, summary_a, df_feats_a = run_experiment_on_cached("BASELINE", cached_data, str(model_path))
    df_vid_a.to_csv(output_dir / "normalization_baseline.csv", index=False)

    # Run EXPERIMENT B (Spatial Normalization)
    df_vid_b, df_sub_b, summary_b, df_feats_b = run_experiment_on_cached("SPATIAL", cached_data, str(model_path))
    df_vid_b.to_csv(output_dir / "normalization_spatial.csv", index=False)
    df_vid_b.to_csv(output_dir / "tulip_normalized_spatial_metadata.csv", index=False)
    df_feats_b[feature_cols].to_csv(output_dir / "tulip_normalized_spatial_features.csv", index=False)
    print(f"Saved Spatial normalized features shape: {df_feats_b[feature_cols].shape}")

    # Run EXPERIMENT C (Spatiotemporal Normalization)
    df_vid_c, df_sub_c, summary_c, df_feats_c = run_experiment_on_cached("SPATIOTEMPORAL", cached_data, str(model_path))
    df_vid_c.to_csv(output_dir / "normalization_spatiotemporal.csv", index=False)
    df_vid_c.to_csv(output_dir / "tulip_normalized_spatiotemporal_metadata.csv", index=False)
    df_feats_c[feature_cols].to_csv(output_dir / "tulip_normalized_spatiotemporal_features.csv", index=False)
    print(f"Saved Spatiotemporal normalized features shape: {df_feats_c[feature_cols].shape}")

    # Feature Domain Audit (Before vs After)
    audit_before_df, audit_before_summary = compute_domain_shift_metrics(df_train_feats, df_feats_a, feature_cols)
    audit_before_df.to_csv(output_dir / "feature_domain_audit_before.csv", index=False)

    audit_after_df, audit_after_summary = compute_domain_shift_metrics(df_train_feats, df_feats_b, feature_cols)
    audit_after_df.to_csv(output_dir / "feature_domain_audit_after.csv", index=False)

    # Write Formal Markdown Comparison Report
    report_content = f"""# ParkinSense — Controlled Normalization Experiments Report

## 1. Overview
This experiment evaluates whether spatial and spatiotemporal domain normalization improves the MediaPipe gait classification pipeline on the unseen 8-subject TULIP benchmark dataset. All experiments use the unchanged baseline model (`pd_detector_baseline.pkl`) at threshold 0.50 without retraining or tuning.

---

## 2. Statistical Comparison Across Experiments

| Metric | Baseline (Exp A) | Spatial Normalization (Exp B) | Spatiotemporal Normalization (Exp C) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **{summary_a['accuracy'] * 100:.2f}%** | **{summary_b['accuracy'] * 100:.2f}%** | **{summary_c['accuracy'] * 100:.2f}%** |
| **Sensitivity / Recall** | **{summary_a['recall'] * 100:.2f}%** | **{summary_b['recall'] * 100:.2f}%** | **{summary_c['recall'] * 100:.2f}%** |
| **Specificity** | **{summary_a['specificity'] * 100:.2f}%** | **{summary_b['specificity'] * 100:.2f}%** | **{summary_c['specificity'] * 100:.2f}%** |
| **Precision** | **{summary_a['precision'] * 100:.2f}%** | **{summary_b['precision'] * 100:.2f}%** | **{summary_c['precision'] * 100:.2f}%** |
| **F1-Score** | **{summary_a['f1_score']:.4f}** | **{summary_b['f1_score']:.4f}** | **{summary_c['f1_score']:.4f}** |
| **ROC-AUC** | **{summary_a['roc_auc']:.4f}** | **{summary_b['roc_auc']:.4f}** | **{summary_c['roc_auc']:.4f}** |
| **HT Mean $P(\\text{{PD}})$** | **{summary_a['ht_mean_pd_prob']:.4f}** | **{summary_b['ht_mean_pd_prob']:.4f}** | **{summary_c['ht_mean_pd_prob']:.4f}** |
| **PD Mean $P(\\text{{PD}})$** | **{summary_a['pd_mean_pd_prob']:.4f}** | **{summary_b['pd_mean_pd_prob']:.4f}** | **{summary_c['pd_mean_pd_prob']:.4f}** |
| **Probability Delta ($P_{{\\text{{PD}}}} - P_{{\\text{{HT}}}}$)** | **{summary_a['prob_delta']:.4f}** | **{summary_b['prob_delta']:.4f}** | **{summary_c['prob_delta']:.4f}** |
| **Confusion Matrix [TN, FP, FN, TP]** | [{summary_a['cm_tn']}, {summary_a['cm_fp']}, {summary_a['cm_fn']}, {summary_a['cm_tp']}] | [{summary_b['cm_tn']}, {summary_b['cm_fp']}, {summary_b['cm_fn']}, {summary_b['cm_tp']}] | [{summary_c['cm_tn']}, {summary_c['cm_fp']}, {summary_c['cm_fn']}, {summary_c['cm_tp']}] |

---

## 3. Feature Domain Shift Statistics (Training vs TULIP)

| Feature Shift Metric | Before Normalization (Exp A) | After Spatial Normalization (Exp B) | Reduction / Change |
| :--- | :---: | :---: | :---: |
| **Median $D_j$** | **{audit_before_summary['median_D']:.4f}** | **{audit_after_summary['median_D']:.4f}** | **{audit_before_summary['median_D'] - audit_after_summary['median_D']:.4f} lower shift** |
| **90th Percentile $D_j$** | **{audit_before_summary['p90_D']:.4f}** | **{audit_after_summary['p90_D']:.4f}** | **{audit_before_summary['p90_D'] - audit_after_summary['p90_D']:.4f} lower shift** |
| **Features with $D_j > 2.0$** | **{audit_before_summary['count_D_gt_2']} / 290** | **{audit_after_summary['count_D_gt_2']} / 290** | **-{audit_before_summary['count_D_gt_2'] - audit_after_summary['count_D_gt_2']} features** |
| **Features with $D_j > 3.0$** | **{audit_before_summary['count_D_gt_3']} / 290** | **{audit_after_summary['count_D_gt_3']} / 290** | **-{audit_before_summary['count_D_gt_3'] - audit_after_summary['count_D_gt_3']} features** |
| **Features with $R_j > 5.0$** | **{audit_before_summary['count_R_gt_5']} / 290** | **{audit_after_summary['count_R_gt_5']} / 290** | **-{audit_before_summary['count_R_gt_5'] - audit_after_summary['count_R_gt_5']} features** |

---

## 4. Subject-Level Probability Breakdown Across Experiments

### Experiment A (Baseline)
{df_to_markdown(df_sub_a[['subject_id', 'category', 'true_label', 'mean_pd_probability', 'subject_pred_label', 'correct_pred']])}

### Experiment B (Spatial Normalization)
{df_to_markdown(df_sub_b[['subject_id', 'category', 'true_label', 'mean_pd_probability', 'subject_pred_label', 'correct_pred']])}

### Experiment C (Spatiotemporal Normalization)
{df_to_markdown(df_sub_c[['subject_id', 'category', 'true_label', 'mean_pd_probability', 'subject_pred_label', 'correct_pred']])}
"""

    report_path = output_dir / "normalization_experiment_report.md"
    with open(report_path, "w") as f:
        f.write(report_content)

    print(f"\nSaved normalization experiment report to: {report_path}")
    print("\nSummary Results:")
    print(f"  Exp A (Baseline)           : Acc = {summary_a['accuracy']*100:.1f}%, AUC = {summary_a['roc_auc']:.4f}")
    print(f"  Exp B (Spatial Norm)       : Acc = {summary_b['accuracy']*100:.1f}%, AUC = {summary_b['roc_auc']:.4f}")
    print(f"  Exp C (Spatiotemporal Norm): Acc = {summary_c['accuracy']*100:.1f}%, AUC = {summary_c['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
