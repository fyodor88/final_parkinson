"""
ParkinSense — TULIP External Evaluation Script (MediaPipe Pipeline)
===================================================================
Evaluates the trained MediaPipe gait classification model on the TULIP external benchmark dataset.

Data hierarchy:
  data/raw/TULIP/gait/
    ├── PD/ (Parkinson's Disease, ground_truth = 1)
    │   ├── subject02/
    │   ├── subject05/
    │   ├── subject08/
    │   ├── subject11/
    │   └── subject13/
    └── HT/ (Healthy Control, ground_truth = 0)
        ├── subject01/
        ├── subject07/
        └── subject12/

Performs video-level inference, subject-level multi-view aggregation, and per-camera analysis.
Uses multiprocessing for fast parallel execution across CPU cores.
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

from src.inference.mediapipe_predict import MediaPipePipeline
from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)


def extract_camera_id(filename: str) -> str:
    """Extract camera identifier from filename, handling typos like 'camrea5' or 'subject14_camera3.mp4'."""
    match = re.search(r"cam(?:era|rea)?\s*(\d+)", filename, re.IGNORECASE)
    if match:
        return f"Camera {match.group(1)}"
    return "Camera Unknown"


def process_single_video(rec: dict) -> dict:
    """Process a single video record using MediaPipePipeline (worker function for multiprocessing)."""
    vpath = Path(rec["video_path"])
    job_id = f"tulip_{rec['subject_id']}_{rec['camera_id'].replace(' ', '_')}"
    
    t0 = time.time()
    try:
        pipeline = MediaPipePipeline()
        res = pipeline.run(vpath, job_id=job_id)
        elapsed = time.time() - t0
        
        pd_info = res.get("parkinsons_detection", {})
        sev_info = res.get("severity_estimation", {})

        return {
            "subject_id": rec["subject_id"],
            "category": rec["category"],
            "true_label": rec["true_label"],
            "camera_id": rec["camera_id"],
            "video_name": rec["video_name"],
            "video_path": rec["video_path"],
            "pred_label": pd_info.get("label", np.nan),
            "pd_probability": pd_info.get("probability_raw", np.nan),
            "risk_level": pd_info.get("risk_level", "N/A"),
            "severity_score": sev_info.get("updrs_gait_score", np.nan),
            "severity_description": sev_info.get("description", ""),
            "detected_frames": res.get("num_frames", 0),
            "total_video_frames": res.get("metadata", {}).get("total_video_frames", res.get("num_frames", 0)),
            "fps": res.get("fps", 0),
            "status": "SUCCESS",
            "inference_sec": round(elapsed, 2),
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "subject_id": rec["subject_id"],
            "category": rec["category"],
            "true_label": rec["true_label"],
            "camera_id": rec["camera_id"],
            "video_name": rec["video_name"],
            "video_path": rec["video_path"],
            "pred_label": np.nan,
            "pd_probability": np.nan,
            "risk_level": "ERROR",
            "severity_score": np.nan,
            "severity_description": str(e),
            "detected_frames": 0,
            "total_video_frames": 0,
            "fps": 0,
            "status": "FAILED",
            "inference_sec": round(elapsed, 2),
        }


def run_tulip_evaluation():
    tulip_dir = PROJECT_ROOT / "data" / "raw" / "TULIP" / "gait"
    output_dir = PROJECT_ROOT / "data" / "processed"
    ensure_dir(output_dir)

    if not tulip_dir.exists():
        raise FileNotFoundError(f"TULIP dataset directory not found at: {tulip_dir}")

    # Discover videos
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

    print(f"Discovered {len(video_records)} videos across 8 subjects.")

    video_results = []
    start_eval_time = time.time()

    # Parallel processing with ProcessPoolExecutor
    max_workers = min(4, os.cpu_count() or 4)
    print(f"Starting parallel evaluation with {max_workers} worker processes...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_rec = {executor.submit(process_single_video, rec): rec for rec in video_records}
        completed_count = 0
        
        for future in as_completed(future_to_rec):
            completed_count += 1
            res = future.result()
            video_results.append(res)
            print(f"[{completed_count}/{len(video_records)}] Processed {res['category']}/{res['subject_id']}/{res['video_name']} ({res['camera_id']}) -> Status: {res['status']} | Pred: {res['pred_label']} | Prob: {res['pd_probability']}")

    total_eval_sec = time.time() - start_eval_time

    # Sort video results for clean output
    video_results.sort(key=lambda x: (x["category"], x["subject_id"], x["camera_id"]))

    # Create Video-Level DataFrame
    df_video = pd.DataFrame(video_results)
    video_csv_path = output_dir / "tulip_video_evaluation.csv"
    df_video.to_csv(video_csv_path, index=False)
    print(f"\nSaved video-level evaluation CSV to: {video_csv_path}")

    # Compute Subject-Level Aggregation
    df_valid = df_video[df_video["status"] == "SUCCESS"].copy()

    subject_rows = []
    for subject_id, group in df_valid.groupby("subject_id"):
        true_label = group["true_label"].iloc[0]
        category = group["category"].iloc[0]
        num_cameras = len(group)
        
        mean_prob = float(group["pd_probability"].mean())
        subj_pred_mean = 1 if mean_prob >= 0.5 else 0
        
        # Majority voting
        maj_pred = int(group["pred_label"].mode()[0])
        
        # Camera level probabilities list
        cam_probs = {row["camera_id"]: round(row["pd_probability"], 4) for _, row in group.iterrows()}
        
        subject_rows.append({
            "subject_id": subject_id,
            "category": category,
            "true_label": true_label,
            "num_cameras_processed": num_cameras,
            "mean_pd_probability": round(mean_prob, 4),
            "subject_pred_label": subj_pred_mean,
            "majority_pred_label": maj_pred,
            "correct_pred": (subj_pred_mean == true_label),
            "camera_probabilities": json.dumps(cam_probs),
        })

    df_subject = pd.DataFrame(subject_rows).sort_values(["category", "subject_id"])
    subject_csv_path = output_dir / "tulip_subject_evaluation.csv"
    df_subject.to_csv(subject_csv_path, index=False)
    print(f"Saved subject-level evaluation CSV to: {subject_csv_path}")

    # Per-Camera Analysis
    camera_rows = []
    for cam_id, group in df_valid.groupby("camera_id"):
        y_true = group["true_label"].values
        y_pred = group["pred_label"].values.astype(int)
        y_prob = group["pd_probability"].values
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        camera_rows.append({
            "camera_id": cam_id,
            "num_videos": len(group),
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "mean_pd_prob_ht": round(float(group[group["true_label"] == 0]["pd_probability"].mean()), 4) if len(group[group["true_label"] == 0]) > 0 else np.nan,
            "mean_pd_prob_pd": round(float(group[group["true_label"] == 1]["pd_probability"].mean()), 4) if len(group[group["true_label"] == 1]) > 0 else np.nan,
        })

    df_camera = pd.DataFrame(camera_rows).sort_values("camera_id")
    camera_csv_path = output_dir / "tulip_camera_evaluation.csv"
    df_camera.to_csv(camera_csv_path, index=False)
    print(f"Saved per-camera evaluation CSV to: {camera_csv_path}")

    # Compute Overall Metrics
    # 1. Subject-Level Metrics
    sub_y_true = df_subject["true_label"].values
    sub_y_pred = df_subject["subject_pred_label"].values
    sub_y_prob = df_subject["mean_pd_probability"].values

    sub_acc = accuracy_score(sub_y_true, sub_y_pred)
    sub_prec = precision_score(sub_y_true, sub_y_pred, zero_division=0)
    sub_rec = recall_score(sub_y_true, sub_y_pred, zero_division=0)
    sub_f1 = f1_score(sub_y_true, sub_y_pred, zero_division=0)
    try:
        sub_auc = roc_auc_score(sub_y_true, sub_y_prob)
    except Exception:
        sub_auc = np.nan
    sub_cm = confusion_matrix(sub_y_true, sub_y_pred, labels=[0, 1])

    # 2. Video-Level Metrics
    vid_y_true = df_valid["true_label"].values
    vid_y_pred = df_valid["pred_label"].values.astype(int)
    vid_y_prob = df_valid["pd_probability"].values

    vid_acc = accuracy_score(vid_y_true, vid_y_pred)
    vid_prec = precision_score(vid_y_true, vid_y_pred, zero_division=0)
    vid_rec = recall_score(vid_y_true, vid_y_pred, zero_division=0)
    vid_f1 = f1_score(vid_y_true, vid_y_pred, zero_division=0)
    try:
        vid_auc = roc_auc_score(vid_y_true, vid_y_prob)
    except Exception:
        vid_auc = np.nan
    vid_cm = confusion_matrix(vid_y_true, vid_y_pred, labels=[0, 1])

    # Print Summary Report
    print("\n" + "=" * 60)
    print("           PARKINSENSE TULIP EVALUATION REPORT           ")
    print("=" * 60)
    print(f"Total Videos Processed : {len(df_video)} ({len(df_valid)} Successful)")
    print(f"Total Subjects         : {len(df_subject)} (3 HT, 5 PD)")
    print(f"Evaluation Runtime     : {round(total_eval_sec, 2)} seconds")
    
    print("\n--- SUBJECT-LEVEL METRICS (Aggregated Multi-View) ---")
    print(f"Accuracy  : {sub_acc * 100:.2f}% ({int(sub_acc * len(sub_y_true))}/{len(sub_y_true)} correct)")
    print(f"Precision : {sub_prec * 100:.2f}%")
    print(f"Recall    : {sub_rec * 100:.2f}%")
    print(f"F1-Score  : {sub_f1:.4f}")
    print(f"ROC-AUC   : {sub_auc if np.isnan(sub_auc) else f'{sub_auc:.4f}'}")
    print("Confusion Matrix (Rows=True, Cols=Pred [HT, PD]):")
    print(f"  [TN={sub_cm[0,0]}  FP={sub_cm[0,1]}]")
    print(f"  [FN={sub_cm[1,0]}  TP={sub_cm[1,1]}]")

    print("\n--- VIDEO-LEVEL METRICS (Individual Camera Views) ---")
    print(f"Accuracy  : {vid_acc * 100:.2f}% ({int(vid_acc * len(vid_y_true))}/{len(vid_y_true)} correct)")
    print(f"Precision : {vid_prec * 100:.2f}%")
    print(f"Recall    : {vid_rec * 100:.2f}%")
    print(f"F1-Score  : {vid_f1:.4f}")
    print(f"ROC-AUC   : {vid_auc if np.isnan(vid_auc) else f'{vid_auc:.4f}'}")
    print("Confusion Matrix (Rows=True, Cols=Pred [HT, PD]):")
    print(f"  [TN={vid_cm[0,0]}  FP={vid_cm[0,1]}]")
    print(f"  [FN={vid_cm[1,0]}  TP={vid_cm[1,1]}]")

    print("\n--- PER-SUBJECT DETAILED BREAKDOWN ---")
    print(df_subject[["subject_id", "category", "true_label", "mean_pd_probability", "subject_pred_label", "correct_pred"]].to_string(index=False))

    print("\n--- PER-CAMERA ANGLE PERFORMANCE BREAKDOWN ---")
    print(df_camera.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    run_tulip_evaluation()
