import os
import sys
import glob
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.insert(0, "/home/pranav/ParkinSense")

from backend.pipeline import analyze_video

RESULTS_DIR = Path("/home/pranav/ParkinSense/data/results/tulip_4dhumans")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path("/home/pranav/ParkinSense/models/pd_detector.pkl")
TEST_VIDEOS_DIR = Path("/home/pranav/ParkinSense/test_videos_extracted/test_videos")

import joblib
model = joblib.load(MODEL_PATH)
# Assuming model is a pipeline or standard model, let's extract expected features.
try:
    expected_features = model.feature_names_in_
    print(f"Expected features: {len(expected_features)}")
except AttributeError:
    # If not scikit-learn >= 1.0 or feature_names_in_ is missing
    expected_features = None
    print("Cannot directly extract expected feature names from model.")

# We know we need 290 features.
model_hash_before = hashlib.md5(open(MODEL_PATH, "rb").read()).hexdigest()
print(f"Model hash before: {model_hash_before}")

# 2. Gather Inventory
all_videos = []
for label_dir, label_val in [("HT", 0), ("PD", 1)]:
    vids = glob.glob(str(TEST_VIDEOS_DIR / label_dir / "**" / "*.mp4"), recursive=True)
    for v in vids:
        # Expected naming: subjectX_cameraY.mp4
        filename = Path(v).name
        subject_id = filename.split('_')[0]
        camera_id = filename.split('_')[1].replace('.mp4', '')
        all_videos.append({
            "subject_id": subject_id,
            "camera_id": camera_id,
            "filename": filename,
            "ground_truth": label_val,
            "filepath": v
        })

inventory_df = pd.DataFrame(all_videos)
inventory_df.to_csv(RESULTS_DIR / "tulip_4dhumans_test_inventory.csv", index=False)
print(f"Total videos found: {len(inventory_df)}")
print(f"Unique subjects: {inventory_df['subject_id'].nunique()}")

# 3. Process every video
results = []
failures = []
all_features_list = []

for idx, row in inventory_df.iterrows():
    print(f"Processing {row['filename']}...")
    try:
        res = analyze_video(row['filepath'], mode="local", cleanup=False)
        p_pd = res.get('probability_pd', 0.0)
        prediction = 1 if p_pd >= 0.50 else 0
        
        results.append({
            "subject_id": row['subject_id'],
            "camera_id": row['camera_id'],
            "filename": row['filename'],
            "ground_truth": row['ground_truth'],
            "status": "success",
            "p_pd": p_pd,
            "prediction": prediction,
            "filepath": row['filepath']
        })
        
        # Also store the extracted features for feature audit
        # Note: analyze_video doesn't return the raw feature vector in its response directly usually,
        # but we can grab it if it's saved somewhere or we can just run the feature extractor.
        
    except Exception as e:
        print(f"Failed {row['filename']}: {e}")
        failures.append({
            "subject_id": row['subject_id'],
            "camera_id": row['camera_id'],
            "filename": row['filename'],
            "error": str(e)
        })
        results.append({
            "subject_id": row['subject_id'],
            "camera_id": row['camera_id'],
            "filename": row['filename'],
            "ground_truth": row['ground_truth'],
            "status": "failed",
            "p_pd": np.nan,
            "prediction": np.nan,
            "filepath": row['filepath']
        })

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_DIR / "tulip_4dhumans_video_predictions.csv", index=False)

if failures:
    failures_df = pd.DataFrame(failures)
    failures_df.to_csv(RESULTS_DIR / "tulip_4dhumans_failures.csv", index=False)

# Evaluate video-level metrics
valid_results = results_df[results_df['status'] == 'success']
y_true = valid_results['ground_truth'].values
y_pred = valid_results['prediction'].values
y_prob = valid_results['p_pd'].values

metrics = {
    "video_level": {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
}

# Evaluate subject-level metrics
subject_df = valid_results.groupby(['subject_id', 'ground_truth']).agg(
    number_of_cameras=('camera_id', 'count'),
    mean_p_pd=('p_pd', 'mean')
).reset_index()

subject_df['prediction'] = (subject_df['mean_p_pd'] >= 0.50).astype(int)
subject_df.to_csv(RESULTS_DIR / "tulip_4dhumans_subject_predictions.csv", index=False)

y_sub_true = subject_df['ground_truth'].values
y_sub_pred = subject_df['prediction'].values
y_sub_prob = subject_df['mean_p_pd'].values

metrics["subject_level"] = {
    "accuracy": accuracy_score(y_sub_true, y_sub_pred),
    "precision": precision_score(y_sub_true, y_sub_pred, zero_division=0),
    "sensitivity": recall_score(y_sub_true, y_sub_pred, zero_division=0),
    "specificity": recall_score(y_sub_true, y_sub_pred, pos_label=0, zero_division=0),
    "f1": f1_score(y_sub_true, y_sub_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_sub_true, y_sub_prob) if len(np.unique(y_sub_true)) > 1 else np.nan,
    "confusion_matrix": confusion_matrix(y_sub_true, y_sub_pred).tolist()
}

with open(RESULTS_DIR / "tulip_4dhumans_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Evaluation completed. Generating report...")
os.system(f"{sys.executable} /home/pranav/ParkinSense/generate_report.py")
print("Check results directory.")
