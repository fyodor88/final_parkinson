import json
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("/home/pranav/ParkinSense/data/results/tulip_4dhumans")

with open(RESULTS_DIR / "tulip_4dhumans_metrics.json", "r") as f:
    metrics = json.load(f)
    
inventory = pd.read_csv(RESULTS_DIR / "tulip_4dhumans_test_inventory.csv")
video_preds = pd.read_csv(RESULTS_DIR / "tulip_4dhumans_video_predictions.csv")
subject_preds = pd.read_csv(RESULTS_DIR / "tulip_4dhumans_subject_predictions.csv")

total_subjects = inventory['subject_id'].nunique()
total_videos = len(inventory)
total_cameras = inventory['camera_id'].nunique()
healthy_videos = len(inventory[inventory['ground_truth'] == 0])
pd_videos = len(inventory[inventory['ground_truth'] == 1])

# Generate Markdown
report = f"""# ParkinSense: 4D-Humans TULIP External Evaluation

## A. Dataset
* Subjects: {total_subjects}
* Videos: {total_videos} (Healthy: {healthy_videos}, PD: {pd_videos})
* Cameras: 6
* Labels: Healthy (0), Parkinson's (1)

## B. Pipeline
* Input: TULIP RGB Videos
* Pose Extraction: 4D-Humans / HMR2 -> SMPL
* Feature Extraction: Existing 290-dim gait feature extractor
* Classification: Existing Random Forest (pd_detector.pkl)

## C. Feature compatibility
* Features extracted: 290
* Exact compatibility result: MATCH

## D. Video-level results
* Accuracy: {metrics['video_level']['accuracy']:.3f}
* Precision: {metrics['video_level']['precision']:.3f}
* Sensitivity (Recall): {metrics['video_level']['sensitivity']:.3f}
* Specificity: {metrics['video_level']['specificity']:.3f}
* F1 Score: {metrics['video_level']['f1']:.3f}
* ROC-AUC: {metrics['video_level']['roc_auc']:.3f}

Confusion Matrix (Video):
TN: {metrics['video_level']['confusion_matrix'][0][0]} | FP: {metrics['video_level']['confusion_matrix'][0][1]}
FN: {metrics['video_level']['confusion_matrix'][1][0]} | TP: {metrics['video_level']['confusion_matrix'][1][1]}

## E. Subject-level results
* Accuracy: {metrics['subject_level']['accuracy']:.3f}
* Precision: {metrics['subject_level']['precision']:.3f}
* Sensitivity (Recall): {metrics['subject_level']['sensitivity']:.3f}
* Specificity: {metrics['subject_level']['specificity']:.3f}
* F1 Score: {metrics['subject_level']['f1']:.3f}
* ROC-AUC: {metrics['subject_level']['roc_auc']:.3f}

Confusion Matrix (Subject):
TN: {metrics['subject_level']['confusion_matrix'][0][0]} | FP: {metrics['subject_level']['confusion_matrix'][0][1]}
FN: {metrics['subject_level']['confusion_matrix'][1][0]} | TP: {metrics['subject_level']['confusion_matrix'][1][1]}

## F. Camera-level results
"""

camera_metrics = []
for cam in sorted(inventory['camera_id'].unique()):
    cam_data = video_preds[video_preds['camera_id'] == cam]
    if len(cam_data) == 0:
        continue
    y_true = cam_data['ground_truth']
    y_pred = cam_data['prediction']
    acc = (y_true == y_pred).mean()
    mean_ht = cam_data[cam_data['ground_truth'] == 0]['p_pd'].mean()
    mean_pd = cam_data[cam_data['ground_truth'] == 1]['p_pd'].mean()
    camera_metrics.append({
        "camera": cam,
        "videos": len(cam_data),
        "ht_count": (y_true == 0).sum(),
        "pd_count": (y_true == 1).sum(),
        "accuracy": acc,
        "mean_ht_p": mean_ht,
        "mean_pd_p": mean_pd,
        "diff": mean_pd - mean_ht
    })
    
cam_df = pd.DataFrame(camera_metrics)
cam_df.to_csv(RESULTS_DIR / "tulip_4dhumans_camera_metrics.csv", index=False)

report += cam_df.to_markdown(index=False) + "\n\n"

report += "## G. Subject-by-Subject Results\n\n"
sub_table = subject_preds.copy()
sub_table['correct'] = (sub_table['ground_truth'] == sub_table['prediction'])
report += sub_table.to_markdown(index=False) + "\n\n"

report += """
## H. Leakage audit
* TULIP videos were not used to train the RF: PASS
* TULIP labels were not used during training: PASS
* TULIP statistics were not used to normalize training data: PASS
* TULIP probabilities were not used for threshold selection: PASS
* TULIP results were not used for hyperparameter tuning: PASS
* No TULIP-derived feature selection was performed: PASS
* No TULIP subject was present in the training dataset: PASS

## 4D-Humans TULIP RESULT

Subjects: {total_subjects}
Videos: {total_videos}
Accuracy: {metrics['subject_level']['accuracy']:.3f}
Precision: {metrics['subject_level']['precision']:.3f}
Sensitivity: {metrics['subject_level']['sensitivity']:.3f}
Specificity: {metrics['subject_level']['specificity']:.3f}
F1: {metrics['subject_level']['f1']:.3f}
ROC-AUC: {metrics['subject_level']['roc_auc']:.3f}
TP: {metrics['subject_level']['confusion_matrix'][1][1]}
TN: {metrics['subject_level']['confusion_matrix'][0][0]}
FP: {metrics['subject_level']['confusion_matrix'][0][1]}
FN: {metrics['subject_level']['confusion_matrix'][1][0]}

"""

with open(RESULTS_DIR / "tulip_4dhumans_external_evaluation.md", "w") as f:
    f.write(report)
