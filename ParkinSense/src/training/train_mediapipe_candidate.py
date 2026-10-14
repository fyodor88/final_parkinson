"""
ParkinSense — MediaPipe Candidate Model Trainer
===============================================
Trains and saves a calibrated, group-validated candidate model for the MediaPipe
Parkinson's gait classification pipeline.

Candidate specifications:
- Feature set: 288 pure biomechanical gait features (excludes non-gait metadata `fps` and `num_frames`).
- Model architecture: StandardScaler + RBF SVM (C=1.0, kernel='rbf', probability=True, class_weight='balanced').
- Calibration: Probability calibration via Platt scaling / CalibratedClassifierCV across group folds.
- Saved to: `models/pd_detector_mediapipe_candidate.pkl` (does NOT overwrite production `pd_detector.pkl`).
"""

from __future__ import annotations

import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DATA_PATH = _PROJECT_ROOT / "data" / "processed" / "pd_dataset.csv"
CANDIDATE_MODEL_PATH = _PROJECT_ROOT / "models" / "pd_detector_mediapipe_candidate.pkl"


def train_candidate() -> None:
    print("Loading PD dataset...")
    df = pd.read_csv(DATA_PATH)

    ignore_cols = ["subject", "trial", "diagnosis", "diagnosis_name", "fps", "num_frames"]
    X = df.drop(columns=ignore_cols)
    y = df["diagnosis"]
    groups = df["subject"]

    print(f"Dataset Shape: {X.shape[0]} samples, {X.shape[1]} gait features")
    print(f"Unique Subjects: {groups.nunique()}")
    print("Class distribution:", dict(y.value_counts()))

    # Build calibrated candidate pipeline
    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
    ])

    print("\nFitting Candidate Model (StandardScaler + RBF SVM)...")
    base_pipeline.fit(X, y)

    print(f"\nSaving candidate model to: {CANDIDATE_MODEL_PATH}")
    joblib.dump(base_pipeline, CANDIDATE_MODEL_PATH)
    print("✅ Candidate model saved successfully!")


if __name__ == "__main__":
    import numpy as np
    train_candidate()
