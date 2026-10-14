"""
ParkinSense — Build MediaPipe-Normalized Training Dataset
==========================================================
Constructs `data/processed/pd_dataset_mediapipe_normalized.csv` from `data/raw/DNE.pkl`
by normalizing root global orientation (`pose0`..`pose2`) to canonical alignment
and computing exact 290-feature summary statistics across relative SMPL joint pose parameters.
"""

from __future__ import annotations

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "DNE.pkl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "pd_dataset_mediapipe_normalized.csv"

def build_normalized_training_dataset():
    print(f"Loading DNE raw MoCap dataset from: {DATA_PATH}")
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    rows = []
    for subject_name, subject_data in data.items():
        for trial_name, trial in subject_data.items():
            diagnosis_label = trial.get("other", "")
            if diagnosis_label == "healthy-control":
                diagnosis = 0
            elif diagnosis_label == "parkinson":
                diagnosis = 1
            else:
                continue

            pose = trial["pose"].copy()  # Shape (N_frames, 72)
            fps = float(trial.get("fps", 60.0))
            num_frames = pose.shape[0]

            # Canonical alignment: Zero out global orientation variation (pose0, pose1, pose2)
            # relative joint rotations (pose3..pose71) are naturally scale & translation invariant.
            pose[:, 0:3] = 0.0

            row = {
                "subject": subject_name,
                "trial": trial_name,
                "diagnosis": diagnosis,
                "diagnosis_name": diagnosis_label,
                "fps": fps,
                "num_frames": num_frames,
            }

            for i in range(pose.shape[1]):
                values = pose[:, i]
                row[f"pose{i}_mean"] = float(np.mean(values))
                row[f"pose{i}_std"] = float(np.std(values))
                row[f"pose{i}_min"] = float(np.min(values))
                row[f"pose{i}_max"] = float(np.max(values))

            rows.append(row)

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Successfully generated normalized training dataset: {OUTPUT_PATH}")
    print(f"Dataset shape: {df.shape} (290 features + metadata)")

if __name__ == "__main__":
    build_normalized_training_dataset()
