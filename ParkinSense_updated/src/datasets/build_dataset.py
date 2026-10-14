import pickle
import pandas as pd
import numpy as np
import os

# ===============================
# CONFIGURATION
# ===============================

DATA_PATH = "data/raw/BMCLab.pkl"
OUTPUT_PATH = "data/processed/dataset.csv"

# ===============================
# LOAD DATASET
# ===============================

print("Loading CARE-PD dataset...")

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

print("Dataset Loaded Successfully!")
print()

rows = []

# ===============================
# PROCESS EVERY SUBJECT
# ===============================

for subject_name, subject_data in data.items():

    print(f"Processing {subject_name}")

    # Every walking trial
    for trial_name, trial in subject_data.items():

        pose = trial["pose"]

        # Basic Information
        row = {
            "subject": subject_name,
            "trial": trial_name,
            "medication": trial["medication"],
            "updrs_gait": trial["UPDRS_GAIT"],
            "fps": trial["fps"],
            "num_frames": pose.shape[0]
        }

        # ==================================
        # Extract statistical features
        # ==================================

        for i in range(pose.shape[1]):

            values = pose[:, i]

            row[f"pose{i}_mean"] = np.mean(values)
            row[f"pose{i}_std"] = np.std(values)
            row[f"pose{i}_min"] = np.min(values)
            row[f"pose{i}_max"] = np.max(values)

        rows.append(row)

print()
print("Finished Processing All Subjects")

# ==================================
# SAVE DATASET
# ==================================

df = pd.DataFrame(rows)

os.makedirs("data/processed", exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print()
print("=" * 50)
print("Dataset Saved Successfully!")
print("=" * 50)

print()

print("Shape:", df.shape)

print()

print(df.head())