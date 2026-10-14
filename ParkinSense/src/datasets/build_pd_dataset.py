import pickle
import pandas as pd
import numpy as np
import os

# ===============================
# CONFIGURATION
# ===============================

DATA_PATH = "data/raw/DNE.pkl"
OUTPUT_PATH = "data/processed/pd_dataset.csv"

print("Loading DNE dataset...")

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

print("Dataset loaded successfully!")
print("Number of subjects:", len(data))

rows = []

# ===============================
# PROCESS SUBJECTS
# ===============================

for subject_name, subject_data in data.items():

    print(f"Processing {subject_name}")

    for trial_name, trial in subject_data.items():

        # Get diagnosis label
        diagnosis_label = trial["other"]

        # ---------------------------------
        # KEEP ONLY HEALTHY AND PARKINSON
        # ---------------------------------

        if diagnosis_label == "healthy-control":
            diagnosis = 0

        elif diagnosis_label == "parkinson":
            diagnosis = 1

        else:
            # Skip other diseases
            continue

        pose = trial["pose"]

        # ===============================
        # BASIC INFORMATION
        # ===============================

        row = {
            "subject": subject_name,
            "trial": trial_name,
            "diagnosis": diagnosis,
            "diagnosis_name": diagnosis_label,
            "fps": trial["fps"],
            "num_frames": pose.shape[0]
        }

        # ===============================
        # EXTRACT SAME POSE FEATURES
        # ===============================

        for i in range(pose.shape[1]):

            values = pose[:, i]

            row[f"pose{i}_mean"] = np.mean(values)
            row[f"pose{i}_std"] = np.std(values)
            row[f"pose{i}_min"] = np.min(values)
            row[f"pose{i}_max"] = np.max(values)

        rows.append(row)

# ===============================
# CREATE DATAFRAME
# ===============================

df = pd.DataFrame(rows)

# Create output folder if needed
os.makedirs("data/processed", exist_ok=True)

# Save dataset
df.to_csv(OUTPUT_PATH, index=False)

# ===============================
# RESULTS
# ===============================

print("\n" + "=" * 50)
print("PD DATASET CREATED SUCCESSFULLY")
print("=" * 50)

print("\nDataset shape:", df.shape)

print("\nDiagnosis distribution:")
print(df["diagnosis_name"].value_counts())

print("\nBinary labels:")
print(df["diagnosis"].value_counts())

print("\nSaved to:")
print(OUTPUT_PATH)