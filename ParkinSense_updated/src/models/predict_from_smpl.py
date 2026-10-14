import pandas as pd
import numpy as np
import joblib
from scipy.spatial.transform import Rotation


# ============================================================
# CONFIGURATION
# ============================================================

SMPL_FILE = "4D-Humans/example_data/smpl.csv"

# Model 1: UPDRS Gait Score
UPDRS_MODEL_FILE = "models/random_forest.pkl"

# Model 2: Parkinson's Detection
PD_MODEL_FILE = "models/pd_detector.pkl"

OUTPUT_FEATURES = "remote_output/prediction_features.csv"


# ============================================================
# LOAD SMPL CSV
# ============================================================

print("Loading SMPL CSV...")

df = pd.read_csv(SMPL_FILE)

print("SMPL shape:", df.shape)


# ============================================================
# CONVERT ROTATION MATRICES TO AXIS-ANGLE
# ============================================================

def rotation_matrix_to_axis_angle(row, prefix, count):

    result = []

    for joint in range(count):

        values = [
            row[f"{prefix}_{joint * 9 + i}"]
            for i in range(9)
        ]

        matrix = np.array(values).reshape(3, 3)

        rotation = Rotation.from_matrix(matrix)

        axis_angle = rotation.as_rotvec()

        result.extend(axis_angle)

    return result


# ============================================================
# CREATE 72-DIMENSION SMPL POSE
# ============================================================

poses = []

for _, row in df.iterrows():

    global_pose = rotation_matrix_to_axis_angle(
        row,
        "global_orient",
        1
    )

    body_pose = rotation_matrix_to_axis_angle(
        row,
        "body_pose",
        23
    )

    pose = global_pose + body_pose

    if len(pose) != 72:
        raise ValueError(
            f"Expected 72 pose values, got {len(pose)}"
        )

    poses.append(pose)


poses = np.array(poses)

print("Converted pose shape:", poses.shape)


# ============================================================
# CALCULATE FEATURES
# ============================================================

features = {}

features["fps"] = 29.75
features["num_frames"] = poses.shape[0]

for i in range(72):

    values = poses[:, i]

    features[f"pose{i}_mean"] = np.mean(values)
    features[f"pose{i}_std"] = np.std(values)
    features[f"pose{i}_min"] = np.min(values)
    features[f"pose{i}_max"] = np.max(values)


feature_df = pd.DataFrame([features])


# ============================================================
# LOAD BOTH MODELS
# ============================================================

print("\nLoading models...")

updrs_model = joblib.load(UPDRS_MODEL_FILE)
pd_model = joblib.load(PD_MODEL_FILE)

print("UPDRS model expects:", updrs_model.n_features_in_, "features")
print("PD model expects:", pd_model.n_features_in_, "features")


# ============================================================
# CREATE CORRECT FEATURE ORDER FOR EACH MODEL
# ============================================================

# UPDRS model features
if hasattr(updrs_model, "feature_names_in_"):
    updrs_features = feature_df[
        list(updrs_model.feature_names_in_)
    ]
else:
    updrs_features = feature_df


# PD detection model features
if hasattr(pd_model, "feature_names_in_"):
    pd_features = feature_df[
        list(pd_model.feature_names_in_)
    ]
else:
    pd_features = feature_df


# ============================================================
# SAVE FEATURES
# ============================================================

feature_df.to_csv(
    OUTPUT_FEATURES,
    index=False
)

print("\nFeature shape:", feature_df.shape)
print("Saved:", OUTPUT_FEATURES)


# ============================================================
# PREDICT PARKINSON'S
# ============================================================

pd_prediction = pd_model.predict(pd_features)[0]
pd_probabilities = pd_model.predict_proba(pd_features)[0]


# ============================================================
# PREDICT UPDRS GAIT SCORE
# ============================================================

updrs_prediction = updrs_model.predict(updrs_features)[0]
updrs_probabilities = updrs_model.predict_proba(updrs_features)[0]


# ============================================================
# FINAL PARKINSENSE RESULT
# ============================================================

print()
print("=" * 55)
print("               PARKINSENSE RESULT")
print("=" * 55)


# -------------------------
# PARKINSON'S DETECTION
# -------------------------

print("\nPARKINSON'S DETECTION")
print("-" * 55)

if pd_prediction == 1:
    print("Result: PARKINSON'S LIKELY")
else:
    print("Result: PARKINSON'S NOT LIKELY")

print("\nProbabilities:")

for cls, probability in zip(
    pd_model.classes_,
    pd_probabilities
):

    if cls == 0:
        label = "Healthy"
    else:
        label = "Parkinson's"

    print(
        f"{label}: {probability * 100:.2f}%"
    )


# -------------------------
# UPDRS GAIT ASSESSMENT
# -------------------------

print("\n" + "-" * 55)
print("GAIT ASSESSMENT")
print("-" * 55)

print(
    f"Predicted UPDRS Gait Score: "
    f"{updrs_prediction}"
)

print("\nScore probabilities:")

for cls, probability in zip(
    updrs_model.classes_,
    updrs_probabilities
):

    print(
        f"Score {cls}: {probability * 100:.2f}%"
    )


# -------------------------
# DISCLAIMER
# -------------------------

print("\n" + "=" * 55)
print(
    "Note: This is an AI research model and is not a clinical diagnosis."
)
print("=" * 55)
