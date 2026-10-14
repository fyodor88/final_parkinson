import pandas as pd
import numpy as np
import os

# ---------------------------------------------------
# Create output folder
# ---------------------------------------------------
os.makedirs("output", exist_ok=True)

# ---------------------------------------------------
# Read landmark file
# ---------------------------------------------------
df = pd.read_csv("output/landmarks.csv")

print("Landmarks Loaded Successfully!")

# ---------------------------------------------------
# Angle Function
# ---------------------------------------------------
def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    cosine = np.clip(cosine, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine))

    return angle


# ---------------------------------------------------
# Landmark IDs
# ---------------------------------------------------

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_ELBOW = 13
RIGHT_ELBOW = 14

LEFT_WRIST = 15
RIGHT_WRIST = 16

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_KNEE = 25
RIGHT_KNEE = 26

LEFT_ANKLE = 27
RIGHT_ANKLE = 28


# ---------------------------------------------------
# Process every frame
# ---------------------------------------------------

frames = df["frame"].unique()

features = []

for frame in frames:

    frame_data = df[df["frame"] == frame]

    try:

        # -----------------------------
        # LEFT LEG
        # -----------------------------
        left_hip = frame_data[frame_data.landmark == LEFT_HIP][["x","y"]].values[0]
        left_knee = frame_data[frame_data.landmark == LEFT_KNEE][["x","y"]].values[0]
        left_ankle = frame_data[frame_data.landmark == LEFT_ANKLE][["x","y"]].values[0]

        # -----------------------------
        # RIGHT LEG
        # -----------------------------
        right_hip = frame_data[frame_data.landmark == RIGHT_HIP][["x","y"]].values[0]
        right_knee = frame_data[frame_data.landmark == RIGHT_KNEE][["x","y"]].values[0]
        right_ankle = frame_data[frame_data.landmark == RIGHT_ANKLE][["x","y"]].values[0]

        # -----------------------------
        # SHOULDERS
        # -----------------------------
        left_shoulder = frame_data[frame_data.landmark == LEFT_SHOULDER][["x","y"]].values[0]
        right_shoulder = frame_data[frame_data.landmark == RIGHT_SHOULDER][["x","y"]].values[0]

        # -----------------------------
        # ELBOWS
        # -----------------------------
        left_elbow = frame_data[frame_data.landmark == LEFT_ELBOW][["x","y"]].values[0]
        right_elbow = frame_data[frame_data.landmark == RIGHT_ELBOW][["x","y"]].values[0]

        # -----------------------------
        # WRISTS
        # -----------------------------
        left_wrist = frame_data[frame_data.landmark == LEFT_WRIST][["x","y"]].values[0]
        right_wrist = frame_data[frame_data.landmark == RIGHT_WRIST][["x","y"]].values[0]

        # ----------------------------------
        # Knee Angles
        # ----------------------------------
        left_knee_angle = calculate_angle(
            left_hip,
            left_knee,
            left_ankle
        )

        right_knee_angle = calculate_angle(
            right_hip,
            right_knee,
            right_ankle
        )

        # ----------------------------------
        # Hip Angles
        # ----------------------------------
        left_hip_angle = calculate_angle(
            left_shoulder,
            left_hip,
            left_knee
        )

        right_hip_angle = calculate_angle(
            right_shoulder,
            right_hip,
            right_knee
        )

        # ----------------------------------
        # Elbow Angles
        # ----------------------------------
        left_elbow_angle = calculate_angle(
            left_shoulder,
            left_elbow,
            left_wrist
        )

        right_elbow_angle = calculate_angle(
            right_shoulder,
            right_elbow,
            right_wrist
        )

        # ----------------------------------
        # Step Width
        # ----------------------------------
        step_width = np.linalg.norm(
            left_ankle - right_ankle
        )

        # ----------------------------------
        # Arm Swing
        # ----------------------------------
        arm_swing = np.linalg.norm(
            left_wrist - right_wrist
        )

        features.append({

            "frame": frame,

            "left_knee_angle": left_knee_angle,
            "right_knee_angle": right_knee_angle,

            "left_hip_angle": left_hip_angle,
            "right_hip_angle": right_hip_angle,

            "left_elbow_angle": left_elbow_angle,
            "right_elbow_angle": right_elbow_angle,

            "step_width": step_width,

            "arm_swing": arm_swing

        })

    except Exception as e:

        print(f"Skipping Frame {frame}: {e}")

# ---------------------------------------------------
# Save Features
# ---------------------------------------------------

features_df = pd.DataFrame(features)

features_df.to_csv(
    "output/features.csv",
    index=False
)

print("\n===================================")
print("Feature Extraction Complete")
print("Saved : output/features.csv")
print("===================================\n")

print(features_df.head())