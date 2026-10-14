"""ParkinSense — MediaPipe End-to-End Inference Pipeline.

Fast inference using MediaPipe Pose instead of 4D-Humans / HMR2.
The project keeps both local MediaPipe and remote GPU workflows available,
with the local pipeline used by default for validation and demo runs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

# Resolve project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import get_logger, ensure_dir
from src.utils.helpers import calculate_angle, euclidean_distance
from src.pose.mediapipe_smpl import (
    mediapipe_to_smpl_rotations,
    rotations_to_smpl_dataframe_row,
)
from src.features.smpl_feature_extractor import SMPLFeatureExtractor
from src.features.explainability import ModelExplainer
from src.inference.prediction_mapping import summarize_binary_prediction

logger = get_logger(__name__)


def _load_model_dataset_performance() -> Dict[str, Any]:
    """Load dataset-level validation metrics from the project CSV results when available."""
    csv_path = _PROJECT_ROOT / "data" / "processed" / "mediapipe_model_comparison.csv"
    if not csv_path.exists():
        return {}

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return {}

        preferred = df[df["model_name"].str.contains("RandomForest_Current_Baseline", na=False)]
        if preferred.empty:
            preferred = df.head(1)

        row = preferred.iloc[0]
        return {
            "accuracy": float(row.get("accuracy", np.nan)),
            "accuracy_std": float(row.get("accuracy_std", np.nan)),
            "sensitivity": float(row.get("sensitivity", np.nan)),
            "specificity": float(row.get("specificity", np.nan)),
            "precision": float(row.get("precision", np.nan)),
            "f1_score": float(row.get("f1_score", np.nan)),
            "roc_auc": float(row.get("roc_auc", np.nan)),
            "model_name": str(row.get("model_name", "Random Forest")),
            "feature_set": str(row.get("feature_set", "290 features")),
            "num_features": int(row.get("num_features", 290)),
        }
    except Exception:
        return {}


# Model paths
_DEFAULT_PD_MODEL = _PROJECT_ROOT / "models" / "pd_detector.pkl"
_DEFAULT_SEV_MODEL = _PROJECT_ROOT / "models" / "random_forest.pkl"

# Minimum number of frames with detected landmarks for a valid run
_MIN_FRAMES_REQUIRED = 10

# MediaPipe model complexity (0=lite/fast, 1=full, 2=heavy/accurate)
_MP_MODEL_COMPLEXITY = 1
_MP_MIN_DETECTION = 0.5
_MP_MIN_TRACKING = 0.5

# UPDRS gait severity descriptions
UPDRS_GAIT_DESCRIPTIONS = {
    0: "Normal Gait (UPDRS 0: No movement difficulty)",
    1: "Slight Impairment (UPDRS 1: Slow walking or minor asymmetry)",
    2: "Moderate Impairment (UPDRS 2: Impaired balance or festive gait)",
}

# Minimum frame distance between selected key gait frames (seconds)
MIN_FRAME_DISTANCE_SECONDS = 1.0

# MediaPipe Pose landmark indices
LANDMARK_NOSE = 0
LANDMARK_LEFT_SHOULDER = 11
LANDMARK_RIGHT_SHOULDER = 12
LANDMARK_LEFT_ELBOW = 13
LANDMARK_RIGHT_ELBOW = 14
LANDMARK_LEFT_WRIST = 15
LANDMARK_RIGHT_WRIST = 16
LANDMARK_LEFT_HIP = 23
LANDMARK_RIGHT_HIP = 24
LANDMARK_LEFT_KNEE = 25
LANDMARK_RIGHT_KNEE = 26
LANDMARK_LEFT_ANKLE = 27
LANDMARK_RIGHT_ANKLE = 28
LANDMARK_LEFT_HEEL = 29
LANDMARK_RIGHT_HEEL = 30
LANDMARK_LEFT_FOOT_INDEX = 31
LANDMARK_RIGHT_FOOT_INDEX = 32

# Body part weights for composite gait variation score
BODY_PART_WEIGHTS = {
    "knee": 0.25,
    "hip": 0.20,
    "ankle": 0.20,
    "arm": 0.15,
    "torso": 0.15,
    "asymmetry": 0.05,
}


# --------------------------------------------------------------------------
# Validation & Math Helpers
# --------------------------------------------------------------------------

def _check_finite(arr: np.ndarray, label: str) -> None:
    """Raise ValueError if arr contains NaN or Inf."""
    n_nan = int(np.sum(np.isnan(arr)))
    n_inf = int(np.sum(np.isinf(arr)))
    if n_nan > 0 or n_inf > 0:
        raise ValueError(
            f"Invalid values in {label}: {n_nan} NaN, {n_inf} Inf. "
            "This may indicate degenerate MediaPipe landmarks or a processing error."
        )


def _check_feature_vector(feature_df: pd.DataFrame) -> None:
    """Validate that the feature DataFrame has exactly 290 finite columns."""
    n_cols = feature_df.shape[1]
    if n_cols != 290:
        raise ValueError(
            f"Feature vector has {n_cols} columns, expected exactly 290. "
            "This indicates a mismatch between the MediaPipe pipeline and the "
            "trained model's expected feature format."
        )
    arr = feature_df.values.astype(float)
    _check_finite(arr, "feature vector")


def _compute_knee_angle(coords: Dict[int, np.ndarray], hip_idx: int, knee_idx: int, ankle_idx: int) -> float:
    """Compute knee joint angle in degrees [0, 180] using hip-knee-ankle geometry."""
    hip = coords.get(hip_idx)
    knee = coords.get(knee_idx)
    ankle = coords.get(ankle_idx)
    if hip is None or knee is None or ankle is None:
        return 0.0
    return calculate_angle(hip, knee, ankle)


def _compute_hip_angle(coords: Dict[int, np.ndarray], shoulder_idx: int, hip_idx: int, knee_idx: int) -> float:
    """Compute hip joint angle in degrees [0, 180] using shoulder-hip-knee geometry."""
    shoulder = coords.get(shoulder_idx)
    hip = coords.get(hip_idx)
    knee = coords.get(knee_idx)
    if shoulder is None or hip is None or knee is None:
        return 0.0
    return calculate_angle(shoulder, hip, knee)


def _compute_arm_angle(coords: Dict[int, np.ndarray], is_left: bool = True) -> float:
    """Compute arm/elbow angle in degrees [0, 180] using shoulder-elbow-wrist geometry."""
    s_idx = LANDMARK_LEFT_SHOULDER if is_left else LANDMARK_RIGHT_SHOULDER
    e_idx = LANDMARK_LEFT_ELBOW if is_left else LANDMARK_RIGHT_ELBOW
    w_idx = LANDMARK_LEFT_WRIST if is_left else LANDMARK_RIGHT_WRIST
    s = coords.get(s_idx)
    e = coords.get(e_idx)
    w = coords.get(w_idx)
    if s is None or e is None or w is None:
        return 0.0
    return calculate_angle(s, e, w)


def _compute_torso_angle(coords: Dict[int, np.ndarray]) -> float:
    """Compute torso tilt angle in degrees relative to vertical vector."""
    lh = coords.get(LANDMARK_LEFT_HIP)
    rh = coords.get(LANDMARK_RIGHT_HIP)
    ls = coords.get(LANDMARK_LEFT_SHOULDER)
    rs = coords.get(LANDMARK_RIGHT_SHOULDER)
    if lh is None or rh is None or ls is None or rs is None:
        return 0.0
    hip_center = (lh + rh) / 2.0
    shoulder_center = (ls + rs) / 2.0
    torso_vec = shoulder_center - hip_center
    norm = np.linalg.norm(torso_vec)
    if norm < 1e-6:
        return 0.0
    torso_norm = torso_vec / norm
    vertical = np.array([0.0, -1.0, 0.0]) if len(torso_norm) >= 3 else np.array([0.0, -1.0])
    cos_val = np.dot(torso_norm[:len(vertical)], vertical)
    cos_val = float(np.clip(cos_val, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_val)))


def _compute_step_width(coords: Dict[int, np.ndarray]) -> float:
    """Compute 3D Euclidean step width distance between left and right ankles."""
    la = coords.get(LANDMARK_LEFT_ANKLE)
    ra = coords.get(LANDMARK_RIGHT_ANKLE)
    if la is None or ra is None:
        return 0.0
    return float(np.linalg.norm(la - ra))


def _compute_arm_swing_distance(coords: Dict[int, np.ndarray]) -> float:
    """Compute 3D Euclidean distance between left and right wrists."""
    lw = coords.get(LANDMARK_LEFT_WRIST)
    rw = coords.get(LANDMARK_RIGHT_WRIST)
    if lw is None or rw is None:
        return 0.0
    return float(np.linalg.norm(lw - rw))


def _compute_frame_displacement(coords: Dict[int, np.ndarray],
                                prev_coords: Optional[Dict[int, np.ndarray]],
                                landmark_indices: List[int]) -> float:
    """Compute average 3D displacement of selected landmarks from previous frame."""
    if prev_coords is None:
        return 0.0
    displacements = []
    for idx in landmark_indices:
        if idx in coords and idx in prev_coords:
            disp = np.linalg.norm(coords[idx] - prev_coords[idx])
            displacements.append(disp)
    if not displacements:
        return 0.0
    return float(np.mean(displacements))


def _compute_asymmetry(coords: Dict[int, np.ndarray]) -> float:
    """Compute left-right knee angle asymmetry in degrees."""
    lk = _compute_knee_angle(coords, LANDMARK_LEFT_HIP, LANDMARK_LEFT_KNEE, LANDMARK_LEFT_ANKLE)
    rk = _compute_knee_angle(coords, LANDMARK_RIGHT_HIP, LANDMARK_RIGHT_KNEE, LANDMARK_RIGHT_ANKLE)
    return abs(lk - rk)


# --------------------------------------------------------------------------
# Frame scoring and key gait frame extraction
# --------------------------------------------------------------------------

def compute_frame_gait_scores(coords_per_frame: List[Dict[int, np.ndarray]],
                               fps: float) -> List[Dict[str, Any]]:
    """Compute per-frame gait variation scores from MediaPipe landmarks."""
    n_frames = len(coords_per_frame)
    if n_frames == 0:
        return []

    scores: List[Dict[str, Any]] = []

    for i in range(n_frames):
        coords = coords_per_frame[i]
        prev_coords = coords_per_frame[i - 1] if i > 0 else None

        lk_angle = _compute_knee_angle(coords, LANDMARK_LEFT_HIP, LANDMARK_LEFT_KNEE, LANDMARK_LEFT_ANKLE)
        rk_angle = _compute_knee_angle(coords, LANDMARK_RIGHT_HIP, LANDMARK_RIGHT_KNEE, LANDMARK_RIGHT_ANKLE)

        prev_lk = _compute_knee_angle(prev_coords, LANDMARK_LEFT_HIP, LANDMARK_LEFT_KNEE, LANDMARK_LEFT_ANKLE) if prev_coords else lk_angle
        prev_rk = _compute_knee_angle(prev_coords, LANDMARK_RIGHT_HIP, LANDMARK_RIGHT_KNEE, LANDMARK_RIGHT_ANKLE) if prev_coords else rk_angle

        knee_change = max(abs(lk_angle - prev_lk), abs(rk_angle - prev_rk))

        hip_move = _compute_frame_displacement(coords, prev_coords, [LANDMARK_LEFT_HIP, LANDMARK_RIGHT_HIP])
        ankle_move = _compute_frame_displacement(coords, prev_coords, [LANDMARK_LEFT_ANKLE, LANDMARK_RIGHT_ANKLE])
        arm_move = _compute_frame_displacement(coords, prev_coords, [LANDMARK_LEFT_WRIST, LANDMARK_RIGHT_WRIST])

        torso_angle = _compute_torso_angle(coords)
        prev_torso = _compute_torso_angle(prev_coords) if prev_coords else torso_angle
        torso_move = abs(torso_angle - prev_torso)

        asymmetry = _compute_asymmetry(coords)

        # Normalize components for composite score
        knee_norm = min(knee_change / 45.0, 1.0)
        hip_norm = min(hip_move * 5.0, 1.0)
        ankle_norm = min(ankle_move * 5.0, 1.0)
        arm_norm = min(arm_move * 5.0, 1.0)
        torso_norm = min(torso_move / 20.0, 1.0)
        asym_norm = min(asymmetry / 30.0, 1.0)

        score = (
            BODY_PART_WEIGHTS["knee"] * knee_norm
            + BODY_PART_WEIGHTS["hip"] * hip_norm
            + BODY_PART_WEIGHTS["ankle"] * ankle_norm
            + BODY_PART_WEIGHTS["arm"] * arm_norm
            + BODY_PART_WEIGHTS["torso"] * torso_norm
            + BODY_PART_WEIGHTS["asymmetry"] * asym_norm
        )

        contributions = {
            "knee": BODY_PART_WEIGHTS["knee"] * knee_norm,
            "hip": BODY_PART_WEIGHTS["hip"] * hip_norm,
            "ankle": BODY_PART_WEIGHTS["ankle"] * ankle_norm,
            "arm": BODY_PART_WEIGHTS["arm"] * arm_norm,
            "torso": BODY_PART_WEIGHTS["torso"] * torso_norm,
            "asymmetry": BODY_PART_WEIGHTS["asymmetry"] * asym_norm,
        }
        sorted_parts = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_parts[0][0] if sorted_parts else "knee"
        secondary = sorted_parts[1][0] if len(sorted_parts) > 1 else "asymmetry"

        indicator_map = {
            "knee": "Knee movement",
            "hip": "Hip movement",
            "ankle": "Step width",
            "arm": "Arm swing",
            "torso": "Postural variation",
            "asymmetry": "Left-right variation",
        }

        scores.append({
            "frame": int(i),
            "score": float(score),
            "knee_change": round(knee_change, 4),
            "hip_move": round(hip_move, 4),
            "ankle_move": round(ankle_move, 4),
            "arm_move": round(arm_move, 4),
            "torso_move": round(torso_move, 4),
            "asymmetry": round(asymmetry, 4),
            "primary_indicator": indicator_map.get(primary, "Gait variation"),
            "secondary_indicator": indicator_map.get(secondary, "Gait variation"),
        })

    return scores


def select_key_gait_frames(frame_scores: List[Dict[str, Any]],
                           fps: float, n_frames: int = 5) -> List[Dict[str, Any]]:
    """Select 3-5 key gait frames using non-maximum suppression."""
    if not frame_scores:
        return []

    sorted_by_score = sorted(frame_scores, key=lambda x: x["score"], reverse=True)
    selected: List[Dict[str, Any]] = []
    min_distance_frames = max(1, round(MIN_FRAME_DISTANCE_SECONDS * fps))

    for candidate in sorted_by_score:
        if len(selected) >= n_frames:
            break

        too_close = False
        for s in selected:
            if abs(candidate["frame"] - s["frame"]) < min_distance_frames:
                too_close = True
                break

        if not too_close:
            selected.append(candidate)

    if len(selected) < 3 and len(frame_scores) > 0:
        remaining = [s for s in frame_scores if s not in selected]
        remaining.sort(key=lambda x: x["score"], reverse=True)
        for s in remaining:
            if len(selected) >= 5:
                break
            too_close = False
            for sel in selected:
                if abs(s["frame"] - sel["frame"]) < min_distance_frames:
                    too_close = True
                    break
            if not too_close:
                selected.append(s)

    selected.sort(key=lambda x: x["frame"])
    return selected


# --------------------------------------------------------------------------
# Structured Explanation Generator (Backend Metadata)
# --------------------------------------------------------------------------

def _build_frame_explanation_metadata(
    primary_indicator: str,
    secondary_indicator: str,
    metrics: Dict[str, Any],
    actual_frame_idx: int,
) -> Dict[str, Any]:
    """Build structured explanation metadata for a key gait frame based strictly on real calculated measurements."""
    primary_lower = primary_indicator.lower()
    ts = metrics.get("timestamp", 0.0)
    score = metrics.get("score", 0.0)

    lk = metrics.get("knee_angle_left", 0.0)
    rk = metrics.get("knee_angle_right", 0.0)
    dlk = metrics.get("delta_knee_left", 0.0)
    drk = metrics.get("delta_knee_right", 0.0)

    lh = metrics.get("hip_angle_left", 0.0)
    rh = metrics.get("hip_angle_right", 0.0)

    la = metrics.get("arm_angle_left", 0.0)
    ra = metrics.get("arm_angle_right", 0.0)
    dla = metrics.get("delta_arm_left", 0.0)
    dra = metrics.get("delta_arm_right", 0.0)

    sw = metrics.get("step_width", 0.0)
    d_sw = metrics.get("delta_step_width", 0.0)
    ank_l = metrics.get("ankle_move_left", 0.0)
    ank_r = metrics.get("ankle_move_right", 0.0)

    arm_l = metrics.get("left_arm_movement", 0.0)
    arm_r = metrics.get("right_arm_movement", 0.0)
    arm_diff = abs(arm_l - arm_r)

    torso_a = metrics.get("torso_angle", 0.0)
    asym = metrics.get("asymmetry", 0.0)

    if "knee" in primary_lower:
        title = "Knee Joint Flexion & Extension Dynamics"
        reason_selected = (
            f"Frame #{actual_frame_idx} (timestamp {ts:.2f}s) was selected because it recorded peak knee kinematic "
            f"variance (variance score: {score:.4f}) during knee flexion and extension."
        )
        analysis_summary = (
            f"Left knee angle was measured at {lk:.1f}° (frame change Δ: {dlk:+.1f}°) and right knee angle at {rk:.1f}° "
            f"(frame change Δ: {drk:+.1f}°). Left-right knee angle asymmetry was recorded at {asym:.1f}°."
        )
        measured_values = [
            {"label": "Left Knee Angle", "value": f"{lk:.1f}°"},
            {"label": "Right Knee Angle", "value": f"{rk:.1f}°"},
            {"label": "Left-Right Knee Difference", "value": f"{asym:.1f}°"},
            {"label": "Left Knee Frame Change (Δ)", "value": f"{dlk:+.1f}°"},
            {"label": "Right Knee Frame Change (Δ)", "value": f"{drk:+.1f}°"},
        ]

    elif "step" in primary_lower or "ankle" in primary_lower:
        title = "Step Width & Base-of-Support Stance"
        reason_selected = (
            f"Frame #{actual_frame_idx} (timestamp {ts:.2f}s) was selected because it captured significant variance "
            f"in lateral stance width (variance score: {score:.4f}), evaluating base-of-support stance stability."
        )
        analysis_summary = (
            f"Lateral step width between 3D ankle landmarks was measured at {sw:.3f} m (frame change Δ: {d_sw:+.3f} m). "
            f"Left ankle displacement: {ank_l:.3f} m, right ankle displacement: {ank_r:.3f} m."
        )
        measured_values = [
            {"label": "Step Width (Ankle Distance)", "value": f"{sw:.3f} m"},
            {"label": "Step Width Frame Change (Δ)", "value": f"{d_sw:+.3f} m"},
            {"label": "Left Ankle Displacement", "value": f"{ank_l:.3f} m"},
            {"label": "Right Ankle Displacement", "value": f"{ank_r:.3f} m"},
        ]

    elif "arm" in primary_lower:
        title = "Arm Swing Range of Motion & Upper Limb Motion"
        reason_selected = (
            f"Frame #{actual_frame_idx} (timestamp {ts:.2f}s) was selected due to high kinematic variation in upper limb "
            f"arm swing and wrist displacement (variance score: {score:.4f})."
        )
        analysis_summary = (
            f"Left arm angle was measured at {la:.1f}° (Δ: {dla:+.1f}°) and right arm angle at {ra:.1f}° (Δ: {dra:+.1f}°). "
            f"Wrist displacement was {arm_l:.3f} m on the left and {arm_r:.3f} m on the right (arm swing difference: {arm_diff:.3f} m)."
        )
        measured_values = [
            {"label": "Left Arm Angle", "value": f"{la:.1f}°"},
            {"label": "Right Arm Angle", "value": f"{ra:.1f}°"},
            {"label": "Left Wrist Displacement", "value": f"{arm_l:.3f} m"},
            {"label": "Right Wrist Displacement", "value": f"{arm_r:.3f} m"},
            {"label": "Arm Swing Difference", "value": f"{arm_diff:.3f} m"},
        ]

    elif "hip" in primary_lower or "torso" in primary_lower or "postural" in primary_lower:
        title = "Hip Flexion & Postural Trunk Alignment"
        reason_selected = (
            f"Frame #{actual_frame_idx} (timestamp {ts:.2f}s) was selected to evaluate hip joint extension "
            f"and trunk postural alignment (variance score: {score:.4f})."
        )
        analysis_summary = (
            f"Left hip angle was measured at {lh:.1f}° and right hip angle at {rh:.1f}°. "
            f"Postural torso tilt angle relative to vertical was recorded at {torso_a:.1f}°."
        )
        measured_values = [
            {"label": "Left Hip Angle", "value": f"{lh:.1f}°"},
            {"label": "Right Hip Angle", "value": f"{rh:.1f}°"},
            {"label": "Torso Posture Tilt", "value": f"{torso_a:.1f}°"},
            {"label": "Left-Right Hip Difference", "value": f"{abs(lh - rh):.1f}°"},
        ]

    else:
        title = "Bilateral Kinematic Gait Asymmetry"
        reason_selected = (
            f"Frame #{actual_frame_idx} (timestamp {ts:.2f}s) was selected because it recorded peak left-right "
            f"bilateral movement variation across gait markers (variance score: {score:.4f})."
        )
        analysis_summary = (
            f"Bilateral knee angle asymmetry was measured at {asym:.1f}° (Left: {lk:.1f}°, Right: {rk:.1f}°). "
            f"Upper limb wrist movement difference was {arm_diff:.3f} m."
        )
        measured_values = [
            {"label": "Bilateral Knee Asymmetry", "value": f"{asym:.1f}°"},
            {"label": "Left Knee Angle", "value": f"{lk:.1f}°"},
            {"label": "Right Knee Angle", "value": f"{rk:.1f}°"},
            {"label": "Arm Swing Movement Difference", "value": f"{arm_diff:.3f} m"},
            {"label": "Kinematic Variance Score", "value": f"{score:.4f}"},
        ]

    return {
        "title": title,
        "reason_selected": reason_selected,
        "analysis_summary": analysis_summary,
        "measured_values": measured_values,
    }


# --------------------------------------------------------------------------
# Skeleton Drawing & Clean Screenshot Generator
# --------------------------------------------------------------------------

def _is_valid_landmark(landmark_px: Dict[int, tuple], idx: int) -> bool:
    """Check if a landmark has valid pixel coordinates."""
    if idx not in landmark_px:
        return False
    x, y = landmark_px[idx]
    return x >= 0 and y >= 0


def _extract_frame_from_video(video_path: str | Path,
                              actual_frame_idx: int,
                              image_coords: Dict[int, np.ndarray],
                              world_coords: Dict[int, np.ndarray],
                              primary_indicator: str,
                              metrics: Dict[str, Any],
                              output_path: Path) -> str | None:
    """Extract a single frame from video, draw MediaPipe skeleton and clean visual overlays,
    and save as a paper-ready clean JPG figure.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return None

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_frame_idx = max(0, min(actual_frame_idx, total_video_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)

    ok, frame_bgr = cap.read()
    cap.release()

    if not ok or frame_bgr is None:
        logger.error("Cannot read frame %d from %s", actual_frame_idx, video_path)
        return None

    original_output_path = output_path.with_name(f"{output_path.stem}_original.jpg")
    cv2.imwrite(str(original_output_path), frame_bgr)

    h, w = frame_bgr.shape[:2]

    # Map normalized 2D image coordinates to exact pixel coordinates
    px_map: Dict[int, Tuple[int, int]] = {}
    for idx, lm in image_coords.items():
        if len(lm) >= 2:
            x_px = int(np.clip(lm[0] * w, 0, w - 1))
            y_px = int(np.clip(lm[1] * h, 0, h - 1))
            px_map[idx] = (x_px, y_px)

    # Skeleton connections
    left_connections = [
        (LANDMARK_LEFT_SHOULDER, LANDMARK_LEFT_ELBOW),
        (LANDMARK_LEFT_ELBOW, LANDMARK_LEFT_WRIST),
        (LANDMARK_LEFT_HIP, LANDMARK_LEFT_KNEE),
        (LANDMARK_LEFT_KNEE, LANDMARK_LEFT_ANKLE),
        (LANDMARK_LEFT_ANKLE, LANDMARK_LEFT_HEEL),
        (LANDMARK_LEFT_HEEL, LANDMARK_LEFT_FOOT_INDEX),
    ]

    right_connections = [
        (LANDMARK_RIGHT_SHOULDER, LANDMARK_RIGHT_ELBOW),
        (LANDMARK_RIGHT_ELBOW, LANDMARK_RIGHT_WRIST),
        (LANDMARK_RIGHT_HIP, LANDMARK_RIGHT_KNEE),
        (LANDMARK_RIGHT_KNEE, LANDMARK_RIGHT_ANKLE),
        (LANDMARK_RIGHT_ANKLE, LANDMARK_RIGHT_HEEL),
        (LANDMARK_RIGHT_HEEL, LANDMARK_RIGHT_FOOT_INDEX),
    ]

    torso_connections = [
        (LANDMARK_LEFT_SHOULDER, LANDMARK_RIGHT_SHOULDER),
        (LANDMARK_LEFT_HIP, LANDMARK_RIGHT_HIP),
        (LANDMARK_LEFT_SHOULDER, LANDMARK_LEFT_HIP),
        (LANDMARK_RIGHT_SHOULDER, LANDMARK_RIGHT_HIP),
    ]

    # Draw Torso (Yellow)
    for a, b in torso_connections:
        if a in px_map and b in px_map:
            cv2.line(frame_bgr, px_map[a], px_map[b], (0, 230, 255), 3, cv2.LINE_AA)

    # Draw Left Side (Green)
    for a, b in left_connections:
        if a in px_map and b in px_map:
            cv2.line(frame_bgr, px_map[a], px_map[b], (0, 255, 128), 3, cv2.LINE_AA)

    # Draw Right Side (Cyan/Orange)
    for a, b in right_connections:
        if a in px_map and b in px_map:
            cv2.line(frame_bgr, px_map[a], px_map[b], (255, 200, 0), 3, cv2.LINE_AA)

    # Draw joint circles
    for idx, pt in px_map.items():
        if idx in [LANDMARK_LEFT_KNEE, LANDMARK_RIGHT_KNEE, LANDMARK_LEFT_ANKLE, LANDMARK_RIGHT_ANKLE,
                   LANDMARK_LEFT_WRIST, LANDMARK_RIGHT_WRIST, LANDMARK_LEFT_HIP, LANDMARK_RIGHT_HIP,
                   LANDMARK_LEFT_SHOULDER, LANDMARK_RIGHT_SHOULDER, LANDMARK_LEFT_ELBOW, LANDMARK_RIGHT_ELBOW]:
            color = (0, 255, 255) if idx in [LANDMARK_LEFT_KNEE, LANDMARK_RIGHT_KNEE] else (255, 255, 255)
            cv2.circle(frame_bgr, pt, 6, color, -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, pt, 8, (0, 0, 0), 2, cv2.LINE_AA)

    # Feature-specific Visual Annotations over joints (clean)
    primary_lower = primary_indicator.lower()

    if "knee" in primary_lower:
        for k_idx in [LANDMARK_LEFT_KNEE, LANDMARK_RIGHT_KNEE]:
            if k_idx in px_map:
                cv2.circle(frame_bgr, px_map[k_idx], 12, (0, 255, 255), 3, cv2.LINE_AA)

        if LANDMARK_LEFT_KNEE in px_map:
            pt = px_map[LANDMARK_LEFT_KNEE]
            val = metrics.get("knee_angle_left", 0.0)
            cv2.putText(frame_bgr, f"L Knee: {val:.1f} deg", (pt[0] - 85, max(25, pt[1] - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2, cv2.LINE_AA)

        if LANDMARK_RIGHT_KNEE in px_map:
            pt = px_map[LANDMARK_RIGHT_KNEE]
            val = metrics.get("knee_angle_right", 0.0)
            cv2.putText(frame_bgr, f"R Knee: {val:.1f} deg", (pt[0] + 15, max(25, pt[1] - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2, cv2.LINE_AA)

    elif "step" in primary_lower or "ankle" in primary_lower:
        if LANDMARK_LEFT_ANKLE in px_map and LANDMARK_RIGHT_ANKLE in px_map:
            pt_l = px_map[LANDMARK_LEFT_ANKLE]
            pt_r = px_map[LANDMARK_RIGHT_ANKLE]
            cv2.line(frame_bgr, pt_l, pt_r, (255, 0, 255), 3, cv2.LINE_AA)
            cv2.circle(frame_bgr, pt_l, 10, (255, 0, 255), 3, cv2.LINE_AA)
            cv2.circle(frame_bgr, pt_r, 10, (255, 0, 255), 3, cv2.LINE_AA)
            mid_x = (pt_l[0] + pt_r[0]) // 2
            mid_y = max(30, (pt_l[1] + pt_r[1]) // 2 - 15)
            sw_val = metrics.get("step_width", 0.0)
            cv2.putText(frame_bgr, f"Step Width: {sw_val:.2f}m", (mid_x - 65, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)

    elif "arm" in primary_lower:
        for w_idx, color in [(LANDMARK_LEFT_WRIST, (0, 255, 128)), (LANDMARK_RIGHT_WRIST, (255, 200, 0))]:
            if w_idx in px_map:
                pt = px_map[w_idx]
                cv2.circle(frame_bgr, pt, 10, color, 3, cv2.LINE_AA)
                elbow_idx = LANDMARK_LEFT_ELBOW if w_idx == LANDMARK_LEFT_WRIST else LANDMARK_RIGHT_ELBOW
                if elbow_idx in px_map:
                    pt_e = px_map[elbow_idx]
                    dir_vec = (pt[0] - pt_e[0], pt[1] - pt_e[1])
                    norm = max(1e-5, math.hypot(dir_vec[0], dir_vec[1]))
                    arrow_target = (int(pt[0] + (dir_vec[0] / norm) * 25), int(pt[1] + (dir_vec[1] / norm) * 25))
                    cv2.arrowedLine(frame_bgr, pt, arrow_target, color, 2, cv2.LINE_AA, tipLength=0.3)

    elif "hip" in primary_lower or "torso" in primary_lower or "postural" in primary_lower:
        if LANDMARK_LEFT_HIP in px_map and LANDMARK_RIGHT_HIP in px_map:
            cv2.circle(frame_bgr, px_map[LANDMARK_LEFT_HIP], 10, (0, 230, 255), 3, cv2.LINE_AA)
            cv2.circle(frame_bgr, px_map[LANDMARK_RIGHT_HIP], 10, (0, 230, 255), 3, cv2.LINE_AA)

    # Save clean JPG image (No bottom HUD text overlay inside image)
    ensure_dir(output_path.parent)
    ok = cv2.imwrite(str(output_path), frame_bgr)
    if ok:
        return str(output_path)
    logger.error("cv2.imwrite failed for %s", output_path)
    return None


def _create_zoom_crop(video_path: Path, actual_frame_idx: int,
                      image_coords: Dict[int, np.ndarray],
                      primary_indicator: str,
                      output_dir: Path) -> str | None:
    """Create a zoomed crop of the relevant body region from the key frame."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_frame_idx = max(0, min(actual_frame_idx, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)

    ok, frame_bgr = cap.read()
    cap.release()

    if not ok or frame_bgr is None:
        return None

    h, w = frame_bgr.shape[:2]
    primary = primary_indicator.lower()

    if "knee" in primary:
        indices = [LANDMARK_LEFT_HIP, LANDMARK_RIGHT_HIP, LANDMARK_LEFT_KNEE, LANDMARK_RIGHT_KNEE, LANDMARK_LEFT_ANKLE, LANDMARK_RIGHT_ANKLE]
    elif "step" in primary or "ankle" in primary:
        indices = [LANDMARK_LEFT_ANKLE, LANDMARK_RIGHT_ANKLE, LANDMARK_LEFT_HEEL, LANDMARK_RIGHT_HEEL, LANDMARK_LEFT_FOOT_INDEX, LANDMARK_RIGHT_FOOT_INDEX]
    elif "arm" in primary:
        indices = [LANDMARK_LEFT_SHOULDER, LANDMARK_RIGHT_SHOULDER, LANDMARK_LEFT_ELBOW, LANDMARK_RIGHT_ELBOW, LANDMARK_LEFT_WRIST, LANDMARK_RIGHT_WRIST]
    else:
        indices = [LANDMARK_LEFT_SHOULDER, LANDMARK_RIGHT_SHOULDER, LANDMARK_LEFT_HIP, LANDMARK_RIGHT_HIP, LANDMARK_LEFT_KNEE, LANDMARK_RIGHT_KNEE]

    pts_x, pts_y = [], []
    for idx in indices:
        if idx in image_coords:
            lm = image_coords[idx]
            pts_x.append(int(lm[0] * w))
            pts_y.append(int(lm[1] * h))

    if not pts_x or not pts_y:
        min_x, max_x, min_y, max_y = 0, w, 0, h
    else:
        pad = 40
        min_x = max(0, min(pts_x) - pad)
        max_x = min(w, max(pts_x) + pad)
        min_y = max(0, min(pts_y) - pad)
        max_y = min(h, max(pts_y) + pad)

    if max_x > min_x and max_y > min_y:
        crop = frame_bgr[min_y:max_y, min_x:max_x]
        crop_resized = cv2.resize(crop, (320, 240))
        zoom_filename = f"frame_{actual_frame_idx:04d}_zoom.jpg"
        ensure_dir(output_dir)
        zoom_path = output_dir / zoom_filename
        if cv2.imwrite(str(zoom_path), crop_resized):
            return str(zoom_path)

    return None


# --------------------------------------------------------------------------
# MediaPipe landmark extraction
# --------------------------------------------------------------------------

def extract_landmarks_from_video(
    video_path: str | Path,
    model_complexity: int = _MP_MODEL_COMPLEXITY,
    min_detection_confidence: float = _MP_MIN_DETECTION,
    min_tracking_confidence: float = _MP_MIN_TRACKING,
) -> Tuple[List[Dict[int, np.ndarray]], List[Dict[int, np.ndarray]], List[Dict[int, float]], List[int], float, int]:
    """Extract 3D world landmarks and 2D normalized image landmarks from a video."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        logger.warning("FPS not available from video metadata, defaulting to 30.0")
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = -1

    logger.info("Extracting MediaPipe Pose landmarks: %s | FPS: %.2f", video_path.name, fps)

    mp_pose = mp.solutions.pose
    world_coords_per_frame: List[Dict[int, np.ndarray]] = []
    image_coords_per_frame: List[Dict[int, np.ndarray]] = []
    visibility_per_frame: List[Dict[int, float]] = []
    frame_indices: List[int] = []

    frame_no = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose_model:
        while cap.isOpened():
            ok, frame_bgr = cap.read()
            if not ok:
                break

            curr_frame_idx = frame_no
            frame_no += 1

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = pose_model.process(frame_rgb)

            world_source = results.pose_world_landmarks
            image_source = results.pose_landmarks

            if world_source is None and image_source is None:
                continue

            world_dict: Dict[int, np.ndarray] = {}
            image_dict: Dict[int, np.ndarray] = {}
            vis_dict: Dict[int, float] = {}

            if world_source is not None:
                for idx, lm in enumerate(world_source.landmark):
                    world_dict[idx] = np.array([lm.x, lm.y, lm.z], dtype=np.float64)

            if image_source is not None:
                for idx, lm in enumerate(image_source.landmark):
                    image_dict[idx] = np.array([lm.x, lm.y, lm.z], dtype=np.float64)
                    vis_dict[idx] = float(getattr(lm, "visibility", 1.0))
                    if idx not in world_dict:
                        world_dict[idx] = image_dict[idx]

            if any(not np.all(np.isfinite(v)) for v in world_dict.values()):
                continue

            world_coords_per_frame.append(world_dict)
            image_coords_per_frame.append(image_dict)
            visibility_per_frame.append(vis_dict)
            frame_indices.append(curr_frame_idx)

    cap.release()

    detected_count = len(world_coords_per_frame)
    logger.info("MediaPipe extraction complete: %d total frames read, %d frames detected", frame_no, detected_count)

    if detected_count == 0:
        raise RuntimeError("MediaPipe could not detect any pose landmarks in this video.")

    if detected_count < _MIN_FRAMES_REQUIRED:
        raise RuntimeError(f"MediaPipe detected landmarks in only {detected_count} frames (minimum required: {_MIN_FRAMES_REQUIRED}).")

    return world_coords_per_frame, image_coords_per_frame, visibility_per_frame, frame_indices, fps, frame_no


# --------------------------------------------------------------------------
# SMPL dataframe builder
# --------------------------------------------------------------------------

def build_smpl_dataframe(
    coords_per_frame: List[Dict[int, np.ndarray]],
    visibility_per_frame: List[Dict[int, float]],
) -> pd.DataFrame:
    """Convert per-frame MediaPipe landmark data into a SMPL rotation matrix DataFrame."""
    n_frames = len(coords_per_frame)
    logger.info("Building SMPL rotation matrix DataFrame for %d frames...", n_frames)

    rows = []
    invalid_frames = 0

    for i, (coords, vis) in enumerate(zip(coords_per_frame, visibility_per_frame)):
        try:
            rotations = mediapipe_to_smpl_rotations(coords, vis)
            if not np.all(np.isfinite(rotations)):
                rotations = np.tile(np.eye(3), (24, 1, 1)).reshape(24, 3, 3)
                invalid_frames += 1

            row = rotations_to_smpl_dataframe_row(rotations)
            rows.append(row)
        except Exception as exc:
            logger.warning("Frame %d rotation failure: %s", i, exc)
            invalid_frames += 1
            continue

    if not rows:
        raise RuntimeError("All frames failed rotation estimation.")

    df = pd.DataFrame(rows)
    if df.isnull().any().any() or np.any(~np.isfinite(df.values)):
        raise RuntimeError("SMPL DataFrame contains invalid values (NaN/Inf).")

    return df


# --------------------------------------------------------------------------
# Main pipeline class
# --------------------------------------------------------------------------

class MediaPipePipeline:
    """End-to-end MediaPipe-based Parkinson's gait analysis pipeline."""

    def __init__(
        self,
        pd_model_path: Optional[str | Path] = None,
        severity_model_path: Optional[str | Path] = None,
        mp_model_complexity: int = _MP_MODEL_COMPLEXITY,
        mp_min_detection: float = _MP_MIN_DETECTION,
        mp_min_tracking: float = _MP_MIN_TRACKING,
    ) -> None:
        self.pd_model_path = Path(pd_model_path or _DEFAULT_PD_MODEL)
        self.severity_model_path = Path(severity_model_path or _DEFAULT_SEV_MODEL)
        self.mp_model_complexity = mp_model_complexity
        self.mp_min_detection = mp_min_detection
        self.mp_min_tracking = mp_min_tracking

        self._pd_model: Any = None
        self._sev_model: Any = None
        self._feature_extractor = SMPLFeatureExtractor()
        self._explainer = ModelExplainer()

    def _load_models(self) -> None:
        if self._pd_model is None:
            if not self.pd_model_path.exists():
                raise FileNotFoundError(f"PD Detector model not found: {self.pd_model_path}")
            self._pd_model = joblib.load(self.pd_model_path)

        if self._sev_model is None:
            if not self.severity_model_path.exists():
                raise FileNotFoundError(f"Severity model not found: {self.severity_model_path}")
            self._sev_model = joblib.load(self.severity_model_path)

    def _align_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        pd_expected = getattr(self._pd_model, "feature_names_in_", None)
        if pd_expected is not None:
            expected_list = list(pd_expected)
            present = set(feature_df.columns)
            missing = set(expected_list) - present
            if missing:
                raise ValueError(f"Feature alignment failed: {len(missing)} missing features.")
            feature_df = feature_df[expected_list]
        return feature_df

    def run(
        self,
        video_path: str | Path,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute full MediaPipe pipeline on walking video."""
        pipeline_start = time.time()
        video_path = Path(video_path)

        logger.info("MEDIAPIPE PIPELINE — Job: %s | Video: %s", job_id or "N/A", video_path.name)

        # 1. Load models
        self._load_models()

        # 2. Extract landmarks
        mp_start = time.time()
        (world_coords_per_frame, image_coords_per_frame, visibility_per_frame,
         frame_indices, fps, total_video_frames) = extract_landmarks_from_video(
            video_path,
            model_complexity=self.mp_model_complexity,
            min_detection_confidence=self.mp_min_detection,
            min_tracking_confidence=self.mp_min_tracking,
        )
        mp_time = time.time() - mp_start
        num_detected_frames = len(world_coords_per_frame)

        # 3. Build SMPL rotation matrices
        smpl_df = build_smpl_dataframe(world_coords_per_frame, visibility_per_frame)
        self.last_smpl_df = smpl_df

        # 4. Extract 290 gait features
        feat_start = time.time()
        feature_df = self._feature_extractor.extract_features(smpl_df, fps=fps)
        _check_feature_vector(feature_df)
        feature_df = self._align_features(feature_df)
        self.last_feature_df = feature_df
        feat_time = time.time() - feat_start

        # 5. ML Inference
        pd_probs = self._pd_model.predict_proba(feature_df)[0]
        prediction_summary = summarize_binary_prediction(self._pd_model.classes_, pd_probs)
        pd_pred_class = prediction_summary["label"]
        pd_probability = prediction_summary["parkinsons_probability"]
        if pd_probability is None:
            pd_probability = prediction_summary["confidence"]
        diagnosis_name = prediction_summary["diagnosis"]
        risk_level = "Not assessed"

        sev_pred_class = int(self._sev_model.predict(feature_df)[0])
        sev_desc = UPDRS_GAIT_DESCRIPTIONS.get(sev_pred_class, f"UPDRS Gait Score {sev_pred_class}")
        sev_probs = None
        if hasattr(self._sev_model, "predict_proba"):
            sev_probs = [float(p) for p in self._sev_model.predict_proba(feature_df)[0]]

        explainable_features = self._explainer.explain(self._pd_model, feature_df, top_k=5)

        # 6. Compute Real Per-Frame Gait Metrics
        knee_angles_left = []
        knee_angles_right = []
        hip_angles_left = []
        hip_angles_right = []
        arm_angles_left = []
        arm_angles_right = []
        step_widths = []
        arm_movements_left = []
        arm_movements_right = []
        ankle_displacements_left = []
        ankle_displacements_right = []
        torso_angles = []
        frame_asymmetries = []

        for i in range(num_detected_frames):
            w_coords = world_coords_per_frame[i]
            prev_w = world_coords_per_frame[i - 1] if i > 0 else None

            lk = _compute_knee_angle(w_coords, LANDMARK_LEFT_HIP, LANDMARK_LEFT_KNEE, LANDMARK_LEFT_ANKLE)
            rk = _compute_knee_angle(w_coords, LANDMARK_RIGHT_HIP, LANDMARK_RIGHT_KNEE, LANDMARK_RIGHT_ANKLE)
            knee_angles_left.append(lk)
            knee_angles_right.append(rk)

            lh = _compute_hip_angle(w_coords, LANDMARK_LEFT_SHOULDER, LANDMARK_LEFT_HIP, LANDMARK_LEFT_KNEE)
            rh = _compute_hip_angle(w_coords, LANDMARK_RIGHT_SHOULDER, LANDMARK_RIGHT_HIP, LANDMARK_RIGHT_KNEE)
            hip_angles_left.append(lh)
            hip_angles_right.append(rh)

            la = _compute_arm_angle(w_coords, is_left=True)
            ra = _compute_arm_angle(w_coords, is_left=False)
            arm_angles_left.append(la)
            arm_angles_right.append(ra)

            sw = _compute_step_width(w_coords)
            step_widths.append(sw)

            arm_m_left = _compute_frame_displacement(w_coords, prev_w, [LANDMARK_LEFT_WRIST])
            arm_m_right = _compute_frame_displacement(w_coords, prev_w, [LANDMARK_RIGHT_WRIST])
            arm_movements_left.append(arm_m_left)
            arm_movements_right.append(arm_m_right)

            ank_m_left = _compute_frame_displacement(w_coords, prev_w, [LANDMARK_LEFT_ANKLE])
            ank_m_right = _compute_frame_displacement(w_coords, prev_w, [LANDMARK_RIGHT_ANKLE])
            ankle_displacements_left.append(ank_m_left)
            ankle_displacements_right.append(ank_m_right)

            torso_a = _compute_torso_angle(w_coords)
            torso_angles.append(torso_a)

            asym = abs(lk - rk)
            frame_asymmetries.append(asym)

        # 7. Select Key Gait Frames
        frame_scores = compute_frame_gait_scores(world_coords_per_frame, fps)
        key_frames = select_key_gait_frames(frame_scores, fps, n_frames=5)

        # Target output folder for screenshots
        gait_frames_dir = _PROJECT_ROOT / "static" / "gait_frames" / str(job_id) if job_id else _PROJECT_ROOT / "static" / "gait_frames"
        ensure_dir(gait_frames_dir)

        key_gait_frames_result = []

        for frame_info in key_frames:
            det_idx = int(frame_info["frame"])
            actual_video_frame = frame_indices[det_idx] if det_idx < len(frame_indices) else det_idx
            primary_indicator = frame_info["primary_indicator"]
            secondary_indicator = frame_info.get("secondary_indicator", "")

            prev_det_idx = max(0, det_idx - 1)

            frame_metrics = {
                "timestamp": round(actual_video_frame / fps, 2),
                "score": float(frame_info["score"]),
                "knee_angle_left": knee_angles_left[det_idx],
                "knee_angle_right": knee_angles_right[det_idx],
                "delta_knee_left": knee_angles_left[det_idx] - knee_angles_left[prev_det_idx],
                "delta_knee_right": knee_angles_right[det_idx] - knee_angles_right[prev_det_idx],
                "hip_angle_left": hip_angles_left[det_idx],
                "hip_angle_right": hip_angles_right[det_idx],
                "arm_angle_left": arm_angles_left[det_idx],
                "arm_angle_right": arm_angles_right[det_idx],
                "delta_arm_left": arm_angles_left[det_idx] - arm_angles_left[prev_det_idx],
                "delta_arm_right": arm_angles_right[det_idx] - arm_angles_right[prev_det_idx],
                "step_width": step_widths[det_idx],
                "delta_step_width": step_widths[det_idx] - step_widths[prev_det_idx],
                "left_arm_movement": arm_movements_left[det_idx],
                "right_arm_movement": arm_movements_right[det_idx],
                "ankle_move_left": ankle_displacements_left[det_idx],
                "ankle_move_right": ankle_displacements_right[det_idx],
                "torso_angle": torso_angles[det_idx],
                "asymmetry": frame_asymmetries[det_idx],
            }

            img_output_path = gait_frames_dir / f"frame_{actual_video_frame:04d}.jpg"

            saved_path = _extract_frame_from_video(
                video_path=video_path,
                actual_frame_idx=actual_video_frame,
                image_coords=image_coords_per_frame[det_idx],
                world_coords=world_coords_per_frame[det_idx],
                primary_indicator=primary_indicator,
                metrics=frame_metrics,
                output_path=img_output_path,
            )

            zoom_saved_path = _create_zoom_crop(
                video_path=video_path,
                actual_frame_idx=actual_video_frame,
                image_coords=image_coords_per_frame[det_idx],
                primary_indicator=primary_indicator,
                output_dir=gait_frames_dir,
            )

            image_url = f"/static/gait_frames/{job_id}/frame_{actual_video_frame:04d}.jpg" if job_id else f"/static/gait_frames/frame_{actual_video_frame:04d}.jpg"
            zoom_url = f"/static/gait_frames/{job_id}/frame_{actual_video_frame:04d}_zoom.jpg" if job_id else f"/static/gait_frames/frame_{actual_video_frame:04d}_zoom.jpg"

            vis_dict = visibility_per_frame[det_idx] if det_idx < len(visibility_per_frame) else {}
            conf = min(vis_dict.values()) if vis_dict else 1.0

            # Generate structured explanation metadata for frontend rendering
            explanation_meta = _build_frame_explanation_metadata(
                primary_indicator=primary_indicator,
                secondary_indicator=secondary_indicator,
                metrics=frame_metrics,
                actual_frame_idx=actual_video_frame,
            )

            key_gait_frames_result.append({
                "frame_index": actual_video_frame,
                "timestamp": frame_metrics["timestamp"],
                "score": round(frame_metrics["score"], 4),
                "primary_indicator": primary_indicator,
                "secondary_indicator": secondary_indicator,
                "title": explanation_meta["title"],
                "reason_selected": explanation_meta["reason_selected"],
                "analysis_summary": explanation_meta["analysis_summary"],
                "measured_values": explanation_meta["measured_values"],
                "original_image_path": image_url.replace(".jpg", "_original.jpg") if saved_path else None,
                "image_path": image_url if saved_path else None,
                "zoom_image_path": zoom_url if zoom_saved_path else None,
                "left_arm_movement": round(frame_metrics["left_arm_movement"], 4),
                "right_arm_movement": round(frame_metrics["right_arm_movement"], 4),
                "arm_difference": round(abs(frame_metrics["left_arm_movement"] - frame_metrics["right_arm_movement"]), 4),
                "confidence": round(conf, 4),
                "knee_angle_left": round(frame_metrics["knee_angle_left"], 2),
                "knee_angle_right": round(frame_metrics["knee_angle_right"], 2),
                "ankle_move_left": round(frame_metrics["ankle_move_left"], 4),
                "ankle_move_right": round(frame_metrics["ankle_move_right"], 4),
                "asymmetry": round(frame_metrics["asymmetry"], 2),
            })

        gait_metrics = {
            "left_knee_angle_mean": round(float(np.nanmean(knee_angles_left)), 2),
            "right_knee_angle_mean": round(float(np.nanmean(knee_angles_right)), 2),
            "left_arm_movement_mean": round(float(np.nanmean(arm_movements_left)), 4),
            "right_arm_movement_mean": round(float(np.nanmean(arm_movements_right)), 4),
            "arm_movement_difference_mean": round(abs(float(np.nanmean(arm_movements_left)) - float(np.nanmean(arm_movements_right))), 4),
            "left_right_asymmetry_mean": round(float(np.nanmean(frame_asymmetries)), 2),
            "step_width_mean": round(float(np.nanmean(step_widths)), 4),
            "measured_frame_count": num_detected_frames,
        }
        sample_step = max(1, math.ceil(num_detected_frames / 120))
        gait_timeseries = [
            {
                "timestamp": round((frame_indices[index] if index < len(frame_indices) else index) / fps, 2),
                "knee_left": round(knee_angles_left[index], 2),
                "knee_right": round(knee_angles_right[index], 2),
                "left_arm": round(arm_movements_left[index], 4),
                "right_arm": round(arm_movements_right[index], 4),
            }
            for index in range(0, num_detected_frames, sample_step)
        ]

        total_time = time.time() - pipeline_start
        visibility_values = []
        for vis in visibility_per_frame:
            visibility_values.extend(list(vis.values()))
        pose_quality = 100.0
        if visibility_values:
            pose_quality = round(float(np.nanmean(np.asarray(visibility_values, dtype=float)) * 100.0), 2)
        valid_landmark_pct = 100.0 if not visibility_values else round(float(np.mean(np.asarray(visibility_values, dtype=float) >= 0.5) * 100.0), 2)
        analysis_duration = float(num_detected_frames / fps) if fps > 0 else 0.0

        model_information = {
            "model": "Random Forest",
            "n_features": int(getattr(self._pd_model, "n_features_in_", feature_df.shape[1])),
            "n_trees": int(getattr(self._pd_model, "n_estimators", 0) or 0),
            "classes": [str(cls) for cls in getattr(self._pd_model, "classes_", [])],
        }
        dataset_performance = _load_model_dataset_performance()

        return {
            "status": "success",
            "pipeline": "mediapipe",
            "job_id": job_id,
            "video_file": video_path.name,
            "num_frames": num_detected_frames,
            "fps": round(fps, 2),
            "video_duration_seconds": round(analysis_duration, 2),
            "processing_time_seconds": round(total_time, 2),
            "mediapipe_time_seconds": round(mp_time, 2),
            "feature_time_seconds": round(feat_time, 2),
            "pose_detection_quality": pose_quality,
            "valid_landmark_percentage": valid_landmark_pct,
            "parkinsons_detection": {
                "label": pd_pred_class,
                "diagnosis": diagnosis_name,
                "probability": round(pd_probability * 100, 1),
                "probability_raw": round(pd_probability, 4),
                "confidence": round(prediction_summary["confidence"] * 100, 1),
                "risk_level": risk_level,
                "class_probabilities": {
                    label: round(probability * 100, 1)
                    for label, probability in prediction_summary["class_probabilities"].items()
                },
            },
            "severity_estimation": {
                "updrs_gait_score": sev_pred_class,
                "description": sev_desc,
                "class_probabilities": sev_probs,
            },
            "explainability": {
                "important_gait_indicators": explainable_features,
            },
            "metadata": {
                "num_frames": num_detected_frames,
                "fps": round(fps, 2),
                "video_duration_seconds": round(analysis_duration, 2),
                "features_extracted": int(feature_df.shape[1]),
                "pd_model_features": int(getattr(self._pd_model, "n_features_in_", 290)),
                "severity_model_features": int(getattr(self._sev_model, "n_features_in_", 290)),
                "mediapipe_model_complexity": self.mp_model_complexity,
                "feature_nan_count": int(np.sum(np.isnan(feature_df.values))),
                "feature_inf_count": int(np.sum(np.isinf(feature_df.values))),
                "pose_detection_quality": pose_quality,
                "valid_landmark_percentage": valid_landmark_pct,
            },
            "model_information": model_information,
            "dataset_performance": dataset_performance,
            "disclaimer": (
                "ParkinSense is a research prototype intended for gait analysis research and educational purposes. "
                "It is not a clinical diagnostic tool and should not be used to diagnose or make medical decisions."
            ),
            "key_gait_frames": key_gait_frames_result,
            "gait_metrics": gait_metrics,
            "gait_timeseries": gait_timeseries,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="ParkinSense — MediaPipe Inference Pipeline")
    parser.add_argument("--video", "-v", required=True, help="Path to walking video")
    parser.add_argument("--complexity", "-c", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--output-json", "-o", type=str, default=None)
    args = parser.parse_args()

    pipeline = MediaPipePipeline(mp_model_complexity=args.complexity)

    try:
        result = pipeline.run(args.video)
    except Exception as err:
        print(f"\nPIPELINE ERROR: {err}", file=sys.stderr)
        logger.exception("Pipeline execution failed")
        sys.exit(1)

    print()
    print("=" * 65)
    print("             PARKINSENSE MEDIAPIPE RESULT")
    print("=" * 65)
    print(f"  Pipeline       : {result['pipeline'].upper()}")
    print(f"  Video          : {result['video_file']}")
    print(f"  Frames         : {result['num_frames']}")
    print(f"  FPS            : {result['fps']}")
    print(f"  MediaPipe time : {result['mediapipe_time_seconds']}s")
    print(f"  Feature time   : {result['feature_time_seconds']}s")
    print(f"  Total time     : {result['processing_time_seconds']}s")
    print("-" * 65)
    pd_info = result["parkinsons_detection"]
    print(f"  Diagnosis      : {pd_info['diagnosis']}")
    print(f"  Probability    : {pd_info['probability']}%")
    print(f"  Risk Level     : {pd_info['risk_level']}")
    print("-" * 65)
    sev_info = result["severity_estimation"]
    print(f"  Severity       : {sev_info['description']}")
    print("-" * 65)
    print(f"  Key Gait Frames: {len(result['key_gait_frames'])} generated")
    for frame in result["key_gait_frames"]:
        print(f"    - Frame #{frame['frame_index']} ({frame['timestamp']}s): {frame['title']} -> {frame['image_path']}")
    print("=" * 65)

    if args.output_json:
        out_path = Path(args.output_json)
        ensure_dir(out_path.parent)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Result saved to: {out_path}")


if __name__ == "__main__":
    main()
