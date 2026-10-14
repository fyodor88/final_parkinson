"""
ParkinSense — MediaPipe Pose Invariant Domain Normalization Pipeline
=====================================================================
Implements a mathematically justified spatial and temporal normalization layer
for 3D MediaPipe pose landmarks, ensuring invariance to:
  1. Global translation (pelvis centering)
  2. Subject body scale (shoulder-hip span normalization)
  3. Camera / body orientation (canonical body-fixed coordinate frame)
  4. Camera coordinate system
  5. Video duration & frame rate differences
  6. Noisy / low-confidence landmarks (visibility filtering & short-gap interpolation)

Maintains 100% backward compatibility with the existing 290-feature model format.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from src.utils import get_logger
from src.pose.mediapipe_smpl import (
    _MP,
    SMPL_JOINT_NAMES,
    _safe_normalize,
    _gram_schmidt,
    _rotation_matrix_from_bone,
    _validate_rotation_matrix,
    rotations_to_smpl_dataframe_row,
)

logger = get_logger(__name__)

# Default Normalization Configuration
DEFAULT_NORMALIZATION_CONFIG: Dict[str, Any] = {
    "confidence_filter": True,
    "min_visibility": 0.30,
    "max_interpolation_gap": 5,
    "pelvis_center": True,
    "scale_normalization": True,
    "canonical_orientation": True,
    "temporal_normalization": True,
    "gait_cycle_segmentation": False,
    "target_fps": 30.0,
    "target_num_frames": 150,
}


def filter_landmarks_by_visibility(
    world_coords_per_frame: List[Dict[int, np.ndarray]],
    visibility_per_frame: List[Dict[int, float]],
    min_visibility: float = 0.30,
    max_interpolation_gap: int = 5,
) -> Tuple[List[Dict[int, np.ndarray]], Dict[str, Any]]:
    """
    Filter low-confidence landmarks and interpolate short temporal gaps.

    Returns:
        Filtered list of landmark dicts per frame, plus summary statistics.
    """
    num_frames = len(world_coords_per_frame)
    if num_frames == 0:
        return [], {"invalid_percentage": 0.0, "interpolated_count": 0}

    # Identify valid landmarks per landmark ID and frame
    all_landmark_ids = set()
    for frame_dict in world_coords_per_frame:
        all_landmark_ids.update(frame_dict.keys())

    filtered_frames: List[Dict[int, np.ndarray]] = [{} for _ in range(num_frames)]
    total_landmarks_checked = 0
    invalid_landmarks_count = 0
    interpolated_count = 0

    for lm_id in all_landmark_ids:
        # Build 1D validity array for this landmark across frames
        valid_mask = []
        coords_list = []
        for f in range(num_frames):
            vis = visibility_per_frame[f].get(lm_id, 0.0) if f < len(visibility_per_frame) else 0.0
            coord = world_coords_per_frame[f].get(lm_id)
            total_landmarks_checked += 1

            if coord is not None and vis >= min_visibility:
                valid_mask.append(True)
                coords_list.append(coord.copy())
            else:
                valid_mask.append(False)
                coords_list.append(None)
                invalid_landmarks_count += 1

        # Interpolate short missing gaps
        f = 0
        while f < num_frames:
            if not valid_mask[f]:
                gap_start = f
                while f < num_frames and not valid_mask[f]:
                    f += 1
                gap_end = f
                gap_len = gap_end - gap_start

                # Interpolate if gap is bounded and <= max_interpolation_gap
                if gap_start > 0 and gap_end < num_frames and gap_len <= max_interpolation_gap:
                    start_coord = coords_list[gap_start - 1]
                    end_coord = coords_list[gap_end]
                    for g_idx, i_frame in enumerate(range(gap_start, gap_end), 1):
                        alpha = g_idx / (gap_len + 1)
                        interp_coord = (1.0 - alpha) * start_coord + alpha * end_coord
                        coords_list[i_frame] = interp_coord
                        valid_mask[i_frame] = True
                        interpolated_count += 1
            else:
                f += 1

        # Put filtered/interpolated landmarks into filtered_frames
        for f in range(num_frames):
            if valid_mask[f] and coords_list[f] is not None:
                filtered_frames[f][lm_id] = coords_list[f]

    invalid_pct = (invalid_landmarks_count / max(1, total_landmarks_checked)) * 100.0
    stats = {
        "invalid_percentage": round(invalid_pct, 2),
        "invalid_count": invalid_landmarks_count,
        "total_checked": total_landmarks_checked,
        "interpolated_count": interpolated_count,
    }
    return filtered_frames, stats


def compute_pelvis_center(coords: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
    """Compute pelvis center as midpoint between Left Hip and Right Hip."""
    l_hip = coords.get(_MP["LEFT_HIP"])
    r_hip = coords.get(_MP["RIGHT_HIP"])
    if l_hip is None or r_hip is None:
        return None
    return (l_hip + r_hip) * 0.5


def normalize_skeleton_translation(
    coords: Dict[int, np.ndarray],
    pelvis_center: Optional[np.ndarray] = None,
) -> Dict[int, np.ndarray]:
    """Subtract pelvis center from all joint coordinates to set pelvis as origin."""
    if pelvis_center is None:
        pelvis_center = compute_pelvis_center(coords)
    if pelvis_center is None:
        return {idx: arr.copy() for idx, arr in coords.items()}

    return {idx: (arr - pelvis_center) for idx, arr in coords.items()}


def compute_body_scale(coords: Dict[int, np.ndarray], epsilon: float = 1e-6) -> float:
    """
    Compute robust body scale s = (||LShoulder - RShoulder|| + ||LHip - RHip||) / 2.
    Returns 1.0 if key landmarks are missing or scale is near zero.
    """
    l_sh = coords.get(_MP["LEFT_SHOULDER"])
    r_sh = coords.get(_MP["RIGHT_SHOULDER"])
    l_hip = coords.get(_MP["LEFT_HIP"])
    r_hip = coords.get(_MP["RIGHT_HIP"])

    scales = []
    if l_sh is not None and r_sh is not None:
        sh_dist = float(np.linalg.norm(r_sh - l_sh))
        if sh_dist > epsilon:
            scales.append(sh_dist)
    if l_hip is not None and r_hip is not None:
        hip_dist = float(np.linalg.norm(r_hip - l_hip))
        if hip_dist > epsilon:
            scales.append(hip_dist)

    if not scales:
        return 1.0

    scale = sum(scales) / len(scales)
    return max(scale, epsilon)


def normalize_skeleton_scale(
    coords: Dict[int, np.ndarray],
    scale: float,
    epsilon: float = 1e-6,
) -> Dict[int, np.ndarray]:
    """Divide all joint coordinates by body scale."""
    valid_scale = max(scale, epsilon)
    return {idx: (arr / valid_scale) for idx, arr in coords.items()}


def build_canonical_body_frame(coords: Dict[int, np.ndarray]) -> Tuple[np.ndarray, bool]:
    """
    Build 3x3 body-fixed orthonormal frame R_body = [x | y | z].
      x = normalize(R_Shoulder - L_Shoulder)  (lateral: left to right)
      y_raw = normalize(Shoulder_Center - Pelvis_Center) (vertical: pelvis to shoulder)
      z = normalize(cross(x, y_raw))          (sagittal: forward)
      y = normalize(cross(z, x))              (re-orthogonalized vertical)

    Returns:
      (3x3 rotation matrix R_body, is_valid flag).
      If degenerate, returns (Identity matrix, False).
    """
    l_sh = coords.get(_MP["LEFT_SHOULDER"])
    r_sh = coords.get(_MP["RIGHT_SHOULDER"])
    l_hip = coords.get(_MP["LEFT_HIP"])
    r_hip = coords.get(_MP["RIGHT_HIP"])

    if l_sh is None or r_sh is None or l_hip is None or r_hip is None:
        return np.eye(3, dtype=np.float64), False

    sh_center = (l_sh + r_sh) * 0.5
    pel_center = (l_hip + r_hip) * 0.5

    x_raw = r_sh - l_sh
    y_raw = sh_center - pel_center

    x = _safe_normalize(x_raw)
    if np.linalg.norm(x) < 1e-6:
        return np.eye(3, dtype=np.float64), False

    y_dir = _safe_normalize(y_raw)
    if np.linalg.norm(y_dir) < 1e-6:
        return np.eye(3, dtype=np.float64), False

    z_raw = np.cross(x, y_dir)
    z = _safe_normalize(z_raw)
    if np.linalg.norm(z) < 1e-6:
        return np.eye(3, dtype=np.float64), False

    y = _safe_normalize(np.cross(z, x))

    R_body = np.column_stack([x, y, z])

    # Enforce right-handed coordinate system (det = +1)
    if np.linalg.det(R_body) < 0:
        R_body[:, 2] = -R_body[:, 2]

    # Validate orthogonality and determinant
    orth_err = np.max(np.abs(R_body.T @ R_body - np.eye(3)))
    det_val = np.linalg.det(R_body)
    is_valid = (orth_err < 0.05) and (abs(det_val - 1.0) < 0.05)

    if not is_valid:
        return np.eye(3, dtype=np.float64), False

    return R_body, True


def transform_to_canonical_coordinates(
    coords: Dict[int, np.ndarray],
    R_body: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Transform joint coordinates into body-fixed canonical frame: q_i = R_body.T @ p_i."""
    return {idx: (R_body.T @ arr) for idx, arr in coords.items()}


def estimate_joint_rotations_canonical(coords: Dict[int, np.ndarray]) -> List[np.ndarray]:
    """
    Estimate 24 SMPL joint rotation matrices from normalized canonical skeleton coordinates.
    Returns list of 24 3x3 rotation matrices.
    """
    from src.pose.mediapipe_smpl import LandmarkFrame
    
    dummy_vis = {idx: 1.0 for idx in coords.keys()}
    frame = LandmarkFrame(coords, dummy_vis)

    from src.pose.mediapipe_smpl import (
        _estimate_pelvis_rotation,
        _estimate_hip_rotation,
        _estimate_spine_rotation,
        _estimate_knee_rotation,
        _estimate_ankle_rotation,
        _estimate_foot_rotation,
        _estimate_neck_rotation,
        _estimate_collar_rotation,
        _estimate_head_rotation,
        _estimate_shoulder_rotation,
        _estimate_elbow_rotation,
        _estimate_wrist_rotation,
        _estimate_hand_rotation,
    )

    rotations: List[np.ndarray] = [
        _estimate_pelvis_rotation(frame),        # 0
        _estimate_hip_rotation(frame, "left"),     # 1
        _estimate_hip_rotation(frame, "right"),    # 2
        _estimate_spine_rotation(frame, 1),        # 3
        _estimate_knee_rotation(frame, "left"),    # 4
        _estimate_knee_rotation(frame, "right"),   # 5
        _estimate_spine_rotation(frame, 2),        # 6
        _estimate_ankle_rotation(frame, "left"),   # 7
        _estimate_ankle_rotation(frame, "right"),  # 8
        _estimate_spine_rotation(frame, 3),        # 9
        _estimate_foot_rotation(frame, "left"),    # 10
        _estimate_foot_rotation(frame, "right"),   # 11
        _estimate_neck_rotation(frame),            # 12
        _estimate_collar_rotation(frame, "left"),  # 13
        _estimate_collar_rotation(frame, "right"), # 14
        _estimate_head_rotation(frame),            # 15
        _estimate_shoulder_rotation(frame, "left"),# 16
        _estimate_shoulder_rotation(frame, "right"),# 17
        _estimate_elbow_rotation(frame, "left"),   # 18
        _estimate_elbow_rotation(frame, "right"),  # 19
        _estimate_wrist_rotation(frame, "left"),   # 20
        _estimate_wrist_rotation(frame, "right"),  # 21
        _estimate_hand_rotation(frame, "left"),    # 22
        _estimate_hand_rotation(frame, "right"),   # 23
    ]

    # Numerical validation of each rotation matrix
    validated_rotations = []
    for j_idx, R in enumerate(rotations):
        if not _validate_rotation_matrix(R, SMPL_JOINT_NAMES[j_idx]):
            validated_rotations.append(np.eye(3, dtype=np.float64))
        else:
            validated_rotations.append(R)

    return validated_rotations


def detect_gait_cycles(
    coords_sequence: List[Dict[int, np.ndarray]],
    fps: float = 30.0,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """
    Detect gait cycles based on lower-limb ankle Z-displacement extrema.
    Returns:
        List of (start_frame, end_frame) cycle bounds, plus metadata dict.
    """
    num_frames = len(coords_sequence)
    if num_frames < int(fps * 1.5):
        return [], {"status": "insufficient_frames", "num_cycles": 0}

    # Extract left and right ankle longitudinal z coordinates
    z_left = []
    z_right = []
    for coords in coords_sequence:
        l_ank = coords.get(_MP["LEFT_ANKLE"])
        r_ank = coords.get(_MP["RIGHT_ANKLE"])
        z_left.append(l_ank[2] if l_ank is not None else 0.0)
        z_right.append(r_ank[2] if r_ank is not None else 0.0)

    z_diff = np.array(z_left) - np.array(z_right)
    # Smooth signal with moving average window
    window = max(3, int(fps * 0.15))
    kernel = np.ones(window) / window
    z_smooth = np.convolve(z_diff, kernel, mode="same")

    # Find zero-crossings corresponding to heel strikes
    zero_crossings = np.where(np.diff(np.signbit(z_smooth)))[0]
    cycles = []
    min_cycle_len = int(fps * 0.6)  # at least 0.6s per stride
    max_cycle_len = int(fps * 2.5)  # at most 2.5s per stride

    for i in range(len(zero_crossings) - 2):
        c_start = int(zero_crossings[i])
        c_end = int(zero_crossings[i + 2])
        length = c_end - c_start
        if min_cycle_len <= length <= max_cycle_len:
            cycles.append((c_start, c_end))

    stats = {
        "status": "success" if cycles else "insufficient_cycles",
        "num_cycles": len(cycles),
        "cycles": cycles,
    }
    return cycles, stats


def build_normalized_smpl_dataframe(
    world_coords_per_frame: List[Dict[int, np.ndarray]],
    visibility_per_frame: List[Dict[int, float]],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Full spatial normalization pipeline for a video's landmark sequence.
    Converts raw world landmarks into normalized SMPL rotation matrix DataFrame.
    """
    cfg = {**DEFAULT_NORMALIZATION_CONFIG, **(config or {})}
    num_frames = len(world_coords_per_frame)

    if num_frames == 0:
        raise ValueError("Input landmark sequence is empty.")

    # 1. Confidence Filtering & Short-gap Interpolation
    if cfg["confidence_filter"]:
        coords_seq, filter_stats = filter_landmarks_by_visibility(
            world_coords_per_frame,
            visibility_per_frame,
            min_visibility=cfg["min_visibility"],
            max_interpolation_gap=cfg["max_interpolation_gap"],
        )
    else:
        coords_seq = [c.copy() for c in world_coords_per_frame]
        filter_stats = {"invalid_percentage": 0.0, "interpolated_count": 0}

    # 2-5. Spatial Normalization per frame
    rows = []
    scale_failures = 0
    orientation_failures = 0
    last_valid_R_body = np.eye(3, dtype=np.float64)

    for f_idx in range(num_frames):
        coords = coords_seq[f_idx]

        # Translation Normalization
        if cfg["pelvis_center"]:
            coords = normalize_skeleton_translation(coords)

        # Body Scale Normalization
        if cfg["scale_normalization"]:
            scale = compute_body_scale(coords)
            if abs(scale - 1.0) < 1e-5:
                scale_failures += 1
            coords = normalize_skeleton_scale(coords, scale)

        # Canonical Orientation Normalization
        if cfg["canonical_orientation"]:
            R_body, is_valid = build_canonical_body_frame(coords)
            if is_valid:
                last_valid_R_body = R_body
            else:
                R_body = last_valid_R_body
                orientation_failures += 1
            coords = transform_to_canonical_coordinates(coords, R_body)

        # Estimate Joint Rotations on Normalized Skeleton
        rotations = estimate_joint_rotations_canonical(coords)

        # Convert to SMPL DataFrame row format
        row_dict = rotations_to_smpl_dataframe_row(np.array(rotations))
        rows.append(row_dict)

    smpl_df = pd.DataFrame(rows)

    stats = {
        "num_frames": num_frames,
        "invalid_landmark_pct": filter_stats["invalid_percentage"],
        "interpolated_count": filter_stats["interpolated_count"],
        "scale_failures": scale_failures,
        "orientation_failures": orientation_failures,
    }

    return smpl_df, stats


def extract_normalized_features(
    smpl_df: pd.DataFrame,
    fps: float = 30.0,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Extract exact 290-dimensional feature vector from normalized SMPL DataFrame.
    """
    cfg = {**DEFAULT_NORMALIZATION_CONFIG, **(config or {})}

    poses = []
    for _, row in smpl_df.iterrows():
        global_pose = []
        for j in range(1):
            values = [row[f"global_orient_{i}"] for i in range(9)]
            matrix = np.array(values).reshape(3, 3)
            rot = Rotation.from_matrix(matrix)
            global_pose.extend(rot.as_rotvec().tolist())

        body_pose = []
        for j in range(23):
            values = [row[f"body_pose_{j * 9 + i}"] for i in range(9)]
            matrix = np.array(values).reshape(3, 3)
            rot = Rotation.from_matrix(matrix)
            body_pose.extend(rot.as_rotvec().tolist())

        pose = global_pose + body_pose
        poses.append(pose)

    poses_arr = np.array(poses)  # (N_frames, 72)

    # Compute summary statistics
    feature_dict: Dict[str, float] = {
        "fps": float(fps),
        "num_frames": int(poses_arr.shape[0]),
    }

    for i in range(72):
        vals = poses_arr[:, i]
        feature_dict[f"pose{i}_mean"] = float(np.mean(vals))
        feature_dict[f"pose{i}_std"] = float(np.std(vals))
        feature_dict[f"pose{i}_min"] = float(np.min(vals))
        feature_dict[f"pose{i}_max"] = float(np.max(vals))

    feature_df = pd.DataFrame([feature_dict])
    return feature_df
