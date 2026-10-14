"""
ParkinSense — MediaPipe → SMPL-like 24-joint Rotation Estimator
================================================================
Converts 3D MediaPipe world landmarks into the exact 24-joint SMPL
rotation representation used by smpl_feature_extractor.py, enabling
the downstream 290-feature pipeline without requiring 4D-Humans / HMR2.

SMPL 24-joint ordering (as used by 4D-Humans output and smpl_feature_extractor.py):
    0  — Pelvis          (global_orient, prefix: global_orient_0..8)
    1  — L_Hip
    2  — R_Hip
    3  — Spine1
    4  — L_Knee
    5  — R_Knee
    6  — Spine2
    7  — L_Ankle
    8  — R_Ankle
    9  — Spine3
    10 — L_Foot
    11 — R_Foot
    12 — Neck
    13 — L_Collar
    14 — R_Collar
    15 — Head
    16 — L_Shoulder
    17 — R_Shoulder
    18 — L_Elbow
    19 — R_Elbow
    20 — L_Wrist
    21 — R_Wrist
    22 — L_Hand  (approximated from wrist)
    23 — R_Hand  (approximated from wrist)

MediaPipe landmark indices used (world landmarks, 3D):
    11 — LEFT_SHOULDER
    12 — RIGHT_SHOULDER
    13 — LEFT_ELBOW
    14 — RIGHT_ELBOW
    15 — LEFT_WRIST
    16 — RIGHT_WRIST
    23 — LEFT_HIP
    24 — RIGHT_HIP
    25 — LEFT_KNEE
    26 — RIGHT_KNEE
    27 — LEFT_ANKLE
    28 — RIGHT_ANKLE
    29 — LEFT_HEEL
    30 — RIGHT_HEEL
    31 — LEFT_FOOT_INDEX
    32 — RIGHT_FOOT_INDEX
     0 — NOSE
     7 — LEFT_EAR
     8 — RIGHT_EAR

Rotation matrix convention:
    Each 3x3 rotation matrix R satisfies R.T @ R approximately I, det(R) approximately +1.

Limitations:
    - This is a geometric approximation, NOT equivalent to the SMPL
      shape-conditioned optimisation performed by 4D-Humans / HMR2.
    - Joints with no direct MediaPipe correspondence (Spine1-3, Collar,
      Hand) are estimated via linear interpolation or anatomical
      heuristics and may carry higher uncertainty.
    - Visibility filtering is applied; low-confidence joints fall back
      to identity rotation.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------
# MediaPipe landmark index constants
# --------------------------------------------------------------------------
_MP = {
    "NOSE":             0,
    "LEFT_EYE":         2,
    "RIGHT_EYE":        5,
    "LEFT_EAR":         7,
    "RIGHT_EAR":        8,
    "LEFT_SHOULDER":   11,
    "RIGHT_SHOULDER":  12,
    "LEFT_ELBOW":      13,
    "RIGHT_ELBOW":     14,
    "LEFT_WRIST":      15,
    "RIGHT_WRIST":     16,
    "LEFT_HIP":        23,
    "RIGHT_HIP":       24,
    "LEFT_KNEE":       25,
    "RIGHT_KNEE":      26,
    "LEFT_ANKLE":      27,
    "RIGHT_ANKLE":     28,
    "LEFT_HEEL":       29,
    "RIGHT_HEEL":      30,
    "LEFT_FOOT_INDEX": 31,
    "RIGHT_FOOT_INDEX": 32,
}

# SMPL 24-joint index -> name
SMPL_JOINT_NAMES: List[str] = [
    "Pelvis",    # 0
    "L_Hip",     # 1
    "R_Hip",     # 2
    "Spine1",    # 3
    "L_Knee",    # 4
    "R_Knee",    # 5
    "Spine2",    # 6
    "L_Ankle",   # 7
    "R_Ankle",   # 8
    "Spine3",    # 9
    "L_Foot",    # 10
    "R_Foot",    # 11
    "Neck",      # 12
    "L_Collar",  # 13
    "R_Collar",  # 14
    "Head",      # 15
    "L_Shoulder",# 16
    "R_Shoulder",# 17
    "L_Elbow",   # 18
    "R_Elbow",   # 19
    "L_Wrist",   # 20
    "R_Wrist",   # 21
    "L_Hand",    # 22
    "R_Hand",    # 23
]

_VISIBILITY_THRESHOLD = 0.3  # landmarks below this are treated as unreliable


# --------------------------------------------------------------------------
# Low-level geometry helpers
# --------------------------------------------------------------------------

def _safe_normalize(v: np.ndarray) -> np.ndarray:
    """Return unit vector; returns zero vector if input is near zero."""
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros(3, dtype=np.float64)
    return v / n


def _gram_schmidt(x_raw: np.ndarray, y_raw: np.ndarray) -> np.ndarray:
    """
    Build a right-handed orthonormal frame from two (possibly non-orthogonal)
    direction vectors using Gram-Schmidt orthogonalisation.

    The first vector x_raw defines the X axis.
    The second vector y_raw is orthogonalized w.r.t. X to give Y.
    Z = X x Y (cross product).

    Returns:
        3x3 rotation matrix [X | Y | Z] where columns are the frame axes.
        Identity matrix if degenerate input.
    """
    x = _safe_normalize(x_raw)
    if np.linalg.norm(x) < 1e-9:
        return np.eye(3)

    # Make y orthogonal to x
    y_ortho = y_raw - np.dot(y_raw, x) * x
    y = _safe_normalize(y_ortho)
    if np.linalg.norm(y) < 1e-9:
        # Pick an arbitrary perpendicular
        y = _safe_normalize(np.cross(x, np.array([0.0, 1.0, 0.0])))
        if np.linalg.norm(y) < 1e-9:
            y = _safe_normalize(np.cross(x, np.array([1.0, 0.0, 0.0])))

    z = np.cross(x, y)
    z = _safe_normalize(z)

    R = np.column_stack([x, y, z])
    # Enforce det = +1 (right-handed frame)
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]
    return R


def _rotation_matrix_from_bone(
    child: np.ndarray,
    parent: np.ndarray,
    lateral_hint: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Construct a 3x3 rotation matrix for a bone segment.

    The primary bone axis (Y in SMPL convention) points from parent -> child.
    The lateral hint provides a secondary direction to complete the frame.

    Args:
        child:        3D position of the child joint.
        parent:       3D position of the parent joint.
        lateral_hint: Optional vector to disambiguate the frame roll.

    Returns:
        3x3 rotation matrix (valid SO(3) element).
    """
    bone_dir = child - parent
    if np.linalg.norm(bone_dir) < 1e-9:
        return np.eye(3, dtype=np.float64)

    # Primary axis: bone direction (Y-axis)
    y_axis = _safe_normalize(bone_dir)

    # Secondary axis: lateral direction (X-axis candidate)
    if lateral_hint is None:
        lateral_hint = np.array([1.0, 0.0, 0.0])

    # Ensure lateral_hint is not parallel to y_axis
    if abs(np.dot(_safe_normalize(lateral_hint), y_axis)) > 0.99:
        lateral_hint = np.array([0.0, 0.0, 1.0])

    return _gram_schmidt(lateral_hint, y_axis)


def _validate_rotation_matrix(R: np.ndarray, joint_name: str = "") -> bool:
    """
    Validate that R is a proper rotation matrix (R.T @ R approx I, det approx +1).
    Returns False and logs warning if invalid.
    """
    if R.shape != (3, 3):
        warnings.warn(f"Joint {joint_name}: rotation matrix is not 3x3")
        return False

    orth_err = np.max(np.abs(R.T @ R - np.eye(3)))
    det_val = np.linalg.det(R)

    if orth_err > 0.05:
        warnings.warn(
            f"Joint {joint_name}: orthogonality error {orth_err:.4f} (threshold 0.05)"
        )
        return False
    if abs(det_val - 1.0) > 0.05:
        warnings.warn(
            f"Joint {joint_name}: determinant {det_val:.4f} (expected ~+1)"
        )
        return False
    return True


# --------------------------------------------------------------------------
# Per-landmark frame extraction from MediaPipe data
# --------------------------------------------------------------------------

class LandmarkFrame:
    """
    Wraps a single frame of MediaPipe landmarks for convenient access.

    Attributes:
        coords:      Dict[int, ndarray(3,)] - world landmark positions
        visibility:  Dict[int, float]       - detection confidence
    """

    def __init__(
        self,
        coords: Dict[int, np.ndarray],
        visibility: Dict[int, float],
    ) -> None:
        self.coords = coords
        self.visibility = visibility

    def get(self, idx: int, default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Get landmark position if visible, else return default."""
        if idx not in self.coords:
            return default
        if self.visibility.get(idx, 0.0) < _VISIBILITY_THRESHOLD:
            return default
        return self.coords[idx]

    def midpoint(self, idx_a: int, idx_b: int) -> Optional[np.ndarray]:
        """Midpoint between two landmarks; None if either is unavailable."""
        a = self.get(idx_a)
        b = self.get(idx_b)
        if a is None or b is None:
            return None
        return (a + b) * 0.5


# --------------------------------------------------------------------------
# SMPL joint rotation estimators
# --------------------------------------------------------------------------

def _estimate_pelvis_rotation(frame: LandmarkFrame) -> np.ndarray:
    """Joint 0: Pelvis (global_orient). From hip plane + torso direction."""
    l_hip = frame.get(_MP["LEFT_HIP"])
    r_hip = frame.get(_MP["RIGHT_HIP"])
    l_shoulder = frame.get(_MP["LEFT_SHOULDER"])
    r_shoulder = frame.get(_MP["RIGHT_SHOULDER"])

    if l_hip is None or r_hip is None:
        return np.eye(3, dtype=np.float64)

    x_raw = l_hip - r_hip  # lateral axis: right -> left
    pelvis_mid = (l_hip + r_hip) * 0.5
    if l_shoulder is not None and r_shoulder is not None:
        shoulder_mid = (l_shoulder + r_shoulder) * 0.5
        y_raw = shoulder_mid - pelvis_mid  # superior axis
    else:
        y_raw = np.array([0.0, -1.0, 0.0])  # fallback: world up (MediaPipe Y is down)

    return _gram_schmidt(x_raw, y_raw)


def _estimate_hip_rotation(frame: LandmarkFrame, side: str) -> np.ndarray:
    """Joints 1 (L_Hip), 2 (R_Hip). Bone: Hip -> Knee."""
    if side == "left":
        hip_idx, knee_idx = _MP["LEFT_HIP"], _MP["LEFT_KNEE"]
    else:
        hip_idx, knee_idx = _MP["RIGHT_HIP"], _MP["RIGHT_KNEE"]

    hip = frame.get(hip_idx)
    knee = frame.get(knee_idx)
    if hip is None or knee is None:
        return np.eye(3, dtype=np.float64)

    l_hip = frame.get(_MP["LEFT_HIP"])
    r_hip = frame.get(_MP["RIGHT_HIP"])
    if l_hip is not None and r_hip is not None:
        lateral = l_hip - r_hip if side == "left" else r_hip - l_hip
    else:
        lateral = np.array([1.0, 0.0, 0.0]) if side == "left" else np.array([-1.0, 0.0, 0.0])

    return _rotation_matrix_from_bone(knee, hip, lateral_hint=lateral)


def _estimate_spine_rotation(frame: LandmarkFrame, level: int) -> np.ndarray:
    """
    Joints 3 (Spine1), 6 (Spine2), 9 (Spine3).
    Interpolated positions along the torso at t = level/4.
    """
    l_hip = frame.get(_MP["LEFT_HIP"])
    r_hip = frame.get(_MP["RIGHT_HIP"])
    l_shoulder = frame.get(_MP["LEFT_SHOULDER"])
    r_shoulder = frame.get(_MP["RIGHT_SHOULDER"])

    if l_hip is None or r_hip is None:
        return np.eye(3, dtype=np.float64)

    pelvis_mid = (l_hip + r_hip) * 0.5
    if l_shoulder is not None and r_shoulder is not None:
        shoulder_mid = (l_shoulder + r_shoulder) * 0.5
    else:
        shoulder_mid = pelvis_mid + np.array([0.0, -0.5, 0.0])

    t = level / 4.0
    spine_pos = pelvis_mid + t * (shoulder_mid - pelvis_mid)
    spine_next = pelvis_mid + (t + 0.25) * (shoulder_mid - pelvis_mid)

    if l_shoulder is not None and r_shoulder is not None:
        lateral = l_shoulder - r_shoulder
    else:
        lateral = l_hip - r_hip

    return _rotation_matrix_from_bone(spine_next, spine_pos, lateral_hint=lateral)


def _estimate_knee_rotation(frame: LandmarkFrame, side: str) -> np.ndarray:
    """Joints 4 (L_Knee), 5 (R_Knee). Bone: Knee -> Ankle."""
    if side == "left":
        knee_idx, ankle_idx = _MP["LEFT_KNEE"], _MP["LEFT_ANKLE"]
    else:
        knee_idx, ankle_idx = _MP["RIGHT_KNEE"], _MP["RIGHT_ANKLE"]

    knee = frame.get(knee_idx)
    ankle = frame.get(ankle_idx)
    if knee is None or ankle is None:
        return np.eye(3, dtype=np.float64)

    l_hip = frame.get(_MP["LEFT_HIP"])
    r_hip = frame.get(_MP["RIGHT_HIP"])
    if l_hip is not None and r_hip is not None:
        lateral = l_hip - r_hip if side == "left" else r_hip - l_hip
    else:
        lateral = np.array([1.0, 0.0, 0.0]) if side == "left" else np.array([-1.0, 0.0, 0.0])

    return _rotation_matrix_from_bone(ankle, knee, lateral_hint=lateral)


def _estimate_ankle_rotation(frame: LandmarkFrame, side: str) -> np.ndarray:
    """Joints 7 (L_Ankle), 8 (R_Ankle). Bone: Ankle -> Heel/Foot direction."""
    if side == "left":
        ankle_idx = _MP["LEFT_ANKLE"]
        heel_idx  = _MP["LEFT_HEEL"]
        foot_idx  = _MP["LEFT_FOOT_INDEX"]
    else:
        ankle_idx = _MP["RIGHT_ANKLE"]
        heel_idx  = _MP["RIGHT_HEEL"]
        foot_idx  = _MP["RIGHT_FOOT_INDEX"]

    ankle = frame.get(ankle_idx)
    heel  = frame.get(heel_idx)
    foot  = frame.get(foot_idx)

    if ankle is None:
        return np.eye(3, dtype=np.float64)

    if heel is not None and foot is not None:
        forward = foot - heel
    elif foot is not None:
        forward = foot - ankle
    else:
        forward = np.array([0.0, 0.0, -1.0])

    lateral = np.array([1.0, 0.0, 0.0]) if side == "left" else np.array([-1.0, 0.0, 0.0])
    return _gram_schmidt(lateral, forward)


def _estimate_foot_rotation(frame: LandmarkFrame, side: str) -> np.ndarray:
    """Joints 10 (L_Foot), 11 (R_Foot). Bone: Heel -> FootIndex."""
    if side == "left":
        heel_idx = _MP["LEFT_HEEL"]
        foot_idx = _MP["LEFT_FOOT_INDEX"]
    else:
        heel_idx = _MP["RIGHT_HEEL"]
        foot_idx = _MP["RIGHT_FOOT_INDEX"]

    heel = frame.get(heel_idx)
    foot = frame.get(foot_idx)
    if heel is None or foot is None:
        return np.eye(3, dtype=np.float64)

    return _rotation_matrix_from_bone(foot, heel)


def _estimate_neck_rotation(frame: LandmarkFrame) -> np.ndarray:
    """Joint 12: Neck. Bone: shoulder midpoint -> head (ear midpoint or nose)."""
    l_shoulder = frame.get(_MP["LEFT_SHOULDER"])
    r_shoulder = frame.get(_MP["RIGHT_SHOULDER"])
    l_ear = frame.get(_MP["LEFT_EAR"])
    r_ear = frame.get(_MP["RIGHT_EAR"])
    nose  = frame.get(_MP["NOSE"])

    if l_shoulder is None or r_shoulder is None:
        return np.eye(3, dtype=np.float64)

    neck_pos = (l_shoulder + r_shoulder) * 0.5
    if l_ear is not None and r_ear is not None:
        head_pos = (l_ear + r_ear) * 0.5
    elif nose is not None:
        head_pos = nose
    else:
        head_pos = neck_pos + np.array([0.0, -0.2, 0.0])

    lateral = l_shoulder - r_shoulder
    return _rotation_matrix_from_bone(head_pos, neck_pos, lateral_hint=lateral)


def _estimate_collar_rotation(frame: LandmarkFrame, side: str) -> np.ndarray:
    """Joints 13 (L_Collar), 14 (R_Collar). Bone: neck -> shoulder."""
    l_shoulder = frame.get(_MP["LEFT_SHOULDER"])
    r_shoulder = frame.get(_MP["RIGHT_SHOULDER"])

    if l_shoulder is None or r_shoulder is None:
        return np.eye(3, dtype=np.float64)

    neck_pos = (l_shoulder + r_shoulder) * 0.5
    shoulder_pos = l_shoulder if side == "left" else r_shoulder
    lateral = l_shoulder - r_shoulder if side == "left" else r_shoulder - l_shoulder
    return _rotation_matrix_from_bone(shoulder_pos, neck_pos, lateral_hint=lateral)


def _estimate_head_rotation(frame: LandmarkFrame) -> np.ndarray:
    """Joint 15: Head. From ear midpoint -> nose direction."""
    l_ear = frame.get(_MP["LEFT_EAR"])
    r_ear = frame.get(_MP["RIGHT_EAR"])
    nose  = frame.get(_MP["NOSE"])

    if l_ear is None or r_ear is None:
        return np.eye(3, dtype=np.float64)

    ear_mid = (l_ear + r_ear) * 0.5
    lateral = l_ear - r_ear
    forward = nose - ear_mid if nose is not None else np.array([0.0, -1.0, 0.0])
    return _gram_schmidt(lateral, forward)


def _estimate_shoulder_rotation(frame: LandmarkFrame, side: str) -> np.ndarray:
    """Joints 16 (L_Shoulder), 17 (R_Shoulder). Bone: Shoulder -> Elbow."""
    if side == "left":
        shoulder_idx = _MP["LEFT_SHOULDER"]
        elbow_idx    = _MP["LEFT_ELBOW"]
    else:
        shoulder_idx = _MP["RIGHT_SHOULDER"]
        elbow_idx    = _MP["RIGHT_ELBOW"]

    shoulder = frame.get(shoulder_idx)
    elbow    = frame.get(elbow_idx)
    if shoulder is None or elbow is None:
        return np.eye(3, dtype=np.float64)

    l_shoulder = frame.get(_MP["LEFT_SHOULDER"])
    r_shoulder = frame.get(_MP["RIGHT_SHOULDER"])
    if l_shoulder is not None and r_shoulder is not None:
        lateral = l_shoulder - r_shoulder if side == "left" else r_shoulder - l_shoulder
    else:
        lateral = np.array([1.0, 0.0, 0.0]) if side == "left" else np.array([-1.0, 0.0, 0.0])

    return _rotation_matrix_from_bone(elbow, shoulder, lateral_hint=lateral)


def _estimate_elbow_rotation(frame: LandmarkFrame, side: str) -> np.ndarray:
    """Joints 18 (L_Elbow), 19 (R_Elbow). Bone: Elbow -> Wrist."""
    if side == "left":
        elbow_idx = _MP["LEFT_ELBOW"]
        wrist_idx = _MP["LEFT_WRIST"]
    else:
        elbow_idx = _MP["RIGHT_ELBOW"]
        wrist_idx = _MP["RIGHT_WRIST"]

    elbow = frame.get(elbow_idx)
    wrist = frame.get(wrist_idx)
    if elbow is None or wrist is None:
        return np.eye(3, dtype=np.float64)

    l_shoulder = frame.get(_MP["LEFT_SHOULDER"])
    r_shoulder = frame.get(_MP["RIGHT_SHOULDER"])
    if l_shoulder is not None and r_shoulder is not None:
        lateral = l_shoulder - r_shoulder if side == "left" else r_shoulder - l_shoulder
    else:
        lateral = np.array([1.0, 0.0, 0.0]) if side == "left" else np.array([-1.0, 0.0, 0.0])

    return _rotation_matrix_from_bone(wrist, elbow, lateral_hint=lateral)


def _estimate_wrist_rotation(frame: LandmarkFrame, side: str) -> np.ndarray:
    """Joints 20 (L_Wrist), 21 (R_Wrist). Same direction as elbow->wrist segment."""
    return _estimate_elbow_rotation(frame, side)


def _estimate_hand_rotation(frame: LandmarkFrame, side: str) -> np.ndarray:
    """
    Joints 22 (L_Hand), 23 (R_Hand).
    MediaPipe 33-point body pose does not provide finger landmarks.
    Approximated as same rotation as wrist.
    Limitation: hand orientation cannot be accurately estimated.
    """
    return _estimate_wrist_rotation(frame, side)


# --------------------------------------------------------------------------
# Main per-frame conversion function
# --------------------------------------------------------------------------

def mediapipe_to_smpl_rotations(
    landmark_coords: Dict[int, np.ndarray],
    landmark_visibility: Dict[int, float],
) -> np.ndarray:
    """
    Convert one frame of MediaPipe world landmarks into 24 SMPL-like
    local joint rotation matrices.

    Args:
        landmark_coords:     Dict mapping MediaPipe landmark index -> (x, y, z).
        landmark_visibility: Dict mapping MediaPipe landmark index -> visibility float.

    Returns:
        np.ndarray of shape (24, 3, 3):
            [0]     -> global_orient (Pelvis)
            [1..23] -> body_pose joints

    All returned matrices satisfy R.T @ R approx I and det(R) approx +1.
    Degenerate/low-visibility joints fall back to identity matrix.
    """
    frame = LandmarkFrame(landmark_coords, landmark_visibility)

    # Build all 24 rotation matrices
    estimators = [
        _estimate_pelvis_rotation(frame),           # 0  Pelvis
        _estimate_hip_rotation(frame, "left"),       # 1  L_Hip
        _estimate_hip_rotation(frame, "right"),      # 2  R_Hip
        _estimate_spine_rotation(frame, 1),          # 3  Spine1
        _estimate_knee_rotation(frame, "left"),      # 4  L_Knee
        _estimate_knee_rotation(frame, "right"),     # 5  R_Knee
        _estimate_spine_rotation(frame, 2),          # 6  Spine2
        _estimate_ankle_rotation(frame, "left"),     # 7  L_Ankle
        _estimate_ankle_rotation(frame, "right"),    # 8  R_Ankle
        _estimate_spine_rotation(frame, 3),          # 9  Spine3
        _estimate_foot_rotation(frame, "left"),      # 10 L_Foot
        _estimate_foot_rotation(frame, "right"),     # 11 R_Foot
        _estimate_neck_rotation(frame),              # 12 Neck
        _estimate_collar_rotation(frame, "left"),    # 13 L_Collar
        _estimate_collar_rotation(frame, "right"),   # 14 R_Collar
        _estimate_head_rotation(frame),              # 15 Head
        _estimate_shoulder_rotation(frame, "left"),  # 16 L_Shoulder
        _estimate_shoulder_rotation(frame, "right"), # 17 R_Shoulder
        _estimate_elbow_rotation(frame, "left"),     # 18 L_Elbow
        _estimate_elbow_rotation(frame, "right"),    # 19 R_Elbow
        _estimate_wrist_rotation(frame, "left"),     # 20 L_Wrist
        _estimate_wrist_rotation(frame, "right"),    # 21 R_Wrist
        _estimate_hand_rotation(frame, "left"),      # 22 L_Hand
        _estimate_hand_rotation(frame, "right"),     # 23 R_Hand
    ]

    # Validate and sanitize each rotation matrix
    rotations = np.zeros((24, 3, 3), dtype=np.float64)
    for i, R in enumerate(estimators):
        if _validate_rotation_matrix(R, SMPL_JOINT_NAMES[i]):
            rotations[i] = R
        else:
            # Fallback to identity — joint cannot be reliably estimated
            rotations[i] = np.eye(3, dtype=np.float64)

    return rotations


def rotations_to_smpl_dataframe_row(rotations: np.ndarray) -> dict:
    """
    Flatten 24 x 3x3 rotation matrices into a DataFrame row dict compatible
    with SMPLFeatureExtractor.

    Column naming matches rotation_matrix_to_axis_angle() in smpl_feature_extractor.py:
        global_orient_0 ... global_orient_8   (joint 0, flattened 3x3, row-major)
        body_pose_0     ... body_pose_206      (joints 1-23, flattened, row-major)

    Args:
        rotations: np.ndarray of shape (24, 3, 3)

    Returns:
        Dict mapping column name -> scalar float value.
    """
    row: dict = {}

    # Joint 0: global_orient (9 values, row-major)
    for i, val in enumerate(rotations[0].flatten(order='C')):
        row[f"global_orient_{i}"] = float(val)

    # Joints 1-23: body_pose (23 x 9 = 207 values, row-major)
    for joint_idx in range(1, 24):
        base = (joint_idx - 1) * 9
        for i, val in enumerate(rotations[joint_idx].flatten(order='C')):
            row[f"body_pose_{base + i}"] = float(val)

    return row
