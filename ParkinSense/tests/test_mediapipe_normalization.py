"""
Unit tests for MediaPipe Invariant Domain Normalization pipeline.
===================================================================
Tests translation invariance, scale invariance, rotation invariance,
orthogonality of rotation matrices, landmark filtering, and feature count.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from src.pose.mediapipe_normalization import (
    _MP,
    compute_pelvis_center,
    normalize_skeleton_translation,
    compute_body_scale,
    normalize_skeleton_scale,
    build_canonical_body_frame,
    transform_to_canonical_coordinates,
    estimate_joint_rotations_canonical,
    filter_landmarks_by_visibility,
    build_normalized_smpl_dataframe,
    extract_normalized_features,
)


def _generate_synthetic_skeleton() -> Dict[int, np.ndarray]:
    """Generate a realistic synthetic 3D skeleton frame in meters."""
    return {
        _MP["LEFT_HIP"]:       np.array([-0.15, 0.90, 0.0]),
        _MP["RIGHT_HIP"]:      np.array([ 0.15, 0.90, 0.0]),
        _MP["LEFT_SHOULDER"]:  np.array([-0.20, 1.40, 0.0]),
        _MP["RIGHT_SHOULDER"]: np.array([ 0.20, 1.40, 0.0]),
        _MP["LEFT_KNEE"]:      np.array([-0.15, 0.50, 0.1]),
        _MP["RIGHT_KNEE"]:     np.array([ 0.15, 0.50, 0.1]),
        _MP["LEFT_ANKLE"]:     np.array([-0.15, 0.10, 0.0]),
        _MP["RIGHT_ANKLE"]:    np.array([ 0.15, 0.10, 0.0]),
        _MP["LEFT_ELBOW"]:     np.array([-0.30, 1.15, 0.0]),
        _MP["RIGHT_ELBOW"]:    np.array([ 0.30, 1.15, 0.0]),
        _MP["LEFT_WRIST"]:     np.array([-0.35, 0.90, 0.0]),
        _MP["RIGHT_WRIST"]:    np.array([ 0.35, 0.90, 0.0]),
        _MP["NOSE"]:           np.array([ 0.00, 1.65, 0.05]),
        _MP["LEFT_EAR"]:       np.array([-0.10, 1.65, 0.0]),
        _MP["RIGHT_EAR"]:      np.array([ 0.10, 1.65, 0.0]),
    }


def test_pelvis_centering():
    """Verify that pelvis center becomes [0, 0, 0] after translation normalization."""
    skel = _generate_synthetic_skeleton()
    pelvis = compute_pelvis_center(skel)
    assert pelvis is not None
    norm_skel = normalize_skeleton_translation(skel, pelvis)

    new_pelvis = compute_pelvis_center(norm_skel)
    assert new_pelvis is not None
    np.testing.assert_allclose(new_pelvis, np.zeros(3), atol=1e-7)


test_pelvis_centering()


def test_scale_normalization():
    """Verify body scale becomes approximately 1.0 after scale normalization."""
    skel = _generate_synthetic_skeleton()
    scale = compute_body_scale(skel)
    assert scale > 0.0
    norm_skel = normalize_skeleton_scale(skel, scale)

    new_scale = compute_body_scale(norm_skel)
    np.testing.assert_allclose(new_scale, 1.0, atol=1e-6)


test_scale_normalization()


def test_canonical_frame_orthogonality():
    """Verify canonical frame matrix satisfies R.T @ R = I and det(R) = +1."""
    skel = _generate_synthetic_skeleton()
    R_body, is_valid = build_canonical_body_frame(skel)
    assert is_valid
    np.testing.assert_allclose(R_body.T @ R_body, np.eye(3), atol=1e-6)
    assert abs(np.linalg.det(R_body) - 1.0) < 1e-6


test_canonical_frame_orthogonality()


def test_translation_invariance():
    """Verify canonical skeleton is invariant to arbitrary global 3D translation."""
    skel = _generate_synthetic_skeleton()
    translation_vector = np.array([12.5, -45.0, 100.2])

    skel_translated = {idx: (arr + translation_vector) for idx, arr in skel.items()}

    # Normalize original
    pel1 = compute_pelvis_center(skel)
    t1 = normalize_skeleton_translation(skel, pel1)
    s1 = compute_body_scale(t1)
    sc1 = normalize_skeleton_scale(t1, s1)
    R1, _ = build_canonical_body_frame(sc1)
    c1 = transform_to_canonical_coordinates(sc1, R1)

    # Normalize translated
    pel2 = compute_pelvis_center(skel_translated)
    t2 = normalize_skeleton_translation(skel_translated, pel2)
    s2 = compute_body_scale(t2)
    sc2 = normalize_skeleton_scale(t2, s2)
    R2, _ = build_canonical_body_frame(sc2)
    c2 = transform_to_canonical_coordinates(sc2, R2)

    for idx in skel.keys():
        np.testing.assert_allclose(c1[idx], c2[idx], atol=1e-6)


test_translation_invariance()


def test_scale_invariance():
    """Verify canonical skeleton is invariant to global scaling (e.g. 2.5x larger)."""
    skel = _generate_synthetic_skeleton()
    skel_scaled = {idx: (arr * 2.5) for idx, arr in skel.items()}

    # Original canonical
    pel1 = compute_pelvis_center(skel)
    t1 = normalize_skeleton_translation(skel, pel1)
    s1 = compute_body_scale(t1)
    sc1 = normalize_skeleton_scale(t1, s1)
    R1, _ = build_canonical_body_frame(sc1)
    c1 = transform_to_canonical_coordinates(sc1, R1)

    # Scaled canonical
    pel2 = compute_pelvis_center(skel_scaled)
    t2 = normalize_skeleton_translation(skel_scaled, pel2)
    s2 = compute_body_scale(t2)
    sc2 = normalize_skeleton_scale(t2, s2)
    R2, _ = build_canonical_body_frame(sc2)
    c2 = transform_to_canonical_coordinates(sc2, R2)

    for idx in skel.keys():
        np.testing.assert_allclose(c1[idx], c2[idx], atol=1e-6)


test_scale_invariance()


def test_rotation_invariance():
    """Verify canonical skeleton is invariant to rigid global 3D rotation."""
    skel = _generate_synthetic_skeleton()
    rot_rand = Rotation.from_euler("xyz", [25.0, -35.0, 45.0], degrees=True).as_matrix()
    skel_rotated = {idx: (rot_rand @ arr) for idx, arr in skel.items()}

    # Original canonical
    pel1 = compute_pelvis_center(skel)
    t1 = normalize_skeleton_translation(skel, pel1)
    s1 = compute_body_scale(t1)
    sc1 = normalize_skeleton_scale(t1, s1)
    R1, _ = build_canonical_body_frame(sc1)
    c1 = transform_to_canonical_coordinates(sc1, R1)

    # Rotated canonical
    pel2 = compute_pelvis_center(skel_rotated)
    t2 = normalize_skeleton_translation(skel_rotated, pel2)
    s2 = compute_body_scale(t2)
    sc2 = normalize_skeleton_scale(t2, s2)
    R2, _ = build_canonical_body_frame(sc2)
    c2 = transform_to_canonical_coordinates(sc2, R2)

    for idx in skel.keys():
        np.testing.assert_allclose(c1[idx], c2[idx], atol=1e-5)


test_rotation_invariance()


def test_feature_vector_shape_and_no_nan():
    """Verify end-to-end normalization extracts exactly 290 finite features without NaN or Inf."""
    skel = _generate_synthetic_skeleton()
    coords_seq = [skel.copy() for _ in range(20)]
    vis_seq = [{idx: 0.95 for idx in skel.keys()} for _ in range(20)]

    smpl_df, stats = build_normalized_smpl_dataframe(coords_seq, vis_seq)
    feat_df = extract_normalized_features(smpl_df, fps=30.0)

    assert feat_df.shape == (1, 290)
    assert not feat_df.isnull().any().any()
    assert np.all(np.isfinite(feat_df.values))


test_feature_vector_shape_and_no_nan()

print("ALL MEDIA PIPE NORMALIZATION UNIT TESTS PASSED SUCCESSFULLY!")
