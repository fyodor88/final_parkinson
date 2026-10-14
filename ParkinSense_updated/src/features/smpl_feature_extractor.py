"""
ParkinSense — SMPL Feature Extractor
====================================
Transforms raw 216-dimensional SMPL rotation matrix sequences into the exact 290-dimensional
feature representation expected by `pd_detector.pkl` and `random_forest.pkl`.

Features generated:
- fps
- num_frames
- pose0_mean, pose0_std, pose0_min, pose0_max
  ...
- pose71_mean, pose71_std, pose71_min, pose71_max
"""

from __future__ import annotations

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from src.utils import get_logger

logger = get_logger(__name__)


def rotation_matrix_to_axis_angle(row: pd.Series, prefix: str, count: int) -> List[float]:
    """
    Convert sequence of 3x3 rotation matrices in row to 3D axis-angle vectors.
    """
    result = []
    for joint in range(count):
        values = [row[f"{prefix}_{joint * 9 + i}"] for i in range(9)]
        matrix = np.array(values).reshape(3, 3)
        rotation = Rotation.from_matrix(matrix)
        axis_angle = rotation.as_rotvec()
        result.extend(axis_angle.tolist())
    return result


class SMPLFeatureExtractor:
    """
    Extracts 290 statistical features from SMPL pose sequences.
    """

    def extract_features(self, smpl_df: pd.DataFrame, fps: float = 29.75) -> pd.DataFrame:
        """
        Convert raw SMPL matrix DataFrame to a 290-column feature vector row.

        Args:
            smpl_df: DataFrame with 216 rotation matrix columns.
            fps:     Frame rate of original video.

        Returns:
            DataFrame with 1 row and 290 columns.
        """
        if smpl_df.empty:
            raise ValueError("Input SMPL DataFrame is empty.")

        logger.info("Extracting SMPL 72-channel axis-angle poses for %d frames...", len(smpl_df))

        poses = []
        for _, row in smpl_df.iterrows():
            global_pose = rotation_matrix_to_axis_angle(row, "global_orient", 1)
            body_pose = rotation_matrix_to_axis_angle(row, "body_pose", 23)
            pose = global_pose + body_pose

            if len(pose) != 72:
                raise ValueError(f"Expected 72 pose values per frame, got {len(pose)}")

            poses.append(pose)

        poses_arr = np.array(poses)  # Shape: (N_frames, 72)

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
        logger.info("✅ SMPL Feature extraction complete. Output shape: %s", feature_df.shape)
        return feature_df
