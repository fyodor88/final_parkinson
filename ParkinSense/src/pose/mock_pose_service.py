"""
ParkinSense — Mock Pose Service
===============================
Provides a local mock pose extraction service for development and testing
when 4D-Humans is not running locally on Mac.

In MOCK mode:
- If data/input/smpl.csv exists, loads and returns real sample SMPL frames.
- Otherwise, generates synthetic 216-dimensional rotation matrix data representing
  24 SMPL joint rotation matrices across N frames.

Returns standard pose dictionary matching the remote 4D-Humans API contract.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOCK_SMPL_CSV = _PROJECT_ROOT / "data" / "input" / "smpl.csv"


class MockPoseService:
    """
    Local mock pose extractor simulating the remote 4D-Humans REST API.
    """

    def __init__(self, sample_csv_path: Path | str | None = None) -> None:
        self.sample_csv_path = Path(sample_csv_path) if sample_csv_path else DEFAULT_MOCK_SMPL_CSV

    def extract_smpl_from_video(self, video_path: str | Path) -> Tuple[pd.DataFrame, float, int]:
        """
        Simulate pose extraction for video_path.

        Args:
            video_path: Path to the walking video.

        Returns:
            Tuple of (smpl_df, fps, num_frames)
            where smpl_df is a DataFrame with 216 rotation matrix columns
            ('global_orient_0'..'global_orient_8', 'body_pose_0'..'body_pose_206').
        """
        video_path = Path(video_path)
        logger.info("[MOCK POSE SERVICE] Processing video: %s", video_path.name)
        logger.warning(
            "[MOCK POSE SERVICE] Operating in LOCAL MOCK mode. "
            "No GPU 4D-Humans inference is performed on this Mac."
        )

        if self.sample_csv_path.exists():
            logger.info("[MOCK POSE SERVICE] Loading baseline sample SMPL data from: %s", self.sample_csv_path)
            smpl_df = pd.read_csv(self.sample_csv_path)
            fps = 29.75
            num_frames = len(smpl_df)
            logger.info("[MOCK POSE SERVICE] Loaded %d frames from sample CSV.", num_frames)
            return smpl_df, fps, num_frames

        # Fallback: Generate synthetic 216 rotation matrix columns across 150 frames
        logger.info("[MOCK POSE SERVICE] Generating synthetic SMPL pose data (150 frames)...")
        num_frames = 150
        fps = 30.0

        # Build column names
        orient_cols = [f"global_orient_{i}" for i in range(9)]
        body_cols = [f"body_pose_{i}" for i in range(207)]
        cols = orient_cols + body_cols

        # Identity rotation matrices (3x3 flattened = [1,0,0, 0,1,0, 0,0,1])
        identity_mat = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        # Replicate for 24 joints (1 global + 23 body) = 24 * 9 = 216
        single_frame = np.tile(identity_mat, 24)

        # Add small random noise to simulate motion dynamics
        rng = np.random.default_rng(seed=42)
        data = np.tile(single_frame, (num_frames, 1))
        noise = rng.normal(loc=0.0, scale=0.02, size=data.shape)
        data += noise

        smpl_df = pd.DataFrame(data, columns=cols)
        return smpl_df, fps, num_frames
