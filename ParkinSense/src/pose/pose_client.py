"""
ParkinSense — Remote & Mock Pose Client
=======================================
Handles communication between the ParkinSense backend (on Mac) and the external
4D-Humans pose extraction service (on Friend's GPU machine).

Environment variables:
- FOURD_HUMANS_URL  (default: "http://localhost:8000")
- POSE_SERVICE_MODE ("remote" or "mock", default: "mock")

API Contract for 4D-Humans Server:
- GET  /health          -> {"status": "ok", "service": "4d-humans"}
- POST /extract-smpl    -> multipart/form-data video file upload
                           returns JSON with {"status": "success", "fps": float, "smpl_data": [...]}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import requests

from src.utils import get_logger
from src.pose.mock_pose_service import MockPoseService

logger = get_logger(__name__)

# Environment variable defaults
DEFAULT_SERVICE_URL = os.getenv("FOURD_HUMANS_URL", "http://localhost:8000")
DEFAULT_SERVICE_MODE = os.getenv("POSE_SERVICE_MODE", "mock").lower()


class PoseClient:
    """
    Client for extracting SMPL pose data from walking videos via remote API or local mock.
    """

    def __init__(
        self,
        server_url: str | None = None,
        mode: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.server_url = (server_url or DEFAULT_SERVICE_URL).rstrip("/")
        self.mode = (mode or DEFAULT_SERVICE_MODE).lower()
        self.timeout = timeout
        self._mock_service = MockPoseService()

    def check_health(self) -> Dict[str, Any]:
        """
        Check health of the connected pose service.

        Returns:
            Dict containing status, mode, and server URL info.
        """
        if self.mode == "mock":
            return {
                "status": "ok",
                "mode": "mock",
                "message": "Local Mock Pose Service active.",
                "server_url": self.server_url,
            }

        health_url = f"{self.server_url}/health"
        try:
            resp = requests.get(health_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                data["mode"] = "remote"
                data["server_url"] = self.server_url
                return data
            else:
                return {
                    "status": "error",
                    "mode": "remote",
                    "server_url": self.server_url,
                    "error": f"Server responded with status HTTP {resp.status_code}",
                }
        except requests.RequestException as err:
            return {
                "status": "offline",
                "mode": "remote",
                "server_url": self.server_url,
                "error": f"Failed to connect to 4D-Humans server at {self.server_url}: {err}",
            }

    def extract_smpl(self, video_path: str | Path) -> Tuple[pd.DataFrame, float, int, str]:
        """
        Extract SMPL rotation matrix DataFrame from video_path.

        Args:
            video_path: Path to local walking video.

        Returns:
            Tuple of (smpl_df, fps, num_frames, mode_used)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Input video file not found: {video_path}")

        if self.mode == "mock":
            logger.info("Using Local Mock Pose Service...")
            df, fps, num_frames = self._mock_service.extract_smpl_from_video(video_path)
            return df, fps, num_frames, "mock"

        # Remote mode
        logger.info("Sending video to remote 4D-Humans server: %s/extract-smpl", self.server_url)
        extract_url = f"{self.server_url}/extract-smpl"

        try:
            with open(video_path, "rb") as vf:
                files = {"file": (video_path.name, vf, "video/mp4")}
                resp = requests.post(extract_url, files=files, timeout=self.timeout)

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Remote 4D-Humans service failed (HTTP {resp.status_code}): {resp.text}"
                )

            data = resp.json()
            if data.get("status") != "success":
                raise RuntimeError(f"4D-Humans extraction error: {data.get('message', 'Unknown error')}")

            fps = float(data.get("fps", 30.0))
            smpl_data = data.get("smpl_data", [])

            if not smpl_data:
                raise ValueError("Remote 4D-Humans returned empty SMPL data.")

            # Convert to DataFrame
            # Column naming matching expectation (216 rotation matrix floats)
            orient_cols = [f"global_orient_{i}" for i in range(9)]
            body_cols = [f"body_pose_{i}" for i in range(207)]
            cols = orient_cols + body_cols

            smpl_df = pd.DataFrame(smpl_data, columns=cols)
            num_frames = len(smpl_df)

            logger.info("Successfully received %d SMPL frames from remote 4D-Humans service.", num_frames)
            return smpl_df, fps, num_frames, "remote"

        except requests.RequestException as err:
            logger.error("Failed to connect to external 4D-Humans server: %s", err)
            raise ConnectionError(
                f"External 4D-Humans GPU server at {self.server_url} is unavailable. "
                f"Set POSE_SERVICE_MODE=mock for local testing. Error: {err}"
            ) from err
