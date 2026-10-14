"""
ParkiSense — MediaPipe Pose Extractor
======================================
Extracts all 33 MediaPipe landmarks from a walking video and:
  • saves a per-frame CSV of landmark coordinates
  • writes an annotated output video with the skeleton overlay

Usage (CLI):
    python src/pose/mediapipe_pose.py --input uploads/video.mp4

Usage (Python API):
    from src.pose.mediapipe_pose import PoseExtractor
    extractor = PoseExtractor()
    extractor.extract_landmarks("uploads/video.mp4")
    extractor.generate_annotated_video("uploads/video.mp4")
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Allow running as a standalone script from any working directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import get_logger, ensure_dir, LANDMARK_NAMES

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_VIDEO_INPUT  = str(_PROJECT_ROOT / "uploads" / "video.mp4")
DEFAULT_LANDMARKS_CSV = str(_PROJECT_ROOT / "outputs" / "landmarks.csv")
DEFAULT_POSE_VIDEO   = str(_PROJECT_ROOT / "outputs" / "pose_video.mp4")


# ---------------------------------------------------------------------------
# PoseExtractor
# ---------------------------------------------------------------------------

class PoseExtractor:
    """
    Wraps MediaPipe Pose to extract 33 body landmarks from a video.

    Attributes:
        min_detection_confidence: Minimum confidence for initial detection.
        min_tracking_confidence:  Minimum confidence for landmark tracking.
        model_complexity:         MediaPipe model complexity (0, 1, or 2).
        landmarks_csv:            Output path for the landmark CSV.
        pose_video:               Output path for the annotated video.
    """

    # Drawing colour palette
    _LANDMARK_COLOUR    = (0, 255, 170)   # Mint green
    _CONNECTION_COLOUR  = (255, 100, 0)   # Orange
    _TEXT_COLOUR        = (255, 255, 255) # White

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float  = 0.5,
        model_complexity: int           = 1,
        landmarks_csv: str              = DEFAULT_LANDMARKS_CSV,
        pose_video: str                 = DEFAULT_POSE_VIDEO,
    ) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence  = min_tracking_confidence
        self.model_complexity         = model_complexity
        self.landmarks_csv            = landmarks_csv
        self.pose_video               = pose_video

        self._mp_pose  = mp.solutions.pose
        self._mp_draw  = mp.solutions.drawing_utils
        self._mp_style = mp.solutions.drawing_styles

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_landmarks(self, video_path: str = DEFAULT_VIDEO_INPUT) -> str:
        """
        Process *video_path* with MediaPipe Pose and write all landmarks to CSV.

        CSV schema:
            frame, landmark_id, landmark_name, x, y, z, visibility

        Args:
            video_path: Path to the input video file.

        Returns:
            Path to the generated CSV file.

        Raises:
            FileNotFoundError: If *video_path* does not exist.
            RuntimeError:      If the video cannot be opened.
        """
        video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        cap = self._open_capture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0

        ensure_dir(Path(self.landmarks_csv).parent)

        logger.info("Extracting landmarks → %s", self.landmarks_csv)
        logger.info("Video: %s | FPS: %.1f | Total frames: %d", video_path, fps, total_frames)

        row_count = 0
        frame_no  = 0

        with open(self.landmarks_csv, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["frame", "landmark_id", "landmark_name", "x", "y", "z", "visibility"])

            with self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            ) as pose:
                with tqdm(total=total_frames, desc="Extracting landmarks", unit="frame") as pbar:
                    while cap.isOpened():
                        ok, frame = cap.read()
                        if not ok:
                            break
                        frame_no += 1
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = pose.process(rgb)

                        if results.pose_landmarks:
                            for idx, lm in enumerate(results.pose_landmarks.landmark):
                                name = LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else f"UNKNOWN_{idx}"
                                writer.writerow([
                                    frame_no, idx, name,
                                    round(lm.x, 6),
                                    round(lm.y, 6),
                                    round(lm.z, 6),
                                    round(lm.visibility, 6),
                                ])
                                row_count += 1
                        pbar.update(1)

        cap.release()

        logger.info("✅ Landmark extraction complete — %d rows written to %s",
                    row_count, self.landmarks_csv)
        return self.landmarks_csv

    def generate_annotated_video(self, video_path: str = DEFAULT_VIDEO_INPUT) -> str:
        """
        Re-process *video_path* and write an annotated video with skeleton overlay.

        Args:
            video_path: Path to the input video file.

        Returns:
            Path to the generated annotated video.

        Raises:
            FileNotFoundError: If *video_path* does not exist.
        """
        video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        cap          = self._open_capture(video_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ensure_dir(Path(self.pose_video).parent)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.pose_video, fourcc, fps, (width, height))
        if not writer.isOpened():
            logger.warning("VideoWriter failed to open — annotated video will NOT be saved.")

        logger.info("Generating annotated video → %s", self.pose_video)

        lm_spec  = self._mp_draw.DrawingSpec(color=self._LANDMARK_COLOUR,   thickness=2, circle_radius=3)
        con_spec = self._mp_draw.DrawingSpec(color=self._CONNECTION_COLOUR,  thickness=2)

        with self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        ) as pose:
            with tqdm(total=total_frames, desc="Rendering pose video", unit="frame") as pbar:
                frame_no = 0
                while cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_no += 1

                    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb)

                    if results.pose_landmarks:
                        self._mp_draw.draw_landmarks(
                            frame,
                            results.pose_landmarks,
                            self._mp_pose.POSE_CONNECTIONS,
                            lm_spec,
                            con_spec,
                        )

                    # Frame counter overlay
                    cv2.putText(
                        frame,
                        f"Frame {frame_no}/{total_frames}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        self._TEXT_COLOUR,
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        "ParkiSense",
                        (20, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        self._TEXT_COLOUR,
                        1,
                        cv2.LINE_AA,
                    )

                    if writer.isOpened():
                        writer.write(frame)
                    pbar.update(1)

        cap.release()
        writer.release()

        logger.info("✅ Annotated video saved → %s", self.pose_video)
        return self.pose_video

    def run_full_pipeline(self, video_path: str = DEFAULT_VIDEO_INPUT) -> tuple[str, str]:
        """
        Convenience method: extract landmarks AND generate the annotated video.

        Args:
            video_path: Path to the input walking video.

        Returns:
            Tuple of (landmarks_csv_path, pose_video_path).
        """
        csv_path   = self.extract_landmarks(video_path)
        video_path_out = self.generate_annotated_video(video_path)
        return csv_path, video_path_out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _open_capture(video_path: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        return cap


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mediapipe_pose",
        description="ParkiSense — MediaPipe Pose Landmark Extractor",
    )
    p.add_argument(
        "--input", "-i",
        default=DEFAULT_VIDEO_INPUT,
        help="Path to input video (default: uploads/video.mp4)",
    )
    p.add_argument(
        "--landmarks-csv", "-l",
        default=DEFAULT_LANDMARKS_CSV,
        help="Output path for landmarks CSV",
    )
    p.add_argument(
        "--pose-video", "-v",
        default=DEFAULT_POSE_VIDEO,
        help="Output path for annotated video",
    )
    p.add_argument(
        "--min-detection", "-d",
        type=float,
        default=0.5,
        help="Min detection confidence (0.0–1.0)",
    )
    p.add_argument(
        "--min-tracking", "-t",
        type=float,
        default=0.5,
        help="Min tracking confidence (0.0–1.0)",
    )
    p.add_argument(
        "--complexity", "-c",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="MediaPipe model complexity (0=lite, 1=full, 2=heavy)",
    )
    p.add_argument(
        "--landmarks-only",
        action="store_true",
        help="Only extract landmarks CSV; skip annotated video generation",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    extractor = PoseExtractor(
        min_detection_confidence=args.min_detection,
        min_tracking_confidence=args.min_tracking,
        model_complexity=args.complexity,
        landmarks_csv=args.landmarks_csv,
        pose_video=args.pose_video,
    )

    if args.landmarks_only:
        extractor.extract_landmarks(args.input)
    else:
        extractor.run_full_pipeline(args.input)


if __name__ == "__main__":
    main()
