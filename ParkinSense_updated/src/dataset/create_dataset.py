"""
ParkiSense — Dataset Builder
==============================
Iterates over every video in ``data/raw/videos/{control,parkinson}/``,
runs the full MediaPipe → Feature Extraction pipeline on each, and
assembles a labelled CSV ready for model training.

Output
------
``data/processed/dataset.csv`` with columns:

    left_knee_angle, right_knee_angle, …, dominant_stride_frequency, label

Usage (CLI):
    python src/dataset/create_dataset.py

    # Custom paths:
    python src/dataset/create_dataset.py \\
        --videos-dir  data/raw/videos \\
        --output-csv  data/processed/dataset.csv

Usage (Python API):
    from src.dataset.create_dataset import DatasetBuilder
    builder = DatasetBuilder()
    builder.build()
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Resolve project root for standalone execution
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import get_logger, ensure_dir
from src.pose.mediapipe_pose import PoseExtractor
from src.features.feature_extractor import GaitFeatureExtractor

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_VIDEOS_DIR  = str(_PROJECT_ROOT / "data" / "raw" / "videos")
DEFAULT_OUTPUT_CSV  = str(_PROJECT_ROOT / "data" / "processed" / "dataset.csv")
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

LABEL_MAP = {
    "control":   0,
    "parkinson": 1,
}


# ---------------------------------------------------------------------------
# DatasetBuilder
# ---------------------------------------------------------------------------

class DatasetBuilder:
    """
    Builds the labelled gait-feature dataset from a directory tree of videos.

    Expected input structure::

        videos_dir/
            control/
                *.mp4
            parkinson/
                *.mp4

    Args:
        videos_dir:    Root directory containing ``control/`` and ``parkinson/``
                       sub-folders.
        output_csv:    Path for the final ``dataset.csv``.
        fps:           Frame-rate passed to :class:`GaitFeatureExtractor`.
        min_detection: MediaPipe min detection confidence.
        min_tracking:  MediaPipe min tracking confidence.
    """

    def __init__(
        self,
        videos_dir:    str   = DEFAULT_VIDEOS_DIR,
        output_csv:    str   = DEFAULT_OUTPUT_CSV,
        fps:           float = 30.0,
        min_detection: float = 0.5,
        min_tracking:  float = 0.5,
    ) -> None:
        self.videos_dir    = Path(videos_dir)
        self.output_csv    = Path(output_csv)
        self.fps           = fps
        self.min_detection = min_detection
        self.min_tracking  = min_tracking

        self._pose_extractor     = PoseExtractor(
            min_detection_confidence=min_detection,
            min_tracking_confidence=min_tracking,
        )
        self._feature_extractor  = GaitFeatureExtractor(
            fps=fps,
            normalize=False,   # Normalization applied globally after all videos
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> pd.DataFrame:
        """
        Run the full build pipeline.

        Returns:
            DataFrame with feature rows and a ``label`` column.

        Raises:
            FileNotFoundError: If *videos_dir* does not exist.
        """
        if not self.videos_dir.exists():
            raise FileNotFoundError(f"Videos directory not found: {self.videos_dir}")

        video_list = self._discover_videos()
        if not video_list:
            logger.error(
                "No videos found in %s. "
                "Place .mp4 files under control/ and parkinson/ sub-folders.",
                self.videos_dir,
            )
            return pd.DataFrame()

        logger.info("Found %d videos — starting feature extraction …", len(video_list))

        rows: List[dict] = []
        failed: List[str] = []

        for video_path, label in tqdm(video_list, desc="Processing videos", unit="video"):
            try:
                feature_row = self._process_single_video(video_path, label)
                rows.append(feature_row)
                logger.info("✅ %s (label=%d) — OK", video_path.name, label)
            except Exception as exc:
                logger.warning("❌ %s — SKIPPED: %s", video_path.name, exc)
                failed.append(str(video_path))

        if not rows:
            logger.error("No features could be extracted. Dataset is empty.")
            return pd.DataFrame()

        dataset = pd.DataFrame(rows)

        # ── Global normalisation ────────────────────────────────────────
        from sklearn.preprocessing import StandardScaler

        feature_cols = [c for c in GaitFeatureExtractor.FEATURE_COLUMNS if c in dataset.columns]
        scaler = StandardScaler()
        dataset[feature_cols] = scaler.fit_transform(dataset[feature_cols])
        logger.info("Normalised %d feature columns with StandardScaler.", len(feature_cols))

        # ── Save ────────────────────────────────────────────────────────
        ensure_dir(self.output_csv.parent)
        dataset.to_csv(str(self.output_csv), index=False)
        logger.info("Dataset saved → %s  (%d rows × %d cols)", self.output_csv, *dataset.shape)

        if failed:
            logger.warning(
                "%d video(s) failed to process:\n%s",
                len(failed),
                "\n".join(f"  • {p}" for p in failed),
            )

        self._print_summary(dataset)
        return dataset

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discover_videos(self) -> List[Tuple[Path, int]]:
        """
        Walk the videos directory and return (video_path, label) pairs.
        """
        result: List[Tuple[Path, int]] = []
        for class_name, label in LABEL_MAP.items():
            class_dir = self.videos_dir / class_name
            if not class_dir.exists():
                logger.warning("Class directory missing: %s", class_dir)
                continue
            for fpath in sorted(class_dir.iterdir()):
                if fpath.suffix.lower() in SUPPORTED_EXTENSIONS:
                    result.append((fpath, label))
        return result

    def _process_single_video(self, video_path: Path, label: int) -> dict:
        """
        Run PoseExtractor + GaitFeatureExtractor on one video.

        Uses a temporary directory so intermediate CSVs don't pollute outputs/.

        Args:
            video_path: Absolute path to the video file.
            label:      Class label (0 or 1).

        Returns:
            Feature dict (including ``label``).
        """
        with tempfile.TemporaryDirectory(prefix="parkisense_") as tmpdir:
            landmarks_csv = os.path.join(tmpdir, "landmarks.csv")
            pose_ext = PoseExtractor(
                min_detection_confidence=self.min_detection,
                min_tracking_confidence=self.min_tracking,
                landmarks_csv=landmarks_csv,
                pose_video=os.path.join(tmpdir, "pose.mp4"),
            )
            pose_ext.extract_landmarks(str(video_path))

            feat_ext = GaitFeatureExtractor(
                features_csv=os.path.join(tmpdir, "features.csv"),
                fps=self.fps,
                normalize=False,
            )
            features = feat_ext.extract_features_from_df(
                pd.read_csv(landmarks_csv),
                label=label,
            )

        features["video_file"] = video_path.name
        return features

    @staticmethod
    def _print_summary(df: pd.DataFrame) -> None:
        """Print a brief class-balance summary."""
        if "label" not in df.columns:
            return
        counts = df["label"].value_counts().sort_index()
        logger.info("── Class distribution ─────────────────────────────")
        for lbl, cnt in counts.items():
            name = {v: k for k, v in LABEL_MAP.items()}.get(lbl, str(lbl))
            logger.info("  %-12s (label=%s): %d samples", name, lbl, cnt)
        logger.info("───────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="create_dataset",
        description="ParkiSense — Dataset Builder",
    )
    p.add_argument("--videos-dir", "-v", default=DEFAULT_VIDEOS_DIR, help="Root videos directory")
    p.add_argument("--output-csv", "-o", default=DEFAULT_OUTPUT_CSV,  help="Output dataset CSV path")
    p.add_argument("--fps",        "-f", type=float, default=30.0,     help="Video FPS for time features")
    p.add_argument("--min-detection", type=float, default=0.5,        help="MediaPipe min detection conf.")
    p.add_argument("--min-tracking",  type=float, default=0.5,        help="MediaPipe min tracking conf.")
    return p


def main() -> None:
    args    = _build_arg_parser().parse_args()
    builder = DatasetBuilder(
        videos_dir=args.videos_dir,
        output_csv=args.output_csv,
        fps=args.fps,
        min_detection=args.min_detection,
        min_tracking=args.min_tracking,
    )
    builder.build()


if __name__ == "__main__":
    main()
