#!/usr/bin/env python3
"""
ParkinSense — Single AI Pipeline Orchestrator
==============================================
Runs the complete ParkinSense end-to-end gait analysis pipeline on a walking video:

    Walking Video (.mp4 / .avi / .mov)
      ↓
    4D-Humans / HMR2 (SMPL pose extraction)
      ↓
    SMPL 216-dim matrix sequence
      ↓
    Gait feature extraction (290 dims)
      ↓
    Trained ML models (Parkinson's Classifier & UPDRS Severity Estimator)
      ↓
    Prediction + UPDRS + Clinical Explainability
      ↓
    Structured JSON Output

Usage:
    python run_pipeline.py path/to/walking_video.mp4
    python run_pipeline.py path/to/walking_video.mp4 --mode mock
    python run_pipeline.py path/to/walking_video.mp4 --mode ssh
    python run_pipeline.py path/to/walking_video.mp4 --output-json result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Resolve project root
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.pipeline import analyze_video, print_result
from src.utils import get_logger, ensure_dir

logger = get_logger("run_pipeline")


def run(
    video_path: str | Path,
    mode: str = "auto",
    job_id: str | None = None,
    cleanup: bool = True,
    output_json: str | Path | None = None,
) -> dict:
    """
    Run the unified ParkinSense AI pipeline programmatically.

    Args:
        video_path:  Path to the input walking video file.
        mode:        Processing mode ("auto", "ssh", "local", "mock").
        job_id:      Optional custom job ID.
        cleanup:     Whether to clean up temporary intermediate files.
        output_json: Optional file path to save output JSON result.

    Returns:
        Dictionary containing prediction results, probabilities, and explainability.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video file not found: {video_path}")

    logger.info("Starting ParkinSense AI Pipeline on: %s", video_path)

    result = analyze_video(
        video_path=video_path,
        job_id=job_id,
        mode=mode,
        cleanup=cleanup,
    )

    if output_json:
        output_path = Path(output_json)
        ensure_dir(output_path.parent)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info("Saved result to: %s", output_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ParkinSense — AI-Powered Parkinson's Detection & Gait Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py video.mp4
  python run_pipeline.py video.mp4 --mode mock
  python run_pipeline.py video.mp4 --mode ssh --output-json outputs/result.json
        """,
    )
    parser.add_argument("video", help="Path to walking video file (.mp4, .avi, .mov, .mkv)")
    parser.add_argument(
        "--mode",
        choices=["auto", "ssh", "local", "mock"],
        default="auto",
        help="Pose service processing mode (default: auto)",
    )
    parser.add_argument("--job-id", type=str, default=None, help="Custom job ID")
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Retain intermediate SMPL and video files",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write final JSON result file",
    )

    args = parser.parse_args()

    try:
        result = run(
            video_path=args.video,
            mode=args.mode,
            job_id=args.job_id,
            cleanup=not args.no_cleanup,
            output_json=args.output_json,
        )

        print_result(result)

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[PIPELINE FAILURE] {e}", file=sys.stderr)
        logger.exception("Pipeline error")
        sys.exit(1)


if __name__ == "__main__":
    main()
