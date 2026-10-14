"""
ParkinSense — Complete Pipeline Orchestrator
=============================================
Runs the full end-to-end pipeline:

    Video
      ↓
    4D-Humans (SMPL extraction)
      ↓
    SMPL CSV
      ↓
    Feature extraction (290 dims)
      ↓
    ML models (PD detection + Severity)
      ↓
    Prediction + Explainability
      ↓
    JSON result

Usage (CLI):
    python backend/pipeline.py path/to/video.mp4
    python backend/pipeline.py path/to/video.mp4 --mode mock
    python backend/pipeline.py path/to/video.mp4 --mode auto

Usage (Python):
    from backend.pipeline import analyze_video
    result = analyze_video("path/to/video.mp4")
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Resolve project root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)

# Import backend modules
from backend.fourd_humans import process_video as run_4d_humans, cleanup_job
from backend.prediction import PredictionEngine

# Output directories
REMOTE_OUTPUT_DIR = _PROJECT_ROOT / "remote_output"


def generate_job_id() -> str:
    """Generate a unique job ID with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"job_{timestamp}_{short_uuid}"


def analyze_video(
    video_path: str | Path,
    job_id: Optional[str] = None,
    mode: Optional[str] = None,
    cleanup: bool = True,
    status_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Run the complete ParkinSense analysis pipeline on a walking video.

    This is the main entry point that orchestrates the full pipeline:

        Video → 4D-Humans → SMPL → Features → ML Models → Prediction

    Args:
        video_path:      Path to the input walking video file.
        job_id:          Optional unique job identifier (auto-generated if not provided).
        mode:            Processing mode ("auto", "local", "remote", "mock").
        cleanup:         Whether to clean up temporary files after completion.
        status_callback: Optional callback func(step_idx, total_steps, message) for status tracking.

    Returns:
        Dictionary containing the complete prediction result with metadata.

    Raises:
        FileNotFoundError: If video file does not exist.
        RuntimeError:       If any pipeline step fails.
    """
    start_time = time.time()

    def update_status(step_idx: int, total_steps: int, msg: str) -> None:
        if status_callback:
            try:
                status_callback(step_idx, total_steps, msg)
            except Exception as cb_err:
                logger.warning("Status callback failed: %s", cb_err)

    # Validate input
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    # Generate job ID
    if job_id is None:
        job_id = generate_job_id()

    logger.info("=" * 60)
    logger.info("PARKINSENSE PIPELINE — Job: %s", job_id)
    logger.info("=" * 60)
    logger.info("Video: %s (%.1f MB)", video_path.name, video_path.stat().st_size / (1024 * 1024))

    result = {
        "status": "success",
        "job_id": job_id,
        "video_file": video_path.name,
    }

    try:
        # ── Step 1: Video Upload & Validation Complete ────────────────
        update_status(1, 4, "Video uploaded & validated. Initializing 4D pose extraction...")

        # ── Step 2: 4D-Humans Processing ──────────────────────────────
        logger.info("Step 2/4 — 4D-Humans SMPL Extraction...")
        update_status(2, 4, "Running 4D-Humans pose extraction (SMPL dynamics)...")
        step_start = time.time()

        smpl_csv_path, fps, mode_used, is_demo = run_4d_humans(video_path, job_id, mode=mode)

        step_duration = time.time() - step_start
        logger.info("  ✓ SMPL generated: %s (%.1fs)", smpl_csv_path.name, step_duration)

        result["smpl_file"] = str(smpl_csv_path.name)
        result["fps"] = round(fps, 2)
        result["pose_service_mode"] = mode_used
        result["is_demo"] = is_demo
        if is_demo:
            result["demo_notice"] = (
                "DEMO / MOCK MODE: 4D-Humans GPU stack unavailable. "
                "Analysis was performed using sample SMPL gait dynamics for pipeline verification."
            )

        # ── Step 3: Feature Extraction + Model Prediction ──────────────
        logger.info("Step 3/4 — Feature Extraction & Model Prediction...")
        update_status(3, 4, "Extracting 290 gait features & running ML classifiers...")
        step_start = time.time()

        engine = PredictionEngine()
        prediction = engine.predict(smpl_csv_path, fps=fps)

        step_duration = time.time() - step_start
        logger.info("  ✓ Prediction complete (%.1fs)", step_duration)

        # Merge prediction into result
        result.update(prediction)

        # ── Step 4: Save Results ───────────────────────────────────────
        logger.info("Step 4/4 — Finalizing & saving results...")
        update_status(4, 4, "Generating explainability dashboard & finalizing results...")
        output_json = REMOTE_OUTPUT_DIR / f"{job_id}_result.json"
        ensure_dir(REMOTE_OUTPUT_DIR)

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

        result["output_file"] = str(output_json.name)
        logger.info("  ✓ Results saved: %s", output_json.name)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error("Pipeline failed: %s", e)
        raise

    finally:
        # Cleanup
        if cleanup:
            try:
                cleanup_job(job_id)
            except Exception as e:
                logger.warning("Cleanup failed (non-fatal): %s", e)

    # Total duration
    total_duration = time.time() - start_time
    result["processing_time_seconds"] = round(total_duration, 2)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE — %.1fs total", total_duration)
    logger.info("=" * 60)

    return result


def print_result(result: Dict[str, Any]) -> None:
    """Pretty-print the prediction result to terminal."""
    print()
    print("=" * 60)
    print("               PARKINSENSE RESULT")
    print("=" * 60)
    print(f"  Job ID        : {result.get('job_id', 'N/A')}")
    print(f"  Video File    : {result.get('video_file', 'N/A')}")
    print(f"  Processing    : {result.get('processing_time_seconds', 'N/A')}s")
    print("-" * 60)

    if result.get("status") == "error":
        print(f"  ERROR: {result.get('error', 'Unknown error')}")
        print("=" * 60)
        return

    pd_info = result.get("parkinsons_detection", {})
    print(f"  PD Detection  : {pd_info.get('diagnosis', 'N/A')}")
    print(f"  Confidence    : {pd_info.get('probability', 'N/A')}%")
    print(f"  Risk Level    : {pd_info.get('risk_level', 'N/A')}")

    probs = pd_info.get("class_probabilities", {})
    if probs:
        print(f"    Healthy     : {probs.get('Healthy', 'N/A')}%")
        print(f"    Parkinsons  : {probs.get('Parkinsons', 'N/A')}%")

    print("-" * 60)

    sev_info = result.get("severity_estimation", {})
    print(f"  Severity      : {sev_info.get('description', 'N/A')}")

    sev_probs = sev_info.get("class_probabilities")
    if sev_probs:
        for i, p in enumerate(sev_probs):
            print(f"    Score {i}      : {round(p * 100, 1)}%")

    print("-" * 60)

    indicators = result.get("explainability", {}).get("important_gait_indicators", [])
    if indicators:
        print("  Top Clinical Gait Indicators:")
        for item in indicators:
            print(f"    • {item.get('indicator', 'N/A')} ({item.get('joint_name', 'N/A')}) — Importance: {item.get('importance', 'N/A')}")

    meta = result.get("metadata", {})
    if meta:
        print("-" * 60)
        print(f"  Frames: {meta.get('num_frames', 'N/A')} | FPS: {meta.get('fps', 'N/A')} | Features: {meta.get('features_extracted', 'N/A')}")

    print("=" * 60)
    print("  Note: This is an AI research model, not a clinical diagnosis.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ParkinSense — Complete Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backend/pipeline.py video/input/walking.mp4
  python backend/pipeline.py video/input/walking.mp4 --mode mock
  python backend/pipeline.py video/input/walking.mp4 --mode auto
        """,
    )
    parser.add_argument("video", help="Path to walking video file")
    parser.add_argument(
        "--mode",
        choices=["auto", "local", "remote", "mock"],
        default="auto",
        help="Processing mode (default: auto-detect)",
    )
    parser.add_argument("--job-id", type=str, default=None, help="Custom job ID")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep temporary files")
    parser.add_argument("--output-json", type=str, default=None, help="Save result to JSON file")
    args = parser.parse_args()

    try:
        result = analyze_video(
            video_path=args.video,
            job_id=args.job_id,
            mode=args.mode,
            cleanup=not args.no_cleanup,
        )

        print_result(result)

        if args.output_json:
            output_path = Path(args.output_json)
            ensure_dir(output_path.parent)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Result saved to: {output_path}")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nPIPELINE FAILED: {e}", file=sys.stderr)
        logger.exception("Pipeline error")
        sys.exit(1)
