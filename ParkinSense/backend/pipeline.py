"""
ParkinSense — Unified Pipeline Orchestrator
============================================
Runs the complete end-to-end walking video analysis workflow:

    Video Upload
         │
         ▼
    Selectable Gait Extractor (MediaPipe OR 4D-Humans)
         │
         ▼
    Feature Adapter & Validation (290-dim contract)
         │
         ▼
    Central Prediction Service (Trained Random Forest Models)
         │
         ▼
    Unified Result JSON Schema (Clinical Dashboard)
"""

from __future__ import annotations

import json
import os
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
from backend.extractors import (
    BaseGaitExtractor,
    GaitExtractionResult,
    MediaPipeGaitExtractor,
    FourDHumansGaitExtractor,
    get_gait_extractor,
)
from backend.prediction import PredictionEngine
from backend.fourd_humans import GPUServiceUnavailableError, cleanup_job as cleanup_4d_job

logger = get_logger(__name__)

# Output directories
REMOTE_OUTPUT_DIR = _PROJECT_ROOT / "remote_output"
ensure_dir(REMOTE_OUTPUT_DIR)

# Supported pipelines
SUPPORTED_PIPELINES = ("mediapipe", "fourd_humans")


def generate_job_id() -> str:
    """Generate a unique job ID with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"job_{timestamp}_{short_uuid}"


def normalize_pipeline_selection(
    pipeline: Optional[str] = None,
    mode: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Resolve requested pipeline name and sub-mode.

    Returns:
        (canonical_pipeline_name, effective_submode)
    """
    # Check pipeline first, fall back to mode
    raw = (pipeline or mode or os.getenv("PIPELINE_MODE", "mediapipe")).lower().strip()

    if raw in ("mediapipe", "mp"):
        return "mediapipe", None

    if raw in ("fourd_humans", "4d-humans", "fourd", "hmr2", "mesh"):
        return "fourd_humans", (mode if mode not in ("fourd_humans", "hmr2") else None)

    if raw in ("ssh", "local", "auto", "mock", "remote"):
        # Legacy mode aliases that imply 4D-Humans
        return "fourd_humans", raw

    raise ValueError(
        f"Unknown pipeline selection '{raw}'. "
        f"Supported options: ['mediapipe', '4D-Humans']."
    )


def analyze_video(
    video_path: str | Path,
    job_id: Optional[str] = None,
    pipeline: Optional[str] = None,
    mode: Optional[str] = None,
    cleanup: bool = True,
    status_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Run the complete ParkinSense analysis pipeline on an uploaded walking video.

    Architecture flow:
      1. Dispatch to selected extractor (MediaPipe or 4D-Humans).
      2. Gait extractor generates 290-dim features and kinematic measurements.
      3. Feature Adapter validates against the strict 290-dim feature contract.
      4. Centralized PredictionEngine scores features using locked models.
      5. Generates standardized result JSON for the unified frontend dashboard.

    Args:
        video_path: Path to walking video file.
        job_id: Optional job identifier (generated automatically if None).
        pipeline: Selected pipeline ('mediapipe' or 'fourd_humans').
        mode: Optional execution sub-mode ('ssh', 'local', 'mock', 'auto').
        cleanup: Whether to clean up temporary execution files on completion.
        status_callback: Optional progress reporter func(step_idx, total_steps, message).

    Returns:
        Standardized result dictionary.
    """
    start_time = time.time()
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    if job_id is None:
        job_id = generate_job_id()

    # Normalize pipeline & sub-mode
    canonical_pipeline, effective_mode = normalize_pipeline_selection(pipeline, mode)

    logger.info("=" * 65)
    logger.info("PARKINSENSE ANALYSIS — Job: %s", job_id)
    logger.info("Pipeline: %s | Submode: %s", canonical_pipeline, effective_mode or "default")
    logger.info("Video: %s (%.1f MB)", video_path.name, video_path.stat().st_size / (1024 * 1024))
    logger.info("=" * 65)

    def cb(step: int, total: int, msg: str) -> None:
        if status_callback:
            try:
                status_callback(step, total, msg)
            except Exception as err:
                logger.warning("Status callback error: %s", err)

    # 1. Instantiate the selected Gait Extractor
    extractor = get_gait_extractor(canonical_pipeline, mode=effective_mode)

    try:
        # 2. Run Gait Extraction
        logger.info("Stage 1/2: Extracting gait via %s extractor...", canonical_pipeline)
        extraction: GaitExtractionResult = extractor.extract_gait(
            video_path=video_path,
            job_id=job_id,
            status_callback=cb,
        )

        # 3. Centralized Prediction Service
        cb(3, 4, "Passing 290 features to centralized classifier & explainability...")
        logger.info("Stage 2/2: Scoring via centralized PredictionEngine...")
        engine = PredictionEngine()
        prediction = engine.predict_features(
            feature_df=extraction.feature_df,
            source_pipeline=canonical_pipeline,
        )

        cb(4, 4, "Finalizing report & explainability dashboard...")
        total_duration = round(time.time() - start_time, 2)

        # 4. Assemble Unified Result Schema
        result: Dict[str, Any] = {
            "status": "success",
            "pipeline": canonical_pipeline,
            "job_id": job_id,
            "video_file": video_path.name,
            "num_frames": extraction.num_frames,
            "fps": extraction.fps,
            "video_duration_seconds": extraction.video_duration_seconds,
            "processing_time_seconds": total_duration,
            "pose_service_mode": extraction.pose_service_mode,
            "is_demo": extraction.is_demo,
            "pose_detection_quality": extraction.pose_detection_quality,
            "valid_landmark_percentage": extraction.valid_landmark_percentage,
            "parkinsons_detection": prediction["parkinsons_detection"],
            "severity_estimation": prediction["severity_estimation"],
            "explainability": prediction["explainability"],
            "model_information": prediction["model_information"],
            "dataset_performance": prediction["dataset_performance"],
            "disclaimer": prediction["disclaimer"],
            "gait_metrics": extraction.gait_metrics,
            "gait_timeseries": extraction.gait_timeseries,
            "key_gait_frames": extraction.key_gait_frames,
            "timing": extraction.timing,
            "metadata": {
                "num_frames": extraction.num_frames,
                "fps": extraction.fps,
                "video_duration_seconds": extraction.video_duration_seconds,
                "features_extracted": int(extraction.feature_df.shape[1]),
                "pd_model_features": prediction["model_information"]["n_features"],
                "pipeline_selected": canonical_pipeline,
                "pose_service_mode": extraction.pose_service_mode,
                "pose_detection_quality": extraction.pose_detection_quality,
                "valid_landmark_percentage": extraction.valid_landmark_percentage,
            },
        }

        if extraction.is_demo:
            result["demo_notice"] = (
                "DEMO / MOCK MODE: 4D-Humans GPU stack unavailable. "
                "Analysis was performed using sample SMPL gait dynamics for pipeline verification."
            )

        # Persist result JSON
        output_json = REMOTE_OUTPUT_DIR / f"{job_id}_result.json"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        result["output_file"] = str(output_json.name)

        logger.info("Pipeline completed successfully in %.2fs. Result saved: %s", total_duration, output_json.name)
        return result

    except GPUServiceUnavailableError:
        logger.error("Job %s failed: 4D-Humans GPU inference service unavailable.", job_id)
        raise
    except Exception as err:
        logger.exception("Job %s pipeline failure: %s", job_id, err)
        raise
    finally:
        if cleanup and canonical_pipeline == "fourd_humans":
            try:
                cleanup_4d_job(job_id)
            except Exception as e:
                logger.warning("Cleanup error (non-fatal): %s", e)
