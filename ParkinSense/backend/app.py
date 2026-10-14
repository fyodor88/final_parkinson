"""
ParkinSense — Flask Backend API Server
======================================
Provides RESTful API endpoints for the ParkinSense web frontend.

Endpoints:
- GET  /api/health        → Check server status & available pipelines
- GET  /api/health/gpu    → Check GPU service health
- POST /api/predict       → Upload video & pipeline selection, start background AI pipeline
- GET  /api/status/<id>   → Poll job status & fetch final result JSON
- GET  /                  → Serve frontend UI
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

from flask import Flask, request, jsonify, send_from_directory

# Resolve project root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import get_logger, ensure_dir
from backend.pipeline import (
    analyze_video,
    generate_job_id,
    normalize_pipeline_selection,
    SUPPORTED_PIPELINES,
)
from backend.fourd_humans import GPUServiceUnavailableError, check_gpu_health

logger = get_logger(__name__)


def _friendly_error_message(raw_message: str | Exception) -> str:
    """Return user-safe error content without exposing internal Python stack traces."""
    text = str(raw_message or "Processing failed.")
    lower = text.lower()

    if (
        isinstance(raw_message, GPUServiceUnavailableError)
        or "gpu service" in lower
        or "gpu machine" in lower
        or "4d-humans" in lower
        or "hydra" in lower
    ):
        return (
            "The 4D-Humans GPU inference service is currently unavailable. "
            "The remote GPU server is offline or unreachable via SSH, and local GPU acceleration "
            "is not supported on macOS M1. Local MediaPipe analysis remains fully available."
        )
    if "media" in lower and ("could not detect" in lower or "no pose" in lower or "landmarks" in lower):
        return "Unable to obtain reliable gait data from this video. Please upload a clearer walking video with the full body visible."
    if "insufficient" in lower or "minimum required" in lower or "detected landmarks" in lower:
        return "Unable to obtain reliable gait data from this video. Please upload a clearer walking video with the full body visible."
    if "not found" in lower or "no video" in lower or "empty" in lower:
        return "No valid video file was provided. Please upload a supported walking video."
    if "unsupported format" in lower or "supported formats" in lower:
        return text
    if "corrupt" in lower or "cannot open video" in lower:
        return "The uploaded video could not be opened or appears to be corrupted. Please try another file."
    if "pipeline selection" in lower or "invalid pipeline" in lower:
        return text
    if "model" in lower or "feature" in lower:
        return "Video processing failed during gait feature extraction. Please try a clearer walking video."
    return "Processing failed. Please check the uploaded video or try a different clip."


app = Flask(__name__, static_folder=str(_PROJECT_ROOT / "frontend" / "dist"), static_url_path="")

# Directories & settings
UPLOAD_DIR = _PROJECT_ROOT / "uploads"
REMOTE_OUTPUT_DIR = _PROJECT_ROOT / "remote_output"
GAIT_FRAMES_DIR = _PROJECT_ROOT / "static" / "gait_frames"
ensure_dir(UPLOAD_DIR)
ensure_dir(REMOTE_OUTPUT_DIR)
ensure_dir(GAIT_FRAMES_DIR)
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# In-memory job state store for background tasks
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


@app.after_request
def add_cors_headers(response):
    """Enable CORS for browser frontend interactions."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/", methods=["GET"])
@app.route("/<path:path>", methods=["GET"])
def serve_index(path: str = ""):
    """Serve the React frontend SPA build if present, or fallback index.html."""
    dist_dir = _PROJECT_ROOT / "frontend" / "dist"
    frontend_dir = _PROJECT_ROOT / "frontend"

    if (dist_dir / "index.html").exists():
        target = dist_dir / path
        if path and target.exists() and target.is_file():
            return send_from_directory(dist_dir, path)
        return send_from_directory(dist_dir, "index.html")

    if (frontend_dir / "index.html").exists():
        target = frontend_dir / path
        if path and target.exists() and target.is_file():
            return send_from_directory(frontend_dir, path)
        return send_from_directory(frontend_dir, "index.html")

    return jsonify({
        "message": "ParkinSense Backend API Server active.",
        "supported_pipelines": list(SUPPORTED_PIPELINES),
        "endpoints": ["GET /api/health", "POST /api/predict", "GET /api/status/<job_id>"],
    })


@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health_check():
    """Health status check endpoint reporting server & pipeline readiness."""
    gpu_health = check_gpu_health()
    return jsonify({
        "status": "healthy",
        "service": "ParkinSense Backend API",
        "timestamp": time.time(),
        "pipelines": {
            "mediapipe": {
                "available": True,
                "description": "Local 33-landmark pose extraction (fast, CPU/local)",
            },
            "fourd_humans": {
                "available": bool(gpu_health.get("available", False)),
                "service": "4D-Humans HMR2",
                "description": "3D human mesh SMPL reconstruction (requires GPU server)",
                "status": "online" if gpu_health.get("available") else "offline",
            },
        },
    }), 200


@app.route("/health/gpu", methods=["GET"])
@app.route("/api/health/gpu", methods=["GET"])
def gpu_health_check():
    """
    GPU inference service health check.
    Verifies SSH reachability & 4D-Humans status on GPU machine.
    """
    try:
        health = check_gpu_health()
        status_code = 200 if health["available"] else 503
        return jsonify(health), status_code
    except Exception as err:
        logger.exception("GPU health check error: %s", err)
        return jsonify({
            "available": False,
            "service": "4D-Humans",
            "error": str(err),
        }), 503


def _run_background_pipeline(
    video_path: Path,
    job_id: str,
    pipeline: str,
    mode: str | None,
) -> None:
    """Background worker function executing the full AI pipeline."""
    def status_cb(step_idx: int, total_steps: int, message: str) -> None:
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["step_index"] = step_idx
                JOBS[job_id]["total_steps"] = total_steps
                JOBS[job_id]["current_step"] = message
                JOBS[job_id]["progress_percent"] = int((step_idx / total_steps) * 100)

    try:
        logger.info("Starting background analysis for job %s (Pipeline: %s)...", job_id, pipeline)
        result = analyze_video(
            video_path=video_path,
            job_id=job_id,
            pipeline=pipeline,
            mode=mode,
            cleanup=False,
            status_callback=status_cb,
        )

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["step_index"] = 4
            JOBS[job_id]["progress_percent"] = 100
            JOBS[job_id]["current_step"] = "Analysis complete!"
            JOBS[job_id]["result"] = result
            JOBS[job_id]["completed_at"] = time.time()

        logger.info("Job %s completed successfully via pipeline '%s'.", job_id, pipeline)

    except GPUServiceUnavailableError as err:
        logger.error("Job %s GPU service unavailable: %s", job_id, err)
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error_code"] = "GPU_SERVICE_UNAVAILABLE"
            JOBS[job_id]["error"] = _friendly_error_message(err)
    except Exception as err:
        logger.exception("Job %s pipeline failure: %s", job_id, err)
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error_code"] = "PIPELINE_ERROR"
            JOBS[job_id]["error"] = _friendly_error_message(err)
    finally:
        if video_path.exists():
            try:
                video_path.unlink()
            except OSError:
                pass


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict_video():
    """
    Video prediction endpoint. Accepts multipart/form-data video upload and selected pipeline.

    Form/Query Parameters:
      - pipeline: 'mediapipe' (default) or 'fourd_humans'
      - mode: optional submode override ('ssh', 'local', 'mock')
      - sync: 'true' to execute synchronously for debugging/tests
    """
    if "file" not in request.files and "video" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No video file provided. Expecting 'file' or 'video' form field.",
        }), 400

    file = request.files.get("file") or request.files.get("video")

    if not file or file.filename == "":
        return jsonify({
            "status": "error",
            "message": "Selected video file is empty.",
        }), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "status": "error",
            "message": f"Unsupported format '{ext}'. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        }), 400

    # Pipeline selection
    raw_pipeline = request.args.get("pipeline") or request.form.get("pipeline")
    raw_mode = request.args.get("mode") or request.form.get("mode")

    try:
        canonical_pipeline, submode = normalize_pipeline_selection(
            pipeline=raw_pipeline,
            mode=raw_mode,
        )
    except ValueError as val_err:
        return jsonify({
            "status": "error",
            "message": str(val_err),
        }), 400

    job_id = generate_job_id()
    job_upload_dir = UPLOAD_DIR / job_id
    ensure_dir(job_upload_dir)
    temp_path = job_upload_dir / file.filename

    is_sync = request.args.get("sync", "").lower() in ("true", "1")

    file.save(str(temp_path))
    logger.info(
        "Uploaded video saved for job %s: %s (Pipeline: %s, Submode: %s)",
        job_id,
        temp_path,
        canonical_pipeline,
        submode or "default",
    )

    # Initialize job state
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "pipeline": canonical_pipeline,
            "status": "processing",
            "step_index": 1,
            "total_steps": 4,
            "progress_percent": 15,
            "current_step": f"Video uploaded. Initializing {canonical_pipeline} gait extraction...",
            "created_at": time.time(),
            "video_file": file.filename,
            "result": None,
            "error": None,
        }

    if is_sync:
        # Synchronous execution mode (useful for integration tests)
        _run_background_pipeline(temp_path, job_id, canonical_pipeline, submode)
        job_data = JOBS.get(job_id, {})
        if job_data.get("status") == "completed":
            return jsonify(job_data["result"]), 200
        else:
            status_code = 503 if job_data.get("error_code") == "GPU_SERVICE_UNAVAILABLE" else 500
            return jsonify({
                "status": "error",
                "message": job_data.get("error", "Processing failed."),
            }), status_code

    # Asynchronous background job mode
    thread = threading.Thread(
        target=_run_background_pipeline,
        args=(temp_path, job_id, canonical_pipeline, submode),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "status": "processing",
        "job_id": job_id,
        "pipeline": canonical_pipeline,
        "message": f"Video uploaded successfully. {canonical_pipeline.capitalize()} pipeline started.",
        "status_url": f"/api/status/{job_id}",
    }), 202


@app.route("/static/gait_frames/<job_id>/<path:filename>", methods=["GET"])
@app.route("/static/gait_frames/<path:filename>", methods=["GET"])
@app.route("/static/<path:filename>", methods=["GET"])
def serve_gait_frame(filename: str, job_id: str | None = None):
    """Serve gait frame images extracted from uploaded videos."""
    if job_id:
        target_dir = _PROJECT_ROOT / "static" / "gait_frames" / job_id
        if (target_dir / filename).exists():
            return send_from_directory(target_dir, filename)

    gait_frames_dir = _PROJECT_ROOT / "static" / "gait_frames"
    if (gait_frames_dir / filename).exists():
        return send_from_directory(gait_frames_dir, filename)

    static_dir = _PROJECT_ROOT / "static"
    return send_from_directory(static_dir, filename)


@app.route("/status/<job_id>", methods=["GET"])
@app.route("/api/status/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    """Status polling endpoint returning live processing state & final results."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)

    if not job:
        result_json_path = REMOTE_OUTPUT_DIR / f"{job_id}_result.json"
        if result_json_path.exists():
            try:
                import json
                with open(result_json_path, "r", encoding="utf-8") as f:
                    res_data = json.load(f)
                return jsonify({
                    "job_id": job_id,
                    "pipeline": res_data.get("pipeline", "unknown"),
                    "status": "completed",
                    "step_index": 4,
                    "total_steps": 4,
                    "progress_percent": 100,
                    "current_step": "Analysis complete!",
                    "result": res_data,
                    "error": None,
                }), 200
            except Exception as e:
                logger.error("Failed to read persisted job result %s: %s", result_json_path, e)

        return jsonify({
            "status": "error",
            "message": f"Job '{job_id}' not found.",
        }), 404

    return jsonify(job), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    logger.info("Starting ParkinSense Backend API Server on port %d...", port)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
