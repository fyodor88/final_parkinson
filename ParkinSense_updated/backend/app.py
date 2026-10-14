"""
ParkinSense — Flask Backend API Server
======================================
Provides RESTful API endpoints for the ParkinSense web frontend.

Endpoints:
- GET  /api/health        → Check server status
- GET  /api/health/gpu    → Check GPU service health
- POST /api/predict       → Upload video, start background AI pipeline, return job_id
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
from backend.pipeline import analyze_video, generate_job_id
from backend.fourd_humans import GPUServiceUnavailableError, check_gpu_health

logger = get_logger(__name__)

app = Flask(__name__, static_folder=str(_PROJECT_ROOT / "frontend" / "dist"), static_url_path="")

# Directories & settings
UPLOAD_DIR = _PROJECT_ROOT / "uploads"
REMOTE_OUTPUT_DIR = _PROJECT_ROOT / "remote_output"
ensure_dir(UPLOAD_DIR)
ensure_dir(REMOTE_OUTPUT_DIR)
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# In-memory job state store for background tasks
# Key: job_id -> Value: dict(status, step_index, total_steps, current_step, result, error, created_at)
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
        "endpoints": ["GET /api/health", "POST /api/predict", "GET /api/status/<job_id>"],
    })


@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health_check():
    """Health status check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "ParkinSense Backend API",
        "timestamp": time.time(),
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


def _run_background_pipeline(video_path: Path, job_id: str, mode: str | None) -> None:
    """Background worker function executing the full AI pipeline."""
    def status_cb(step_idx: int, total_steps: int, message: str) -> None:
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["step_index"] = step_idx
                JOBS[job_id]["total_steps"] = total_steps
                JOBS[job_id]["current_step"] = message
                JOBS[job_id]["progress_percent"] = int((step_idx / total_steps) * 100)

    try:
        logger.info("Starting background analysis for job %s...", job_id)
        result = analyze_video(
            video_path=video_path,
            job_id=job_id,
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

        logger.info("Job %s completed successfully.", job_id)

    except GPUServiceUnavailableError as err:
        logger.error("Job %s GPU service unavailable: %s", job_id, err)
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error_code"] = "GPU_SERVICE_UNAVAILABLE"
            JOBS[job_id]["error"] = "The gait analysis GPU service is currently unavailable. Please try again later."
    except Exception as err:
        logger.exception("Job %s pipeline failure: %s", job_id, err)
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error_code"] = "PIPELINE_ERROR"
            JOBS[job_id]["error"] = str(err)
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
    Video prediction endpoint. Accepts multipart/form-data video upload.

    Spawns background AI pipeline job and returns job_id for status polling.
    If query param ?sync=true is provided, runs synchronously.
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

    job_id = generate_job_id()
    job_upload_dir = UPLOAD_DIR / job_id
    ensure_dir(job_upload_dir)
    temp_path = job_upload_dir / file.filename

    mode = request.args.get("mode") or request.form.get("mode")
    is_sync = request.args.get("sync", "").lower() in ("true", "1")

    file.save(str(temp_path))
    logger.info("Uploaded video saved for job %s: %s", job_id, temp_path)

    # Initialize job state
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "step_index": 1,
            "total_steps": 4,
            "progress_percent": 15,
            "current_step": "Video uploaded. Initializing AI pipeline...",
            "created_at": time.time(),
            "video_file": file.filename,
            "result": None,
            "error": None,
        }

    if is_sync:
        # Synchronous execution mode
        _run_background_pipeline(temp_path, job_id, mode)
        job_data = JOBS.get(job_id, {})
        if job_data.get("status") == "completed":
            return jsonify(job_data["result"]), 200
        else:
            return jsonify({
                "status": "error",
                "message": job_data.get("error", "Processing failed."),
            }), 500

    # Asynchronous background job mode
    thread = threading.Thread(
        target=_run_background_pipeline,
        args=(temp_path, job_id, mode),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "status": "processing",
        "job_id": job_id,
        "message": "Video uploaded successfully. AI pipeline started.",
        "status_url": f"/api/status/{job_id}",
    }), 202


@app.route("/status/<job_id>", methods=["GET"])
@app.route("/api/status/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    """
    Status polling endpoint. Returns live processing state & final results.
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)

    if not job:
        # Fallback check for persisted job output file on disk
        result_json_path = REMOTE_OUTPUT_DIR / f"{job_id}_result.json"
        if result_json_path.exists():
            try:
                import json
                with open(result_json_path, "r", encoding="utf-8") as f:
                    res_data = json.load(f)
                return jsonify({
                    "job_id": job_id,
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
    app.run(host="0.0.0.0", port=port, debug=False)
