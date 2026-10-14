"""
ParkinSense — 4D-Humans Integration Module
==========================================
Handles video → SMPL extraction using 4D-Humans.

Modes (controlled by POSE_SERVICE_MODE):
  "mock"  — Load pre-existing SMPL CSV (for development/testing only)
  "ssh"   — SCP video to remote GPU machine, run track.py via SSH, SCP result back.
            FAILS HARD if GPU machine is unavailable — never falls back to mock.
  "auto"  — Try local → ssh → mock (development convenience only)

The mode is set via environment variable POSE_SERVICE_MODE (default: "auto").

SAFETY RULE:
  When POSE_SERVICE_MODE=ssh, the system MUST use real GPU inference.
  If the GPU machine is unavailable, it raises GPUServiceUnavailableError.
  It MUST NOT silently fall back to mock mode and return a misleading prediction.
"""

from __future__ import annotations

import glob
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Resolve project root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)


class GPUServiceUnavailableError(Exception):
    """Raised when the GPU inference service is required but unavailable.

    This exception is never caught and silently replaced with mock data.
    It surfaces to the API layer, which returns a clean error to the frontend.
    """

    pass


# Directories
REMOTE_INPUT_DIR = _PROJECT_ROOT / "remote_input"
REMOTE_OUTPUT_DIR = _PROJECT_ROOT / "remote_output"
SAMPLE_SMPL_CSV = _PROJECT_ROOT / "data" / "input" / "smpl.csv"

# 4D-Humans paths (local)
FOURD_HUMANS_ROOT = os.getenv("FOURD_HUMANS_ROOT", str(_PROJECT_ROOT / "4D-Humans"))

# SSH remote settings (Pranav's Ubuntu GPU laptop)
SSH_HOST = os.getenv("FOURD_HUMANS_SSH_HOST", "10.40.16.253")
SSH_USER = os.getenv("FOURD_HUMANS_SSH_USER", "pranav")
SSH_KEY = os.getenv("FOURD_HUMANS_SSH_KEY", "")  # empty = use default ssh key
SSH_PORT = os.getenv("FOURD_HUMANS_SSH_PORT", "22")
REMOTE_4D_HUMANS_DIR = os.getenv(
    "FOURD_HUMANS_REMOTE_DIR", "~/ParkinSense/4D-Humans"
)
REMOTE_TEMP_DIR = os.getenv(
    "FOURD_HUMANS_REMOTE_TEMP", "/tmp/parkinsense_jobs"
)

# Mode control
POSE_SERVICE_MODE = os.getenv("POSE_SERVICE_MODE", "auto").lower()


import time

_GPU_HEALTH_CACHE: dict = {"timestamp": 0, "result": None}


def _ssh_cmd_base() -> list[str]:
    """Build base SSH command with identity/key if specified."""
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=2"]
    if SSH_PORT != "22":
        cmd.extend(["-p", SSH_PORT])
    if SSH_KEY:
        cmd.extend(["-i", SSH_KEY])
    cmd.append(f"{SSH_USER}@{SSH_HOST}")
    return cmd


def _scp_cmd_base() -> list[str]:
    """Build base SCP command with identity/key if specified."""
    cmd = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=2"]
    if SSH_PORT != "22":
        cmd.extend(["-P", SSH_PORT])
    if SSH_KEY:
        cmd.extend(["-i", SSH_KEY])
    return cmd


def _check_ssh_connectivity() -> bool:
    """Check if SSH to the remote GPU machine works."""
    try:
        cmd = _ssh_cmd_base() + ["echo", "ok"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and "ok" in result.stdout:
            logger.info("SSH to %s@%s successful.", SSH_USER, SSH_HOST)
            return True
        logger.warning(
            "SSH check failed (rc=%d): %s", result.returncode, result.stderr.strip()
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("SSH check failed: %s", e)
    return False


def _check_4d_humans_local() -> bool:
    """Check if 4D-Humans repo is available locally."""
    root = Path(FOURD_HUMANS_ROOT)
    track_py = root / "track.py"
    if track_py.exists():
        logger.info("4D-Humans found locally at: %s", root)
        return True
    return False


def _check_remote_4d_humans_on_ssh() -> bool:
    """Check if 4D-Humans repo exists on the remote machine."""
    try:
        cmd = _ssh_cmd_base() + [
            "test", "-f", f"{REMOTE_4D_HUMANS_DIR}/track.py", "&&", "echo", "ok"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and "ok" in result.stdout:
            logger.info(
                "4D-Humans found on remote machine at: %s@%s:%s",
                SSH_USER, SSH_HOST, REMOTE_4D_HUMANS_DIR,
            )
            return True
    except Exception as e:
        logger.warning("Remote 4D-Humans check failed: %s", e)
    return False


def check_gpu_health() -> dict:
    """
    Health check for the GPU inference service.

    Returns a dict with:
      - "available": bool — whether the GPU service is reachable and usable
      - "service":   str  — always "4D-Humans"

    This function is called by GET /health/gpu.
    """
    now = time.time()
    if _GPU_HEALTH_CACHE["result"] is not None and (now - _GPU_HEALTH_CACHE["timestamp"]) < 30:
        return _GPU_HEALTH_CACHE["result"]

    ssh_ok = _check_ssh_connectivity()
    repo_ok = _check_remote_4d_humans_on_ssh() if ssh_ok else False

    available = ssh_ok and repo_ok
    detail_parts = []

    if not ssh_ok:
        detail_parts.append(f"SSH to {SSH_HOST} failed")
    if ssh_ok and not repo_ok:
        detail_parts.append(f"4D-Humans track.py not found at {REMOTE_4D_HUMANS_DIR}")

    detail = "; ".join(detail_parts) if detail_parts else "All checks passed"

    if available:
        logger.info("GPU health check: available (ssh OK, repo OK)")
    else:
        logger.warning("GPU health check: unavailable — %s", detail)

    res = {
        "available": available,
        "service": "4D-Humans",
    }
    _GPU_HEALTH_CACHE["timestamp"] = now
    _GPU_HEALTH_CACHE["result"] = res
    return res


def detect_mode() -> str:
    """
    Determine the processing mode.

    Mode rules:
      "mock"          → always mock (intentional dev/testing choice)
      "ssh" / "remote"→ must use GPU machine; raises GPUServiceUnavailableError if unavailable
      "local"         → must use local 4D-Humans; raises FileNotFoundError if not installed
      "auto"          → try local → ssh → mock (dev convenience; explicit modes above take priority)

    NEVER silently falls back from an explicit mode to mock.
    """
    if POSE_SERVICE_MODE in ("mock", "local_mock"):
        return "mock"

    if POSE_SERVICE_MODE in ("ssh", "ssh_remote", "remote"):
        if _check_ssh_connectivity() and _check_remote_4d_humans_on_ssh():
            return "ssh"
        raise GPUServiceUnavailableError(
            "GPU inference service is currently unavailable. "
            "The remote GPU machine is offline or 4D-Humans is not installed on it. "
            "Please try again later or contact the administrator."
        )

    if POSE_SERVICE_MODE in ("local", "subprocess"):
        if _check_4d_humans_local():
            return "local"
        raise GPUServiceUnavailableError(
            "Local 4D-Humans installation not found. "
            "Set FOURD_HUMANS_ROOT or install 4D-Humans locally."
        )

    # Auto-detect: try local → ssh → mock (dev convenience only)
    if _check_4d_humans_local():
        return "local"
    if _check_ssh_connectivity() and _check_remote_4d_humans_on_ssh():
        return "ssh"
    logger.warning(
        "4D-Humans not available locally or via SSH. "
        "Falling back to MOCK mode. "
        "Set POSE_SERVICE_MODE=ssh to require real GPU inference."
    )
    return "mock"


# ─── LOCAL SUBPROCESS ───────────────────────────────────────────────


def _run_4d_humans_local(video_path: Path, output_dir: Path) -> Path:
    """Run 4D-Humans locally via `python track.py`."""
    root = Path(FOURD_HUMANS_ROOT)
    track_py = root / "track.py"

    if not track_py.exists():
        raise FileNotFoundError(
            f"4D-Humans track.py not found at: {track_py}\n"
            f"Clone: git clone https://github.com/shubham-goel/4D-Humans.git {root}"
        )

    ensure_dir(output_dir)

    cmd = [
        sys.executable, "track.py",
        f'video.source={video_path}',
    ]

    logger.info("Running 4D-Humans (local): %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        logger.error("4D-Humans failed (rc=%d)", result.returncode)
        logger.error("STDOUT: %s", result.stdout[-2000:] if result.stdout else "")
        logger.error("STDERR: %s", result.stderr[-2000:] if result.stderr else "")
        raise RuntimeError(f"4D-Humans failed: {result.stderr[-500:]}")

    logger.info("4D-Humans completed successfully.")
    return output_dir


# ─── SSH REMOTE EXECUTION ───────────────────────────────────────────


def _run_4d_humans_ssh(video_path: Path, job_id: str) -> Tuple[Path, str]:
    """
    Run 4D-Humans on the remote GPU machine via SSH/SCP.

    Flow:
      1. mkdir temp dir on remote
      2. SCP video → remote
      3. SSH: cd 4D-Humans && python track.py video.source=<path>
      4. SCP output .pkl ← remote
      5. Cleanup remote temp

    Returns:
        (local_pkl_path, video_filename_on_remote)
    """
    remote_job_dir = f"{REMOTE_TEMP_DIR}/{job_id}"
    remote_video = f"{remote_job_dir}/{video_path.name}"
    remote_output_dir = f"{remote_job_dir}/outputs"

    # 1. Create temp dir on remote
    logger.info("Creating remote job dir: %s@%s:%s", SSH_USER, SSH_HOST, remote_job_dir)
    cmd = _ssh_cmd_base() + ["mkdir", "-p", remote_job_dir]
    subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)

    # 2. SCP video to remote
    logger.info("SCP video → remote: %s → %s", video_path.name, remote_video)
    cmd = _scp_cmd_base() + [str(video_path), f"{SSH_USER}@{SSH_HOST}:{remote_video}"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"SCP upload failed: {result.stderr}")

    # 3. Run 4D-Humans on remote
    #    Output goes to ./outputs/<video_stem>/
    #    The track.py default output dir is ./outputs/
    logger.info(
        "Running 4D-Humans on remote: %s@%s — video.source=%s",
        SSH_USER, SSH_HOST, remote_video,
    )
    run_cmd = (
        f"cd {REMOTE_4D_HUMANS_DIR} && "
        f"python track.py video.source={remote_video} "
        f"render.show=False "
        f"save_video=False "
        f"render.fps=30"
    )
    cmd = _ssh_cmd_base() + [run_cmd]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)

    if result.returncode != 0:
        logger.error("Remote 4D-Humans failed (rc=%d)", result.returncode)
        logger.error("STDOUT: %s", result.stdout[-2000:] if result.stdout else "")
        logger.error("STDERR: %s", result.stderr[-2000:] if result.stderr else "")
        raise RuntimeError(f"Remote 4D-Humans failed: {result.stderr[-500:]}")

    logger.info("Remote 4D-Humans completed successfully.")

    # 4. Find and SCP the .pkl output back
    #    Default output: ./outputs/<video_stem>/*.pkl
    video_stem = video_path.stem
    remote_pkl_pattern = (
        f"{REMOTE_4D_HUMANS_DIR}/outputs/{video_stem}/*.pkl"
    )
    # Also check for tracklet file (PHALP outputs tracklets.pkl)
    remote_pkl_pattern2 = (
        f"{REMOTE_4D_HUMANS_DIR}/outputs/{video_stem}/tracklets.pkl"
    )

    local_output_dir = REMOTE_OUTPUT_DIR / job_id
    ensure_dir(local_output_dir)

    # SCP any .pkl files from remote
    cmd = _scp_cmd_base() + [
        f"{SSH_USER}@{SSH_HOST}:{remote_pkl_pattern}",
        str(local_output_dir) + "/",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        # Try the specific tracklets.pkl
        cmd = _scp_cmd_base() + [
            f"{SSH_USER}@{SSH_HOST}:{remote_pkl_pattern2}",
            str(local_output_dir) + "/",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            # List what's in the output dir on remote to debug
            list_cmd = _ssh_cmd_base() + [
                "ls", "-la",
                f"{REMOTE_4D_HUMANS_DIR}/outputs/{video_stem}/",
            ]
            list_result = subprocess.run(
                list_cmd, capture_output=True, text=True, timeout=15
            )
            logger.error(
                "Remote output dir contents: %s",
                list_result.stdout if list_result.returncode == 0 else "N/A",
            )
            raise RuntimeError(
                f"No .pkl output found on remote at: "
                f"{REMOTE_4D_HUMANS_DIR}/outputs/{video_stem}/"
            )

    # Find the local .pkl
    local_pkls = list(local_output_dir.glob("*.pkl"))
    if not local_pkls:
        raise FileNotFoundError("SCP succeeded but no .pkl found locally")
    local_pkl = local_pkls[0]
    logger.info("Retrieved output: %s (%.1f KB)", local_pkl, local_pkl.stat().st_size / 1024)

    # 5. Cleanup remote temp dir (best effort)
    cleanup_cmd = _ssh_cmd_base() + ["rm", "-rf", remote_job_dir]
    try:
        subprocess.run(cleanup_cmd, capture_output=True, text=True, timeout=15)
    except Exception:
        pass

    return local_pkl, remote_video


# ─── PKL → SMPL CSV CONVERSION ──────────────────────────────────────


def _pkl_to_smpl_csv(pkl_path: Path) -> pd.DataFrame:
    """
    Convert 4D-Humans/PHALP .pkl output to ParkinSense SMPL CSV.

    The .pkl file from `track.py` contains a list of tracklet dictionaries.
    Each tracklet has keys like:
        - 'smpl_param': dict with 'body_pose' (N,23,3,3) and 'global_orient' (N,1,3,3)
        - 'frame_id': list of frame indices
        - 'track_id': int

    We pick the tracklet with the most frames (longest-tracked person).
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # data can be:
    #   - a dict of tracklets keyed by track_id
    #   - a list of tracklet dicts
    #   - a dict with 'tracklets' or similar key
    tracklets = _extract_tracklets(data)

    if not tracklets:
        raise ValueError(
            f"No tracklets found in {pkl_path}. Data type: {type(data)}, "
            f"keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}"
        )

    # Pick the longest tracklet (most frames)
    best = max(tracklets, key=lambda t: _tracklet_frame_count(t))
    logger.info(
        "Selected tracklet with %d frames (out of %d tracklets)",
        _tracklet_frame_count(best), len(tracklets),
    )

    return _tracklet_to_smpl_csv(best)


def _extract_tracklets(data) -> list[dict]:
    """Extract tracklet list from various possible PHALP output formats."""
    if isinstance(data, dict):
        # Format 1: dict keyed by track_id → each value is a tracklet
        if all(isinstance(v, dict) for v in data.values()):
            first_val = next(iter(data.values()))
            if "smpl_param" in first_val or "body_pose" in first_val:
                return list(data.values())
        # Format 2: dict with 'tracklets' key
        if "tracklets" in data:
            return data["tracklets"]
        # Format 3: dict with 'results' key
        if "results" in data:
            return data["results"]
        # Format 4: single tracklet dict (wrap in list)
        if "smpl_param" in data or "body_pose" in data:
            return [data]

    if isinstance(data, list):
        # Format 5: list of tracklet dicts
        if data and isinstance(data[0], dict):
            return data

    return []


def _tracklet_frame_count(tracklet: dict) -> int:
    """Count frames in a tracklet."""
    smpl = tracklet.get("smpl_param", tracklet)
    if isinstance(smpl, dict):
        bp = smpl.get("body_pose", None)
        if bp is not None:
            arr = np.array(bp)
            return arr.shape[0]
    # Fallback: check frame_id list
    frames = tracklet.get("frame_id", tracklet.get("frames", []))
    return len(frames) if frames else 0


def _tracklet_to_smpl_csv(tracklet: dict) -> pd.DataFrame:
    """Convert a single tracklet dict to ParkinSense SMPL CSV format."""
    # Extract SMPL params — try multiple key patterns
    smpl = tracklet.get("smpl_param", tracklet)

    global_orient = None
    body_pose = None

    if isinstance(smpl, dict):
        global_orient = smpl.get("global_orient", None)
        body_pose = smpl.get("body_pose", None)

    # Fallback: check top-level keys
    if global_orient is None:
        global_orient = tracklet.get("global_orient", None)
    if body_pose is None:
        body_pose = tracklet.get("body_pose", None)

    if global_orient is None or body_pose is None:
        raise ValueError(
            f"Cannot find global_orient/body_pose in tracklet. "
            f"Keys: {list(tracklet.keys())}"
        )

    go_arr = np.array(global_orient)  # (N, 1, 3, 3) or (N, 9)
    bp_arr = np.array(body_pose)      # (N, 23, 3, 3) or (N, 207)

    n_frames = go_arr.shape[0]

    # Flatten rotation matrices
    if go_arr.ndim == 4:  # (N, 1, 3, 3)
        go_flat = go_arr.reshape(n_frames, 9)
    elif go_arr.ndim == 3 and go_arr.shape[1] == 9:  # (N, 9)
        go_flat = go_arr
    elif go_arr.ndim == 3 and go_arr.shape[1] == 3:  # (N, 3, 3)
        go_flat = go_arr.reshape(n_frames, 9)
    else:
        go_flat = go_arr.reshape(n_frames, -1)

    if bp_arr.ndim == 4:  # (N, 23, 3, 3)
        bp_flat = bp_arr.reshape(n_frames, 207)
    elif bp_arr.ndim == 2 and bp_arr.shape[1] == 207:  # (N, 207)
        bp_flat = bp_arr
    elif bp_arr.ndim == 3 and bp_arr.shape[1] == 23:  # (N, 23, 9)?
        bp_flat = bp_arr.reshape(n_frames, -1)
    else:
        bp_flat = bp_arr.reshape(n_frames, -1)

    # Build column names
    orient_cols = [f"global_orient_{i}" for i in range(go_flat.shape[1])]
    body_cols = [f"body_pose_{i}" for i in range(bp_flat.shape[1])]
    all_cols = orient_cols + body_cols

    data = np.concatenate([go_flat, bp_flat], axis=1)
    df = pd.DataFrame(data, columns=all_cols)
    df.insert(0, "frame", range(1, n_frames + 1))
    df.insert(1, "person_id", 0)

    logger.info(
        "Converted tracklet to SMPL CSV: %d frames, %d columns",
        n_frames, len(all_cols),
    )
    return df


# ─── MOCK MODE ───────────────────────────────────────────────────────


def _load_mock_smpl(job_id: Optional[str] = None, video_path: Optional[Path] = None) -> Tuple[pd.DataFrame, float]:
    """
    Load the sample SMPL CSV for mock mode with deterministic per-job variation.
    
    This ensures that each uploaded video in MOCK mode produces a unique, distinct
    feature vector and prediction result for testing multi-video flows, while still
    executing the full 290-dim feature extraction and trained ML models.
    """
    if not SAMPLE_SMPL_CSV.exists():
        raise FileNotFoundError(
            f"Sample SMPL CSV not found: {SAMPLE_SMPL_CSV}\n"
            "Cannot use mock mode without sample data."
        )
    logger.info("Loading sample SMPL data from: %s", SAMPLE_SMPL_CSV)
    df = pd.read_csv(SAMPLE_SMPL_CSV)
    fps = 29.75

    # If job_id or video_path is provided, apply a deterministic subtle variation
    if job_id or video_path:
        import hashlib
        seed_str = f"{job_id or ''}_{video_path.name if video_path else ''}"
        seed_val = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.RandomState(seed_val)

        # Apply subtle joint angular variation to body_pose columns (cols 11 onwards)
        body_cols = [c for c in df.columns if c.startswith("body_pose_")]
        if body_cols:
            noise_scale = 0.015 + rng.uniform(-0.005, 0.008)
            noise = rng.normal(loc=0.0, scale=noise_scale, size=(len(df), len(body_cols)))
            df[body_cols] = df[body_cols].values + noise

        # Slightly trim or slice frame window deterministically to vary frame count & dynamics
        trim_start = rng.randint(0, min(15, max(1, len(df) - 150)))
        trim_len = len(df) - rng.randint(0, 20)
        df = df.iloc[trim_start:trim_start + trim_len].copy()
        df["frame"] = range(1, len(df) + 1)
        logger.info("Applied mock variation for job %s: %d frames (seed %d)", job_id, len(df), seed_val)

    logger.info("Loaded %d frames from sample SMPL CSV.", len(df))
    return df, fps


# ─── PUBLIC API ──────────────────────────────────────────────────────


def process_video(video_path: str | Path, job_id: str, mode: Optional[str] = None) -> Tuple[Path, float, str, bool]:
    """
    Process a video through 4D-Humans and produce SMPL output.

    Args:
        video_path: Path to the input walking video.
        job_id:     Unique job identifier for file naming.
        mode:       Override mode ("local", "ssh", "mock", or "auto").
                    When "ssh" is specified and GPU is unavailable, raises
                    GPUServiceUnavailableError. Never silently falls back.

    Returns:
        Tuple of (smpl_csv_path, fps, mode_used, is_demo)

    Raises:
        GPUServiceUnavailableError: If mode requires GPU and it's unavailable.
        FileNotFoundError:          If video or required files are missing.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # If caller passes explicit mode, honor it directly (no env-var override)
    if mode and mode != "auto":
        use_mode = mode
        # Validate that the explicit mode is actually available
        if use_mode == "ssh":
            if not (_check_ssh_connectivity() and _check_remote_4d_humans_on_ssh()):
                raise GPUServiceUnavailableError(
                    "GPU inference service is currently unavailable. "
                    "The remote GPU machine is offline or 4D-Humans is not installed. "
                    "Please try again later."
                )
        elif use_mode == "local":
            if not _check_4d_humans_local():
                raise GPUServiceUnavailableError(
                    "Local 4D-Humans installation not found."
                )
        elif use_mode not in ("mock",):
            # Unknown mode — treat as auto
            use_mode = detect_mode()
    else:
        use_mode = detect_mode()

    logger.info("4D-Humans processing mode for job %s: %s", job_id, use_mode)

    ensure_dir(REMOTE_INPUT_DIR)
    ensure_dir(REMOTE_OUTPUT_DIR)

    is_demo = False

    if use_mode == "local":
        job_output_dir = REMOTE_OUTPUT_DIR / job_id
        ensure_dir(job_output_dir)

        _run_4d_humans_local(video_path, job_output_dir)

        # Find .pkl output
        local_pkls = list(job_output_dir.glob("*.pkl"))
        if not local_pkls:
            raise FileNotFoundError(f"No .pkl output found in {job_output_dir}")
        smpl_df = _pkl_to_smpl_csv(local_pkls[0])

    elif use_mode == "ssh":
        local_pkl, _ = _run_4d_humans_ssh(video_path, job_id)
        smpl_df = _pkl_to_smpl_csv(local_pkl)

    else:
        logger.warning("Using MOCK mode for job %s — sample SMPL data, not real inference.", job_id)
        smpl_df, _ = _load_mock_smpl(job_id=job_id, video_path=video_path)
        is_demo = True

    # Save SMPL CSV per job
    smpl_csv_path = REMOTE_OUTPUT_DIR / f"{job_id}_smpl.csv"
    smpl_df.to_csv(str(smpl_csv_path), index=False)
    logger.info("SMPL CSV saved for job %s: %s (%d frames)", job_id, smpl_csv_path, len(smpl_df))

    # FPS
    if use_mode == "mock":
        fps = 29.75
    else:
        fps = _detect_fps(video_path)

    return smpl_csv_path, fps, use_mode, is_demo


def _detect_fps(video_path: Path) -> float:
    """Detect video FPS using OpenCV."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps > 0:
            return float(fps)
    except Exception as e:
        logger.warning("Could not detect FPS: %s. Using default 30.0", e)
    return 30.0


def cleanup_job(job_id: str) -> None:
    """Clean up temporary files for a completed job."""
    input_file = REMOTE_INPUT_DIR / f"{job_id}.mp4"
    output_csv = REMOTE_OUTPUT_DIR / f"{job_id}_smpl.csv"

    if input_file.exists():
        try:
            input_file.unlink()
            logger.info("Cleaned up input video: %s", input_file)
        except OSError as e:
            logger.warning("Failed to clean up %s: %s", input_file, e)
