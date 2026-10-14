"""
ParkinSense — Unified Pipeline Integration & Regression Tests
==============================================================
Validates:
  1. Pipeline selection ('mediapipe' vs 'fourd_humans').
  2. Strict 290-dim FeatureContract validation & NaN/order checks.
  3. FeatureAdapter conformance.
  4. Centralized PredictionEngine scoring & consistent result schema.
  5. 4D-Humans GPU unavailable safety (never fakes success).
  6. Flask API endpoints (/api/health, /api/predict, /api/status).
  7. Existing MediaPipe regression tests remain fully passing.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.features.feature_contract import (
    FEATURE_CONTRACT_COLUMNS,
    EXPECTED_FEATURE_COUNT,
    FeatureContractError,
    validate_feature_vector,
    conform_to_feature_contract,
)
from src.features.feature_adapter import FeatureAdapter
from backend.extractors import (
    BaseGaitExtractor,
    MediaPipeGaitExtractor,
    FourDHumansGaitExtractor,
    get_gait_extractor,
    GaitExtractionResult,
)
from backend.pipeline import (
    normalize_pipeline_selection,
    analyze_video,
    generate_job_id,
)
from backend.prediction import PredictionEngine
from backend.fourd_humans import GPUServiceUnavailableError
from backend.app import app


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def valid_feature_df() -> pd.DataFrame:
    """Create a synthetically valid 1x290 feature vector DataFrame."""
    data = {"fps": 30.0, "num_frames": 100}
    for i in range(72):
        data[f"pose{i}_mean"] = 0.05 * (i % 5)
        data[f"pose{i}_std"] = 0.02 * (i % 3)
        data[f"pose{i}_min"] = -0.10
        data[f"pose{i}_max"] = 0.15
    df = pd.DataFrame([data])
    return df[list(FEATURE_CONTRACT_COLUMNS)]


@pytest.fixture
def test_video_path(tmp_path: Path) -> Path:
    """Create a minimal valid mp4 test file using OpenCV."""
    import cv2
    video_file = tmp_path / "test_walk.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_file), fourcc, 10.0, (64, 64))
    for _ in range(15):
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        out.write(frame)
    out.release()
    return video_file


@pytest.fixture
def client():
    """Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


# ------------------------------------------------------------------------------
# 1. Feature Contract & Adapter Tests
# ------------------------------------------------------------------------------

def test_feature_contract_valid(valid_feature_df: pd.DataFrame):
    """Valid 290-dim DataFrame must pass contract validation."""
    validate_feature_vector(valid_feature_df)
    assert valid_feature_df.shape == (1, 290)


def test_feature_contract_dimension_mismatch(valid_feature_df: pd.DataFrame):
    """Contract must reject vectors with incorrect column counts."""
    # Drop one column -> 289
    df_289 = valid_feature_df.drop(columns=["pose71_max"])
    with pytest.raises(FeatureContractError, match="Feature count mismatch"):
        validate_feature_vector(df_289)

    # Extra column -> 291
    df_291 = valid_feature_df.copy()
    df_291["extra_col"] = 1.0
    with pytest.raises(FeatureContractError, match="Feature count mismatch"):
        validate_feature_vector(df_291)


def test_feature_contract_nan_and_inf(valid_feature_df: pd.DataFrame):
    """Contract must reject NaN or Inf values."""
    df_nan = valid_feature_df.copy()
    df_nan["pose0_mean"] = np.nan
    with pytest.raises(FeatureContractError, match="non-finite"):
        validate_feature_vector(df_nan)

    df_inf = valid_feature_df.copy()
    df_inf["pose10_std"] = np.inf
    with pytest.raises(FeatureContractError, match="non-finite"):
        validate_feature_vector(df_inf)


def test_feature_contract_bounds(valid_feature_df: pd.DataFrame):
    """Contract must enforce positive FPS and frame counts."""
    df_bad_fps = valid_feature_df.copy()
    df_bad_fps["fps"] = -5.0
    with pytest.raises(FeatureContractError, match="Invalid video FPS"):
        validate_feature_vector(df_bad_fps)

    df_bad_frames = valid_feature_df.copy()
    df_bad_frames["num_frames"] = 0
    with pytest.raises(FeatureContractError, match="Invalid frame count"):
        validate_feature_vector(df_bad_frames)


def test_feature_adapter_reordering(valid_feature_df: pd.DataFrame):
    """FeatureAdapter must reorder scrambled columns to match strict contract."""
    cols = list(valid_feature_df.columns)
    # Reverse columns
    reversed_df = valid_feature_df[cols[::-1]].copy()
    
    # In strict mode, validate fails on wrong sequence
    with pytest.raises(FeatureContractError, match="Feature ordering mismatch"):
        validate_feature_vector(reversed_df, strict_order=True)

    # But conform_to_feature_contract and FeatureAdapter align it
    adapter = FeatureAdapter(strict=True)
    aligned = adapter.adapt(reversed_df, source_pipeline="mediapipe")
    assert tuple(aligned.columns) == FEATURE_CONTRACT_COLUMNS
    validate_feature_vector(aligned, strict_order=True)


# ------------------------------------------------------------------------------
# 2. Pipeline Selection & Extractor Factory Tests
# ------------------------------------------------------------------------------

def test_pipeline_normalization():
    """Test resolution of various pipeline names and aliases."""
    assert normalize_pipeline_selection("mediapipe") == ("mediapipe", None)
    assert normalize_pipeline_selection("MediaPipe") == ("mediapipe", None)
    assert normalize_pipeline_selection("fourd_humans") == ("fourd_humans", None)
    assert normalize_pipeline_selection("4D-Humans") == ("fourd_humans", None)
    assert normalize_pipeline_selection("hmr2") == ("fourd_humans", None)
    assert normalize_pipeline_selection(mode="ssh") == ("fourd_humans", "ssh")
    assert normalize_pipeline_selection(mode="mock") == ("fourd_humans", "mock")

    with pytest.raises(ValueError, match="Unknown pipeline selection"):
        normalize_pipeline_selection("invalid_extractor")


def test_extractor_factory():
    """Factory must instantiate the correct extractor subclass."""
    mp_ext = get_gait_extractor("mediapipe")
    assert isinstance(mp_ext, MediaPipeGaitExtractor)

    fd_ext = get_gait_extractor("fourd_humans")
    assert isinstance(fd_ext, FourDHumansGaitExtractor)

    with pytest.raises(ValueError):
        get_gait_extractor("nonexistent")


# ------------------------------------------------------------------------------
# 3. Centralized Prediction Service Tests
# ------------------------------------------------------------------------------

def test_central_prediction_service(valid_feature_df: pd.DataFrame):
    """PredictionEngine must score valid features and return the full standard schema."""
    engine = PredictionEngine()
    prediction = engine.predict_features(valid_feature_df, source_pipeline="mediapipe")

    # Core required keys
    assert "parkinsons_detection" in prediction
    assert "severity_estimation" in prediction
    assert "explainability" in prediction
    assert "model_information" in prediction
    assert "dataset_performance" in prediction
    assert "disclaimer" in prediction

    # Check parkinsons_detection structure
    pd_info = prediction["parkinsons_detection"]
    assert pd_info["label"] in (0, 1)
    assert "diagnosis" in pd_info
    assert 0.0 <= pd_info["probability"] <= 100.0
    assert "Healthy" in pd_info["class_probabilities"]
    assert "Parkinsons" in pd_info["class_probabilities"]

    # Check severity
    sev = prediction["severity_estimation"]
    assert sev["updrs_gait_score"] in (0, 1, 2)
    assert "description" in sev

    # Check explainability
    assert len(prediction["explainability"]["important_gait_indicators"]) > 0


# ------------------------------------------------------------------------------
# 4. 4D-Humans Safety & Mock Pipeline Execution Tests
# ------------------------------------------------------------------------------

def test_fourd_humans_gpu_unavailable_raises_error(test_video_path: Path):
    """
    CRITICAL SAFETY REQUIREMENT:
    When 4D-Humans is selected in SSH mode and the GPU server is offline,
    the pipeline MUST raise GPUServiceUnavailableError.
    It MUST NOT silently fall back to mock data or claim real inference succeeded.
    """
    with patch("backend.fourd_humans._check_ssh_connectivity", return_value=False):
        extractor = FourDHumansGaitExtractor(mode="ssh")
        with pytest.raises(GPUServiceUnavailableError, match="offline|unavailable"):
            extractor.extract_gait(test_video_path, job_id="test_gpu_check")


def test_fourd_humans_mock_pipeline_execution(test_video_path: Path):
    """
    When explicitly configured in mock mode for offline testing,
    4D-Humans runs through the full feature contract and central prediction service.
    """
    result = analyze_video(
        video_path=test_video_path,
        job_id="test_mock_job",
        pipeline="fourd_humans",
        mode="mock",
        cleanup=True,
    )

    assert result["status"] == "success"
    assert result["pipeline"] == "fourd_humans"
    assert result["is_demo"] is True
    assert "demo_notice" in result
    assert "parkinsons_detection" in result
    assert "severity_estimation" in result
    assert "gait_timeseries" in result
    assert len(result["gait_timeseries"]) > 0
    assert result["metadata"]["features_extracted"] == 290


# ------------------------------------------------------------------------------
# 5. Flask API Endpoint Tests
# ------------------------------------------------------------------------------

def test_api_health_endpoint(client):
    """GET /api/health must report server status and pipeline readiness."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "pipelines" in data
    assert "mediapipe" in data["pipelines"]
    assert data["pipelines"]["mediapipe"]["available"] is True
    assert "fourd_humans" in data["pipelines"]


def test_api_predict_no_file(client):
    """POST /api/predict without video file returns 400."""
    res = client.post("/api/predict")
    assert res.status_code == 400
    data = res.get_json()
    assert data["status"] == "error"
    assert "No video file" in data["message"]


def test_api_predict_invalid_pipeline(client):
    """POST /api/predict with unsupported pipeline returns 400."""
    data = {
        "file": (io.BytesIO(b"dummy_video_content"), "test.mp4"),
        "pipeline": "invalid_vision_ai",
    }
    res = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    res_data = res.get_json()
    assert "Unknown pipeline" in res_data["message"]


def test_api_predict_gpu_offline_returns_503(client):
    """
    POST /api/predict?pipeline=fourd_humans with sync=true when GPU is offline
    must return HTTP 503 with informative diagnostic message.
    """
    with patch("backend.fourd_humans._check_ssh_connectivity", return_value=False):
        data = {
            "file": (io.BytesIO(b"fake_video_bytes"), "walking.mp4"),
            "pipeline": "fourd_humans",
            "mode": "ssh",
        }
        res = client.post(
            "/api/predict?pipeline=fourd_humans&mode=ssh&sync=true",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 503
        res_data = res.get_json()
        assert res_data["status"] == "error"
        assert "GPU" in res_data["message"]


def test_api_predict_async_success(client):
    """POST /api/predict async creates job and returns 202."""
    data = {
        "file": (io.BytesIO(b"fake_video_bytes"), "walking.mp4"),
        "pipeline": "mediapipe",
    }
    # Mock background runner to avoid waiting for heavy inference
    with patch("backend.app._run_background_pipeline"):
        res = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert res.status_code == 202
        res_data = res.get_json()
        assert res_data["status"] == "processing"
        assert "job_id" in res_data
        assert res_data["pipeline"] == "mediapipe"
        assert "status_url" in res_data
