"""
ParkinSense — System Integration & End-to-End Test Suite
=========================================================
Validates the complete application behavior per Phase 7 requirements:
1. Valid MediaPipe video inference on real video.
2. Invalid or unreadable video handling.
3. Feature dimension, ordering, and contract compatibility.
4. Model loading and prediction schema verification.
5. 4D-Humans route error handling when GPU service is offline.
6. Dashboard rendering and expected API response format.
"""

import os
import sys
import tempfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline import analyze_video
from backend.prediction import PredictionEngine
from backend.fourd_humans import check_gpu_health, GPUServiceUnavailableError
from backend.app import app, _friendly_error_message
from src.features.feature_contract import (
    FEATURE_CONTRACT_COLUMNS,
    EXPECTED_FEATURE_COUNT,
    conform_to_feature_contract,
    validate_feature_vector,
)

SAMPLE_VIDEO = PROJECT_ROOT / "data" / "raw" / "Tulip" / "gait" / "HT" / "subject01" / "subject1_camera1.mp4"


def test_feature_contract_schema_and_ordering():
    """Verify that feature contract strictly enforces 290 dimensions and correct columns."""
    assert len(FEATURE_CONTRACT_COLUMNS) == 290
    assert FEATURE_CONTRACT_COLUMNS[0] == "fps"
    assert FEATURE_CONTRACT_COLUMNS[1] == "num_frames"
    assert FEATURE_CONTRACT_COLUMNS[2] == "pose0_mean"
    assert FEATURE_CONTRACT_COLUMNS[-1] == "pose71_max"

    # Create conforming dataframe
    data = {col: [0.0] for col in FEATURE_CONTRACT_COLUMNS}
    data["fps"] = [30.0]
    data["num_frames"] = [100]
    df = pd.DataFrame(data)

    validated = conform_to_feature_contract(df)
    assert validated.shape == (1, 290)
    validate_feature_vector(validated)


def test_production_model_loading_and_compatibility():
    """Verify that models/pd_detector.pkl is preserved and compatible."""
    model_path = PROJECT_ROOT / "models" / "pd_detector.pkl"
    assert model_path.exists(), "models/pd_detector.pkl must exist"

    model = joblib.load(model_path)
    assert hasattr(model, "predict_proba")
    assert hasattr(model, "classes_")
    assert list(model.classes_) == [0, 1]
    assert model.n_features_in_ == 290


def test_prediction_engine_output_schema():
    """Verify PredictionEngine produces all expected fields for dashboard."""
    engine = PredictionEngine()
    data = {col: [0.0] for col in FEATURE_CONTRACT_COLUMNS}
    data["fps"] = [60.0]
    data["num_frames"] = [150]
    df = pd.DataFrame(data)

    res = engine.predict_features(df, source_pipeline="mediapipe")
    assert "parkinsons_detection" in res
    assert "severity_estimation" in res
    assert "explainability" in res
    assert "model_information" in res
    assert "dataset_performance" in res
    assert "disclaimer" in res

    pd_info = res["parkinsons_detection"]
    assert "label" in pd_info
    assert "diagnosis" in pd_info
    assert "probability" in pd_info
    assert "confidence" in pd_info
    assert "class_probabilities" in pd_info


def test_real_mediapipe_video_inference():
    """Real inference test using an existing TULIP MP4 video."""
    assert SAMPLE_VIDEO.exists(), f"Sample video not found at {SAMPLE_VIDEO}"

    res = analyze_video(SAMPLE_VIDEO, pipeline="mediapipe", cleanup=False)

    assert res["status"] in ("completed", "success")
    assert res["pipeline"] == "mediapipe"
    assert res["num_frames"] > 50
    assert res["fps"] > 0
    assert "parkinsons_detection" in res
    assert "severity_estimation" in res
    assert "gait_metrics" in res
    assert "key_gait_frames" in res
    assert len(res["key_gait_frames"]) > 0


def test_invalid_video_error_handling():
    """Verify that non-existent or corrupted files raise appropriate errors."""
    with pytest.raises(FileNotFoundError):
        analyze_video("non_existent_path.mp4", pipeline="mediapipe")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"not a valid mp4 header")
        bad_path = f.name

    try:
        with pytest.raises(RuntimeError):
            analyze_video(bad_path, pipeline="mediapipe")
    finally:
        Path(bad_path).unlink(missing_ok=True)


def test_fourd_humans_gpu_unreachable_error_handling():
    """Verify 4D-Humans route handles unreachable remote GPU gracefully."""
    gpu_status = check_gpu_health()
    assert isinstance(gpu_status, dict)
    assert "available" in gpu_status

    if not gpu_status["available"]:
        msg = _friendly_error_message(GPUServiceUnavailableError("GPU unreachable"))
        assert "4D-Humans GPU inference service is currently unavailable" in msg
        assert "Local MediaPipe analysis remains fully available" in msg


def test_flask_api_health_endpoint():
    """Verify Flask /api/health endpoint returns expected structure."""
    with app.test_client() as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] in ("ok", "healthy")
        assert "pipelines" in data
        assert "mediapipe" in data["pipelines"]


def test_smpl_feature_extractor_input_validation():
    """Verify SMPLFeatureExtractor validates empty data, invalid fps, and NaN values."""
    from src.features.smpl_feature_extractor import SMPLFeatureExtractor
    extractor = SMPLFeatureExtractor()

    # Empty DataFrame
    with pytest.raises(ValueError, match="empty"):
        extractor.extract_features(pd.DataFrame())

    # Invalid FPS
    cols = {f"global_orient_{i}": [0.0] for i in range(9)}
    cols.update({f"body_pose_{i}": [0.0] for i in range(23 * 9)})
    dummy_df = pd.DataFrame(cols)

    with pytest.raises(ValueError, match="Invalid fps"):
        extractor.extract_features(dummy_df, fps=-1.0)

    # DataFrame with NaN
    nan_df = dummy_df.copy()
    nan_df.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or non-finite"):
        extractor.extract_features(nan_df, fps=30.0)


def test_severity_model_artifact_provenance():
    """Verify models/random_forest.pkl exists, matches 290 features, and has classes [0, 1, 2]."""
    sev_path = PROJECT_ROOT / "models" / "random_forest.pkl"
    assert sev_path.exists(), "models/random_forest.pkl must exist"
    model = joblib.load(sev_path)
    assert hasattr(model, "predict")
    assert list(model.classes_) == [0, 1, 2]
    assert model.n_features_in_ == 290


def test_candidate_model_metadata_and_rejection():
    """Verify candidate model metadata accurately records external validation failure."""
    import json
    meta_path = PROJECT_ROOT / "data" / "processed" / "improved_model" / "candidate_model_metadata.json"
    assert meta_path.exists(), "candidate_model_metadata.json must exist"
    with open(meta_path) as f:
        meta = json.load(f)

    assert meta["candidate_name"] == "pd_detector_improved.pkl"
    assert meta["validation_status"] == "FAILED_EXTERNAL_GENERALIZATION"
    assert meta["validation_verdict"] == "REJECTED_FOR_PRODUCTION_DEPLOYMENT"
    assert "models/pd_detector.pkl" in meta["production_decision"]

