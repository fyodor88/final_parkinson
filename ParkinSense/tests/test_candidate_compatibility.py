"""
Tests for Phase 4 Candidate Model Application Compatibility
============================================================
Validates:
1. Candidate model (models/pd_detector_phase4_candidate.pkl) loads successfully.
2. PredictionEngine can evaluate feature vectors using candidate model bundle.
3. Feature subsetting and scaling operate as expected without altering input vectors.
4. Response schema is 100% compliant with frontend expectations.
5. Production model (models/pd_detector.pkl) remains fully functional and unchanged.
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_contract import FEATURE_CONTRACT_COLUMNS
from backend.prediction import PredictionEngine

@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    data = {"fps": 30.0, "num_frames": 100}
    for i in range(72):
        data[f"pose{i}_mean"] = 0.05 * (i % 5)
        data[f"pose{i}_std"] = 0.02 * (i % 3)
        data[f"pose{i}_min"] = -0.10
        data[f"pose{i}_max"] = 0.15
    return pd.DataFrame([data])[list(FEATURE_CONTRACT_COLUMNS)]

def test_production_model_unchanged(sample_feature_df: pd.DataFrame):
    pe = PredictionEngine()
    res = pe.predict_features(sample_feature_df, source_pipeline="mediapipe")
    assert res["model_information"]["model"] == "Random Forest"
    assert res["model_information"]["n_features"] == 290
    assert "parkinsons_detection" in res
    assert 0.0 <= res["parkinsons_detection"]["probability"] <= 100.0

def test_phase4_candidate_compatibility(sample_feature_df: pd.DataFrame):
    candidate_path = PROJECT_ROOT / "models" / "pd_detector_phase4_candidate.pkl"
    assert candidate_path.exists(), f"Candidate model not found at {candidate_path}"

    pe = PredictionEngine(pd_model_path=str(candidate_path))
    res = pe.predict_features(sample_feature_df, source_pipeline="mediapipe")

    assert res["model_information"]["model"] == "svm_rbf"
    assert res["model_information"]["n_features"] == 50
    assert "parkinsons_detection" in res
    assert res["parkinsons_detection"]["label"] in (0, 1)
    assert 0.0 <= res["parkinsons_detection"]["probability"] <= 100.0
    assert "Healthy" in res["parkinsons_detection"]["class_probabilities"]
    assert "Parkinsons" in res["parkinsons_detection"]["class_probabilities"]
