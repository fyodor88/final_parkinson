"""
ParkinSense — Prediction Module
================================
Loads trained models and runs prediction on SMPL feature vectors.
Wraps the existing feature extraction and model inference code.

Models used:
- pd_detector.pkl     — Binary PD detection (0=Healthy, 1=Parkinson's)
- random_forest.pkl    — UPDRS gait severity (0, 1, 2)

Feature extraction:
- SMPLFeatureExtractor — 290-dim features from SMPL rotation matrices
- ModelExplainer       — Clinical gait indicator explanation
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Resolve project root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import get_logger
from src.features.smpl_feature_extractor import SMPLFeatureExtractor
from src.features.explainability import ModelExplainer

logger = get_logger(__name__)

# Model paths
DEFAULT_PD_MODEL_PATH = str(_PROJECT_ROOT / "models" / "pd_detector.pkl")
DEFAULT_SEVERITY_MODEL_PATH = str(_PROJECT_ROOT / "models" / "random_forest.pkl")

# UPDRS gait score descriptions
UPDRS_GAIT_DESCRIPTIONS = {
    0: "Normal Gait (UPDRS 0: No movement difficulty)",
    1: "Slight Impairment (UPDRS 1: Slow walking or minor asymmetry)",
    2: "Moderate Impairment (UPDRS 2: Impaired balance or festive gait)",
}


class PredictionEngine:
    """
    Loads trained models and runs inference on SMPL feature vectors.
    """

    def __init__(
        self,
        pd_model_path: Optional[str] = None,
        severity_model_path: Optional[str] = None,
    ) -> None:
        self.pd_model_path = pd_model_path or DEFAULT_PD_MODEL_PATH
        self.severity_model_path = severity_model_path or DEFAULT_SEVERITY_MODEL_PATH
        self.feature_extractor = SMPLFeatureExtractor()
        self.explainer = ModelExplainer()
        self._pd_model: Any = None
        self._severity_model: Any = None

    def _load_models(self) -> None:
        """Load trained models from disk."""
        if self._pd_model is None:
            if not Path(self.pd_model_path).exists():
                raise FileNotFoundError(f"PD Detector model not found: {self.pd_model_path}")
            logger.info("Loading PD Detector model: %s", self.pd_model_path)
            self._pd_model = joblib.load(self.pd_model_path)

        if self._severity_model is None:
            if not Path(self.severity_model_path).exists():
                raise FileNotFoundError(f"Severity model not found: {self.severity_model_path}")
            logger.info("Loading Severity model: %s", self.severity_model_path)
            self._severity_model = joblib.load(self.severity_model_path)

    def predict(
        self,
        smpl_csv_path: str | Path,
        fps: float = 29.75,
    ) -> Dict[str, Any]:
        """
        Run full prediction pipeline on an SMPL CSV file.

        Args:
            smpl_csv_path: Path to SMPL CSV (216 rotation matrix columns).
            fps:           Video frame rate.

        Returns:
            Dictionary containing prediction results, probabilities, and explainability.
        """
        smpl_csv_path = Path(smpl_csv_path)
        if not smpl_csv_path.exists():
            raise FileNotFoundError(f"SMPL CSV not found: {smpl_csv_path}")

        self._load_models()

        # Step 1: Load SMPL data
        logger.info("Loading SMPL CSV: %s", smpl_csv_path)
        smpl_df = pd.read_csv(smpl_csv_path)
        num_frames = len(smpl_df)
        logger.info("Loaded %d SMPL frames.", num_frames)

        # Step 2: Extract 290-dim features
        logger.info("Extracting 290 SMPL gait features...")
        feature_df = self.feature_extractor.extract_features(smpl_df, fps=fps)

        # Ensure exact feature column alignment for each model
        pd_expected = getattr(self._pd_model, "feature_names_in_", None)
        if pd_expected is not None:
            feature_df = feature_df[list(pd_expected)]

        # Step 3: PD Detection
        logger.info("Running PD Detection model...")
        pd_pred_class = int(self._pd_model.predict(feature_df)[0])
        pd_probs = self._pd_model.predict_proba(feature_df)[0]
        pd_probability = float(pd_probs[1]) if len(pd_probs) > 1 else (1.0 if pd_pred_class == 1 else 0.0)

        diagnosis_name = "Parkinson's Likely" if pd_pred_class == 1 else "Healthy / Control"
        risk_level = "High" if pd_probability >= 0.70 else ("Medium" if pd_probability >= 0.40 else "Low")

        # Step 4: Severity Estimation
        logger.info("Running Severity Estimation model...")
        sev_pred_class = int(self._severity_model.predict(feature_df)[0])
        sev_desc = UPDRS_GAIT_DESCRIPTIONS.get(sev_pred_class, f"UPDRS Gait Score {sev_pred_class}")

        sev_probs = None
        if hasattr(self._severity_model, "predict_proba"):
            sev_probs = [float(p) for p in self._severity_model.predict_proba(feature_df)[0]]

        # Step 5: Explainability
        logger.info("Computing clinical explainability...")
        explainable_features = self.explainer.explain(self._pd_model, feature_df, top_k=5)

        # Build result
        result = {
            "parkinsons_detection": {
                "label": pd_pred_class,
                "diagnosis": diagnosis_name,
                "probability": round(pd_probability * 100, 1),
                "probability_raw": round(pd_probability, 4),
                "risk_level": risk_level,
                "class_probabilities": {
                    "Healthy": round(float(pd_probs[0]) * 100, 1),
                    "Parkinsons": round(float(pd_probs[1]) * 100, 1),
                },
            },
            "severity_estimation": {
                "updrs_gait_score": sev_pred_class,
                "description": sev_desc,
                "class_probabilities": sev_probs,
            },
            "explainability": {
                "important_gait_indicators": explainable_features,
            },
            "metadata": {
                "num_frames": num_frames,
                "fps": round(fps, 2),
                "features_extracted": feature_df.shape[1],
                "pd_model_features": int(self._pd_model.n_features_in_),
                "severity_model_features": int(self._severity_model.n_features_in_),
            },
        }

        logger.info(
            "Prediction complete: %s (confidence: %.1f%%, risk: %s)",
            diagnosis_name,
            pd_probability * 100,
            risk_level,
        )

        return result


def run_prediction(smpl_csv_path: str | Path, fps: float = 29.75) -> Dict[str, Any]:
    """
    Convenience function to run prediction on an SMPL CSV.

    Args:
        smpl_csv_path: Path to SMPL CSV file.
        fps:           Video frame rate.

    Returns:
        Prediction result dictionary.
    """
    engine = PredictionEngine()
    return engine.predict(smpl_csv_path, fps=fps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ParkinSense Prediction Engine")
    parser.add_argument("smpl_csv", help="Path to SMPL CSV file")
    parser.add_argument("--fps", type=float, default=29.75, help="Video FPS")
    args = parser.parse_args()

    result = run_prediction(args.smpl_csv, fps=args.fps)

    print("\n" + "=" * 55)
    print("               PARKINSENSE PREDICTION")
    print("=" * 55)
    print(f"\nPD Detection  : {result['parkinsons_detection']['diagnosis']}")
    print(f"Confidence    : {result['parkinsons_detection']['probability']}%")
    print(f"Risk Level    : {result['parkinsons_detection']['risk_level']}")
    print(f"\nSeverity      : {result['severity_estimation']['description']}")
    print("-" * 55)
    print("Top Clinical Gait Indicators:")
    for item in result["explainability"]["important_gait_indicators"]:
        print(f"  - {item['indicator']} ({item['joint_name']}) — Importance: {item['importance']}")
    print("=" * 55)
