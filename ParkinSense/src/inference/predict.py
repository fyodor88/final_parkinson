"""
ParkinSense — End-to-End Inference Pipeline
============================================
Runs the full ParkinSense pipeline on a walking video:

    Video file
        ↓
    PoseClient (Remote 4D-Humans GPU server OR Local Mock)
        ↓
    216-dim SMPL pose sequence
        ↓
    SMPLFeatureExtractor (290 features)
        ↓
    PD Detector Classifier (models/pd_detector.pkl)
        ↓
    Severity Estimator (models/random_forest.pkl)
        ↓
    ModelExplainer (Clinical Gait Indicators)
        ↓
    JSON Output & Prediction Summary

Usage CLI:
    python -m src.inference.predict --video video/input/walking.mp4 --mode mock

Usage Python:
    from src.inference.predict import ParkinsonPredictor
    predictor = ParkinsonPredictor()
    result = predictor.predict("video/input/walking.mp4")
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

# Resolve project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import get_logger, ensure_dir
from src.pose.pose_client import PoseClient
from src.features.smpl_feature_extractor import SMPLFeatureExtractor
from src.features.explainability import ModelExplainer
from src.inference.prediction_mapping import summarize_binary_prediction

logger = get_logger(__name__)

# Defaults
DEFAULT_PD_MODEL_PATH = str(_PROJECT_ROOT / "models" / "pd_detector.pkl")
DEFAULT_SEVERITY_MODEL_PATH = str(_PROJECT_ROOT / "models" / "random_forest.pkl")
DEFAULT_OUTPUT_JSON = str(_PROJECT_ROOT / "outputs" / "prediction.json")

UPDRS_GAIT_DESCRIPTIONS = {
    0: "Normal Gait (UPDRS Gait Score 0: No movement difficulty)",
    1: "Slight Impairment (UPDRS Gait Score 1: Slow walking or minor asymmetry)",
    2: "Mild/Moderate Impairment (UPDRS Gait Score 2: Impaired balance or festive gait)",
}


class ParkinsonPredictor:
    """
    End-to-End ParkinSense Inference Engine.
    """

    def __init__(
        self,
        pd_model_path: str = DEFAULT_PD_MODEL_PATH,
        severity_model_path: str = DEFAULT_SEVERITY_MODEL_PATH,
        output_json: str = DEFAULT_OUTPUT_JSON,
        server_url: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> None:
        self.pd_model_path = pd_model_path
        self.severity_model_path = severity_model_path
        self.output_json = output_json
        self.pose_client = PoseClient(server_url=server_url, mode=mode)
        self.feature_extractor = SMPLFeatureExtractor()
        self.explainer = ModelExplainer()

        self._pd_model: Any = None
        self._severity_model: Any = None

    def _load_models(self) -> None:
        """Load trained models from disk if not already loaded."""
        if self._pd_model is None:
            if not os.path.exists(self.pd_model_path):
                raise FileNotFoundError(f"PD Detector model file missing: {self.pd_model_path}")
            logger.info("Loading PD Detector model from: %s", self.pd_model_path)
            self._pd_model = joblib.load(self.pd_model_path)

        if self._severity_model is None:
            if not os.path.exists(self.severity_model_path):
                raise FileNotFoundError(f"Severity model file missing: {self.severity_model_path}")
            logger.info("Loading Severity model from: %s", self.severity_model_path)
            self._severity_model = joblib.load(self.severity_model_path)

    def predict(self, video_path: str | Path) -> Dict[str, Any]:
        """
        Run full prediction pipeline on input video.

        Args:
            video_path: Path to walking video file.

        Returns:
            Dict containing prediction status, diagnosis, probability, severity score,
            explainable gait indicators, and processing metadata.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")

        self._load_models()

        # Step 1: Extract SMPL Pose
        logger.info("Step 1/4 — Pose Extraction via PoseClient (Mode: %s)...", self.pose_client.mode)
        smpl_df, fps, num_frames, mode_used = self.pose_client.extract_smpl(video_path)

        # Step 2: Feature Extraction (290 dimensions)
        logger.info("Step 2/4 — Extracting 290 SMPL gait features...")
        feature_df = self.feature_extractor.extract_features(smpl_df, fps=fps)

        # Ensure exact feature column alignment expected by models
        pd_expected = getattr(self._pd_model, "feature_names_in_", None)
        if pd_expected is not None:
            feature_df = feature_df[list(pd_expected)]

        # Step 3: Model Inference
        logger.info("Step 3/4 — Running PD Detection & Severity Models...")
        pd_probs = self._pd_model.predict_proba(feature_df)[0]
        prediction_summary = summarize_binary_prediction(self._pd_model.classes_, pd_probs)
        pd_pred_class = prediction_summary["label"]
        pd_probability = prediction_summary["parkinsons_probability"]
        if pd_probability is None:
            pd_probability = prediction_summary["confidence"]

        # Class names & risk levels
        diagnosis_name = prediction_summary["diagnosis"]
        risk_level = "Not assessed"

        # Severity prediction
        sev_pred_class = int(self._severity_model.predict(feature_df)[0])
        sev_desc = UPDRS_GAIT_DESCRIPTIONS.get(sev_pred_class, f"UPDRS Gait Score {sev_pred_class}")

        sev_probs = None
        if hasattr(self._severity_model, "predict_proba"):
            sev_probs = [float(p) for p in self._severity_model.predict_proba(feature_df)[0]]

        # Step 4: Explainability
        logger.info("Step 4/4 — Computing Clinical Explainability...")
        explainable_features = self.explainer.explain(self._pd_model, feature_df, top_k=5)

        result = {
            "status": "success",
            "video_file": video_path.name,
            "pose_service_mode": mode_used,
            "num_frames": num_frames,
            "fps": round(fps, 2),
            "parkinsons_detection": {
                "label": pd_pred_class,
                "diagnosis": diagnosis_name,
                "probability": round(pd_probability * 100, 1),
                "probability_raw": round(pd_probability, 4),
                "confidence": round(prediction_summary["confidence"] * 100, 1),
                "class_probabilities": {
                    label: round(probability * 100, 1)
                    for label, probability in prediction_summary["class_probabilities"].items()
                },
                "risk_level": risk_level,
            },
            "severity_estimation": {
                "updrs_gait_score": sev_pred_class,
                "description": sev_desc,
                "class_probabilities": sev_probs,
            },
            "explainability": {
                "important_gait_indicators": explainable_features,
            },
        }

        # Save JSON output
        ensure_dir(Path(self.output_json).parent)
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        logger.info("✅ Prediction complete! Result saved to %s", self.output_json)
        return result


def main():
    parser = argparse.ArgumentParser(description="ParkinSense End-to-End Inference")
    parser.add_argument("--video", type=str, default="video/input/walking.mp4", help="Path to walking video")
    parser.add_argument("--mode", type=str, choices=["mock", "remote"], default=None, help="Pose service mode")
    parser.add_argument("--url", type=str, default=None, help="Remote 4D-Humans server URL")
    args = parser.parse_args()

    predictor = ParkinsonPredictor(server_url=args.url, mode=args.mode)
    res = predictor.predict(args.video)

    print("\n" + "=" * 55)
    print("               PARKINSENSE RESULT")
    print("=" * 55)
    print(f"Video File       : {res['video_file']}")
    print(f"Pose Mode        : {res['pose_service_mode'].upper()}")
    print("-" * 55)
    print(f"PD Diagnosis     : {res['parkinsons_detection']['diagnosis']}")
    print(f"Confidence       : {res['parkinsons_detection']['probability']}%")
    print(f"Risk Level       : {res['parkinsons_detection']['risk_level']}")
    print("-" * 55)
    print(f"Severity (UPDRS) : {res['severity_estimation']['description']}")
    print("-" * 55)
    print("Top Clinical Gait Indicators:")
    for item in res['explainability']['important_gait_indicators']:
        print(f"  • {item['indicator']} ({item['joint_name']}) — Importance: {item['importance']}")
    print("=" * 55)


if __name__ == "__main__":
    main()
