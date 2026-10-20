"""
ParkinSense — Centralized Prediction Service
============================================
Provides unified model inference and clinical explainability for both
MediaPipe and 4D-Humans pipelines.

Guarantees:
- Strict validation via FeatureAdapter and FeatureContract (290 features).
- Uniform prediction schema for all pipelines.
- Consistent model loading from models/pd_detector.pkl & models/random_forest.pkl.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
from src.features.feature_adapter import FeatureAdapter
from src.inference.prediction_mapping import summarize_binary_prediction

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

STANDARD_DISCLAIMER = (
    "ParkinSense is a research prototype intended for gait analysis research and educational purposes. "
    "It is not a clinical diagnostic tool and should not be used to diagnose or make medical decisions."
)


def _load_model_dataset_performance() -> Dict[str, Any]:
    """Load benchmark dataset metrics from project CSV when present."""
    csv_path = _PROJECT_ROOT / "data" / "processed" / "mediapipe_model_comparison.csv"
    if not csv_path.exists():
        return {}

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return {}

        preferred = df[df["model_name"].str.contains("RandomForest_Current_Baseline", na=False)]
        if preferred.empty:
            preferred = df.head(1)

        row = preferred.iloc[0]
        return {
            "accuracy": float(row.get("accuracy", np.nan)),
            "accuracy_std": float(row.get("accuracy_std", np.nan)),
            "sensitivity": float(row.get("sensitivity", np.nan)),
            "specificity": float(row.get("specificity", np.nan)),
            "precision": float(row.get("precision", np.nan)),
            "f1_score": float(row.get("f1_score", np.nan)),
            "roc_auc": float(row.get("roc_auc", np.nan)),
            "model_name": str(row.get("model_name", "Random Forest")),
            "feature_set": str(row.get("feature_set", "290 features")),
            "num_features": int(row.get("num_features", 290)),
        }
    except Exception as err:
        logger.warning("Could not read dataset performance CSV: %s", err)
        return {}


class PredictionEngine:
    """
    Centralized prediction engine loading trained ML classifiers
    and scoring standardized 290-dim gait feature vectors.
    """

    def __init__(
        self,
        pd_model_path: Optional[str] = None,
        severity_model_path: Optional[str] = None,
    ) -> None:
        self.pd_model_path = pd_model_path or DEFAULT_PD_MODEL_PATH
        self.severity_model_path = severity_model_path or DEFAULT_SEVERITY_MODEL_PATH
        self.feature_extractor = SMPLFeatureExtractor()
        self.feature_adapter = FeatureAdapter(strict=True)
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

    def predict_features(
        self,
        feature_df: pd.DataFrame,
        source_pipeline: str = "common",
    ) -> Dict[str, Any]:
        """
        Run model inference directly on an extracted feature DataFrame.

        Args:
            feature_df: Extracted feature vector DataFrame.
            source_pipeline: Pipeline identifier ('mediapipe', 'fourd_humans', etc.).

        Returns:
            Dictionary containing prediction results, probabilities, and explainability.
        """
        self._load_models()

        # Step 1: Adapt and validate against the 290-dim contract
        aligned_feature_df = self.feature_adapter.adapt(
            feature_df,
            source_pipeline=source_pipeline,
        )

        # Step 2: Binary PD Detection
        logger.info("Running PD Detection classifier (Pipeline: %s)...", source_pipeline)
        if isinstance(self._pd_model, dict):
            clf = self._pd_model["model"]
            feat_names = self._pd_model.get("feature_names", list(aligned_feature_df.columns))
            scaler = self._pd_model.get("scaler")
            selected_features_df = aligned_feature_df[feat_names]
            X_eval = scaler.transform(selected_features_df) if scaler is not None else selected_features_df.values
            pd_probs = clf.predict_proba(X_eval)[0]
            classes = clf.classes_
            explainer_model = clf
            explainer_df = selected_features_df
            model_type = self._pd_model.get("model_name", type(clf).__name__)
            n_feat = len(feat_names)
            n_trees = int(getattr(clf, "n_estimators", 0) or 0)
        else:
            clf = self._pd_model
            pd_probs = clf.predict_proba(aligned_feature_df)[0]
            classes = clf.classes_
            explainer_model = clf
            explainer_df = aligned_feature_df
            model_type = "Random Forest" if "Forest" in type(clf).__name__ else type(clf).__name__
            n_feat = int(getattr(clf, "n_features_in_", aligned_feature_df.shape[1]))
            n_trees = int(getattr(clf, "n_estimators", 0) or 0)

        prediction_summary = summarize_binary_prediction(classes, pd_probs)
        pd_pred_class = prediction_summary["label"]
        pd_probability = prediction_summary["parkinsons_probability"]
        if pd_probability is None:
            pd_probability = prediction_summary["confidence"]

        diagnosis_name = prediction_summary["diagnosis"]
        risk_level = "Not assessed"

        # Step 3: Severity Estimation
        logger.info("Running Severity Estimation classifier...")
        sev_pred_class = int(self._severity_model.predict(aligned_feature_df)[0])
        sev_desc = UPDRS_GAIT_DESCRIPTIONS.get(sev_pred_class, f"UPDRS Gait Score {sev_pred_class}")

        sev_probs = None
        if hasattr(self._severity_model, "predict_proba"):
            sev_probs = [float(p) for p in self._severity_model.predict_proba(aligned_feature_df)[0]]

        # Step 4: Clinical Explainability
        logger.info("Computing clinical explainability...")
        explainable_features = self.explainer.explain(explainer_model, explainer_df, top_k=5)

        # Step 5: Metadata & Model Information
        model_info = {
            "model": model_type,
            "n_features": n_feat,
            "n_trees": n_trees,
            "classes": [str(cls) for cls in getattr(clf, "classes_", [])],
        }

        dataset_performance = _load_model_dataset_performance()

        return {
            "parkinsons_detection": {
                "label": pd_pred_class,
                "diagnosis": diagnosis_name,
                "probability": round(pd_probability * 100, 1),
                "probability_raw": round(pd_probability, 4),
                "confidence": round(prediction_summary["confidence"] * 100, 1),
                "risk_level": risk_level,
                "class_probabilities": {
                    label: round(probability * 100, 1)
                    for label, probability in prediction_summary["class_probabilities"].items()
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
            "model_information": model_info,
            "dataset_performance": dataset_performance,
            "disclaimer": STANDARD_DISCLAIMER,
        }

    def predict(
        self,
        smpl_csv_path: str | Path,
        fps: float = 29.75,
    ) -> Dict[str, Any]:
        """
        Legacy entry point: run prediction on an SMPL CSV file.
        Preserves complete backward compatibility for existing callers.
        """
        smpl_csv_path = Path(smpl_csv_path)
        if not smpl_csv_path.exists():
            raise FileNotFoundError(f"SMPL CSV not found: {smpl_csv_path}")

        smpl_df = pd.read_csv(smpl_csv_path)
        feature_df = self.feature_extractor.extract_features(smpl_df, fps=fps)

        prediction_result = self.predict_features(feature_df, source_pipeline="smpl_csv")

        # Include metadata required by legacy tests/callers
        prediction_result["metadata"] = {
            "num_frames": len(smpl_df),
            "fps": round(fps, 2),
            "features_extracted": feature_df.shape[1],
            "pd_model_features": int(self._pd_model.n_features_in_),
            "severity_model_features": int(self._severity_model.n_features_in_),
        }
        return prediction_result



def run_prediction(smpl_csv_path: str | Path, fps: float = 29.75) -> Dict[str, Any]:
    """Convenience function to run prediction on an SMPL CSV."""
    engine = PredictionEngine()
    return engine.predict(smpl_csv_path, fps=fps)


# ---------------------------------------------------------------------------
# Candidate Model Prediction Engine (TULIP-trained, opt-in research use)
# ---------------------------------------------------------------------------

DEFAULT_CANDIDATE_MODEL_PATH = str(_PROJECT_ROOT / "models" / "pd_detector_tulip_candidate.pkl")

CANDIDATE_DISCLAIMER = (
    "This prediction uses an experimental candidate model trained on 8 clinical participants "
    "(3 healthy, 5 Parkinson's) from the TULIP gait dataset. Results are research estimates only, "
    "not clinical diagnoses. The small training cohort is insufficient for reliable generalization. "
    "Cross-validated subject-level accuracy is approximately 50% (sensitivity 60%, specificity 33%). "
    "Do not use this output for medical decisions."
)


class CandidatePredictionEngine:
    """
    Opt-in prediction engine for the TULIP-trained candidate model.

    Key differences from the production PredictionEngine:
    - Uses 288-dim pose-only features (excludes fps and num_frames).
    - Loads model from models/pd_detector_tulip_candidate.pkl.
    - Does NOT run severity estimation (no TULIP-trained severity model exists).
    - All outputs are explicitly labelled as research estimates.
    - Never overwrites or interferes with production model behavior.
    """

    def __init__(self, candidate_model_path: Optional[str] = None) -> None:
        self.candidate_model_path = candidate_model_path or DEFAULT_CANDIDATE_MODEL_PATH
        self.explainer = ModelExplainer()
        self._model: Any = None
        self._feature_names: List[str] = []
        self._metadata: Dict[str, Any] = {}

    def is_available(self) -> bool:
        """Check whether the candidate model artifact exists on disk."""
        return Path(self.candidate_model_path).exists()

    def _load_model(self) -> None:
        """Load the candidate model artifact dictionary from disk."""
        if self._model is not None:
            return

        model_path = Path(self.candidate_model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Candidate model not found at {model_path}. "
                "Run scripts/train_tulip_mediapipe_model.py first."
            )

        logger.info("Loading candidate model: %s", model_path)
        artifact = joblib.load(model_path)

        if not isinstance(artifact, dict) or "model" not in artifact:
            raise ValueError(
                f"Invalid candidate model artifact at {model_path}. "
                "Expected a dictionary with 'model' and 'feature_names' keys."
            )

        self._model = artifact["model"]
        self._feature_names = artifact.get("feature_names", [])
        self._metadata = {
            k: v for k, v in artifact.items()
            if k not in ("model",)
        }

        expected_n = self._model.n_features_in_
        if len(self._feature_names) != expected_n:
            raise ValueError(
                f"Candidate feature_names length ({len(self._feature_names)}) "
                f"does not match model.n_features_in_ ({expected_n})."
            )

        logger.info(
            "Candidate model loaded: %s, %d features, classes=%s",
            type(self._model).__name__,
            expected_n,
            list(self._model.classes_),
        )

    def _extract_pose_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the 288-dim pose-only subset from a full 290-dim feature DataFrame.

        The production pipeline always produces 290 features (fps + num_frames + 288 pose).
        The candidate model expects only the 288 pose columns in the exact same order.
        """
        present = set(feature_df.columns)
        missing = [c for c in self._feature_names if c not in present]
        if missing:
            raise ValueError(
                f"Feature DataFrame is missing {len(missing)} columns required by "
                f"the candidate model. First missing: {missing[:5]}"
            )

        pose_df = feature_df[self._feature_names].copy()

        # Validate finite values
        arr = pose_df.values
        n_nan = int(np.sum(np.isnan(arr)))
        n_inf = int(np.sum(np.isinf(arr)))
        if n_nan > 0 or n_inf > 0:
            raise ValueError(
                f"Pose feature vector contains non-finite values ({n_nan} NaN, {n_inf} Inf)."
            )

        return pose_df

    def predict_features(
        self,
        feature_df: pd.DataFrame,
        source_pipeline: str = "mediapipe",
    ) -> Dict[str, Any]:
        """
        Run candidate model inference on an extracted feature DataFrame.

        The feature_df should be the standard 290-dim output from the MediaPipe
        or 4D-Humans extractor. This method extracts the 288 pose-only columns
        automatically.

        Returns:
            Dictionary with candidate prediction results clearly marked as research estimates.
        """
        self._load_model()

        # Extract 288-dim pose-only subset
        pose_df = self._extract_pose_features(feature_df)

        logger.info("Running candidate PD detection (Pipeline: %s)...", source_pipeline)
        probs = self._model.predict_proba(pose_df)[0]
        classes = list(self._model.classes_)
        pd_idx = classes.index(1) if 1 in classes else 1
        ht_idx = classes.index(0) if 0 in classes else 0

        pd_prob = float(probs[pd_idx])
        ht_prob = float(probs[ht_idx])
        pred_class = int(classes[np.argmax(probs)])
        diagnosis = "Parkinson's Disease (Research Estimate)" if pred_class == 1 else "Healthy Control (Research Estimate)"

        # Explainability
        explainable_features = self.explainer.explain(self._model, pose_df, top_k=5)

        # Training provenance from artifact metadata
        training_data = self._metadata.get("training_data", {})
        cv_perf = self._metadata.get("cv_performance", {})

        return {
            "parkinsons_detection": {
                "label": pred_class,
                "diagnosis": diagnosis,
                "probability": round(pd_prob * 100, 1),
                "probability_raw": round(pd_prob, 4),
                "confidence": round(max(pd_prob, ht_prob) * 100, 1),
                "risk_level": "Research estimate only",
                "class_probabilities": {
                    "Healthy Control": round(ht_prob * 100, 1),
                    "Parkinson's Disease": round(pd_prob * 100, 1),
                },
            },
            "severity_estimation": {
                "updrs_gait_score": None,
                "description": "Severity estimation not available for candidate model.",
                "class_probabilities": None,
            },
            "explainability": {
                "important_gait_indicators": explainable_features,
            },
            "model_information": {
                "model": "Random Forest (TULIP Candidate)",
                "model_version": self._metadata.get("version", "unknown"),
                "n_features": int(self._model.n_features_in_),
                "n_trees": int(getattr(self._model, "n_estimators", 0)),
                "classes": [str(c) for c in classes],
                "training_source": training_data.get("source", "TULIP Clinical Gait Cohort"),
                "training_samples": training_data.get("n_samples", "unknown"),
                "training_participants": training_data.get("n_participants", "unknown"),
                "is_candidate": True,
            },
            "dataset_performance": {
                "logo_subject_accuracy": cv_perf.get("logo_subj_accuracy"),
                "logo_subject_sensitivity": cv_perf.get("logo_subj_sensitivity"),
                "logo_subject_specificity": cv_perf.get("logo_subj_specificity"),
                "logo_video_accuracy": cv_perf.get("logo_vid_accuracy"),
                "logo_video_auc": cv_perf.get("logo_vid_auc"),
                "validation_method": "Leave-One-Subject-Out (8 clinical participants)",
                "note": "Insufficient sample size for reliable generalization claims.",
            },
            "disclaimer": CANDIDATE_DISCLAIMER,
        }

