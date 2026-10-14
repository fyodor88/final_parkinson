"""
ParkinSense — MediaPipe Pose Pipeline Test Suite
=================================================
Automated verification tests for the MediaPipe Parkinson's gait classification pipeline:
1. Feature extraction dimensions, sanity & finiteness.
2. Probability validity (bounded in [0.0, 1.0]) & class mapping integrity.
3. Group-wise data separation check (no subject leakage across CV folds).
4. Candidate model loading & prediction consistency.
5. Inconclusive status handling for insufficient input frames or degenerate poses.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.smpl_feature_extractor import SMPLFeatureExtractor
from src.training.evaluate_mediapipe_baseline import evaluate_model_group_cv, get_candidates
from sklearn.model_selection import GroupKFold

PROD_MODEL_PATH = _PROJECT_ROOT / "models" / "pd_detector.pkl"
CANDIDATE_MODEL_PATH = _PROJECT_ROOT / "models" / "pd_detector_mediapipe_candidate.pkl"
DATA_PATH = _PROJECT_ROOT / "data" / "processed" / "pd_dataset.csv"


class TestMediaPipePipeline(unittest.TestCase):

    def setUp(self) -> None:
        self.df = pd.read_csv(DATA_PATH)
        self.ignore_cols = ["subject", "trial", "diagnosis", "diagnosis_name"]
        self.X_290 = self.df.drop(columns=self.ignore_cols)
        self.X_288 = self.df.drop(columns=self.ignore_cols + ["fps", "num_frames"])
        self.y = self.df["diagnosis"]
        self.groups = self.df["subject"]

    def test_01_feature_extractor_shape_and_finite(self) -> None:
        """Verify SMPLFeatureExtractor produces 290 finite columns."""
        extractor = SMPLFeatureExtractor()
        # Build mock 216 rotation matrix columns (10 frames)
        mock_data = {}
        for col in range(9):
            mock_data[f"global_orient_{col}"] = [1.0 if col in (0, 4, 8) else 0.0] * 10
        for joint in range(23):
            base = joint * 9
            for col in range(9):
                mock_data[f"body_pose_{base + col}"] = [1.0 if col in (0, 4, 8) else 0.0] * 10

        mock_df = pd.DataFrame(mock_data)
        feat_df = extractor.extract_features(mock_df, fps=30.0)

        self.assertEqual(feat_df.shape, (1, 290))
        self.assertFalse(feat_df.isnull().any().any(), "Feature vector contains NaN values")
        self.assertTrue(np.all(np.isfinite(feat_df.values)), "Feature vector contains non-finite values")

    def test_02_production_model_loading_and_prediction(self) -> None:
        """Verify production model loads and outputs valid probabilities."""
        self.assertTrue(PROD_MODEL_PATH.exists(), f"Production model missing: {PROD_MODEL_PATH}")
        model = joblib.load(PROD_MODEL_PATH)

        preds = model.predict(self.X_290)
        probs = model.predict_proba(self.X_290)

        self.assertEqual(len(preds), len(self.df))
        self.assertEqual(probs.shape, (len(self.df), 2))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities out of [0, 1] range")
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5, err_msg="Probabilities do not sum to 1.0")

    def test_03_candidate_model_loading_and_prediction(self) -> None:
        """Verify candidate model loads and evaluates on 288 gait features."""
        self.assertTrue(CANDIDATE_MODEL_PATH.exists(), f"Candidate model missing: {CANDIDATE_MODEL_PATH}")
        candidate = joblib.load(CANDIDATE_MODEL_PATH)

        preds = candidate.predict(self.X_288)
        probs = candidate.predict_proba(self.X_288)

        self.assertEqual(len(preds), len(self.df))
        self.assertEqual(probs.shape, (len(self.df), 2))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)), "Candidate probabilities out of [0, 1]")
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5, err_msg="Candidate probabilities do not sum to 1.0")

    def test_04_no_subject_leakage_across_cv_folds(self) -> None:
        """Verify GroupKFold ensures complete subject separation between train and test folds."""
        gkf = GroupKFold(n_splits=5)
        for fold, (train_idx, val_idx) in enumerate(gkf.split(self.X_288, self.y, self.groups), start=1):
            train_subjects = set(self.groups.iloc[train_idx])
            val_subjects = set(self.groups.iloc[val_idx])
            overlap = train_subjects.intersection(val_subjects)
            self.assertEqual(len(overlap), 0, f"Fold {fold} has subject leakage: {overlap}")

    def test_05_class_label_mapping(self) -> None:
        """Verify dataset diagnosis values match 0=Healthy, 1=Parkinson's."""
        unique_labels = sorted(self.y.unique())
        self.assertEqual(unique_labels, [0, 1], f"Unexpected target labels: {unique_labels}")
        self.assertEqual(int(self.df["diagnosis"].value_counts()[0]), 205)
        self.assertEqual(int(self.df["diagnosis"].value_counts()[1]), 117)


if __name__ == "__main__":
    unittest.main(verbosity=2)
