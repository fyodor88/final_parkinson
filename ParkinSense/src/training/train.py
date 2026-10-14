"""
ParkiSense — Multi-Model Trainer
==================================
Trains 7 classifiers on ``data/processed/dataset.csv`` using 5-fold
Stratified Cross-Validation, evaluates every model, selects the best
one by F1-score, and persists it for inference.

Models trained
--------------
  RandomForest · XGBoost · LightGBM · SVM · KNN · DecisionTree · LogisticRegression

Metrics reported (per fold + mean)
------------------------------------
  Accuracy · Precision · Recall · F1 · ROC-AUC

Outputs
-------
  models/best_model.pkl
  models/scaler.pkl
  outputs/training_results.csv
  outputs/model_comparison.csv
  outputs/confusion_matrices/<ModelName>.png

Usage (CLI):
    python src/training/train.py

    python src/training/train.py \\
        --dataset data/processed/dataset.csv \\
        --models-dir models \\
        --output-dir outputs

Usage (Python API):
    from src.training.train import ModelTrainer
    trainer = ModelTrainer()
    trainer.train()
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional heavy deps — fail gracefully so the file still imports
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import get_logger, ensure_dir
from src.features.feature_extractor import GaitFeatureExtractor

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATASET   = str(_PROJECT_ROOT / "data" / "processed" / "dataset.csv")
DEFAULT_MODELS_DIR = str(_PROJECT_ROOT / "models")
DEFAULT_OUTPUT_DIR = str(_PROJECT_ROOT / "outputs")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _build_model_registry() -> Dict[str, Any]:
    """
    Return an ordered dict of model_name → unfitted sklearn-compatible estimator.
    Heavy optional models (XGBoost, LightGBM) are included only if installed.
    """
    registry: Dict[str, Any] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=-1,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
    }

    if _HAS_XGB:
        registry["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    else:
        logger.warning("XGBoost not installed — skipping.")

    if _HAS_LGB:
        registry["LightGBM"] = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        logger.warning("LightGBM not installed — skipping.")

    return registry


# ---------------------------------------------------------------------------
# ModelTrainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Trains, evaluates, and persists the best Parkinson's disease classifier.

    Args:
        dataset_path:  Path to ``dataset.csv`` produced by :mod:`src.dataset.create_dataset`.
        models_dir:    Directory where ``best_model.pkl`` and ``scaler.pkl`` are saved.
        output_dir:    Directory for CSV reports and confusion-matrix PNGs.
        n_folds:       Number of Stratified K-Fold cross-validation folds.
        selection_metric:  Metric used to elect the best model
                           (``"f1"`` | ``"accuracy"`` | ``"roc_auc"``).
    """

    _METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc")

    def __init__(
        self,
        dataset_path:     str = DEFAULT_DATASET,
        models_dir:       str = DEFAULT_MODELS_DIR,
        output_dir:       str = DEFAULT_OUTPUT_DIR,
        n_folds:          int = 5,
        selection_metric: str = "f1",
    ) -> None:
        self.dataset_path     = Path(dataset_path)
        self.models_dir       = Path(models_dir)
        self.output_dir       = Path(output_dir)
        self.n_folds          = n_folds
        self.selection_metric = selection_metric

        self._cm_dir  = self.output_dir / "confusion_matrices"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, float]:
        """
        Execute the full training pipeline.

        Steps:
          1. Load and validate dataset.
          2. Split into features (X) and labels (y).
          3. For each model, run n_folds-Fold StratifiedKFold CV.
          4. Collect per-fold metrics and average them.
          5. Save confusion matrix PNG for each model.
          6. Select and persist the best model.
          7. Write training_results.csv and model_comparison.csv.

        Returns:
            Dict mapping metric name → score for the **best** model.
        """
        X, y = self._load_dataset()
        models = _build_model_registry()

        logger.info("Starting %d-Fold CV training for %d models …", self.n_folds, len(models))

        all_results:  List[Dict]  = []
        comparison:   List[Dict]  = []
        best_name:    str         = ""
        best_score:   float       = -1.0
        best_model:   Any         = None
        best_scaler:  StandardScaler | None = None

        for model_name, model in models.items():
            logger.info("── Training: %s ──────────────────────────────────", model_name)
            fold_metrics, final_model, final_scaler, cm = self._cross_validate(
                model, model_name, X, y
            )
            all_results.extend(fold_metrics)

            mean_scores = self._mean_scores(fold_metrics)
            comparison.append({"model": model_name, **mean_scores})

            self._save_confusion_matrix(cm, model_name)

            score = mean_scores[self.selection_metric]
            logger.info(
                "%s | Accuracy=%.4f  F1=%.4f  ROC-AUC=%.4f",
                model_name,
                mean_scores["accuracy"],
                mean_scores["f1"],
                mean_scores["roc_auc"],
            )

            if score > best_score:
                best_score  = score
                best_name   = model_name
                best_model  = final_model
                best_scaler = final_scaler

        # ── Persist best model ─────────────────────────────────────────
        ensure_dir(self.models_dir)
        best_model_path  = self.models_dir / "best_model.pkl"
        scaler_path      = self.models_dir / "scaler.pkl"

        joblib.dump(best_model,  str(best_model_path))
        joblib.dump(best_scaler, str(scaler_path))

        logger.info("🏆 Best model: %s  (%s=%.4f)", best_name, self.selection_metric, best_score)
        logger.info("✅ Saved → %s", best_model_path)
        logger.info("✅ Saved → %s", scaler_path)

        # ── Write CSV reports ──────────────────────────────────────────
        ensure_dir(self.output_dir)

        training_df   = pd.DataFrame(all_results)
        comparison_df = pd.DataFrame(comparison).sort_values(self.selection_metric, ascending=False)

        training_df.to_csv(  self.output_dir / "training_results.csv",  index=False)
        comparison_df.to_csv(self.output_dir / "model_comparison.csv",  index=False)

        logger.info("✅ training_results.csv saved")
        logger.info("✅ model_comparison.csv saved")
        logger.info("\n%s", comparison_df.to_string(index=False))

        return {k: comparison_df[comparison_df["model"] == best_name][k].values[0]
                for k in self._METRIC_KEYS}

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def _cross_validate(
        self,
        model:      Any,
        model_name: str,
        X:          np.ndarray,
        y:          np.ndarray,
    ) -> Tuple[List[Dict], Any, StandardScaler, np.ndarray]:
        """
        Run Stratified K-Fold CV for *model*.

        Returns:
            fold_metrics:  List of per-fold metric dicts.
            final_model:   Model retrained on the entire dataset.
            final_scaler:  Scaler fitted on the entire dataset.
            cumulative_cm: Sum of confusion matrices across all folds.
        """
        skf          = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        fold_metrics: List[Dict] = []
        cum_cm        = np.zeros((2, 2), dtype=int)

        for fold_idx, (train_idx, val_idx) in enumerate(
            tqdm(
                skf.split(X, y),
                total=self.n_folds,
                desc=f"  {model_name} folds",
                unit="fold",
                leave=False,
            ),
            start=1,
        ):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Scale per-fold (avoid data leakage)
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val   = scaler.transform(X_val)

            import copy
            fold_model = copy.deepcopy(model)
            fold_model.fit(X_train, y_train)

            y_pred  = fold_model.predict(X_val)
            y_proba = self._get_proba(fold_model, X_val)

            metrics = self._compute_metrics(y_val, y_pred, y_proba)
            metrics.update({"model": model_name, "fold": fold_idx})
            fold_metrics.append(metrics)

            cum_cm += confusion_matrix(y_val, y_pred, labels=[0, 1])

        # ── Final model retrained on ALL data ─────────────────────────
        final_scaler = StandardScaler()
        X_scaled     = final_scaler.fit_transform(X)

        import copy
        final_model = copy.deepcopy(model)
        final_model.fit(X_scaled, y)

        return fold_metrics, final_model, final_scaler, cum_cm

    # ------------------------------------------------------------------
    # Metrics & plots
    # ------------------------------------------------------------------

    @staticmethod
    def _get_proba(model: Any, X: np.ndarray) -> np.ndarray:
        """Return probability estimates, falling back to decision_function."""
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        if hasattr(model, "decision_function"):
            df = model.decision_function(X)
            # Normalise to [0,1] with sigmoid
            return 1.0 / (1.0 + np.exp(-df))
        return model.predict(X).astype(float)

    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        return {
            "accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0, 4),
        }

    @staticmethod
    def _mean_scores(fold_metrics: List[Dict]) -> Dict[str, float]:
        metric_keys = ("accuracy", "precision", "recall", "f1", "roc_auc")
        return {k: round(float(np.mean([m[k] for m in fold_metrics])), 4) for k in metric_keys}

    def _save_confusion_matrix(self, cm: np.ndarray, model_name: str) -> None:
        """Save a styled confusion matrix PNG."""
        ensure_dir(self._cm_dir)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Control", "Parkinson"],
            yticklabels=["Control", "Parkinson"],
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        plt.tight_layout()
        save_path = self._cm_dir / f"{model_name}.png"
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
        logger.info("  Confusion matrix saved → %s", save_path)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load ``dataset.csv``, extract feature matrix X and label vector y.

        Returns:
            (X, y) as NumPy arrays.

        Raises:
            FileNotFoundError: If ``dataset.csv`` does not exist.
            ValueError:        If fewer than 2 samples or no label column.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}\n"
                "Run  python src/dataset/create_dataset.py  first."
            )

        df = pd.read_csv(str(self.dataset_path))
        logger.info("Loaded dataset: %d rows × %d cols from %s", *df.shape, self.dataset_path)

        if "label" not in df.columns:
            raise ValueError("Dataset CSV must contain a 'label' column.")

        # Drop non-feature columns
        drop_cols = [c for c in ("label", "video_file") if c in df.columns]
        feature_cols = [c for c in GaitFeatureExtractor.FEATURE_COLUMNS if c in df.columns]

        if not feature_cols:
            # Fall back to all numeric columns except label
            feature_cols = [c for c in df.columns if c not in drop_cols]

        X = df[feature_cols].values.astype(float)
        y = df["label"].values.astype(int)

        logger.info("Features: %d  |  Samples: %d  |  Classes: %s",
                    X.shape[1], X.shape[0], dict(zip(*np.unique(y, return_counts=True))))

        if len(y) < 2:
            raise ValueError("Dataset has fewer than 2 samples — cannot train.")

        return X, y


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train",
        description="ParkiSense — Multi-Model Trainer with 5-Fold CV",
    )
    p.add_argument("--dataset",    "-d", default=DEFAULT_DATASET,    help="Path to dataset.csv")
    p.add_argument("--models-dir", "-m", default=DEFAULT_MODELS_DIR, help="Directory to save models")
    p.add_argument("--output-dir", "-o", default=DEFAULT_OUTPUT_DIR, help="Directory for reports/plots")
    p.add_argument("--folds",      "-k", type=int, default=5,         help="Number of CV folds")
    p.add_argument(
        "--metric",
        choices=["f1", "accuracy", "roc_auc"],
        default="f1",
        help="Metric used to select the best model",
    )
    return p


def main() -> None:
    args    = _build_arg_parser().parse_args()
    trainer = ModelTrainer(
        dataset_path=args.dataset,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        n_folds=args.folds,
        selection_metric=args.metric,
    )
    trainer.train()


if __name__ == "__main__":
    main()
