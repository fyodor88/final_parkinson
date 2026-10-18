"""Phase 4 model-comparison and external-generalization experiments.

Builds on scripts/run_phase4_feature_experiments.py and reuses its dataset
validation, participant-grouped outer splits, fold-local selectors, and metric
definitions. Adds:

- multi-model comparison under identical participant-grouped folds
  (RF baseline, regularized RF, SVM-RBF, L2 logistic regression);
- inner grouped-CV tuning for SVM/LogReg using outer-training data only;
- fold-wise metrics with mean +/- SD and pooled OOF metrics;
- external evaluation of DNE-only-trained candidates on the verified clinical
  TULIP candidate (plus continuity 44-row and descriptive unverified YouTube
  subsets), including the current production model;
- DNE-OOF-based Platt calibration applied to external scores (never fitted on
  TULIP labels);
- domain-shift and collapse diagnostics, confusion matrices, plots;
- experimental model artifacts saved separately from production models.

TULIP labels are never used for fitting, tuning, selection, calibration, or
model choice. This is a research prototype, not a clinical validation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_phase4_feature_experiments import (  # noqa: E402
    calculate_metrics,
    fit_selector,
    make_outer_splits,
    sha256_file,
    validate_dataset,
)

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

DNE_PATH = PROJECT_ROOT / "data" / "processed" / "pd_dataset.csv"
TULIP_CLINICAL_PATH = PROJECT_ROOT / "data" / "processed" / "improved_model" / "phase2_tulip_clinical_candidate.csv"
TULIP_CONTINUITY_PATH = PROJECT_ROOT / "data" / "processed" / "improved_model" / "tulip_training_dataset.csv"
TULIP_YOUTUBE_PATH = PROJECT_ROOT / "data" / "processed" / "improved_model" / "tulip_youtube_eval_dataset.csv"
RUN4_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "phase4_feature_experiments_run4" / "phase4_metrics.csv"
PRODUCTION_MODEL_PATH = PROJECT_ROOT / "models" / "pd_detector.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "phase4_model_experiments_run5"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "experiments" / "phase4_run5"

K_COUNTS = (10, 20, 30, 50)
DNE_SEEDS = (42, 123, 2026)
TULIP_SEEDS = (42,)
ALL_MODELS = ("rf_baseline", "rf_regularized", "svm_rbf", "logreg_l2")
PRIMARY_SELECTOR = "corr_rf_importance"
SELECTORS = ("corr_rf_importance", "l1_logistic")
THRESHOLD = 0.5
CONTRACT_METADATA_COLUMNS = ("fps", "num_frames")
SVM_GRID = {"svc__C": [0.5, 1.0, 10.0], "svc__gamma": ["scale", 0.01]}
LOGREG_GRID = {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]}
CALIBRATION_EPS = 1e-6


def positive_probability(estimator: BaseEstimator, features: pd.DataFrame) -> np.ndarray:
    """Return P(class=1) from a fitted binary estimator."""
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if 1 not in classes:
        raise ValueError("Estimator does not expose class 1")
    return np.asarray(estimator.predict_proba(features))[:, classes.index(1)]


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), CALIBRATION_EPS, 1.0 - CALIBRATION_EPS)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def metric_extensions(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Probability-quality metrics; undefined values are returned as None."""
    probabilities = np.asarray(probabilities, dtype=float)
    clipped = np.clip(probabilities, CALIBRATION_EPS, 1.0 - CALIBRATION_EPS)
    brier = float(brier_score_loss(np.asarray(y_true, dtype=int), probabilities))
    loss = float(log_loss(np.asarray(y_true, dtype=int), clipped, labels=[0, 1]))
    return {"brier_score": brier, "log_loss": loss}


def collapse_flag(metrics: dict[str, Any]) -> str:
    """Flag single-class prediction collapse at the evaluated unit level."""
    predicted_positive = metrics["tp"] + metrics["fp"]
    predicted_negative = metrics["tn"] + metrics["fn"]
    if predicted_positive == 0:
        return "all_predicted_healthy"
    if predicted_negative == 0:
        return "all_predicted_pd"
    return ""


def calculate_metrics_with_scores(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    unit: str,
) -> dict[str, Any]:
    """Participant/recording-level metrics plus Brier, log loss, and collapse flag."""
    metrics = calculate_metrics(y_true, probabilities)
    if unit == "recordings":
        metrics["n_recordings"] = metrics.pop("n_participants")
    metrics.update(metric_extensions(y_true, probabilities))
    metrics["collapse_flag"] = collapse_flag(metrics)
    return metrics


def aggregate_participant_predictions(recording_frame: pd.DataFrame) -> pd.DataFrame:
    """Mean-aggregate recording scores to one row per participant and configuration."""
    participant_frame = (
        recording_frame.groupby(
            ["track", "seed", "model", "feature_set", "participant_id"],
            as_index=False,
            dropna=False,
        )
        .agg(
            true_label=("true_label", "first"),
            n_recordings=("recording_id", "count"),
            mean_pd_score_uncalibrated=("pd_score_uncalibrated", "mean"),
            outer_folds=("fold", lambda values: ",".join(str(value) for value in sorted(set(values)))),
            selector=("selector", "first"),
            requested_count=("requested_count", "first"),
            feature_count=("feature_count", "first"),
        )
    )
    participant_frame["predicted_label"] = (
        participant_frame["mean_pd_score_uncalibrated"] >= THRESHOLD
    ).astype(int)
    return participant_frame


def inner_splitter(y_train: np.ndarray, groups_train: np.ndarray, seed: int) -> StratifiedGroupKFold | None:
    """Inner grouped CV splitter sized to the rarest class's group count."""
    groups_by_class: dict[int, set[str]] = {}
    for group, label in zip(groups_train, y_train):
        groups_by_class.setdefault(int(label), set()).add(group)
    min_groups = min(len(values) for values in groups_by_class.values())
    n_splits = min(3, min_groups)
    if n_splits < 2:
        return None
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def make_feature_sets(rankings: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Build the feature-set matrix: full-288 plus selector top-k sets."""
    feature_sets: list[dict[str, Any]] = [
        {"feature_set": "full_288", "selector": None, "requested_count": None, "columns": None}
    ]
    for selector, ranked in rankings.items():
        for requested_count in K_COUNTS:
            if len(ranked) < requested_count:
                continue
            feature_sets.append({
                "feature_set": f"{selector}_top{requested_count}",
                "selector": selector,
                "requested_count": requested_count,
                "columns": ranked[:requested_count],
            })
    return feature_sets


def model_applies_to_feature_set(model_name: str, feature_set: str) -> bool:
    """RF baseline reproduces both selector arms; other models use the primary selector."""
    if model_name == "rf_baseline":
        return True
    return feature_set == "full_288" or feature_set.startswith(PRIMARY_SELECTOR)


def fit_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    seed: int,
    tune: bool,
) -> tuple[BaseEstimator, dict[str, Any]]:
    """Fit one of the four predeclared model families with leakage-safe tuning."""
    metadata: dict[str, Any] = {"model": model_name, "outer_seed": seed, "tuned": tune}
    if model_name == "rf_baseline":
        estimator = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        )
        metadata["config"] = "existing production-family RF (300 trees, balanced)"
        metadata["tuning_status"] = "fixed_config_not_tuned"
        estimator.fit(X_train, y_train)
        return estimator, metadata
    if model_name == "rf_regularized":
        estimator = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        metadata["config"] = "existing repo RF_Regularized (min_samples_leaf=3, max_depth=15)"
        metadata["tuning_status"] = "fixed_config_not_tuned"
        estimator.fit(X_train, y_train)
        return estimator, metadata
    if model_name not in {"svm_rbf", "logreg_l2"}:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name == "svm_rbf":
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", class_weight="balanced", random_state=42)),
        ])
        grid = SVM_GRID
    else:
        base = Pipeline([
            ("scaler", StandardScaler()),
            (
                "logisticregression",
                LogisticRegression(
                    penalty="l2", class_weight="balanced", max_iter=5000, random_state=42
                ),
            ),
        ])
        grid = LOGREG_GRID

    splitter = inner_splitter(y_train, groups_train, seed) if tune else None
    if splitter is None:
        if model_name == "svm_rbf":
            final = Pipeline([
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ])
            final.fit(X_train, y_train)
        else:
            base.fit(X_train, y_train)
            final = base
        metadata["tuning"] = "disabled_uniform_defaults"
        metadata["tuning_status"] = "tuning_disabled_uniform_defaults"
        return final, metadata

    search = GridSearchCV(base, grid, scoring="roc_auc", cv=splitter, n_jobs=1, refit=True)
    search.fit(X_train, y_train, groups=groups_train)
    best_score = float(search.best_score_)
    if math.isfinite(best_score):
        best = dict(search.best_params_)
        tuning_status = "tuned_on_inner_grouped_cv_roc_auc"
    else:
        # StratifiedGroupKFold can place an entire class in one inner test fold
        # when very few groups remain (tiny TULIP LOSO training sets), making
        # roc_auc undefined. Fall back to the predeclared default
        # hyperparameters instead of an arbitrary nan-ranked grid cell.
        if model_name == "svm_rbf":
            best = {"svc__C": 1.0, "svc__gamma": "scale"}
        else:
            best = {"logisticregression__C": 1.0}
        tuning_status = "fallback_defaults_single_class_inner_fold"
    metadata.update({
        "grid": grid,
        "inner_cv_folds": int(splitter.n_splits),
        "inner_best_roc_auc": best_score,
        "best_params": best,
        "tuning_status": tuning_status,
    })
    if model_name == "svm_rbf":
        final = Pipeline([
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                    C=best["svc__C"],
                    gamma=best["svc__gamma"],
                ),
            ),
        ])
    else:
        final = Pipeline([
            ("scaler", StandardScaler()),
            (
                "logisticregression",
                LogisticRegression(
                    penalty="l2",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42,
                    C=best["logisticregression__C"],
                ),
            ),
        ])
    final.fit(X_train, y_train)
    return final, metadata


def run_cross_validation_track(
    track: str,
    frame: pd.DataFrame,
    models: tuple[str, ...],
    seeds: tuple[int, ...],
    tune_inner: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Participant-grouped OOF evaluation for one track."""
    pose_columns, group_column, label_column = validate_dataset(frame, track)
    recording_column = "trial" if track == "dne" else "recording_id"
    metric_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    recording_rows: list[dict[str, Any]] = []
    selected_feature_rows: list[dict[str, Any]] = []
    fit_metadata_rows: list[dict[str, Any]] = []

    dataset_participants = int(frame[group_column].nunique())
    participant_classes = frame.groupby(group_column)[label_column].first().value_counts().to_dict()
    class_rows = frame.groupby(label_column).size().to_dict()

    for seed in seeds:
        splits = make_outer_splits(frame, track, group_column, label_column, seed)
        for fold_number, (train_indices, test_indices) in enumerate(splits, start=1):
            train = frame.iloc[train_indices]
            test = frame.iloc[test_indices]
            y_train = train[label_column].astype(int).to_numpy()
            train_groups = train[group_column].astype(str).to_numpy()
            test_groups = test[group_column].astype(str).to_numpy()

            rankings: dict[str, list[str]] = {}
            selector_metadata: dict[str, Any] = {}
            for selector in SELECTORS:
                ranked, selector_meta = fit_selector(
                    selector, train[pose_columns], y_train, train_groups, track
                )
                rankings[selector] = ranked
                selector_metadata[selector] = selector_meta

            feature_sets = make_feature_sets(rankings)
            for feature_set in feature_sets:
                columns = feature_set["columns"]
                if columns is None:
                    columns = pose_columns
                    eligible_count = len(pose_columns)
                else:
                    eligible_count = len(rankings[feature_set["selector"]])
                selected_feature_rows.append({
                    "track": track,
                    "seed": seed,
                    "fold": fold_number,
                    "fit_scope": "outer_train_fold",
                    "selector": feature_set["selector"],
                    "requested_count": feature_set["requested_count"],
                    "eligible_after_filter": eligible_count,
                    "selected_count": len(columns),
                    "participants_train": ";".join(sorted(set(train_groups))),
                    "participants_test": ";".join(sorted(set(test_groups))),
                    "features": ";".join(columns),
                    "selector_metadata": json.dumps(selector_metadata.get(feature_set["selector"], {}), default=str),
                })
                for model_name in models:
                    if not model_applies_to_feature_set(model_name, feature_set["feature_set"]):
                        continue
                    estimator, fit_meta = fit_model(
                        model_name, train[columns], y_train, train_groups, seed, tune_inner
                    )
                    fit_metadata_rows.append({
                        "track": track,
                        "seed": seed,
                        "fold": fold_number,
                        "feature_set": feature_set["feature_set"],
                        **fit_meta,
                    })
                    probabilities = positive_probability(estimator, test[columns])
                    fold_identifier = f"{track}_s{seed}_f{fold_number}_{model_name}_{feature_set['feature_set']}"
                    for local_index, row_index in enumerate(test_indices):
                        row = frame.iloc[row_index]
                        recording_rows.append({
                            "track": track,
                            "seed": seed,
                            "fold": fold_number,
                            "model": model_name,
                            "feature_set": feature_set["feature_set"],
                            "selector": feature_set["selector"],
                            "requested_count": feature_set["requested_count"],
                            "feature_count": len(columns),
                            "recording_id": f"{row[group_column]}::{row[recording_column]}",
                            "participant_id": str(row[group_column]),
                            "true_label": int(row[label_column]),
                            "pd_score_uncalibrated": float(probabilities[local_index]),
                            "predicted_label": int(probabilities[local_index] >= THRESHOLD),
                            "fold_identifier": fold_identifier,
                        })

    recording_frame = pd.DataFrame(recording_rows)
    if recording_frame.empty:
        raise RuntimeError(f"{track}: no held-out predictions were generated")

    participant_frame = aggregate_participant_predictions(recording_frame)

    for (seed, model_name, feature_set_name), subset in participant_frame.groupby(
        ["seed", "model", "feature_set"], dropna=False
    ):
        y_true = subset["true_label"].astype(int).to_numpy()
        scores = subset["mean_pd_score_uncalibrated"].to_numpy(dtype=float)
        metrics = calculate_metrics_with_scores(y_true, scores, unit="participants")
        metric_rows.append({
            "evaluation_level": "dne_cv_participant_oof_pooled" if track == "dne" else "tulip_cv_participant_loso_pooled",
            "track": track,
            "seed": int(seed),
            "model": str(model_name),
            "feature_set": str(feature_set_name),
            "dataset_recordings": len(frame),
            "dataset_participants": dataset_participants,
            "dataset_healthy_recordings": int(class_rows.get(0, 0)),
            "dataset_pd_recordings": int(class_rows.get(1, 0)),
            "dataset_healthy_participants": int(participant_classes.get(0, 0)),
            "dataset_pd_participants": int(participant_classes.get(1, 0)),
            "split_strategy": "5-fold StratifiedGroupKFold" if track == "dne" else "LeaveOneGroupOut",
            "threshold": THRESHOLD,
            "score_type": "uncalibrated_model_score",
            **metrics,
        })

    if track == "dne":
        for (seed, model_name, feature_set_name, fold_number), subset in participant_frame.groupby(
            ["seed", "model", "feature_set", "outer_folds"], dropna=False
        ):
            y_true = subset["true_label"].astype(int).to_numpy()
            scores = subset["mean_pd_score_uncalibrated"].to_numpy(dtype=float)
            metrics = calculate_metrics_with_scores(y_true, scores, unit="participants")
            fold_metric_rows.append({
                "track": track,
                "seed": int(seed),
                "fold": int(str(fold_number).split(",")[0]),
                "model": str(model_name),
                "feature_set": str(feature_set_name),
                "n_participants": metrics["n_participants"],
                "accuracy": metrics["accuracy"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "brier_score": metrics["brier_score"],
                "collapse_flag": metrics["collapse_flag"],
            })

    return metric_rows, fold_metric_rows, recording_rows, selected_feature_rows, fit_metadata_rows


def recording_level_metrics(recording_rows: list[dict[str, Any]], level: str, track: str) -> list[dict[str, Any]]:
    recording_frame = pd.DataFrame(recording_rows)
    rows: list[dict[str, Any]] = []
    if recording_frame.empty:
        return rows
    for (seed, model_name, feature_set_name), subset in recording_frame.groupby(
        ["seed", "model", "feature_set"], dropna=False
    ):
        y_true = subset["true_label"].astype(int).to_numpy()
        scores = subset["pd_score_uncalibrated"].to_numpy(dtype=float)
        metrics = calculate_metrics_with_scores(y_true, scores, unit="recordings")
        rows.append({
            "evaluation_level": level,
            "track": track,
            "seed": int(seed),
            "model": str(model_name),
            "feature_set": str(feature_set_name),
            "threshold": THRESHOLD,
            "score_type": "uncalibrated_model_score",
            **metrics,
        })
    return rows


def contract_columns(frame: pd.DataFrame, pose_columns: list[str]) -> list[str]:
    columns = list(CONTRACT_METADATA_COLUMNS) + list(pose_columns)
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"External table is missing contract columns: {missing[:5]}")
    return columns


def fit_platt_calibrator(scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(_logit(scores), np.asarray(labels, dtype=int))
    return calibrator


def apply_platt(calibrator: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(_logit(scores))[:, 1]


def external_evaluation(
    dne: pd.DataFrame,
    external_tables: dict[str, pd.DataFrame],
    models: tuple[str, ...],
    tune_inner: bool,
    oof_recording_frame: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Train on all DNE participants, evaluate on external tables, calibrate on DNE OOF."""
    pose_columns, group_column, label_column = validate_dataset(dne, "dne")
    y_dne = dne[label_column].astype(int).to_numpy()
    groups_dne = dne[group_column].astype(str).to_numpy()
    rankings = {}
    selector_metadata = {}
    for selector in SELECTORS:
        ranked, selector_meta = fit_selector(selector, dne[pose_columns], y_dne, groups_dne, "dne")
        rankings[selector] = ranked
        selector_metadata[selector] = selector_meta

    feature_sets = make_feature_sets(rankings)
    feature_sets.append({
        "feature_set": "legacy_290_metadata",
        "selector": None,
        "requested_count": None,
        "columns": contract_columns(dne, pose_columns),
    })

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selected_feature_rows: list[dict[str, Any]] = []
    fit_metadata_rows: list[dict[str, Any]] = []
    saved_artifacts: dict[str, dict[str, Any]] = {}

    dne_oof = oof_recording_frame[oof_recording_frame["track"] == "dne"]

    for feature_set in feature_sets:
        feature_set_name = feature_set["feature_set"]
        columns = feature_set["columns"] if feature_set["columns"] is not None else pose_columns
        selected_feature_rows.append({
            "track": "dne",
            "seed": 42,
            "fold": None,
            "fit_scope": "all_dne_external_fit",
            "selector": feature_set["selector"],
            "requested_count": feature_set["requested_count"],
            "eligible_after_filter": len(rankings.get(feature_set["selector"], pose_columns) or pose_columns),
            "selected_count": len(columns),
            "participants_train": ";".join(sorted(set(groups_dne))),
            "participants_test": "",
            "features": ";".join(columns),
            "selector_metadata": json.dumps(selector_metadata.get(feature_set["selector"], {}), default=str),
        })
        for model_name in models:
            if not model_applies_to_feature_set(model_name, feature_set_name):
                continue
            estimator, fit_meta = fit_model(model_name, dne[columns], y_dne, groups_dne, 42, tune_inner)
            fit_metadata_rows.append({
                "track": "dne_external_fit",
                "seed": 42,
                "fold": None,
                "feature_set": feature_set_name,
                **fit_meta,
            })

            per_table_scores: dict[str, pd.DataFrame] = {}
            for table_name, table in external_tables.items():
                table_scores = positive_probability(estimator, table[columns])
                per_table_scores[table_name] = table_scores

            oof_subset = dne_oof[
                (dne_oof["model"] == model_name) & (dne_oof["feature_set"] == feature_set_name)
            ]
            calibrator = None
            if not oof_subset.empty:
                calibrator = fit_platt_calibrator(
                    oof_subset["pd_score_uncalibrated"].to_numpy(dtype=float),
                    oof_subset["true_label"].astype(int).to_numpy(),
                )

            for table_name, table in external_tables.items():
                scores = per_table_scores[table_name]
                calibrated = apply_platt(calibrator, scores) if calibrator is not None else np.full(len(scores), np.nan)
                group_column_external = "subject_id"
                for position, (_, row) in enumerate(table.iterrows()):
                    prediction_rows.append({
                        "table": table_name,
                        "cohort": row.get("cohort", "Clinical"),
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "selector": feature_set["selector"],
                        "requested_count": feature_set["requested_count"],
                        "feature_count": len(columns),
                        "subject_id": str(row[group_column_external]),
                        "recording_id": str(row["recording_id"]),
                        "true_label": int(row["verified_label"]),
                        "pd_score_uncalibrated": float(scores[position]),
                        "pd_score_platt_dne_oof": float(calibrated[position]) if calibrator is not None else None,
                        "predicted_label_uncalibrated": int(scores[position] >= THRESHOLD),
                        "predicted_label_platt": (
                            int(calibrated[position] >= THRESHOLD) if calibrator is not None else None
                        ),
                    })

                participant_scores = (
                    pd.DataFrame({
                        "subject_id": table[group_column_external].astype(str).to_numpy(),
                        "true_label": table["verified_label"].astype(int).to_numpy(),
                        "score": scores,
                        "score_platt": calibrated,
                    })
                    .groupby("subject_id", as_index=False)
                    .agg(
                        true_label=("true_label", "first"),
                        n_recordings=("score", "count"),
                        mean_pd_score_uncalibrated=("score", "mean"),
                        mean_pd_score_platt=("score_platt", "mean"),
                    )
                )
                y_true = participant_scores["true_label"].astype(int).to_numpy()
                metrics = calculate_metrics_with_scores(
                    y_true, participant_scores["mean_pd_score_uncalibrated"].to_numpy(dtype=float), unit="participants"
                )
                metric_rows.append({
                    "evaluation_level": f"external_{table_name}_participant",
                    "track": "tulip",
                    "seed": 42,
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "n_recordings": int(participant_scores["n_recordings"].sum()),
                    "threshold": THRESHOLD,
                    "score_type": "uncalibrated_model_score",
                    **metrics,
                })
                if calibrator is not None:
                    calibrated_metrics = calculate_metrics_with_scores(
                        y_true, participant_scores["mean_pd_score_platt"].to_numpy(dtype=float), unit="participants"
                    )
                    metric_rows.append({
                        "evaluation_level": f"external_{table_name}_participant",
                        "track": "tulip",
                        "seed": 42,
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "n_recordings": int(participant_scores["n_recordings"].sum()),
                        "threshold": THRESHOLD,
                        "score_type": "platt_calibrated_on_dne_oof",
                        **calibrated_metrics,
                    })
                rec_metrics = calculate_metrics_with_scores(
                    table["verified_label"].astype(int).to_numpy(), scores, unit="recordings"
                )
                metric_rows.append({
                    "evaluation_level": f"external_{table_name}_recording",
                    "track": "tulip",
                    "seed": 42,
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "threshold": THRESHOLD,
                    "score_type": "uncalibrated_model_score",
                    **rec_metrics,
                })

            saved_artifacts[f"{model_name}__{feature_set_name}"] = {
                "model": estimator,
                "feature_names": list(columns),
                "feature_set": feature_set_name,
                "model_name": model_name,
                "trained_on": "all DNE participants (322 recordings, 58 participants)",
                "fit_metadata": fit_meta,
                "calibrator_platt_dne_oof": calibrator,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

    return metric_rows, prediction_rows, selected_feature_rows, fit_metadata_rows, saved_artifacts


def production_model_evaluation(
    external_tables: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate the current production pd_detector.pkl on external tables."""
    if not PRODUCTION_MODEL_PATH.exists():
        raise FileNotFoundError(f"Production model not found: {PRODUCTION_MODEL_PATH}")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model = joblib.load(PRODUCTION_MODEL_PATH)
    model_sklearn = str(captured[0].message) if captured else ""
    feature_names = [str(name) for name in model.feature_names_in_]
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for table_name, table in external_tables.items():
        missing = [name for name in feature_names if name not in table.columns]
        if missing:
            raise ValueError(f"Production contract mismatch for {table_name}: missing {missing[:5]}")
        scores = positive_probability(model, table[feature_names])
        for position, (_, row) in enumerate(table.iterrows()):
            prediction_rows.append({
                "table": table_name,
                "cohort": row.get("cohort", "Clinical"),
                "model": "production_pd_detector",
                "feature_set": "production_290_contract",
                "selector": None,
                "requested_count": None,
                "feature_count": len(feature_names),
                "subject_id": str(row["subject_id"]),
                "recording_id": str(row["recording_id"]),
                "true_label": int(row["verified_label"]),
                "pd_score_uncalibrated": float(scores[position]),
                "pd_score_platt_dne_oof": None,
                "predicted_label_uncalibrated": int(scores[position] >= THRESHOLD),
                "predicted_label_platt": None,
            })
        participant_scores = (
            pd.DataFrame({
                "subject_id": table["subject_id"].astype(str).to_numpy(),
                "true_label": table["verified_label"].astype(int).to_numpy(),
                "score": scores,
            })
            .groupby("subject_id", as_index=False)
            .agg(
                true_label=("true_label", "first"),
                n_recordings=("score", "count"),
                mean_pd_score_uncalibrated=("score", "mean"),
            )
        )
        y_true = participant_scores["true_label"].astype(int).to_numpy()
        metrics = calculate_metrics_with_scores(
            y_true, participant_scores["mean_pd_score_uncalibrated"].to_numpy(dtype=float), unit="participants"
        )
        metric_rows.append({
            "evaluation_level": f"external_{table_name}_participant",
            "track": "tulip",
            "seed": None,
            "model": "production_pd_detector",
            "feature_set": "production_290_contract",
            "n_recordings": int(participant_scores["n_recordings"].sum()),
            "threshold": THRESHOLD,
            "score_type": "uncalibrated_model_score",
            **metrics,
        })
        rec_metrics = calculate_metrics_with_scores(
            table["verified_label"].astype(int).to_numpy(), scores, unit="recordings"
        )
        metric_rows.append({
            "evaluation_level": f"external_{table_name}_recording",
            "track": "tulip",
            "seed": None,
            "model": "production_pd_detector",
            "feature_set": "production_290_contract",
            "threshold": THRESHOLD,
            "score_type": "uncalibrated_model_score",
            **rec_metrics,
        })
    info = {
        "path": str(PRODUCTION_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(PRODUCTION_MODEL_PATH),
        "type": type(model).__name__,
        "n_features": len(feature_names),
        "feature_names_first": feature_names[:3],
        "feature_names_last": feature_names[-3:],
        "sklearn_pickle_warning": model_sklearn,
    }
    return metric_rows, prediction_rows, info


def domain_shift_diagnostics(
    dne: pd.DataFrame, tulip: pd.DataFrame, pose_columns: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-feature distribution shift plus a label-free domain classifier."""
    from scipy.stats import ks_2samp

    shift_rows = []
    for column in pose_columns:
        reference = dne[column].to_numpy(dtype=float)
        external = tulip[column].to_numpy(dtype=float)
        pooled_sd = float(np.sqrt((reference.var(ddof=1) + external.var(ddof=1)) / 2.0))
        smd = float((external.mean() - reference.mean()) / pooled_sd) if pooled_sd > 0 else np.nan
        ks_statistic, ks_pvalue = ks_2samp(external, reference)
        shift_rows.append({
            "feature": column,
            "dne_mean": float(reference.mean()),
            "dne_sd": float(reference.std(ddof=1)),
            "tulip_mean": float(external.mean()),
            "tulip_sd": float(external.std(ddof=1)),
            "standardized_mean_difference": smd,
            "absolute_smd": abs(smd) if np.isfinite(smd) else np.nan,
            "ks_statistic": float(ks_statistic),
            "ks_pvalue": float(ks_pvalue),
        })
    shift_frame = pd.DataFrame(shift_rows).sort_values("absolute_smd", ascending=False)

    combined = pd.concat(
        [
            dne[pose_columns].assign(_domain=0, _group="dne::" + dne["subject"].astype(str)),
            tulip[pose_columns].assign(_domain=1, _group="tulip::" + tulip["subject_id"].astype(str)),
        ],
        ignore_index=True,
    )
    y_domain = combined["_domain"].astype(int).to_numpy()
    groups = combined["_group"].astype(str).to_numpy()
    X_domain = combined[pose_columns]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_scores = np.zeros(len(combined))
    for train_indices, test_indices in splitter.split(X_domain, y_domain, groups):
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(penalty="l2", C=1.0, max_iter=5000, random_state=42)),
        ])
        if len(np.unique(y_domain[train_indices])) < 2:
            oof_scores[test_indices] = float(y_domain[train_indices][0])
            continue
        pipeline.fit(X_domain.iloc[train_indices], y_domain[train_indices])
        oof_scores[test_indices] = pipeline.predict_proba(X_domain.iloc[test_indices])[:, 1]
    from sklearn.metrics import roc_auc_score

    domain_metrics = {
        "domain_classifier": "StandardScaler + L2 LogisticRegression(C=1.0)",
        "split": "5-fold StratifiedGroupKFold over combined subject groups",
        "roc_auc_dne_vs_tulip": float(roc_auc_score(y_domain, oof_scores)),
        "accuracy_dne_vs_tulip": float(((oof_scores >= 0.5).astype(int) == y_domain).mean()),
        "n_dne_recordings": int((y_domain == 0).sum()),
        "n_tulip_recordings": int((y_domain == 1).sum()),
        "note": "Uses dataset identity only; no PD labels involved.",
    }
    return shift_frame, domain_metrics


def reproduction_check(metric_frame: pd.DataFrame) -> pd.DataFrame:
    """Compare this run's RF arms against run 4 for the same folds and seeds."""
    if not RUN4_METRICS_PATH.exists():
        return pd.DataFrame()
    run4 = pd.read_csv(RUN4_METRICS_PATH)
    mine = metric_frame[
        (metric_frame["model"] == "rf_baseline")
        & (metric_frame["evaluation_level"].isin(["dne_cv_participant_oof_pooled", "tulip_cv_participant_loso_pooled"]))
        & (metric_frame["score_type"] == "uncalibrated_model_score")
    ].copy()
    mine["requested_count"] = mine["feature_set"].str.extract(r"_top(\d+)$").astype(float)
    mine["selector_lookup"] = mine["feature_set"].str.replace(r"_top\d+$", "", regex=True)
    mine.loc[mine["feature_set"] == "full_288", "selector_lookup"] = "full_288"
    mine.loc[mine["feature_set"] == "legacy_290_metadata", "selector_lookup"] = "legacy_290_metadata"
    mine["requested_count"] = mine["requested_count"].fillna(-1.0)

    run4_compare = run4.rename(columns={"method": "selector_lookup"})[
        [
            "track", "seed", "selector_lookup", "requested_count",
            "accuracy", "sensitivity", "specificity", "f1", "roc_auc", "tn", "fp", "fn", "tp",
        ]
    ].copy()
    run4_compare["requested_count"] = run4_compare["requested_count"].fillna(-1.0)
    merged = mine.merge(
        run4_compare,
        on=["track", "seed", "selector_lookup", "requested_count"],
        how="inner",
        suffixes=("_run5", "_run4"),
    )
    if merged.empty:
        return pd.DataFrame()
    reproduced = pd.DataFrame({
        "track": merged["track"],
        "seed": merged["seed"].astype(int),
        "selector": merged["selector_lookup"],
        "requested_count": merged["requested_count"],
        "accuracy_run4": merged["accuracy_run4"],
        "accuracy_run5": merged["accuracy_run5"],
        "accuracy_delta": merged["accuracy_run5"] - merged["accuracy_run4"],
        "sensitivity_delta": merged["sensitivity_run5"] - merged["sensitivity_run4"],
        "specificity_delta": merged["specificity_run5"] - merged["specificity_run4"],
        "roc_auc_delta": merged["roc_auc_run5"] - merged["roc_auc_run4"],
        "reproduced_exactly": (
            np.isclose(merged["accuracy_run5"], merged["accuracy_run4"], atol=1e-9)
            & np.isclose(merged["specificity_run5"], merged["specificity_run4"], atol=1e-9)
            & np.isclose(merged["roc_auc_run5"], merged["roc_auc_run4"], atol=1e-9, equal_nan=True)
        ),
    })
    return reproduced


def selection_stability(selected_feature_rows: list[dict[str, Any]], track: str, selector: str, requested_count: int) -> pd.DataFrame:
    counts: dict[str, int] = {}
    folds = 0
    for row in selected_feature_rows:
        if row["track"] != track or row["selector"] != selector:
            continue
        if row["requested_count"] != requested_count:
            continue
        if row["fit_scope"] != "outer_train_fold":
            continue
        folds += 1
        for feature in str(row["features"]).split(";"):
            counts[feature] = counts.get(feature, 0) + 1
    if folds == 0:
        return pd.DataFrame()
    frame = pd.DataFrame({"feature": list(counts), "times_selected": list(counts.values())})
    frame["selection_frequency"] = frame["times_selected"] / folds
    frame["track"] = track
    frame["selector"] = selector
    frame["requested_count"] = requested_count
    return frame.sort_values(["selection_frequency", "feature"], ascending=[False, True]).reset_index(drop=True)


def build_summary_frames(
    metric_frame: pd.DataFrame,
    fold_metric_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assemble the baseline-vs-experiment summary, DNE CV summary, and TULIP external summary."""
    dne_pooled = metric_frame[metric_frame["evaluation_level"] == "dne_cv_participant_oof_pooled"]
    dne_fold = fold_metric_frame
    external_participant = metric_frame[
        metric_frame["evaluation_level"].isin([
            "external_tulip_clinical_participant",
            "external_tulip_continuity44_participant",
            "external_tulip_youtube_unverified_participant",
        ])
    ]
    external_uncalibrated = external_participant[external_participant["score_type"] == "uncalibrated_model_score"]
    external_calibrated = external_participant[external_participant["score_type"] == "platt_calibrated_on_dne_oof"]

    def feature_count_for(feature_set_name: str) -> int | None:
        if feature_set_name == "full_288":
            return 288
        if feature_set_name == "legacy_290_metadata":
            return 290
        match = re.search(r"_top(\d+)$", feature_set_name)
        return int(match.group(1)) if match else None

    comparison_rows = []
    configs = metric_frame[metric_frame["evaluation_level"] == "dne_cv_participant_oof_pooled"][
        ["model", "feature_set"]
    ].drop_duplicates()
    for _, config in configs.iterrows():
        model_name, feature_set_name = config["model"], config["feature_set"]
        pooled = dne_pooled[(dne_pooled["model"] == model_name) & (dne_pooled["feature_set"] == feature_set_name)]
        folds = dne_fold[(dne_fold["model"] == model_name) & (dne_fold["feature_set"] == feature_set_name)]
        fold_seed42 = folds[folds["seed"] == 42]
        external = external_uncalibrated[
            (external_uncalibrated["model"] == model_name)
            & (external_uncalibrated["feature_set"] == feature_set_name)
            & (external_uncalibrated["evaluation_level"] == "external_tulip_clinical_participant")
        ]
        external_cal = external_calibrated[
            (external_calibrated["model"] == model_name)
            & (external_calibrated["feature_set"] == feature_set_name)
            & (external_calibrated["evaluation_level"] == "external_tulip_clinical_participant")
        ]
        row: dict[str, Any] = {
            "model": model_name,
            "feature_set": feature_set_name,
            "selector": None if feature_set_name in ("full_288", "legacy_290_metadata") else feature_set_name.split("_top")[0],
            "k": None if feature_set_name in ("full_288", "legacy_290_metadata") else int(feature_set_name.split("_top")[1]),
            "n_features": feature_count_for(feature_set_name),
            "dne_pooled_acc_seed42": _value(pooled, 42, "accuracy"),
            "dne_pooled_auc_seed42": _value(pooled, 42, "roc_auc"),
            "dne_pooled_acc_mean_3seeds": float(pooled["accuracy"].mean()) if not pooled.empty else np.nan,
            "dne_pooled_acc_sd_3seeds": float(pooled["accuracy"].std(ddof=1)) if len(pooled) > 1 else np.nan,
            "dne_pooled_auc_mean_3seeds": float(pooled["roc_auc"].mean()) if not pooled.empty else np.nan,
            "dne_pooled_auc_sd_3seeds": float(pooled["roc_auc"].std(ddof=1)) if len(pooled) > 1 else np.nan,
            "dne_fold_acc_mean_seed42": float(fold_seed42["accuracy"].mean()) if not fold_seed42.empty else np.nan,
            "dne_fold_acc_sd_seed42": float(fold_seed42["accuracy"].std(ddof=1)) if not fold_seed42.empty else np.nan,
            "dne_fold_auc_mean_seed42": float(fold_seed42["roc_auc"].mean()) if not fold_seed42.empty else np.nan,
            "dne_fold_auc_sd_seed42": float(fold_seed42["roc_auc"].std(ddof=1)) if not fold_seed42.empty else np.nan,
            "tulip_ext_acc": float(external["accuracy"].iloc[0]) if not external.empty else np.nan,
            "tulip_ext_sensitivity": float(external["sensitivity"].iloc[0]) if not external.empty else np.nan,
            "tulip_ext_specificity": float(external["specificity"].iloc[0]) if not external.empty else np.nan,
            "tulip_ext_f1": float(external["f1"].iloc[0]) if not external.empty else np.nan,
            "tulip_ext_auc": float(external["roc_auc"].iloc[0]) if not external.empty and pd.notna(external["roc_auc"].iloc[0]) else np.nan,
            "tulip_ext_collapse": str(external["collapse_flag"].iloc[0]) if not external.empty else "",
            "tulip_ext_acc_platt": float(external_cal["accuracy"].iloc[0]) if not external_cal.empty else np.nan,
            "tulip_ext_auc_platt": float(external_cal["roc_auc"].iloc[0]) if not external_cal.empty and pd.notna(external_cal["roc_auc"].iloc[0]) else np.nan,
            "notes": _config_note(model_name, feature_set_name),
        }
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)

    baseline_mask = (comparison["model"] == "rf_baseline") & (comparison["feature_set"] == "full_288")
    if baseline_mask.any():
        baseline_acc = float(comparison.loc[baseline_mask, "dne_fold_acc_mean_seed42"].iloc[0])
        baseline_auc = float(comparison.loc[baseline_mask, "dne_fold_auc_mean_seed42"].iloc[0])
        baseline_ext_acc = float(comparison.loc[baseline_mask, "tulip_ext_acc"].iloc[0])
        comparison["delta_dne_fold_acc_vs_rf_full"] = comparison["dne_fold_acc_mean_seed42"] - baseline_acc
        comparison["delta_dne_fold_auc_vs_rf_full"] = comparison["dne_fold_auc_mean_seed42"] - baseline_auc
        comparison["delta_tulip_ext_acc_vs_rf_full"] = comparison["tulip_ext_acc"] - baseline_ext_acc

    production_external = metric_frame[
        (metric_frame["model"] == "production_pd_detector")
        & (metric_frame["evaluation_level"] == "external_tulip_clinical_participant")
    ]
    production_row = {
        "model": "production_pd_detector",
        "feature_set": "production_290_contract",
        "selector": None,
        "k": None,
        "n_features": 290,
        "dne_pooled_acc_seed42": np.nan,
        "dne_pooled_auc_seed42": np.nan,
        "tulip_ext_acc": float(production_external["accuracy"].iloc[0]) if not production_external.empty else np.nan,
        "tulip_ext_sensitivity": float(production_external["sensitivity"].iloc[0]) if not production_external.empty else np.nan,
        "tulip_ext_specificity": float(production_external["specificity"].iloc[0]) if not production_external.empty else np.nan,
        "tulip_ext_f1": float(production_external["f1"].iloc[0]) if not production_external.empty else np.nan,
        "tulip_ext_auc": float(production_external["roc_auc"].iloc[0]) if not production_external.empty and pd.notna(production_external["roc_auc"].iloc[0]) else np.nan,
        "tulip_ext_collapse": str(production_external["collapse_flag"].iloc[0]) if not production_external.empty else "",
        "notes": "Current production artifact (not retrained); evaluated on verified clinical candidate features",
    }
    production_frame = pd.DataFrame([production_row]).reindex(columns=comparison.columns)
    with warnings.catch_warnings():
        # pandas deprecation about all-NA rows in concat; dtype outcome is
        # irrelevant because the frame is written straight to CSV.
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*concatenation with empty or all-NA.*")
        comparison = pd.concat([comparison, production_frame], ignore_index=True)

    dne_summary = (
        dne_fold.groupby(["model", "feature_set", "seed"], as_index=False)
        .agg(
            n_folds=("fold", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            sensitivity_mean=("sensitivity", "mean"),
            sensitivity_sd=("sensitivity", "std"),
            specificity_mean=("specificity", "mean"),
            specificity_sd=("specificity", "std"),
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_sd=("roc_auc", "std"),
            brier_mean=("brier_score", "mean"),
        )
    )
    tulip_external_summary = external_participant.copy()
    return comparison, dne_summary, tulip_external_summary


def _value(frame: pd.DataFrame, seed: int, column: str) -> float:
    subset = frame[frame["seed"] == seed]
    if subset.empty:
        return np.nan
    return float(subset[column].iloc[0])


def _config_note(model_name: str, feature_set_name: str) -> str:
    notes = []
    if model_name == "rf_baseline":
        notes.append("existing production-family RF")
    elif model_name == "rf_regularized":
        notes.append("existing repo RF_Regularized config, untuned")
    elif model_name == "svm_rbf":
        notes.append("inner grouped-CV tuned C/gamma; scaler fitted per fold")
    elif model_name == "logreg_l2":
        notes.append("inner grouped-CV tuned C; scaler fitted per fold")
    if feature_set_name == "legacy_290_metadata":
        notes.append("includes fps/num_frames metadata (audit Finding A)")
    return "; ".join(notes)


def make_plots(
    output_dir: Path,
    metric_frame: pd.DataFrame,
    fold_metric_frame: pd.DataFrame,
    participant_frame: pd.DataFrame,
    external_prediction_frame: pd.DataFrame,
    shift_frame: pd.DataFrame,
    stability_frame: pd.DataFrame,
    models: tuple[str, ...],
) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    color_map = {"rf_baseline": "#1f77b4", "rf_regularized": "#2ca02c", "svm_rbf": "#d62728", "logreg_l2": "#9467bd"}

    def save(fig, name: str) -> None:
        path = figures_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        created.append(str(path.relative_to(output_dir)))

    dne_pooled = metric_frame[
        (metric_frame["evaluation_level"] == "dne_cv_participant_oof_pooled") & (metric_frame["seed"] == 42)
    ]
    if not dne_pooled.empty:
        fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.6))
        if len(models) == 1:
            axes = [axes]
        for axis, model_name in zip(axes, models):
            row = dne_pooled[(dne_pooled["model"] == model_name) & (dne_pooled["feature_set"] == "full_288")]
            if row.empty:
                axis.axis("off")
                continue
            row = row.iloc[0]
            matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=float)
            axis.imshow(matrix, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    axis.text(j, i, int(matrix[i, j]), ha="center", va="center", color="#111111")
            axis.set_xticks([0, 1], labels=["Pred HT", "Pred PD"])
            axis.set_yticks([0, 1], labels=["True HT", "True PD"])
            axis.set_title(f"{model_name}\nDNE OOF participants", fontsize=9)
        fig.suptitle("DNE 5-fold grouped CV — pooled participant-level confusion (full-288, seed 42)", fontsize=11)
        fig.tight_layout()
        save(fig, "fig_confusion_dne_full288.png")

    external_clinical = metric_frame[
        (metric_frame["evaluation_level"] == "external_tulip_clinical_participant")
        & (metric_frame["score_type"] == "uncalibrated_model_score")
    ]
    if not external_clinical.empty:
        fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.6))
        if len(models) == 1:
            axes = [axes]
        for axis, model_name in zip(axes, models):
            row = external_clinical[
                (external_clinical["model"] == model_name) & (external_clinical["feature_set"] == "full_288")
            ]
            if row.empty:
                axis.axis("off")
                continue
            row = row.iloc[0]
            matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=float)
            axis.imshow(matrix, cmap="Oranges")
            for i in range(2):
                for j in range(2):
                    axis.text(j, i, int(matrix[i, j]), ha="center", va="center", color="#111111")
            axis.set_xticks([0, 1], labels=["Pred HT", "Pred PD"])
            axis.set_yticks([0, 1], labels=["True HT", "True PD"])
            axis.set_title(f"{model_name}\nTULIP external participants", fontsize=9)
        fig.suptitle("External TULIP clinical — participant-level confusion (DNE-trained, full-288)", fontsize=11)
        fig.tight_layout()
        save(fig, "fig_confusion_tulip_external.png")

    dne_metrics = participant_frame[participant_frame["track"] == "dne"]
    if not dne_metrics.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fold_curves = fold_metric_frame.groupby(["model", "feature_set"], as_index=False).agg(
            auc_mean=("roc_auc", "mean"), auc_sd=("roc_auc", "std"), acc_mean=("accuracy", "mean"), acc_sd=("accuracy", "std")
        )
        k_order = {"full_288": 288, "legacy_290_metadata": 290, "corr_rf_importance_top10": 10, "corr_rf_importance_top20": 20, "corr_rf_importance_top30": 30, "corr_rf_importance_top50": 50, "l1_logistic_top10": 10, "l1_logistic_top20": 20, "l1_logistic_top30": 30, "l1_logistic_top50": 50}
        for model_name in models:
            subset = fold_curves[
                (fold_curves["model"] == model_name)
                & fold_curves["feature_set"].str.startswith(("corr_rf_importance", "full_288"))
            ].copy()
            if subset.empty:
                continue
            subset["k"] = subset["feature_set"].map(k_order)
            subset = subset.sort_values("k")
            axes[0].errorbar(subset["k"], subset["acc_mean"], yerr=subset["acc_sd"], marker="o", capsize=3, label=model_name, color=color_map.get(model_name))
            axes[1].errorbar(subset["k"], subset["auc_mean"], yerr=subset["auc_sd"], marker="o", capsize=3, label=model_name, color=color_map.get(model_name))
        for axis, label in zip(axes, ["Accuracy", "ROC-AUC"]):
            axis.set_xlabel("number of pose features (corr+RF ranking; 288 = full)")
            axis.set_ylabel(f"{label} (mean +/- fold SD)")
            axis.grid(alpha=0.3)
        axes[0].legend(fontsize=7, loc="lower right")
        fig.suptitle("DNE grouped CV — feature-count curves (all seeds, fold mean +/- SD, participant level)", fontsize=11)
        fig.tight_layout()
        save(fig, "fig_feature_count_curves.png")

    if not external_prediction_frame.empty:
        clinical_points = external_prediction_frame[external_prediction_frame["table"] == "tulip_clinical"].copy()
        if not clinical_points.empty:
            participant_points = (
                clinical_points.groupby(["model", "feature_set", "subject_id"], as_index=False)
                .agg(mean_score=("pd_score_uncalibrated", "mean"), true_label=("true_label", "first"))
            )
            full_points = participant_points[participant_points["feature_set"] == "full_288"]
            available_models = [m for m in models if m in set(full_points["model"])]
            if available_models:
                fig, axes = plt.subplots(1, len(available_models), figsize=(3.4 * len(available_models), 3.6), sharey=True)
                if len(available_models) == 1:
                    axes = [axes]
                rng = np.random.default_rng(0)
                for axis, model_name in zip(axes, available_models):
                    subset = full_points[full_points["model"] == model_name]
                    for label, color, offset in ((0, "#2ca02c", -0.12), (1, "#d62728", 0.12)):
                        points = subset[subset["true_label"] == label]
                        jitter = rng.uniform(-0.06, 0.06, size=len(points)) + offset
                        axis.scatter(points["mean_score"], jitter, color=color, s=36, label=f"true {'PD' if label else 'HT'}")
                    axis.axvline(THRESHOLD, color="#444444", linestyle="--", linewidth=1)
                    axis.set_title(model_name, fontsize=9)
                    axis.set_xlim(0, 1)
                    axis.set_xlabel("mean P(PD) (participant)")
                axes[0].set_ylabel("participants (jittered by class)")
                axes[-1].legend(fontsize=7, loc="center right")
                fig.suptitle("External TULIP clinical — participant scores per model (full-288, uncalibrated; dashed = 0.5)", fontsize=11)
                fig.tight_layout()
                save(fig, "fig_tulip_probability_distributions.png")

    reliability_rows = metric_frame[
        (metric_frame["evaluation_level"] == "dne_cv_participant_oof_pooled")
        & (metric_frame["model"] == "rf_baseline")
        & (metric_frame["feature_set"] == "full_288")
        & (metric_frame["seed"] == 42)
    ]
    dne_participants = participant_frame[
        (participant_frame["track"] == "dne")
        & (participant_frame["model"] == "rf_baseline")
        & (participant_frame["feature_set"] == "full_288")
        & (participant_frame["seed"] == 42)
    ]
    if not dne_participants.empty:
        bins = np.linspace(0.0, 1.0, 11)
        bin_ids = np.clip(np.digitize(dne_participants["mean_pd_score_uncalibrated"], bins) - 1, 0, 9)
        working = dne_participants.assign(bin=bin_ids)
        grouped = working.groupby("bin").agg(
            mean_score=("mean_pd_score_uncalibrated", "mean"),
            fraction_pd=("true_label", "mean"),
            count=("true_label", "count"),
        )
        fig, axis = plt.subplots(figsize=(4.6, 4.4))
        axis.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
        axis.scatter(grouped["mean_score"], grouped["fraction_pd"], s=18 + grouped["count"] * 6, color="#1f77b4")
        for _, row in grouped.iterrows():
            axis.annotate(int(row["count"]), (row["mean_score"], row["fraction_pd"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        axis.set_xlabel("mean predicted P(PD) (DNE OOF participants)")
        axis.set_ylabel("observed fraction PD")
        axis.set_title("DNE OOF reliability — rf_baseline full-288 (marker size = participants)", fontsize=10)
        axis.grid(alpha=0.3)
        if not reliability_rows.empty:
            axis.text(0.02, 0.95, f"Brier = {reliability_rows['brier_score'].iloc[0]:.3f}", transform=axis.transAxes, fontsize=9)
        fig.tight_layout()
        save(fig, "fig_reliability_dne_oof.png")

    if not shift_frame.empty:
        top = shift_frame.head(25).iloc[::-1]
        fig, axis = plt.subplots(figsize=(6.4, 7))
        axis.barh(top["feature"], top["standardized_mean_difference"], color="#8c564b")
        axis.axvline(0, color="#333333", linewidth=0.8)
        axis.set_xlabel("standardized mean difference (TULIP - DNE)")
        axis.set_title("Largest per-feature distribution shifts (288 pose stats)", fontsize=10)
        axis.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        save(fig, "fig_domain_shift_top_features.png")

    if not stability_frame.empty:
        top = stability_frame.head(20).iloc[::-1]
        fig, axis = plt.subplots(figsize=(6.4, 6))
        axis.barh(top["feature"], top["selection_frequency"], color="#17becf")
        axis.set_xlabel("selection frequency across outer folds")
        axis.set_title(
            f"Selection stability — {stability_frame['track'].iloc[0]}, {stability_frame['selector'].iloc[0]}, "
            f"top-{int(stability_frame['requested_count'].iloc[0])}",
            fontsize=10,
        )
        axis.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        save(fig, "fig_selection_stability.png")

    if not fold_metric_frame.empty:
        full_folds = fold_metric_frame[fold_metric_frame["feature_set"] == "full_288"]
        if not full_folds.empty:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for axis, column, label in ((axes[0], "accuracy", "accuracy"), (axes[1], "roc_auc", "ROC-AUC")):
                data = [full_folds[full_folds["model"] == model_name][column].dropna().to_numpy() for model_name in models]
                data = [values for values in data if len(values)]
                labels = [model_name for model_name in models if len(full_folds[full_folds["model"] == model_name])]
                if data:
                    axis.boxplot(data, tick_labels=labels, showmeans=True)
                    axis.tick_params(axis="x", rotation=20, labelsize=8)
                    axis.set_ylabel(f"fold {label}")
                    axis.grid(alpha=0.3, axis="y")
            fig.suptitle("DNE grouped CV — per-fold metrics at full-288 (all seeds)", fontsize=11)
            fig.tight_layout()
            save(fig, "fig_fold_metrics_dne.png")

    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--quick", action="store_true", help="Reduced smoke-run: one DNE seed, two models")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    models_dir = args.models_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 4 output directory: {output_dir}")
    if models_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing experimental models directory: {models_dir}")

    models = ("rf_baseline", "svm_rbf") if args.quick else ALL_MODELS
    dne_seeds = (42,) if args.quick else DNE_SEEDS

    print("Loading and validating datasets...")
    dne = pd.read_csv(DNE_PATH)
    tulip_clinical = pd.read_csv(TULIP_CLINICAL_PATH)
    tulip_continuity = pd.read_csv(TULIP_CONTINUITY_PATH)
    tulip_youtube = pd.read_csv(TULIP_YOUTUBE_PATH)
    pose_columns, _, _ = validate_dataset(dne, "dne")
    validate_dataset(tulip_clinical, "tulip")
    validate_dataset(tulip_continuity, "tulip")
    youtube_frame = tulip_youtube.copy()
    youtube_frame["cohort"] = "YouTube"
    validate_dataset(youtube_frame, "tulip")
    if set(tulip_clinical["recording_id"]) & set(tulip_youtube["recording_id"]):
        raise ValueError("Unexpected recording overlap between clinical candidate and YouTube tables")
    if not set(tulip_clinical["recording_id"]).issubset(set(tulip_continuity["recording_id"])):
        raise ValueError("Clinical candidate rows are not a subset of the 44-row continuity table")

    output_dir.mkdir(parents=True, exist_ok=False)

    print("Stage A — participant-grouped cross-validation (DNE + TULIP)...")
    dne_metrics, dne_fold_metrics, dne_recordings, dne_selected, dne_fit_meta = run_cross_validation_track(
        "dne", dne, models, dne_seeds, tune_inner=not args.quick
    )
    tulip_metrics, tulip_fold_metrics, tulip_recordings, tulip_selected, tulip_fit_meta = run_cross_validation_track(
        "tulip", tulip_clinical, models, TULIP_SEEDS, tune_inner=not args.quick
    )
    recording_metric_rows = recording_level_metrics(dne_recordings, "dne_cv_recording_oof_pooled", "dne") + recording_level_metrics(
        tulip_recordings, "tulip_cv_recording_loso_pooled", "tulip"
    )

    recording_frame = pd.DataFrame(dne_recordings + tulip_recordings)
    participant_frame = aggregate_participant_predictions(recording_frame)

    print("Stage B — external TULIP evaluation (DNE-only training)...")
    external_tables = {
        "tulip_clinical": tulip_clinical,
        "tulip_continuity44": tulip_continuity,
        "tulip_youtube_unverified": youtube_frame,
    }
    external_metrics, external_predictions, external_selected, external_fit_meta, artifacts = external_evaluation(
        dne, external_tables, models, tune_inner=not args.quick, oof_recording_frame=recording_frame
    )
    production_metrics, production_predictions, production_info = production_model_evaluation(external_tables)

    print("Stage C — diagnostics...")
    shift_frame, domain_metrics = domain_shift_diagnostics(dne, tulip_clinical, pose_columns)
    metadata_rows = []
    for name, frame in (("dne", dne), ("tulip_clinical", tulip_clinical), ("tulip_youtube_unverified", youtube_frame)):
        metadata_rows.append({
            "dataset": name,
            "recordings": len(frame),
            "fps_min": float(frame["fps"].min()),
            "fps_median": float(frame["fps"].median()),
            "fps_max": float(frame["fps"].max()),
            "num_frames_min": float(frame["num_frames"].min()),
            "num_frames_median": float(frame["num_frames"].median()),
            "num_frames_max": float(frame["num_frames"].max()),
        })
    metadata_frame = pd.DataFrame(metadata_rows)
    production_importance_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        production_model = joblib.load(PRODUCTION_MODEL_PATH)
    if hasattr(production_model, "feature_importances_"):
        importances = pd.Series(production_model.feature_importances_, index=production_model.feature_names_in_)
        ranked = importances.sort_values(ascending=False)
        for name in ("fps", "num_frames"):
            if name in ranked.index:
                production_importance_rows.append({
                    "feature": name,
                    "importance": float(ranked[name]),
                    "rank_of_290": int(list(ranked.index).index(name)) + 1,
                })

    print("Stage D — baseline reproduction check vs run 4...")
    all_metric_rows = dne_metrics + tulip_metrics + recording_metric_rows + external_metrics + production_metrics
    metric_frame = pd.DataFrame(all_metric_rows)
    reproduction = reproduction_check(metric_frame)

    stability = selection_stability(dne_selected, "dne", PRIMARY_SELECTOR, 30)

    print("Stage E — summary tables, models, and plots...")
    comparison, dne_summary, tulip_external_summary = build_summary_frames(metric_frame, pd.DataFrame(dne_fold_metrics))

    selected_rows = []
    for row in dne_selected + tulip_selected:
        for rank, feature in enumerate(str(row["features"]).split(";"), start=1):
            selected_rows.append({
                "track": row["track"],
                "seed": row["seed"],
                "fold": row["fold"],
                "fit_scope": row["fit_scope"],
                "selector": row["selector"],
                "requested_count": row["requested_count"],
                "feature_count": row["selected_count"],
                "rank": rank,
                "feature": feature,
            })
    selected_frame = pd.DataFrame(selected_rows)

    metric_frame.to_csv(output_dir / "phase4_model_metrics.csv", index=False)
    pd.DataFrame(dne_fold_metrics).to_csv(output_dir / "phase4_fold_metrics.csv", index=False)
    recording_frame.to_csv(output_dir / "phase4_cv_recording_predictions.csv", index=False)
    participant_frame.to_csv(output_dir / "phase4_cv_participant_predictions.csv", index=False)
    external_prediction_frame = pd.DataFrame(external_predictions + production_predictions)
    external_prediction_frame.to_csv(output_dir / "phase4_external_predictions.csv", index=False)
    selected_frame.to_csv(output_dir / "phase4_selected_features.csv", index=False)
    stability.to_csv(output_dir / "phase4_selection_stability.csv", index=False)
    comparison.to_csv(output_dir / "phase4_baseline_vs_experiments.csv", index=False)
    dne_summary.to_csv(output_dir / "phase4_dne_cv_summary.csv", index=False)
    tulip_external_summary.to_csv(output_dir / "phase4_tulip_external_summary.csv", index=False)
    shift_frame.to_csv(output_dir / "phase4_domain_shift_summary.csv", index=False)
    metadata_frame.to_csv(output_dir / "phase4_metadata_comparison.csv", index=False)
    if not reproduction.empty:
        reproduction.to_csv(output_dir / "phase4_reproduction_check.csv", index=False)
    pd.DataFrame(dne_fit_meta + tulip_fit_meta + external_fit_meta).to_csv(
        output_dir / "phase4_fit_metadata.csv", index=False
    )
    pd.DataFrame(production_importance_rows).to_csv(
        output_dir / "phase4_production_metadata_importance.csv", index=False
    )

    figure_paths = make_plots(
        output_dir,
        metric_frame,
        pd.DataFrame(dne_fold_metrics),
        participant_frame,
        external_prediction_frame,
        shift_frame,
        stability,
        models,
    )

    models_dir.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for artifact_name, artifact in artifacts.items():
        model_path = models_dir / f"{artifact_name}.pkl"
        joblib.dump(artifact, model_path)
        manifest[artifact_name] = {
            "path": str(model_path.relative_to(PROJECT_ROOT)),
            "feature_names": artifact["feature_names"][:5] + ["..."],
            "n_features": len(artifact["feature_names"]),
            "trained_on": artifact["trained_on"],
            "notes": "Experimental artifact only. Not a production model.",
        }
    (models_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    run_config = {
        "scope": "Exploratory DNE-grouped CV, TULIP LOSO, and DNE-only-trained external TULIP evaluation; no combined training.",
        "dne_path": str(DNE_PATH.relative_to(PROJECT_ROOT)),
        "dne_sha256": sha256_file(DNE_PATH),
        "tulip_clinical_path": str(TULIP_CLINICAL_PATH.relative_to(PROJECT_ROOT)),
        "tulip_clinical_sha256": sha256_file(TULIP_CLINICAL_PATH),
        "tulip_continuity_path": str(TULIP_CONTINUITY_PATH.relative_to(PROJECT_ROOT)),
        "tulip_continuity_sha256": sha256_file(TULIP_CONTINUITY_PATH),
        "tulip_youtube_path": str(TULIP_YOUTUBE_PATH.relative_to(PROJECT_ROOT)),
        "tulip_youtube_sha256": sha256_file(TULIP_YOUTUBE_PATH),
        "run4_metrics_path": str(RUN4_METRICS_PATH.relative_to(PROJECT_ROOT)) if RUN4_METRICS_PATH.exists() else None,
        "seeds": {"dne_outer": list(dne_seeds), "tulip_outer": list(TULIP_SEEDS), "model_random_state": 42, "selector_random_state": 42},
        "feature_counts": list(K_COUNTS),
        "selectors": {
            "corr_rf_importance": "training-fold constant removal + Spearman |rho|>=0.95 filter, then RF impurity importance top-k",
            "l1_logistic": "training-fold constant removal + StandardScaler + L1 logistic ranking top-k (RF baseline only)",
        },
        "models": {
            "rf_baseline": "RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)",
            "rf_regularized": "RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_depth=15, class_weight='balanced', random_state=42)",
            "svm_rbf": f"Pipeline(StandardScaler, SVC(rbf, probability=True)); GridSearchCV({SVM_GRID}) in inner StratifiedGroupKFold",
            "logreg_l2": f"Pipeline(StandardScaler, LogisticRegression(l2)); GridSearchCV({LOGREG_GRID}) in inner StratifiedGroupKFold",
        },
        "inner_cv": "StratifiedGroupKFold(n_splits=min(3, rarest-class group count), shuffle, seed=outer seed) fitted only on outer-training participants",
        "threshold": THRESHOLD,
        "calibration": "Platt scaling fitted on DNE OOF scores only; TULIP labels never used",
        "probability_interpretation": "model scores, not clinical probabilities",
        "python": sys.version,
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "production_model": production_info,
        "domain_classifier": domain_metrics,
        "quick_mode": bool(args.quick),
        "note": "TULIP has project-level historical exposure; results are exploratory external-domain evaluation, not independent clinical validation.",
    }
    (output_dir / "phase4_experiment_config.json").write_text(
        json.dumps(run_config, indent=2, default=str) + "\n", encoding="utf-8"
    )

    if not reproduction.empty:
        reproduced_count = int(reproduction["reproduced_exactly"].sum())
        print(f"Reproduction: {reproduced_count}/{len(reproduction)} run-4 cells matched exactly")
    if not comparison.empty:
        print(comparison.to_string(index=False))
    print(f"Phase 4 model experiments written under {output_dir}")
    print(f"Experimental models (not production) written under {models_dir}")


if __name__ == "__main__":
    main()
