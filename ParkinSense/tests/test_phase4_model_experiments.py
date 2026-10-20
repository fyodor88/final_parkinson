from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_phase4_model_experiments import (  # noqa: E402
    aggregate_participant_predictions,
    apply_platt,
    build_summary_frames,
    calculate_metrics_with_scores,
    collapse_flag,
    contract_columns,
    fit_model,
    fit_platt_calibrator,
    inner_splitter,
    make_feature_sets,
    model_applies_to_feature_set,
)


def test_make_feature_sets_includes_full_and_topk_only_when_available():
    rankings = {
        "corr_rf_importance": [f"pose{i}_mean" for i in range(30)],
        "l1_logistic": [f"pose{i}_std" for i in range(12)],
    }
    feature_sets = make_feature_sets(rankings)
    names = [item["feature_set"] for item in feature_sets]
    assert names[0] == "full_288"
    assert "corr_rf_importance_top10" in names
    assert "corr_rf_importance_top30" in names
    assert "corr_rf_importance_top50" not in names
    assert "l1_logistic_top10" in names
    assert "l1_logistic_top20" not in names
    top10 = next(item for item in feature_sets if item["feature_set"] == "corr_rf_importance_top10")
    assert len(top10["columns"]) == 10
    assert top10["requested_count"] == 10


def test_model_feature_set_matrix_matches_design():
    assert model_applies_to_feature_set("rf_baseline", "l1_logistic_top10")
    assert model_applies_to_feature_set("rf_baseline", "full_288")
    assert model_applies_to_feature_set("svm_rbf", "full_288")
    assert model_applies_to_feature_set("svm_rbf", "corr_rf_importance_top20")
    assert not model_applies_to_feature_set("svm_rbf", "l1_logistic_top20")
    assert not model_applies_to_feature_set("logreg_l2", "legacy_290_metadata")


def test_collapse_flag_detects_single_class_predictions():
    assert collapse_flag({"tp": 0, "fp": 0, "tn": 3, "fn": 5}) == "all_predicted_healthy"
    assert collapse_flag({"tp": 5, "fp": 3, "tn": 0, "fn": 0}) == "all_predicted_pd"
    assert collapse_flag({"tp": 5, "fp": 1, "tn": 2, "fn": 1}) == ""


def test_inner_splitter_keeps_groups_and_classes():
    groups: list[str] = []
    labels: list[int] = []
    for group_index in range(12):
        label = group_index % 2
        for _ in range(3):
            groups.append(f"g{group_index}")
            labels.append(label)
    splitter = inner_splitter(np.array(labels), np.array(groups), seed=42)
    assert splitter is not None
    X = np.zeros((len(labels), 4))
    for train_indices, test_indices in splitter.split(X, np.array(labels), np.array(groups)):
        train_groups = set(np.array(groups)[train_indices])
        test_groups = set(np.array(groups)[test_indices])
        assert not train_groups & test_groups
        assert len(np.unique(np.array(labels)[train_indices])) == 2


def test_inner_splitter_returns_none_when_too_few_groups():
    labels = np.array([0, 0, 1, 1])
    groups = np.array(["a", "a", "b", "b"])
    assert inner_splitter(labels, groups, seed=42) is None


def _tulip_loso_fold4_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Mirror the real TULIP LOSO fold that leaves 2 healthy + 5 PD groups."""
    layout = [
        ("subject01", 0, 5),
        ("subject02", 1, 6),
        ("subject05", 1, 5),
        ("subject08", 1, 5),
        ("subject11", 1, 5),
        ("subject12", 0, 6),
        ("subject13", 1, 5),
    ]
    labels: list[int] = []
    groups: list[str] = []
    for group, label, count in layout:
        labels.extend([label] * count)
        groups.extend([group] * count)
    return np.array(labels), np.array(groups)


def test_fit_model_falls_back_to_defaults_when_inner_fold_is_single_class(monkeypatch):
    labels, groups = _tulip_loso_fold4_training_data()
    idx_h = np.where(labels == 0)[0]
    idx_p = np.where(labels == 1)[0]
    mid = len(idx_p) // 2

    class SingleClassInnerSplitter:
        n_splits = 2
        def get_n_splits(self, X=None, y=None, groups=None):
            return 2
        def split(self, X, y=None, groups=None):
            yield np.concatenate([idx_h, idx_p[:mid]]), idx_p[mid:]
            yield np.concatenate([idx_h, idx_p[mid:]]), idx_p[:mid]

    monkeypatch.setattr(
        "scripts.run_phase4_model_experiments.inner_splitter",
        lambda y, g, s: SingleClassInnerSplitter(),
    )
    splitter = SingleClassInnerSplitter()
    single_class_folds = 0
    X_zero = np.zeros((len(labels), 4))
    for _, test_indices in splitter.split(X_zero, labels, groups):
        if len(np.unique(labels[test_indices])) == 1:
            single_class_folds += 1
    assert single_class_folds >= 1, "premise: real TULIP fold yields a single-class inner test fold"

    rng = np.random.default_rng(0)
    features = pd.DataFrame(rng.normal(size=(len(labels), 6)), columns=[f"f{i}" for i in range(6)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator, metadata = fit_model("logreg_l2", features, labels, groups, seed=42, tune=True)
    assert metadata["tuning_status"] == "fallback_defaults_single_class_inner_fold"
    assert metadata["best_params"] == {"logisticregression__C": 1.0}
    assert not np.isfinite(metadata["inner_best_roc_auc"])
    probabilities = estimator.predict_proba(features)
    assert probabilities.shape == (len(labels), 2)


def test_fit_model_uses_tuned_params_when_inner_cv_roc_auc_is_finite():
    groups: list[str] = []
    labels: list[int] = []
    for group_index in range(12):
        label = group_index % 2
        for _ in range(3):
            groups.append(f"g{group_index}")
            labels.append(label)
    labels_array = np.array(labels)
    groups_array = np.array(groups)
    rng = np.random.default_rng(1)
    features = pd.DataFrame(
        rng.normal(size=(len(labels_array), 6)), columns=[f"f{i}" for i in range(6)]
    )
    estimator, metadata = fit_model(
        "logreg_l2", features, labels_array, groups_array, seed=42, tune=True
    )
    assert metadata["tuning_status"] == "tuned_on_inner_grouped_cv_roc_auc"
    assert np.isfinite(metadata["inner_best_roc_auc"])
    assert set(metadata["best_params"]) == {"logisticregression__C"}
    assert estimator.predict_proba(features).shape == (len(labels_array), 2)


def test_participant_aggregation_uses_recording_mean():
    recordings = pd.DataFrame(
        [
            {
                "track": "tulip", "seed": 42, "model": "rf_baseline", "feature_set": "full_288",
                "participant_id": "subject01", "true_label": 0, "recording_id": "r1",
                "pd_score_uncalibrated": 0.2, "predicted_label": 0, "fold": 1,
                "selector": None, "requested_count": None, "feature_count": 288,
            },
            {
                "track": "tulip", "seed": 42, "model": "rf_baseline", "feature_set": "full_288",
                "participant_id": "subject01", "true_label": 0, "recording_id": "r2",
                "pd_score_uncalibrated": 0.4, "predicted_label": 0, "fold": 1,
                "selector": None, "requested_count": None, "feature_count": 288,
            },
        ]
    )
    participants = aggregate_participant_predictions(recordings)
    assert len(participants) == 1
    assert participants.loc[0, "n_recordings"] == 2
    assert participants.loc[0, "mean_pd_score_uncalibrated"] == pytest.approx(0.3)
    assert participants.loc[0, "predicted_label"] == 0


def test_calculate_metrics_with_scores_adds_probability_metrics():
    metrics = calculate_metrics_with_scores(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]), unit="participants"
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["log_loss"] > 0
    assert metrics["collapse_flag"] == ""


def test_recording_unit_renames_count_column():
    metrics = calculate_metrics_with_scores(
        np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]), unit="recordings"
    )
    assert "n_recordings" in metrics
    assert "n_participants" not in metrics


def test_platt_calibration_is_monotonic_and_labels_are_training_only():
    training_scores = np.array([0.2, 0.3, 0.7, 0.8, 0.25, 0.75])
    training_labels = np.array([0, 0, 1, 1, 0, 1])
    calibrator = fit_platt_calibrator(training_scores, training_labels)
    external_scores = np.array([0.1, 0.5, 0.95])
    calibrated = apply_platt(calibrator, external_scores)
    assert np.all(np.diff(calibrated) > 0)
    assert 0.0 <= calibrated.min() and calibrated.max() <= 1.0


def test_contract_columns_requires_all_290():
    pose_columns = [f"pose{i}_{stat}" for i in range(72) for stat in ("mean", "std", "min", "max")]
    frame = pd.DataFrame(columns=["fps", "num_frames"] + pose_columns)
    columns = contract_columns(frame, pose_columns)
    assert len(columns) == 290
    assert columns[:2] == ["fps", "num_frames"]
    with pytest.raises(ValueError, match="missing contract columns"):
        contract_columns(pd.DataFrame(columns=["fps"] + pose_columns), pose_columns)


def test_build_summary_frames_computes_deltas_and_flags():
    metric_rows = []
    for seed in (42, 123, 2026):
        metric_rows.append({
            "evaluation_level": "dne_cv_participant_oof_pooled", "track": "dne", "seed": seed,
            "model": "rf_baseline", "feature_set": "full_288", "accuracy": 0.85, "roc_auc": 0.95,
        })
    metric_rows.append({
        "evaluation_level": "dne_cv_participant_oof_pooled", "track": "dne", "seed": 42,
        "model": "svm_rbf", "feature_set": "full_288", "accuracy": 0.90, "roc_auc": 0.97,
    })
    metric_rows.append({
        "evaluation_level": "external_tulip_clinical_participant", "track": "tulip", "seed": 42,
        "model": "rf_baseline", "feature_set": "full_288", "score_type": "uncalibrated_model_score",
        "accuracy": 0.50, "sensitivity": 0.8, "specificity": 0.0, "f1": 0.67, "roc_auc": 0.46,
        "collapse_flag": "all_predicted_pd",
    })
    metric_rows.append({
        "evaluation_level": "external_tulip_clinical_participant", "track": "tulip", "seed": 42,
        "model": "production_pd_detector", "feature_set": "production_290_contract",
        "score_type": "uncalibrated_model_score", "accuracy": 0.375, "sensitivity": 0.0,
        "specificity": 1.0, "f1": 0.0, "roc_auc": 0.667, "collapse_flag": "all_predicted_healthy",
    })
    metric_frame = pd.DataFrame(metric_rows)
    fold_rows = []
    for fold, accuracy in ((1, 0.8), (2, 0.9)):
        fold_rows.append({
            "model": "rf_baseline", "feature_set": "full_288", "seed": 42, "fold": fold,
            "accuracy": accuracy, "sensitivity": 0.8, "specificity": 0.9, "f1": 0.85,
            "roc_auc": 0.93, "brier_score": 0.1, "collapse_flag": "",
        })
    fold_frame = pd.DataFrame(fold_rows)
    comparison, dne_summary, tulip_summary = build_summary_frames(metric_frame, fold_frame)
    rf_row = comparison[comparison["model"] == "rf_baseline"].iloc[0]
    assert rf_row["dne_fold_acc_mean_seed42"] == pytest.approx(0.85)
    assert rf_row["tulip_ext_collapse"] == "all_predicted_pd"
    svm_row = comparison[comparison["model"] == "svm_rbf"].iloc[0]
    assert pd.isna(svm_row["delta_dne_fold_acc_vs_rf_full"])
    assert "production_pd_detector" in set(comparison["model"])
    assert not dne_summary.empty
    assert "external_tulip_clinical_participant" in set(tulip_summary["evaluation_level"])
