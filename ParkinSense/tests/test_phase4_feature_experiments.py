from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_phase4_feature_experiments import (  # noqa: E402
    aggregate_participant_oof,
    calculate_metrics,
    correlation_filter,
    make_outer_splits,
    validate_dataset,
)


def make_dne_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for participant, label in (("h1", 0), ("h2", 0), ("h3", 0), ("p1", 1), ("p2", 1), ("p3", 1)):
        for trial in range(2):
            row = {"subject": participant, "trial": f"{participant}-{trial}", "diagnosis": label}
            row["fps"] = 60.0
            row["num_frames"] = 100 + trial
            for feature_index in range(72):
                for statistic in ("mean", "std", "min", "max"):
                    row[f"pose{feature_index}_{statistic}"] = float(rng.normal(label, 0.2))
            rows.append(row)
    return pd.DataFrame(rows)


def test_dataset_validation_checks_schema_and_labels():
    frame = make_dne_frame()
    features, group_column, label_column = validate_dataset(frame, "dne")
    assert len(features) == 288
    assert group_column == "subject"
    assert label_column == "diagnosis"


def test_outer_splits_keep_participant_groups_together():
    frame = make_dne_frame()
    _, group_column, label_column = validate_dataset(frame, "dne")
    splits = make_outer_splits(frame, "dne", group_column, label_column, seed=42)
    assert len(splits) == 5
    groups = frame[group_column].to_numpy()
    for train_indices, test_indices in splits:
        assert not set(groups[train_indices]) & set(groups[test_indices])


def test_correlation_filter_removes_constants_and_redundancy():
    rng = np.random.default_rng(11)
    base = rng.normal(size=60)
    frame = pd.DataFrame({
        "pose0_mean": base,
        "pose0_std": base * 2 + rng.normal(scale=0.001, size=60),
        "pose1_mean": rng.normal(size=60),
        "pose1_std": np.ones(60),
    })
    retained, removed = correlation_filter(frame, threshold=0.95)
    assert "pose1_std" not in retained
    assert "pose0_mean" in retained or "pose0_std" in retained
    assert len({"pose0_mean", "pose0_std"} & set(retained)) == 1
    assert removed


def test_metric_auc_is_unavailable_for_single_class():
    metrics = calculate_metrics(np.array([1, 1]), np.array([0.8, 0.7]))
    assert metrics["roc_auc"] is None
    assert metrics["n_participants"] == 2
    assert metrics["tp"] == 2


def test_participant_aggregation_preserves_full_feature_baseline():
    recordings = pd.DataFrame([
        {"track": "tulip", "seed": 42, "method": "full_288", "requested_count": None,
         "feature_count": 288, "participant_id": "s1", "true_label": 0,
         "recording_id": "s1-c1", "pd_score_uncalibrated": 0.2, "fold": 1},
        {"track": "tulip", "seed": 42, "method": "full_288", "requested_count": None,
         "feature_count": 288, "participant_id": "s1", "true_label": 0,
         "recording_id": "s1-c2", "pd_score_uncalibrated": 0.4, "fold": 1},
    ])
    participants = aggregate_participant_oof(recordings)
    assert len(participants) == 1
    assert pd.isna(participants.loc[0, "requested_count"])
    assert participants.loc[0, "n_recordings"] == 2
    assert participants.loc[0, "mean_pd_score_uncalibrated"] == pytest.approx(0.3)


def test_tulip_validation_rejects_mixed_participant_labels():
    frame = make_dne_frame().rename(columns={"subject": "subject_id", "trial": "recording_id", "diagnosis": "verified_label"})
    frame["cohort"] = "Clinical"
    frame.loc[frame.index[1], "verified_label"] = 1
    with pytest.raises(ValueError, match="contradictory labels"):
        validate_dataset(frame, "tulip")