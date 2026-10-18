"""Run leakage-safe, exploratory Phase 4 feature-count experiments.

This script uses DNE and the Phase 2 TULIP candidate as separate tracks. All
supervised ranking and fitting occur within participant-grouped training data.
It writes predictions and metadata only; it never serializes a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
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
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DNE_PATH = PROJECT_ROOT / "data" / "processed" / "pd_dataset.csv"
TULIP_PATH = PROJECT_ROOT / "data" / "processed" / "improved_model" / "phase2_tulip_clinical_candidate.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "phase4_feature_experiments_run4"
FEATURE_PATTERN = re.compile(r"\Apose\d+_(?:mean|std|min|max)\Z")
K_COUNTS = (10, 20, 30, 50)
SEEDS = (42, 123, 2026)
CORRELATION_THRESHOLD = 0.95


def feature_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if FEATURE_PATTERN.fullmatch(column)]


def validate_dataset(frame: pd.DataFrame, track: str) -> tuple[list[str], str, str]:
    if track == "dne":
        group_column, label_column = "subject", "diagnosis"
        recording_column = "trial"
    elif track == "tulip":
        group_column, label_column = "subject_id", "verified_label"
        recording_column = "recording_id"
    else:
        raise ValueError(f"Unknown track: {track}")

    required = {group_column, label_column, recording_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"{track}: missing required columns {sorted(required - set(frame.columns))}")
    if frame[group_column].isna().any() or frame[recording_column].isna().any():
        raise ValueError(f"{track}: missing participant or recording identifier")
    if frame[recording_column].duplicated().any():
        raise ValueError(f"{track}: duplicate recording identifiers")
    if set(frame[label_column].astype(int).unique()) != {0, 1}:
        raise ValueError(f"{track}: expected binary labels 0=Healthy, 1=Parkinson's")
    group_labels = frame.groupby(group_column)[label_column].nunique()
    if (group_labels != 1).any():
        raise ValueError(f"{track}: participant has contradictory labels")

    pose_columns = feature_columns(list(frame.columns))
    if len(pose_columns) != 288:
        raise ValueError(f"{track}: expected exactly 288 pose-statistic columns, got {len(pose_columns)}")
    numeric = frame[pose_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{track}: pose features contain missing or non-finite values")
    return pose_columns, group_column, label_column


def make_outer_splits(
    frame: pd.DataFrame,
    track: str,
    group_column: str,
    label_column: str,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = frame[group_column].astype(str).to_numpy()
    y = frame[label_column].astype(int).to_numpy()
    if track == "tulip":
        splitter = LeaveOneGroupOut()
        splits = list(splitter.split(frame, y, groups))
    else:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = list(splitter.split(frame, y, groups))
    for train_indices, test_indices in splits:
        if set(groups[train_indices]) & set(groups[test_indices]):
            raise AssertionError(f"{track}: participant leakage in outer split")
        if len(np.unique(y[train_indices])) != 2:
            raise ValueError(f"{track}: an outer training fold contains only one class")
    return splits
def correlation_filter(
    X_train: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
) -> tuple[list[str], dict[str, str]]:
    """Greedily retain nonconstant, low-redundancy columns using training only."""
    variances = X_train.var(axis=0, ddof=0)
    schema_order = {name: index for index, name in enumerate(X_train.columns)}
    candidates = [name for name in X_train.columns if np.isfinite(variances[name]) and variances[name] > 0]
    candidates.sort(key=lambda name: (-float(variances[name]), schema_order[name]))
    correlations = X_train[candidates].corr(method="spearman").abs()
    retained: list[str] = []
    removed_by: dict[str, str] = {}
    for candidate in candidates:
        blockers = [name for name in retained if correlations.loc[candidate, name] >= threshold]
        if blockers:
            removed_by[candidate] = blockers[0]
        else:
            retained.append(candidate)
    return retained, removed_by


def rf_importance_rank(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    eligible: list[str],
    track: str,
) -> list[str]:
    """Rank by training-fold RF importance after training-fold correlation filtering."""
    if len(eligible) == 0:
        return []
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train[eligible], y_train)
    scores = pd.Series(model.feature_importances_, index=eligible)
    order = {name: index for index, name in enumerate(eligible)}
    return sorted(eligible, key=lambda name: (-float(scores[name]), order[name]))


def l1_rank(X_train: pd.DataFrame, y_train: np.ndarray) -> list[str]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_train)
    selector = LogisticRegression(
        penalty="l1",
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )
    selector.fit(scaled, y_train)
    coefficients = np.abs(selector.coef_[0])
    positions = {name: index for index, name in enumerate(X_train.columns)}
    return sorted(X_train.columns, key=lambda name: (-float(coefficients[positions[name]]), positions[name]))


def fit_selector(
    method: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    track: str,
) -> tuple[list[str], dict[str, Any]]:
    if method == "corr_rf_importance":
        eligible, removed = correlation_filter(X_train)
        ranked = rf_importance_rank(X_train, y_train, groups_train, eligible, track)
        return ranked, {
            "importance_type": "Random Forest mean decrease impurity, fitted on outer training participants",
            "correlation_threshold": CORRELATION_THRESHOLD,
            "eligible_after_correlation": len(eligible),
            "removed_correlated_or_constant": len(removed) + (X_train.shape[1] - len(eligible) - len(removed)),
            "removed_by": removed,
        }
    if method == "l1_logistic":
        variances = X_train.var(axis=0, ddof=0)
        eligible = [name for name in X_train.columns if np.isfinite(variances[name]) and variances[name] > 0]
        if not eligible:
            return [], {"eligible_after_constant_filter": 0}
        ranked = l1_rank(X_train[eligible], y_train)
        return ranked, {"eligible_after_constant_filter": len(eligible)}
    raise ValueError(f"Unknown selector: {method}")


def pd_probability(model: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    class_index = list(model.classes_).index(1)
    return model.predict_proba(X)[:, class_index]


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    auc: float | None = None
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, probabilities))
    specificity = float(tn / (tn + fp)) if (tn + fp) else None
    return {
        "n_participants": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "sensitivity": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "roc_auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def aggregate_participant_oof(recording_frame: pd.DataFrame) -> pd.DataFrame:
    participant_frame = (
        recording_frame.groupby(
            ["track", "seed", "method", "requested_count", "feature_count", "participant_id"],
            as_index=False,
            dropna=False,
        )
        .agg(
            true_label=("true_label", "first"),
            n_recordings=("recording_id", "count"),
            mean_pd_score_uncalibrated=("pd_score_uncalibrated", "mean"),
            outer_folds=("fold", lambda values: ",".join(str(value) for value in sorted(set(values)))),
        )
    )
    participant_frame["predicted_label"] = (
        participant_frame["mean_pd_score_uncalibrated"] >= 0.5
    ).astype(int)
    return participant_frame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_track(track: str, frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pose_columns, group_column, label_column = validate_dataset(frame, track)
    recording_column = "trial" if track == "dne" else "recording_id"
    methods = ("corr_rf_importance", "l1_logistic")
    metric_rows: list[dict[str, Any]] = []
    recording_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    seeds = SEEDS if track == "dne" else (42,)

    for seed in seeds:
        splits = make_outer_splits(frame, track, group_column, label_column, seed)
        experiment_specs: list[tuple[str, int | None, list[str]]] = [
            ("full_288", None, pose_columns),
        ]
        if track == "dne":
            legacy_features = [name for name in ("fps", "num_frames") if name in frame.columns] + pose_columns
            if len(legacy_features) == 290:
                experiment_specs.append(("legacy_290_metadata", None, legacy_features))

        for fold_number, (train_indices, test_indices) in enumerate(splits, start=1):
            train = frame.iloc[train_indices]
            test = frame.iloc[test_indices]
            y_train = train[label_column].astype(int).to_numpy()
            train_groups = train[group_column].astype(str).to_numpy()
            fold_info = {
                "track": track,
                "seed": seed,
                "fold": fold_number,
                "train_participants": sorted(set(train_groups)),
                "test_participants": sorted(set(test[group_column].astype(str))),
                "train_recordings": len(train),
                "test_recordings": len(test),
            }

            fold_specs = list(experiment_specs)
            for method in methods:
                ranked, selector_metadata = fit_selector(
                    method,
                    train[pose_columns],
                    y_train,
                    train_groups,
                    track,
                )
                for requested_count in K_COUNTS:
                    if len(ranked) < requested_count:
                        feature_rows.append({
                            **fold_info,
                            "method": method,
                            "requested_count": requested_count,
                            "selected_count": len(ranked),
                            "status": "insufficient_eligible_features",
                            "selected_features": ranked,
                            "selector_metadata": selector_metadata,
                        })
                        continue
                    selected = ranked[:requested_count]
                    fold_specs.append((method, requested_count, selected))
                    feature_rows.append({
                        **fold_info,
                        "method": method,
                        "requested_count": requested_count,
                        "selected_count": len(selected),
                        "status": "ok",
                        "selected_features": selected,
                        "selector_metadata": selector_metadata,
                    })

            for method_name, count, selected_features in fold_specs:
                classifier = RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )
                classifier.fit(train[selected_features], y_train)
                probabilities = pd_probability(classifier, test[selected_features])
                prediction = (probabilities >= 0.5).astype(int)
                for local_index, row_index in enumerate(test_indices):
                    row = frame.iloc[row_index]
                    recording_rows.append({
                        **fold_info,
                        "method": method_name,
                        "requested_count": count,
                        "feature_count": len(selected_features),
                        "recording_id": f"{row[group_column]}::{row[recording_column]}",
                        "participant_id": str(row[group_column]),
                        "true_label": int(row[label_column]),
                        "pd_score_uncalibrated": float(probabilities[local_index]),
                        "predicted_label": int(prediction[local_index]),
                    })

    recording_frame = pd.DataFrame(recording_rows)
    if recording_frame.empty:
        raise RuntimeError(f"{track}: no held-out predictions were generated")
    participant_frame = aggregate_participant_oof(recording_frame)

    for (seed, method, count, feature_count), subset in participant_frame.groupby(
        ["seed", "method", "requested_count", "feature_count"], dropna=False
    ):
        y_true = subset["true_label"].astype(int).to_numpy()
        probs = subset["mean_pd_score_uncalibrated"].to_numpy(dtype=float)
        metrics = calculate_metrics(y_true, probs)
        class_rows = frame.groupby(label_column).size().to_dict()
        participant_class_counts = frame.groupby(group_column)[label_column].first().value_counts().to_dict()
        metric_rows.append({
            "track": track,
            "seed": int(seed),
            "method": str(method),
            "requested_count": None if pd.isna(count) else int(count),
            "feature_count": int(feature_count),
            "dataset_recordings": len(frame),
            "dataset_participants": frame[group_column].nunique(),
            "dataset_healthy_recordings": int(class_rows.get(0, 0)),
            "dataset_pd_recordings": int(class_rows.get(1, 0)),
            "dataset_healthy_participants": int(participant_class_counts.get(0, 0)),
            "dataset_pd_participants": int(participant_class_counts.get(1, 0)),
            "split_strategy": "5-fold StratifiedGroupKFold" if track == "dne" else "LeaveOneGroupOut",
            "threshold": 0.5,
            "probability_status": "uncalibrated model score; not clinical probability",
            **metrics,
        })
    return metric_rows, recording_rows, feature_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 4 output directory: {output_dir}")

    dne = pd.read_csv(DNE_PATH)
    tulip = pd.read_csv(TULIP_PATH)
    validate_dataset(dne, "dne")
    validate_dataset(tulip, "tulip")
    output_dir.mkdir(parents=True, exist_ok=False)

    all_metrics: list[dict[str, Any]] = []
    all_recordings: list[dict[str, Any]] = []
    all_features: list[dict[str, Any]] = []
    for track, frame in (("dne", dne), ("tulip", tulip)):
        metrics, recordings, features = run_track(track, frame)
        all_metrics.extend(metrics)
        all_recordings.extend(recordings)
        all_features.extend(features)

    pd.DataFrame(all_metrics).to_csv(output_dir / "phase4_metrics.csv", index=False)
    pd.DataFrame(all_recordings).to_csv(output_dir / "phase4_recording_oof_predictions.csv", index=False)
    all_participants = []
    for track, frame in (("dne", dne), ("tulip", tulip)):
        validate_dataset(frame, track)
        recording_predictions = pd.DataFrame(
            [row for row in all_recordings if row["track"] == track]
        )
        grouped = aggregate_participant_oof(recording_predictions)
        all_participants.append(grouped)
    pd.concat(all_participants, ignore_index=True).to_csv(
        output_dir / "phase4_participant_oof_predictions.csv", index=False
    )
    pd.DataFrame(all_features).to_json(
        output_dir / "phase4_selected_features.jsonl",
        orient="records",
        lines=True,
    )

    run_metadata = {
        "scope": "Separate exploratory DNE-only and TULIP-only experiments; no combined data.",
        "dne_path": str(DNE_PATH.relative_to(PROJECT_ROOT)),
        "dne_sha256": sha256_file(DNE_PATH),
        "tulip_path": str(TULIP_PATH.relative_to(PROJECT_ROOT)),
        "tulip_sha256": sha256_file(TULIP_PATH),
        "seeds": {"dne": list(SEEDS), "tulip": [42]},
        "classifier": "RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)",
        "selectors": {
            "corr_rf_importance": "constant removal and abs(Spearman rho)>=0.95 filter fitted on outer training fold; RF mean decrease impurity importance fitted on outer training participants; top-k; correlated features can bias/split importance",
            "l1_logistic": "constant removal, StandardScaler and L1 LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear') fitted on outer training participants; top-k by absolute coefficient",
        },
        "threshold": 0.5,
        "probability_interpretation": "predict_proba-derived uncalibrated scores, not clinical probabilities",
        "python": sys.version,
        "scikit_learn": sklearn.__version__,
        "note": "No model artifacts saved. DNE results are exploratory internal reanalysis with collection/session confounding. TULIP results are exploratory only and are not independent external validation.",
    }
    (output_dir / "phase4_run_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Phase 4 results written under {output_dir}")


if __name__ == "__main__":
    main()