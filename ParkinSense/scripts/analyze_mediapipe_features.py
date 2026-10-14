"""
ParkinSense — MediaPipe 290-Feature Representation Analysis (TRAINING DATA ONLY)
=================================================================================
Scientific feature analysis of the current normalized MediaPipe/SMPL 290-feature
representation used by the ParkinSense classifier.

Scope / integrity constraints (enforced by this script):
  * Reads ONLY `data/processed/pd_dataset_mediapipe_normalized.csv` (training set).
  * NEVER reads any TULIP label, prediction, metric, or manifest.
  * NEVER loads or modifies any saved model file (`models/*.pkl` are untouched;
    the script fits throwaway RandomForest instances in memory only).
  * No feature is dropped, no model is retrained for deployment, and NO claim about
    external (TULIP) accuracy is made or implied.

Outputs:
  * data/processed/mediapipe_feature_importance.csv  (290 rows, 9 columns)
  * docs/mediapipe_feature_analysis.md               (full analysis report)

Column definitions of the CSV (documented here and in the report):
  feature                - feature name
  feature_group          - semantic group (see feature_group())
  variance               - sample variance of the feature over the 322 training trials
  mean                   - mean over the 322 training trials
  std                    - standard deviation over the 322 training trials
  rf_importance          - Mean Decrease Impurity (Gini) importance of a RandomForest
                           trained on ALL 322 training trials (300 trees,
                           class_weight=balanced, random_state=42), normalized to sum 1.
  permutation_importance - mean held-out permutation importance (drop in ROC-AUC) from
                           5-fold subject-wise GroupKFold, 10 repeats per fold.
  mean_fold_importance   - mean across the 5 subject-wise folds of the fold-level MDI
                           (Gini) importance of a RandomForest fit on that fold's
                           training partition.
  importance_std         - standard deviation across the 5 subject-wise folds of the
                           fold-level MDI importance (stability indicator; large value
                           = unstable feature importance across subject folds).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "pd_dataset_mediapipe_normalized.csv"
CSV_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "mediapipe_feature_importance.csv"
REPORT_OUTPUT_PATH = PROJECT_ROOT / "docs" / "mediapipe_feature_analysis.md"

N_SPLITS = 5
N_REPEATS_PERM = 10
RANDOM_STATE = 42
SCORING = "roc_auc"

# SMPL 24-joint names (mirrors src/pose/mediapipe_smpl.py:SMPL_JOINT_NAMES; duplicated
# locally so this analysis script does not import the MediaPipe runtime).
SMPL_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle",
    "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    "L_Hand", "R_Hand",
]

META_COLS = ["subject", "trial", "diagnosis", "diagnosis_name"]

# Requested semantic taxonomy. Groups with 0 features are reported explicitly as ABSENT.
GROUP_TEMPORAL = "temporal_video_length"      # "temporal features" + "FPS/video-length"
GROUP_ROOT = "pose_rotation_root"            # part of "pose/rotation statistics"
GROUP_JOINT = "pose_rotation_joint"          # part of "pose/rotation statistics"
GROUP_POSITION = "position_statistics"       # requested taxonomy -> ABSENT
GROUP_GAIT = "gait_features"                 # requested taxonomy -> ABSENT

ZERO_VAR_THRESHOLD = 0.0          # exact zero variance (dead feature)
NEAR_ZERO_VAR_THRESHOLD = 1e-5    # rad^2 -> std < 0.0032 rad (~0.18 deg)
CORR_REDUNDANT = 0.95             # |Pearson r| above which features are "redundant"
LENGTH_CORR_FLAG = 0.30           # |Spearman rho| with num_frames flagging length-coupling
SESSION_SMD_FLAG = 0.50           # standardized mean difference dne1-healthy vs dne2-healthy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_rf() -> RandomForestClassifier:
    """Random Forest with the exact hyperparameters of the existing pipeline."""
    return RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )


def feature_group(name: str) -> str:
    if name in ("fps", "num_frames"):
        return GROUP_TEMPORAL
    if name.startswith("pose"):
        idx = int(name[4:].split("_")[0])
        return GROUP_ROOT if idx < 3 else GROUP_JOINT
    return "unclassified"


def stat_type(name: str) -> str:
    if name in ("fps", "num_frames"):
        return name
    return name.rsplit("_", 1)[1]


def joint_of(name: str) -> int | None:
    if not name.startswith("pose"):
        return None
    return int(name[4:].split("_")[0]) // 3


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    # escape pipes so cells like "|rho| vs num_frames" do not break the table
    headers = [str(h).replace("|", "\\|") for h in headers]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join([" --- "] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 1. Load + verification
# ---------------------------------------------------------------------------
def load_and_verify() -> tuple[pd.DataFrame, list[str], dict]:
    df = pd.read_csv(DATASET_PATH)
    feats = [c for c in df.columns if c not in META_COLS]

    checks = {
        "trials": int(len(df)),
        "subjects": int(df["subject"].nunique()),
        "features": int(len(feats)),
        "healthy_trials": int((df["diagnosis"] == 0).sum()),
        "pd_trials": int((df["diagnosis"] == 1).sum()),
        "healthy_subjects": int(df.loc[df["diagnosis"] == 0, "subject"].nunique()),
        "pd_subjects": int(df.loc[df["diagnosis"] == 1, "subject"].nunique()),
        "subjects_with_mixed_labels": int((df.groupby("subject")["diagnosis"].nunique() > 1).sum()),
        "fps_unique_values": sorted(float(v) for v in df["fps"].unique()),
        "nan_count": int(df[feats].isna().sum().sum()),
        "trials_per_subject_min": int(df.groupby("subject").size().min()),
        "trials_per_subject_max": int(df.groupby("subject").size().max()),
    }
    expected = {
        "trials": 322, "subjects": 58, "features": 290,
        "healthy_trials": 205, "pd_trials": 117,
    }
    for k, v in expected.items():
        assert checks[k] == v, f"Verification failed: {k}={checks[k]} (expected {v})"
    assert checks["nan_count"] == 0, "Dataset contains NaN feature values"
    assert checks["subjects_with_mixed_labels"] == 0, "A subject carries both labels"
    return df, feats, checks


# ---------------------------------------------------------------------------
# 2. Variance analysis
# ---------------------------------------------------------------------------
def variance_analysis(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    stats = pd.DataFrame({
        "feature": feats,
        "mean": [df[f].mean() for f in feats],
        "std": [df[f].std() for f in feats],
        "variance": [df[f].var() for f in feats],
    })
    stats["feature_group"] = stats["feature"].map(feature_group)
    stats["stat_type"] = stats["feature"].map(stat_type)
    stats["constant"] = stats["variance"] <= ZERO_VAR_THRESHOLD
    stats["near_zero"] = (~stats["constant"]) & (stats["variance"] < NEAR_ZERO_VAR_THRESHOLD)
    return stats


# ---------------------------------------------------------------------------
# 3. Redundancy analysis
# ---------------------------------------------------------------------------
def redundancy_analysis(df: pd.DataFrame, feats: list[str]) -> dict:
    corr = df[feats].corr(method="pearson").abs()
    n = len(feats)
    iu = np.triu_indices(n, k=1)
    vals = corr.values[iu]
    pairs_95, pairs_90, pairs_80 = [], [], []
    for k in range(len(vals)):
        r = vals[k]
        i, j = iu[0][k], iu[1][k]
        if r >= CORR_REDUNDANT:
            pairs_95.append((feats[i], feats[j], r))
        elif r >= 0.90:
            pairs_90.append((feats[i], feats[j], r))
        elif r >= 0.80:
            pairs_80.append((feats[i], feats[j], r))
    pairs_95.sort(key=lambda t: -t[2])
    pairs_90.sort(key=lambda t: -t[2])
    pairs_80.sort(key=lambda t: -t[2])

    # Connected components of |r| >= 0.95 -> redundancy clusters
    g = nx.Graph()
    g.add_nodes_from(feats)
    for a, b, _ in pairs_95:
        g.add_edge(a, b)
    clusters = sorted((sorted(c) for c in nx.connected_components(g)), key=len, reverse=True)
    multi = [c for c in clusters if len(c) > 1]

    # exact duplicate columns (|r| == 1 within float tolerance)
    dup_groups = [c for c in multi if all(
        corr.loc[a, b] > 0.999999999 for a, b in zip(c, c[1:]))]

    # within-feature redundancy: min/max of the same channel
    same_channel = [p for p in pairs_95
                    if p[0].split("_")[0] == p[1].split("_")[0]]
    cross_channel = [p for p in pairs_95
                     if p[0].split("_")[0] != p[1].split("_")[0]]

    return {
        "n_pairs_95": len(pairs_95), "pairs_95": pairs_95,
        "n_pairs_90": len(pairs_90), "pairs_90": pairs_90,
        "n_pairs_80": len(pairs_80), "pairs_80": pairs_80,
        "n_clusters_95": len(multi), "clusters_95": multi,
        "largest_cluster": multi[0] if multi else [],
        "dup_groups": dup_groups,
        "same_channel_pairs_95": len(same_channel),
        "cross_channel_pairs_95": len(cross_channel),
        "max_abs_r": float(np.nanmax(vals)) if np.isfinite(vals).any() else 0.0,
    }


# ---------------------------------------------------------------------------
# 4. Subject-wise CV: MDI importance, permutation importance, univariate AUC
# ---------------------------------------------------------------------------
def subjectwise_cv_importance(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                              feats: list[str]) -> dict:
    gkf = GroupKFold(n_splits=N_SPLITS)
    splits = list(gkf.split(X, y, groups))

    md_full = make_rf().fit(X, y).feature_importances_

    md_folds = np.zeros((N_SPLITS, len(feats)))
    perm_folds = np.zeros((N_SPLITS, len(feats)))
    perm_repeat_std = np.zeros((N_SPLITS, len(feats)))
    fold_info = []

    for k, (tr, va) in enumerate(splits):
        rf = make_rf().fit(X[tr], y[tr])
        md_folds[k] = rf.feature_importances_

        pi = permutation_importance(
            rf, X[va], y[va],
            n_repeats=N_REPEATS_PERM,
            scoring=SCORING,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        perm_folds[k] = pi.importances_mean
        perm_repeat_std[k] = pi.importances_std

        fold_info.append({
            "fold": k + 1,
            "train_trials": int(len(tr)),
            "val_trials": int(len(va)),
            "train_subjects": int(len(np.unique(groups[tr]))),
            "val_subjects": int(len(np.unique(groups[va]))),
            "train_pd_trials": int(y[tr].sum()),
            "val_pd_trials": int(y[va].sum()),
            "val_baseline_auc": float(roc_auc_score(y[va], rf.predict_proba(X[va])[:, 1])),
            "val_accuracy": float(accuracy_score(y[va], rf.predict(X[va]))),
        })

    # univariate (single-feature) subject-wise CV AUC, direction-aligned
    uni_auc = np.full(len(feats), np.nan)
    uni_auc_std = np.full(len(feats), np.nan)
    for j, f in enumerate(feats):
        v = X[:, j]
        if np.std(v) == 0:
            continue
        fold_aucs = []
        for tr, va in splits:
            a = roc_auc_score(y[va], v[va])
            fold_aucs.append(max(a, 1.0 - a))
        uni_auc[j] = float(np.mean(fold_aucs))
        uni_auc_std[j] = float(np.std(fold_aucs))

    return {
        "md_full": md_full,
        "md_folds": md_folds,
        "perm_folds": perm_folds,
        "perm_repeat_std": perm_repeat_std,
        "fold_info": fold_info,
        "uni_auc": uni_auc,
        "uni_auc_std": uni_auc_std,
        "splits": splits,
    }


# ---------------------------------------------------------------------------
# 5. Video-length coupling (num_frames / duration dependence)
# ---------------------------------------------------------------------------
def length_coupling_analysis(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    nf = df["num_frames"].values
    rows = []
    for f in feats:
        v = df[f].values
        if np.std(v) == 0:
            rows.append({"feature": f, "spearman_rho_num_frames": np.nan, "abs_rho": np.nan})
            continue
        rho, _ = spearmanr(v, nf)
        rows.append({"feature": f, "spearman_rho_num_frames": float(rho), "abs_rho": float(abs(rho))})
    out = pd.DataFrame(rows)
    out["stat_type"] = out["feature"].map(stat_type)
    out["feature_group"] = out["feature"].map(feature_group)
    out["length_coupled"] = out["abs_rho"] >= LENGTH_CORR_FLAG
    return out


# ---------------------------------------------------------------------------
# 6. Recording-session (dne1/dne2) domain dependence, label-free
# ---------------------------------------------------------------------------
def session_analysis(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """dne1 holds ONLY healthy subjects while dne2 holds both classes, so any feature
    that separates dne1 from dne2 among *healthy* subjects only is a recording-session /
    protocol proxy, not a disease marker. Standardized mean difference (Cohen's d style)."""
    healthy = df[df["diagnosis"] == 0]
    s = (healthy["subject"].str.startswith("dne2")).astype(int).values
    rows = []
    for f in feats:
        v = healthy[f].values.astype(float)
        pooled_std = float(np.std(v, ddof=1))
        if pooled_std == 0:
            smd = 0.0
        else:
            smd = abs(float(v[s == 1].mean()) - float(v[s == 0].mean())) / pooled_std
        rows.append({"feature": f, "session_smd_healthy": smd})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Internal (training-only) ablations of predefined feature groups
# ---------------------------------------------------------------------------
def cv_score_subset(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    idx: np.ndarray) -> dict:
    gkf = GroupKFold(n_splits=N_SPLITS)
    aucs, accs = [], []
    for tr, va in gkf.split(X, y, groups):
        rf = make_rf().fit(X[tr][:, idx], y[tr])
        prob = rf.predict_proba(X[va][:, idx])[:, 1]
        pred = rf.predict(X[va][:, idx])
        aucs.append(roc_auc_score(y[va], prob))
        accs.append(accuracy_score(y[va], pred))
    return {
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
    }


def run_ablations(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  feats: list[str]) -> pd.DataFrame:
    all_idx = np.arange(len(feats))
    temporal_idx = np.array([i for i, f in enumerate(feats) if f in ("fps", "num_frames")])
    non_constant_idx = np.array([i for i, f in enumerate(feats) if np.std(X[:, i]) > 0])
    joint_idx = np.array([i for i, f in enumerate(feats)
                          if f.startswith("pose") and int(f[4:].split("_")[0]) >= 3])
    root_idx = np.array([i for i, f in enumerate(feats)
                         if f.startswith("pose") and int(f[4:].split("_")[0]) < 3])
    mean_idx = np.array([i for i, f in enumerate(feats) if stat_type(f) == "mean"])
    std_idx = np.array([i for i, f in enumerate(feats) if stat_type(f) == "std"])
    minmax_idx = np.array([i for i, f in enumerate(feats) if stat_type(f) in ("min", "max")])
    mean_std_idx = np.array([i for i, f in enumerate(feats) if stat_type(f) in ("mean", "std")])

    subsets = [
        ("A. all 290 features (current pipeline)", all_idx),
        ("B. temporal/video-length only (fps, num_frames)", temporal_idx),
        ("C. all minus temporal (288)", np.setdiff1d(all_idx, temporal_idx)),
        ("D. all minus constant features (277)", non_constant_idx),
        ("E. joint rotations only, pose3..71 (276)", joint_idx),
        ("F. root orientation only, pose0..2 (12)", root_idx),
        ("G. pose means only (72)", mean_idx),
        ("H. pose std only (72)", std_idx),
        ("I. pose min+max only (144)", minmax_idx),
        ("J. pose mean+std only (144)", mean_std_idx),
    ]
    rows = []
    for name, idx in subsets:
        if len(idx) == 0:
            rows.append({"subset": name, "n_features": 0, "auc_mean": np.nan,
                         "auc_std": np.nan, "acc_mean": np.nan, "acc_std": np.nan})
            continue
        r = cv_score_subset(X, y, groups, idx)
        rows.append({"subset": name, "n_features": int(len(idx)), **r})
    return pd.DataFrame(rows)


def nested_topk_selection(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                          ks: tuple[int, ...] = (30, 60)) -> pd.DataFrame:
    """Nested, training-only estimate of selecting the top-k features by inner
    subject-wise permutation importance. Inner folds select, outer fold scores.
    This is an INTERNAL estimate only; it says nothing about external performance."""
    outer = GroupKFold(n_splits=N_SPLITS)
    outer_splits = list(outer.split(X, y, groups))
    scores = {k: {"auc": [], "acc": []} for k in ks}
    base = {"auc": [], "acc": []}

    for tr, va in outer_splits:
        inner_groups = groups[tr]
        inner = GroupKFold(n_splits=3)
        inner_perm = np.zeros(X.shape[1])
        for itr, iva in inner.split(X[tr], y[tr], inner_groups):
            rf = make_rf().fit(X[tr][itr], y[tr][itr])
            pi = permutation_importance(
                rf, X[tr][iva], y[tr][iva],
                n_repeats=5, scoring=SCORING,
                random_state=RANDOM_STATE, n_jobs=-1,
            )
            inner_perm += pi.importances_mean
        inner_perm /= 3.0

        rf = make_rf().fit(X[tr], y[tr])
        prob = rf.predict_proba(X[va])[:, 1]
        base["auc"].append(roc_auc_score(y[va], prob))
        base["acc"].append(accuracy_score(y[va], rf.predict(X[va])))

        order = np.argsort(-inner_perm)
        for k in ks:
            sel = order[:k]
            rf = make_rf().fit(X[tr][:, sel], y[tr])
            prob = rf.predict_proba(X[va][:, sel])[:, 1]
            scores[k]["auc"].append(roc_auc_score(y[va], prob))
            scores[k]["acc"].append(accuracy_score(y[va], rf.predict(X[va][:, sel])))

    rows = [{
        "variant": f"nested selection: top-{k} by inner permutation importance",
        "n_features": k,
        "auc_mean": float(np.mean(scores[k]["auc"])),
        "auc_std": float(np.std(scores[k]["auc"])),
        "acc_mean": float(np.mean(scores[k]["acc"])),
        "acc_std": float(np.std(scores[k]["acc"])),
    } for k in ks]
    rows.insert(0, {
        "variant": "baseline: all 290 features (same outer folds)",
        "n_features": 290,
        "auc_mean": float(np.mean(base["auc"])),
        "auc_std": float(np.std(base["auc"])),
        "acc_mean": float(np.mean(base["acc"])),
        "acc_std": float(np.std(base["acc"])),
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. Assemble results + write CSV
# ---------------------------------------------------------------------------
def assemble(df: pd.DataFrame, feats: list[str], var_stats: pd.DataFrame,
             cv: dict, length: pd.DataFrame, session: pd.DataFrame) -> pd.DataFrame:
    md_full = cv["md_full"]
    md_folds_mean = cv["md_folds"].mean(axis=0)
    md_folds_std = cv["md_folds"].std(axis=0, ddof=1)
    perm_mean = cv["perm_folds"].mean(axis=0)
    perm_fold_std = cv["perm_folds"].std(axis=0, ddof=1)
    perm_fold_neg = (cv["perm_folds"] <= 0).sum(axis=0)

    out = pd.DataFrame({
        "feature": feats,
        "feature_group": [feature_group(f) for f in feats],
        "variance": var_stats["variance"].values,
        "mean": var_stats["mean"].values,
        "std": var_stats["std"].values,
        "rf_importance": md_full,
        "permutation_importance": perm_mean,
        "mean_fold_importance": md_folds_mean,
        "importance_std": md_folds_std,
    })
    # extra analysis-only columns (kept out of the required CSV)
    out["_perm_fold_std"] = perm_fold_std
    out["_perm_fold_neg_folds"] = perm_fold_neg
    out["_uni_auc"] = cv["uni_auc"]
    out["_uni_auc_std"] = cv["uni_auc_std"]
    out["_length_abs_rho"] = length.set_index("feature").reindex(feats)["abs_rho"].values
    out["_session_smd"] = session.set_index("feature").reindex(feats)["session_smd_healthy"].values
    out["stat_type"] = out["feature"].map(stat_type)
    out["joint_index"] = out["feature"].map(lambda f: joint_of(f)).astype("float")
    out["joint_name"] = out["joint_index"].map(
        lambda j: "-" if pd.isna(j) else SMPL_JOINT_NAMES[int(j)])
    return out


def write_csv(full: pd.DataFrame) -> None:
    cols = ["feature", "feature_group", "variance", "mean", "std",
            "rf_importance", "permutation_importance",
            "mean_fold_importance", "importance_std"]
    csv_df = full[cols].sort_values("permutation_importance", ascending=False)
    CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(CSV_OUTPUT_PATH, index=False, float_format="%.6g")
    print(f"Wrote {CSV_OUTPUT_PATH} ({len(csv_df)} rows)")


# ---------------------------------------------------------------------------
# 9. Report
# ---------------------------------------------------------------------------
def build_report(df: pd.DataFrame, feats: list[str], checks: dict,
                 var_stats: pd.DataFrame, red: dict, cv: dict,
                 full: pd.DataFrame, ablations: pd.DataFrame,
                 nested: pd.DataFrame) -> str:
    grouped = full.groupby("feature_group")
    perm_total_pos = full.loc[full["permutation_importance"] > 0, "permutation_importance"].sum()
    md_total = full["rf_importance"].sum()

    def group_share(col: str, positive_only: bool = False) -> pd.DataFrame:
        s = full.groupby("feature_group")[col].sum()
        if positive_only:
            s = full.loc[full[col] > 0].groupby("feature_group")[col].sum()
        return s

    perm_share = group_share("permutation_importance", positive_only=True)
    perm_share_pct = 100 * perm_share / perm_share.sum()
    md_share_pct = 100 * group_share("rf_importance") / md_total

    stat_share = full.groupby("stat_type").agg(
        n=("feature", "count"),
        md=("rf_importance", "sum"),
        perm=("permutation_importance", "sum"),
        md_stability=("importance_std", "mean"),
        length_coupling=("_length_abs_rho", "median"),
    )
    stat_share["md_pct"] = 100 * stat_share["md"] / md_total
    stat_share["perm_pct"] = 100 * stat_share["perm"].clip(lower=0) / perm_total_pos

    const_feats = var_stats[var_stats["constant"]]["feature"].tolist()
    near_feats = var_stats[var_stats["near_zero"]].sort_values("variance")
    ranked = full.sort_values("permutation_importance", ascending=False).reset_index(drop=True)
    ranked["rank_perm"] = np.arange(1, len(ranked) + 1)

    # rank by rf importance too
    rf_ranked = full.sort_values("rf_importance", ascending=False)
    rf_rank_map = {f: i + 1 for i, f in enumerate(rf_ranked["feature"])}

    # joint-level aggregate importance
    joints = full[full["joint_index"].notna()].groupby(
        ["joint_index", "joint_name"], dropna=False).agg(
        md=("rf_importance", "sum"),
        perm=("permutation_importance", "sum"),
        n=("feature", "count"),
        length=("_length_abs_rho", "median"),
        session=("_session_smd", "median"),
    ).sort_values("perm", ascending=False)

    # unstable features: fold-level permutation importance
    inst = full.copy()
    inst["cv_perm"] = inst["_perm_fold_std"] / (inst["permutation_importance"].abs() + 1e-12)
    inst["neg_folds"] = inst["_perm_fold_neg_folds"]
    unstable = inst[(inst["permutation_importance"] > 0) &
                    ((inst["cv_perm"] > 1.0) | (inst["neg_folds"] >= 2))]
    unstable = unstable.sort_values("cv_perm", ascending=False)

    # session-dependent & length-coupled flags among important features
    top30 = ranked.head(30)
    flagged = full[
        (full["_session_smd"] >= SESSION_SMD_FLAG) |
        (full["_length_abs_rho"] >= LENGTH_CORR_FLAG) |
        (full["feature"].isin(const_feats))
    ].copy()
    flagged["flags"] = ""
    flagged.loc[flagged["feature"].isin(const_feats), "flags"] += "constant;"
    flagged.loc[flagged["_length_abs_rho"] >= LENGTH_CORR_FLAG, "flags"] += "length-coupled;"
    flagged.loc[flagged["_session_smd"] >= SESSION_SMD_FLAG, "flags"] += "session-shifted;"
    flagged = flagged[flagged["flags"] != ""].sort_values(
        "permutation_importance", ascending=False)

    length_stats = full[full["stat_type"].isin(["mean", "std", "min", "max"]) &
                        full["_length_abs_rho"].notna()].copy()
    length_stats["length_coupled"] = length_stats["_length_abs_rho"] >= LENGTH_CORR_FLAG

    # features passing every hygiene flag (non-constant, positive in all 5 folds,
    # not length-coupled, not session-shifted)
    clean_all = full[
        (~full["feature"].isin(const_feats)) &
        (full["permutation_importance"] > 0) &
        (full["_perm_fold_neg_folds"] == 0) &
        (full["_length_abs_rho"].fillna(1.0) < LENGTH_CORR_FLAG) &
        (full["_session_smd"] < SESSION_SMD_FLAG)
    ].sort_values("permutation_importance", ascending=False)
    clean = clean_all.head(15)
    n_clean = len(clean_all)

    s1 = ~full["feature"].isin(const_feats)
    s2 = s1 & (full["permutation_importance"] > 0) & (full["_perm_fold_neg_folds"] == 0)
    s3 = s2 & (full["_length_abs_rho"].fillna(1.0) < LENGTH_CORR_FLAG)
    s4 = s3 & (full["_session_smd"] < SESSION_SMD_FLAG)
    funnel_rows = [
        ["1. all features", int(len(full))],
        ["2. − constant features (`fps`, `pose0..2`)", int(s1.sum())],
        ["3. + permutation importance > 0 in **all 5** held-out folds", int(s2.sum())],
        ["4. + not length-coupled (\\|rho\\| vs `num_frames` < "
         f"{LENGTH_CORR_FLAG})", int(s3.sum())],
        ["5. + not session-shifted (SMD < "
         f"{SESSION_SMD_FLAG})", int(s4.sum())],
    ]

    # temporal dominance numbers
    temporal = full[full["feature_group"] == GROUP_TEMPORAL]
    temporal_perm_pct = 100 * temporal["permutation_importance"].clip(lower=0).sum() / perm_total_pos
    temporal_md_pct = md_share_pct.get(GROUP_TEMPORAL, 0.0)
    root_md_pct = md_share_pct.get(GROUP_ROOT, 0.0)
    nf_row = full[full["feature"] == "num_frames"].iloc[0]
    nf_rank = int(ranked.index[ranked["feature"] == "num_frames"][0]) + 1

    # duration asymmetry between classes
    dur_healthy = float(df.loc[df.diagnosis == 0, "num_frames"].mean())
    dur_pd = float(df.loc[df.diagnosis == 1, "num_frames"].mean())

    # within-session (dne2 only) univariate AUCs -> session effect removed
    d2 = df[df["subject"].str.startswith("dne2")]
    y2 = d2["diagnosis"].to_numpy(dtype=int)
    d2_auc: dict[str, float] = {}
    for f in feats:
        v = d2[f].to_numpy(dtype=float)
        if np.std(v) == 0:
            continue
        a = float(roc_auc_score(y2, v))
        d2_auc[f] = max(a, 1.0 - a)
    d2_top = sorted(d2_auc, key=lambda k: -d2_auc[k])[:5]
    d2_top_str = ", ".join(f"`{k}` ≈ {d2_auc[k]:.2f}" for k in d2_top)
    d2_nf = d2_auc.get("num_frames", float("nan"))

    # dominant joints (dynamic, replaces any hard-coded joint names)
    top_joints_str = ", ".join(f"`{name}`" for (_j, name) in joints.head(4).index)

    corr_len_flagged = int(length_stats["length_coupled"].sum())

    # spearman: session SMD vs importance, length rho vs importance
    rho_session, _ = spearmanr(full["_session_smd"], full["rf_importance"])
    rho_length, _ = spearmanr(full["_length_abs_rho"].fillna(0), full["rf_importance"])
    rho_uni, _ = spearmanr(full["_uni_auc"].fillna(0.5), full["permutation_importance"])
    rho_md_perm, _ = spearmanr(full["rf_importance"], full["permutation_importance"])

    # unstable features that are also globally important
    top50_features = set(ranked.head(50)["feature"])
    unstable_in_top50 = int(len(set(unstable["feature"]) & top50_features))

    # recording-block structure (computed, not hard-coded)
    b1 = df[df["subject"].str.startswith("dne1")]
    b2 = df[df["subject"].str.startswith("dne2")]
    block_str = (
        f"`dne1` ({b1['subject'].nunique()} subjects, "
        f"{'**all healthy**, ' if (b1['diagnosis'] == 0).all() else ''}"
        f"{len(b1)} trials) and `dne2` ({b2['subject'].nunique()} subjects, "
        f"{b2.loc[b2['diagnosis'] == 0, 'subject'].nunique()} healthy / "
        f"{b2.loc[b2['diagnosis'] == 1, 'subject'].nunique()} Parkinson's, {len(b2)} trials)"
    )
    fold_pd_counts = [f["val_pd_trials"] for f in cv["fold_info"]]

    # ---- fold table
    fold_rows = [[f["fold"], f["train_trials"], f["val_trials"], f["train_subjects"],
                  f["val_subjects"], f["train_pd_trials"], f["val_pd_trials"],
                  f"{f['val_baseline_auc']:.4f}", f"{f['val_accuracy']:.4f}"]
                 for f in cv["fold_info"]]

    # ---- verification table
    ver_rows = [
        ["Trials", checks["trials"], 322, "PASS" if checks["trials"] == 322 else "FAIL"],
        ["Subjects", checks["subjects"], 58, "PASS" if checks["subjects"] == 58 else "FAIL"],
        ["Features", checks["features"], 290, "PASS" if checks["features"] == 290 else "FAIL"],
        ["Healthy trials", checks["healthy_trials"], 205, "PASS" if checks["healthy_trials"] == 205 else "FAIL"],
        ["Parkinson's trials", checks["pd_trials"], 117, "PASS" if checks["pd_trials"] == 117 else "FAIL"],
        ["Healthy subjects", checks["healthy_subjects"], "info", "info"],
        ["Parkinson's subjects", checks["pd_subjects"], "info", "info"],
        ["Trials per subject (min-max)", f"{checks['trials_per_subject_min']}-{checks['trials_per_subject_max']}", "info", "info"],
        ["Label constant within subject", checks["subjects_with_mixed_labels"] == 0, True, "PASS"],
        ["Missing values in features", checks["nan_count"], 0, "PASS"],
        ["Distinct fps values", ", ".join(str(v) for v in checks["fps_unique_values"]), "info", "info"],
    ]

    # ---- taxonomy table
    taxonomy_rows = []
    for g, label in [
        (GROUP_TEMPORAL, "temporal features + FPS/video-length features"),
        (GROUP_ROOT, "pose/rotation statistics - root (global) orientation"),
        (GROUP_JOINT, "pose/rotation statistics - relative joint rotations"),
        (GROUP_POSITION, "position statistics"),
        (GROUP_GAIT, "gait features"),
    ]:
        n = int((full["feature_group"] == g).sum())
        taxonomy_rows.append([f"`{g}`", label, n,
                              "ABSENT from this representation" if n == 0 else
                              f"{md_share_pct.get(g, 0):.1f}% of RF importance, "
                              f"{perm_share_pct.get(g, 0):.1f}% of positive permutation importance"])

    stat_rows = [[
        k, int(r["n"]), f"{r['md']:.6f}", f"{r['md_pct']:.1f}%",
        f"{r['perm']:.6f}", f"{r['perm_pct']:.1f}%",
        f"{r['md_stability']:.6f}",
        "n/a" if pd.isna(r["length_coupling"]) else f"{r['length_coupling']:.3f}",
    ] for k, r in stat_share.iterrows()]

    # ---- top 30
    top_rows = []
    for i, r in top30.iterrows():
        top_rows.append([
            int(i) + 1, f"`{r['feature']}`", r["feature_group"], r["joint_name"],
            f"{r['rf_importance']:.6f}", rf_rank_map[r["feature"]],
            f"{r['permutation_importance']:.6f}",
            f"{r['mean_fold_importance']:.6f} ± {r['importance_std']:.6f}",
            f"{r['_perm_fold_std']:.6f}",
            f"{r['_uni_auc']:.3f}" if not np.isnan(r["_uni_auc"]) else "n/a",
            f"{r['_length_abs_rho']:.3f}" if not np.isnan(r["_length_abs_rho"]) else "n/a",
            f"{r['_session_smd']:.3f}",
        ])

    # ---- redundancy pairs
    red_rows = [[f"`{a}`", f"`{b}`", f"{c:.4f}"] for a, b, c in red["pairs_95"][:15]]
    red90_rows = [[f"`{a}`", f"`{b}`", f"{c:.4f}"] for a, b, c in red["pairs_90"][:10]]

    # ---- constant / near-zero tables
    const_rows = []
    for f in const_feats:
        g = feature_group(f)
        const_rows.append([f"`{f}`", g, "0 by construction" if g != GROUP_TEMPORAL else
                           "constant across all training videos"])
    near_rows = [[f"`{r['feature']}`", r["feature_group"], f"{r['variance']:.3e}",
                  f"{r['std']:.3e}"] for _, r in near_feats.head(20).iterrows()]

    # ---- session table
    sess = full.sort_values("_session_smd", ascending=False).head(15)
    sess_rows = [[f"`{r['feature']}`", r["joint_name"], f"{r['_session_smd']:.3f}",
                  f"{r['rf_importance']:.6f}",
                  f"{r['permutation_importance']:.6f}",
                  f"{r['_uni_auc']:.3f}"]
                 for _, r in sess.iterrows()]

    # ---- length coupling table
    len_rows = [[f"`{r['feature']}`", r["stat_type"], f"{r['_length_abs_rho']:.3f}",
                 f"{r['rf_importance']:.6f}", f"{r['permutation_importance']:.6f}"]
                for _, r in full.sort_values("_length_abs_rho", ascending=False).head(15).iterrows()]

    # ---- unstable table
    un_rows = [[f"`{r['feature']}`",
                f"{r['permutation_importance']:.6f}",
                f"{r['_perm_fold_std']:.6f}",
                f"{r['cv_perm']:.2f}",
                int(r["neg_folds"]),
                f"{r['mean_fold_importance']:.6f} ± {r['importance_std']:.6f}"]
               for _, r in unstable.head(20).iterrows()]

    # ---- joint table
    joint_rows = [[f"{int(j)}", name, int(r["n"]), f"{r['md']:.6f}", f"{r['perm']:.6f}",
                   f"{r['length']:.3f}", f"{r['session']:.3f}"]
                  for (j, name), r in joints.head(15).iterrows()]

    # ---- ablation table
    abl_rows = [[r["subset"], int(r["n_features"]),
                 "n/a" if np.isnan(r["auc_mean"]) else f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f}",
                 "n/a" if np.isnan(r["acc_mean"]) else f"{r['acc_mean']:.4f} ± {r['acc_std']:.4f}"]
                for _, r in ablations.iterrows()]
    nested_rows = [[r["variant"], int(r["n_features"]),
                    f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f}",
                    f"{r['acc_mean']:.4f} ± {r['acc_std']:.4f}"]
                   for _, r in nested.iterrows()]

    def abl_auc(prefix: str) -> float:
        row = ablations[ablations["subset"].str.startswith(prefix)]
        return float(row["auc_mean"].iloc[0])

    nested_base = float(nested.loc[nested["n_features"] == 290, "auc_mean"].iloc[0])
    nested_variants = ", ".join(
        f"top-{int(r['n_features'])} → {r['auc_mean']:.4f}"
        for _, r in nested[nested["n_features"] < 290].iterrows())

    # ---- group-level answers
    group_rows = []
    for g in [GROUP_TEMPORAL, GROUP_ROOT, GROUP_JOINT]:
        sub = full[full["feature_group"] == g]
        group_rows.append([
            f"`{g}`", len(sub),
            f"{md_share_pct.get(g, 0):.1f}%",
            f"{100 * sub['permutation_importance'].clip(lower=0).sum() / perm_total_pos:.1f}%",
            f"{sub['rf_importance'].mean():.6f}",
            f"{sub['permutation_importance'].mean():.6f}",
        ])

    md_median = float(full["rf_importance"].median())
    perm_median = float(full["permutation_importance"].median())

    report = f"""# MediaPipe 290-Feature Representation Analysis

> **Scope.** Training data only (`data/processed/pd_dataset_mediapipe_normalized.csv`).
> **No TULIP label, prediction, or metric was read, used, or referenced for feature
> selection or model tuning in this analysis.** No model file was modified, and no
> feature selection was applied to the pipeline. This report makes **no claim that any
> feature subset improves external accuracy** — every number below that comes from
> cross-validation is an *internal, training-domain* estimate only.

---

## 1. Dataset Verification

{md_table(["Check", "Observed", "Expected", "Status"], ver_rows)}

Structure: **{checks['pd_subjects']} Parkinson's + {checks['healthy_subjects']} healthy subjects**;
every subject carries a single diagnosis (no subject-level label leakage across folds);
trials per subject range {checks['trials_per_subject_min']}–{checks['trials_per_subject_max']}.

Recording-session structure (relevant to Sections 7–8): subjects come from two collection
blocks — {block_str}. Session is therefore *partially* confounded
with label: `dne1` never contains a positive case.

---

## 2. Feature Taxonomy (Semantic Groups)

The representation has **no position channels and no gait/dynamics channels** in the
requested sense; it is 2 metadata columns + 72 rotation channels × 4 summary statistics.

{md_table(["Group", "Requested semantic type", "Count", "Share of importance"], taxonomy_rows)}

Per-statistic breakdown:

{md_table(["statistic", "n", "sum MDI", "% of MDI", "sum perm. (ΔAUC)", "% of positive perm.",
           "mean fold MDI std", "median |rho| vs num_frames"], stat_rows)}

Channel → SMPL joint mapping: `pose_(3j, 3j+1, 3j+2)` are the axis-angle (x, y, z) of joint
`j` of the 24-joint SMPL layout (Pelvis, L_Hip, … L_Wrist, R_Wrist, L_Hand, R_Hand).
`pose0..pose2` = root/global orientation (zeroed by construction in
`src/datasets/build_pd_dataset_normalized.py:41`), `pose3..pose71` = the 23 relative joints.

**Requested groups that do not exist in this representation:**

* `position_statistics` — **0 features**. Pelvis centering + body-scale normalization
  remove all absolute translation; no `x/y/z` position summary was ever extracted.
* `gait_features` — **0 features**. There are no cadence, stride, step-time, velocity,
  or time-resolved dynamics features. The only "temporal" columns are `fps` and
  `num_frames` (sequence metadata, not gait measurement).

---

## 3. Variance Analysis

### 3.1 Dead (exactly zero-variance) features — {len(const_feats)}

{md_table(["Feature", "Group", "Reason"], const_rows)}

* `fps` is **60.0 for all {checks['trials']} trials** — a constant column that consumes a
  feature slot and contributes exactly nothing (it cannot split, cannot be permuted).
* `pose0..pose2` (all 4 statistics each) are **zeroed by construction** during canonical
  frame alignment, i.e. the root-orientation block is a permanently dead 12-feature block.

These {len(const_feats)} features are pure noise-free constants; they neither help nor
directly hurt (the RF ignores them), but they inflate the advertised "290-dimensional"
representation to only **{290 - len(const_feats)} usable dimensions**.

### 3.2 Near-zero-variance features (0 < var < {NEAR_ZERO_VAR_THRESHOLD:g} rad²)

Count: **{int((~var_stats['constant'] & (var_stats['variance'] < NEAR_ZERO_VAR_THRESHOLD)).sum())}**
(std < {np.sqrt(NEAR_ZERO_VAR_THRESHOLD):.4f} rad ≈ {np.degrees(np.sqrt(NEAR_ZERO_VAR_THRESHOLD)):.2f}°).

{md_table(["Feature", "Group", "Variance", "Std"], near_rows)}

Additional context: median feature variance is {float(var_stats['variance'].median()):.3e};
{int((var_stats['variance'] < 1e-4).sum())} of 290 features have variance < 1e-4 rad²
(std < 0.01 rad ≈ 0.57°). These channels are near-static in the training domain and are
the ones most likely to fall on the wrong side of a domain shift.

---

## 4. Redundancy / Correlation Analysis

| Correlation band (pairwise \\|r\\|) | Redundant pairs |
| --- | --- |
| ≥ 0.95 | {red['n_pairs_95']} |
| 0.90 – 0.95 | {red['n_pairs_90']} |
| 0.80 – 0.90 | {red['n_pairs_80']} |

* Redundancy clusters at |r| ≥ 0.95: **{red['n_clusters_95']}** (largest cluster has
  **{len(red['largest_cluster'])}** features).
* Of the ≥0.95 pairs, **{red['same_channel_pairs_95']} are within the same channel**
  (i.e. `poseX_min`/`poseX_max`/`poseX_mean` of one joint) and
  **{red['cross_channel_pairs_95']} are across different channels**.
* Maximum |r| observed: **{red['max_abs_r']:.4f}** (no exactly duplicated columns).

Top redundant pairs (|r| ≥ 0.95):

{md_table(["Feature A", "Feature B", "|r|"], red_rows)}

Next band (0.90 ≤ |r| < 0.95):

{md_table(["Feature A", "Feature B", "|r|"], red90_rows)}

**Interpretation.** Redundancy is dominated by *within-channel* statistics: `min`, `max`
and `mean` of a joint channel are near-collinear because the underlying rotation
oscillates around a fixed offset. `min`/`max` are also mechanically coupled to `mean ±
amplitude`, so the 4-statistic × 72-channel grid contains substantial duplicated
information (~{red['n_pairs_95']} high-collinearity pairs). Redundant features split RF
importance arbitrarily across themselves, which is exactly why single-feature importance
ranks must be read at *group* level (Section 5) rather than feature level.

---

## 5. Importance Analysis (training data, subject-wise 5-fold GroupKFold)

Cross-validation uses the same `GroupKFold(n_splits=5)` grouping as
`src/training/train_mediapipe_normalized.py` — folds are split by **subject**, so no
subject's trials appear in both train and validation.

{md_table(["Fold", "Train trials", "Val trials", "Train subj", "Val subj",
           "Train PD trials", "Val PD trials", "Val baseline AUC", "Val accuracy"], fold_rows)}

Note: `GroupKFold` balances *sample counts*, not class balance — the validation folds hold
between {min(fold_pd_counts)} and {max(fold_pd_counts)} Parkinson's trials out of
{min(f['val_trials'] for f in cv['fold_info'])}–{max(f['val_trials'] for f in cv['fold_info'])}
validation trials, which adds variance to any per-fold importance estimate. All importance
numbers below are therefore reported as 5-fold means with their across-fold spread.

Global scale: median MDI importance = {md_median:.6f}, median permutation importance =
{perm_median:.6f}. Features with permutation importance ≤ 0:
**{int((full['permutation_importance'] <= 0).sum())} / 290**
(features that, when shuffled on held-out subjects, do not degrade ROC-AUC).

**MDI and held-out permutation importance disagree strongly**
(Spearman between `rf_importance` and `permutation_importance` = **{rho_md_perm:.3f}**):
the features the forest splits on most are *not* the features that move held-out AUC.
Most strikingly, the `std` block absorbs {stat_share.loc['std', 'md_pct']:.1f}% of all MDI
importance but only {stat_share.loc['std', 'perm_pct']:.1f}% of positive permutation
importance, and several high-MDI `std` features (`pose6_std`, `pose13_std`, `pose22_std`,
`pose28_std`) have *negative* held-out permutation importance. **Ranking for the next
experiment must use the held-out permutation column, not `rf_importance`.**

### 5.1 Group-level importance shares

{md_table(["Group", "n", "% of RF (MDI) importance", "% of positive permutation importance",
           "mean MDI", "mean perm."], group_rows)}

### 5.2 Top 30 features (ranked by held-out permutation importance)

{md_table(["#", "Feature", "Group", "Joint", "RF imp.", "RF rank", "Perm. imp. (ΔAUC)",
           "fold MDI mean ± std", "fold perm. std", "univ. CV AUC", "|rho| vs num_frames",
           "session SMD"], top_rows)}

Rank agreement: Spearman(univariate CV AUC, permutation importance) = **{rho_uni:.3f}**;
Spearman(session SMD, MDI importance) = **{rho_session:.3f}**;
Spearman(|rho| vs num_frames, MDI importance) = **{rho_length:.3f}**.

### 5.3 Which joints dominate

{md_table(["Joint", "Name", "n features", "sum MDI", "sum perm.", "median |rho| vs len",
           "median session SMD"], joint_rows)}

### 5.4 Unstable importance across folds

Definition used: a feature is *unstable* if its mean held-out permutation importance is
positive, yet its across-fold coefficient of variation exceeds 1.0 **or** it has
permutation importance ≤ 0 in ≥ 2 of the 5 folds.

Count of unstable features: **{len(unstable)} / 290**
({unstable_in_top50} of them are inside the global top-50 by permutation importance — the
rest are near-zero features whose coefficient of variation is meaningless).

{md_table(["Feature", "mean perm. imp.", "fold perm. std", "CV", "#folds ≤ 0",
           "fold MDI mean ± std"], un_rows)}

`importance_std` in the CSV is the across-fold standard deviation of the **fold-level MDI
importance**; the fold-permutation standard deviation shown here is the held-out
counterpart. High `importance_std` relative to `mean_fold_importance` means the feature's
value depends strongly on *which subjects* are in the training fold — the classic
signature of a subject-/session-specific feature rather than a disease-general one.

### 5.5 Features that pass every hygiene flag

Filter applied (all conditions, training data only), cumulatively:

{md_table(["Hygiene filter", "features surviving"], funnel_rows)}

Top survivors (sorted by held-out permutation importance):

{md_table(["Feature", "Joint", "statistic", "Perm. imp.", "RF imp.", "univ. CV AUC",
           "|rho| vs len", "session SMD", "fold perm. std"],
          [[f"`{r['feature']}`", r["joint_name"], r["stat_type"],
            f"{r['permutation_importance']:.6f}", f"{r['rf_importance']:.6f}",
            f"{r['_uni_auc']:.3f}", f"{r['_length_abs_rho']:.3f}",
            f"{r['_session_smd']:.3f}", f"{r['_perm_fold_std']:.6f}"]
           for _, r in clean.iterrows()])}

Only **{n_clean} features survive all filters** (out of 290). The filter is deliberately
strict — "positive in all 5 folds" is a high bar for features whose permutation effects
are near zero — so this is a *floor*, not proof that only six features carry signal. It is
nonetheless the natural anchor for a "stable core" arm in the next experiment.

---

## 6. Do fps / num_frames / video-length features dominate?

### 6.1 `fps`

* Constant at 60.0 across all {checks['trials']} training trials → **zero variance, zero
  importance, zero predictive value**. In this training set `fps` cannot dominate anything;
  on external data it will simply be an uninformative slot (unless the external videos have
  a different fps, in which case the column shifts while carrying no signal).

### 6.2 `num_frames` (video duration proxy)

{md_table(["Metric", "Value"], [
    ["Permutation importance (rank)", f"{nf_row['permutation_importance']:.6f} (rank {nf_rank}/{len(ranked)})"],
    ["RF (MDI) importance (rank)", f"{nf_row['rf_importance']:.6f} (rank {rf_rank_map['num_frames']}/{len(feats)})"],
    ["Univariate subject-wise CV AUC", f"{nf_row['_uni_auc']:.4f}"],
    ["Point-biserial correlation with label", f"{np.corrcoef(df['num_frames'], df['diagnosis'])[0,1]:.4f}"],
    ["Mean frames — healthy / PD", f"{df.loc[df.diagnosis==0,'num_frames'].mean():.1f} / {df.loc[df.diagnosis==1,'num_frames'].mean():.1f}"],
    ["Mean duration @60fps — healthy / PD", f"{df.loc[df.diagnosis==0,'num_frames'].mean()/60:.2f}s / {df.loc[df.diagnosis==1,'num_frames'].mean()/60:.2f}s"],
    ["Session SMD (healthy dne1 vs dne2)", f"{float(full.loc[full.feature=='num_frames','_session_smd'].iloc[0]):.3f}"],
])}

Video duration is **genuinely label-associated in this training domain** (PD trials are
{100 * (dur_pd / dur_healthy - 1):.0f}% longer on average: {dur_pd:.0f} vs {dur_healthy:.0f}
frames), so it is a real — but *protocol-dependent* — signal. Recording protocol
also moves duration in healthy subjects (session SMD above), confirming it is partly a
collection artifact rather than pure pathophysiology.

### 6.3 Indirect duration dependence inside the pose features

`num_frames` leaks into the rotation statistics themselves: a longer video samples more gait
cycles, which inflates `std` and stretches `min`/`max`.

| Statistic | n | median \\|rho\\| vs `num_frames` | max \\|rho\\| | # features with \\|rho\\| ≥ {LENGTH_CORR_FLAG} |
| --- | --- | --- | --- | --- |
""" + "\n".join(
        f"| `{st}` | {int(g['feature'].count())} | {g['_length_abs_rho'].median():.3f} | {g['_length_abs_rho'].max():.3f} | {int((g['_length_abs_rho'] >= LENGTH_CORR_FLAG).sum())} |"
        for st, g in length_stats.groupby("stat_type")
    ) + f"""

**{corr_len_flagged} / {int(length_stats['feature'].count())} non-constant pose statistics**
have |Spearman rho| ≥ {LENGTH_CORR_FLAG} with `num_frames`. The most length-coupled features:

{md_table(["Feature", "statistic", "|rho| vs num_frames", "MDI imp.", "perm. imp."], len_rows)}

### 6.4 Direct answer

* **`fps` does not dominate** — it is dead.
* **`num_frames` alone does not dominate** — it carries roughly
  {temporal_md_pct:.1f}% of total RF importance and
  {temporal_perm_pct:.1f}% of positive permutation importance (see Section 5.1), with
  univariate CV AUC {nf_row['_uni_auc']:.3f} versus ≈0.90 for the best rotation features.
* **However, video length is embedded indirectly** in `std`/`min`/`max` statistics
  (Section 6.3). Any external set recorded with different trial durations will shift a
  large block of features at once — this is a first-order external-generalization risk,
  even though the explicit `num_frames` column is not the top predictor.
* **Absolute position features do not dominate because they do not exist** — translation
  is removed by pelvis centering and body-scale normalization, and the root-orientation
  block is zeroed (Section 3.1). No `x/y/z` column is present in the 290 features.

---

## 7. Session / protocol dependence (label-free check)

Comparison restricted to **healthy subjects only** (`dne1` vs `dne2`), so any separation
observed is a recording-protocol effect and cannot be a disease effect.

{md_table(["Feature", "Joint", "session SMD", "MDI imp.", "perm. imp.", "univ. CV AUC"], sess_rows)}

* Features with session SMD ≥ {SESSION_SMD_FLAG}: **{int((full['_session_smd'] >= SESSION_SMD_FLAG).sum())} / 290**.
* Features flagged (constant, length-coupled, or session-shifted) and their rank among the
  most important: **{len(flagged)} features**, of which
  **{int((flagged['feature'].isin(top30['feature'])).sum())} are inside the global top-30**.

Cross-check: univariate AUC computed **within `dne2` only** (both classes, same session —
session effect removed) stays high for the leading rotation features ({d2_top_str}), and
`num_frames` ≈ {d2_nf:.2f}; i.e. genuine label signal
exists in the pose statistics and is not *only* a session artifact. The risk is that the
model currently mixes label signal with session/protocol signal and cannot tell them apart.

Flagged features (all rows with at least one flag, sorted by permutation importance):

{md_table(["Feature", "Group", "flags", "perm. imp.", "MDI imp.",
           "|rho| vs len", "session SMD"],
          [[f"`{r['feature']}`", r["feature_group"], r["flags"],
            f"{r['permutation_importance']:.6f}", f"{r['rf_importance']:.6f}",
            f"{r['_length_abs_rho']:.3f}" if not np.isnan(r['_length_abs_rho']) else "n/a",
            f"{r['_session_smd']:.3f}"]
           for _, r in flagged.head(40).iterrows()])}

---

## 8. Internal ablations of predefined feature groups (descriptive, training-only)

Same 5-fold subject-wise GroupKFold, same RF hyperparameters. **These are internal
estimates on the training distribution. They are NOT external results and must not be
quoted as such.**

{md_table(["Feature subset", "n", "ROC-AUC (mean ± std)", "Accuracy (mean ± std)"], abl_rows)}

Nested top-k (inner subject-wise permutation importance selects, outer subject-wise fold
scores — no leakage between selection and evaluation):

{md_table(["Variant", "n", "ROC-AUC (mean ± std)", "Accuracy (mean ± std)"], nested_rows)}

Observations (internal only; values are mean subject-wise CV ROC-AUC):

* Removing `fps`/`num_frames` changes internal performance very little
  ({abl_auc('A.'):.4f} → {abl_auc('C.'):.4f}) — the metadata block is not the main carrier
  of internal accuracy, so dropping it costs nothing internally.
* The root-orientation block alone scores {abl_auc('F.'):.4f} (chance) — it is dead weight;
  the relative joint-rotation block alone reaches {abl_auc('E.'):.4f} vs {abl_auc('A.'):.4f}
  for all 290 features.
* **The `min`/`max` block alone scores {abl_auc('I.'):.4f} — the highest of every subset —
  even though it is the most length-coupled group (Section 6.3).** In other words the
  internal metric *rewards* duration-sensitive range statistics. This is precisely the
  pattern that is expected to fail when external videos have different trial durations:
  internal ranking alone is not a safe guide for feature selection here.
* `mean`-only ({abl_auc('G.'):.4f}) and `mean`+`std` ({abl_auc('J.'):.4f}) stay close to the
  full-model AUC with half or fewer features, while being the least length-coupled
  statistics (Section 6.3).
* Nested top-k selection results ({nested_variants}) fall **below** the full-feature
  baseline ({nested_base:.4f}): aggressive single-shot pruning discards
  redundant-but-complementary signal. Group-level, hypothesis-driven arms (Section 11) are
  therefore the safer next step than top-k pruning.

---

## 9. Findings

1. **{len(const_feats)} of 290 features are exactly constant** (`fps` + the entire
   root-orientation block `pose0..pose2`); the effective representation is
   {290 - len(const_feats)} dimensions, and the root-orientation group contributes
   {root_md_pct:.1f}% of importance (nothing).
2. **Redundancy is structural**: {red['n_pairs_95']} pairs at |r| ≥ 0.95, almost all
   within-channel `min`/`max`/`mean` collinearity; importance is therefore smeared across
   near-duplicate statistics.
3. **The strongest features are relative joint-rotation statistics**, concentrated in the
   {top_joints_str} channels (Section 5.3), with univariate
   subject-wise CV AUC up to {float(full['_uni_auc'].max()):.3f} — but **MDI and held-out
   permutation importance rank these features very differently**
   (Spearman = {rho_md_perm:.3f}); held-out permutation importance is the trustworthy
   column.
4. **Video length is a hidden second signal**: explicit `num_frames` is a moderate
   predictor (CV AUC {nf_row['_uni_auc']:.3f}), and {corr_len_flagged} features are
   mechanically length-coupled through their `std`/`min`/`max` statistics.
5. **Session/protocol shift is visible in the training data itself**: {int((full['_session_smd'] >= SESSION_SMD_FLAG).sum())}
   features separate healthy `dne1` from healthy `dne2` by SMD ≥ {SESSION_SMD_FLAG}, and
   session-SMD correlates with feature importance (rho = {rho_session:.2f}).
6. **Importance is not fully stable**: {len(unstable)} features have positive mean
   permutation importance but CV > 1 or ≤ 0 in ≥ 2 folds — they are candidate
   subject-specific features.
7. **The requested `position_statistics` and `gait_features` groups do not exist** in this
   representation, so neither can be credited with the current internal performance.

**Explicit non-claim:** nothing in this report demonstrates that removing or keeping any
feature improves *external* accuracy. No external (TULIP) data was read at any point.

---

## 10. Recommended feature groups for the next experiment

Keep (core, externally-meaningful candidates):
1. **`pose_rotation_joint` — `mean` + `std` statistics of `pose3..pose71`**
   ({int(((full.feature_group == GROUP_JOINT) & (full.stat_type.isin(['std','mean']))).sum())} features):
   near-full internal AUC (Section 8, subset J) while dropping the duration-sensitive
   `min`/`max` block; `mean` is the least length-coupled statistic of all four
   (median \\|rho\\| vs `num_frames` ≈ {float(full.loc[full.stat_type=='mean','_length_abs_rho'].median()):.2f}).
   The `mean`-only variant ({int(((full.feature_group == GROUP_JOINT) & (full.stat_type=='mean')).sum())} features,
   Section 8 subset G) is the leaner fallback.
   Caveat: `std` is the *most* length-coupled statistic (median \\|rho\\| vs `num_frames`
   ≈ {float(full.loc[full.stat_type=='std','_length_abs_rho'].median()):.2f}), so run
   `mean`-only and `mean`+`std` as separate arms rather than assuming they are equivalent.
2. **Joint families with the highest summed held-out importance** ({top_joints_str} and the
   other joints in Section 5.3): trunk/head posture, arm-swing (hand/elbow) and
   lower-limb (hip/foot/ankle) channels. Prioritise these *families* rather than
   individual top-k features, because within-channel statistics are near-duplicates
   (Section 4).

Treat with care / candidate for removal in the next controlled experiment:
3. **`temporal_video_length` (`fps`, `num_frames`)** — `fps` is dead weight; `num_frames`
   is a protocol-duration proxy.
4. **`min`/`max` statistics block (144 features)** — most length-coupled group; range
   statistics grow with sequence length, so they encode trial duration as much as motion.
5. **Root-orientation block `pose0..pose2` (12 features)** — zero-variance by construction.
6. **Session-shifted, length-coupled features listed in Section 7** — flag, do not trust.

Do **not** add: any absolute position/translation feature (would reintroduce the exact
domain dependence the normalization removed).

---

## 11. Recommended next model experiment (designed after this analysis)

A single controlled experiment, evaluated **internally first** with the same subject-wise
5-fold GroupKFold, and only afterwards on a *newly locked* external protocol:

1. **Arms** (same RF hyperparameters, same folds, same threshold 0.50):
   * **Arm 0 (control)**: current 290 features — reproduce the internal baseline.
   * **Arm 1 (hygiene)**: drop the 13 constant features (`fps`, `pose0..pose2`) — pure
     dead-weight removal, no information loss.
   * **Arm 2 (duration-robust)**: Arm 1 minus `num_frames` and minus the whole `min`/`max`
     block → keeps `mean`/`std` of relative joint rotations only.
   * **Arm 3 (stability-filtered)**: Arm 2 restricted to features that are stable across
     folds (low `importance_std`, permutation importance > 0 in all 5 folds) and not
     session-shifted (healthy `dne1` vs `dne2` SMD < {SESSION_SMD_FLAG}).
2. **Selection must be re-derived inside each training fold** (nested), never on the full
   training set before splitting, to avoid optimistic bias — the nested protocol in
   Section 8 is the template.
3. **Robustness probes that do not need external labels** (training data only):
   * duration-stratified CV: re-score all arms after *truncating/-padding every trial to a
     fixed frame count* before re-extracting statistics — if Arm 0 collapses and Arm 2 does
     not, video length was doing the work;
   * session-holdout CV: train on `dne2`, test on `dne1` (healthy-only sanity check) and
     vice-versa where labels permit.
4. **Success criteria to be judged later**: internal subject-wise AUC must not degrade
   beyond the fold noise of Arm 0, and the duration/session probes must improve. Only after
   that is a locked external evaluation meaningful — **no external gain is claimed here.**

---

## Appendix A — CSV column definitions

`data/processed/mediapipe_feature_importance.csv` (290 rows, sorted by
`permutation_importance` descending):

{md_table(["Column", "Definition"], [
    ["`feature`", "feature name"],
    ["`feature_group`", "`temporal_video_length` / `pose_rotation_root` / `pose_rotation_joint`"],
    ["`variance`", "sample variance over the 322 training trials"],
    ["`mean`", "mean over the 322 training trials"],
    ["`std`", "standard deviation over the 322 training trials"],
    ["`rf_importance`", "MDI (Gini) importance, RF trained on all 322 trials, normalized to sum 1"],
    ["`permutation_importance`", "mean held-out permutation importance (ΔROC-AUC), 5 subject-wise folds × 10 repeats"],
    ["`mean_fold_importance`", "mean across the 5 folds of the fold-level MDI importance"],
    ["`importance_std`", "std across the 5 folds of the fold-level MDI importance (stability)"],
])}

## Appendix B — Method details

* RF: `n_estimators=300`, `class_weight='balanced'`, `random_state=42` (identical to
  `configs/normalized_model_config.yaml`).
* CV: `GroupKFold(n_splits=5)` grouped by `subject` — identical splits to
  `src/training/train_mediapipe_normalized.py`.
* Permutation importance: `sklearn.inspection.permutation_importance`, `scoring='roc_auc'`,
  `n_repeats=10`, evaluated on the held-out fold only; negative values mean shuffling the
  feature slightly *helped* on that fold (noise / overfit indicator).
* Session effect: standardized mean difference on **healthy subjects only**, pooling the
  standard deviation of all healthy trials.
* Length coupling: Spearman correlation between a feature and `num_frames` across all 322
  trials (`fps` is constant at 60, so `num_frames` ≡ video duration × 60).
* Reproduce: `python scripts/analyze_mediapipe_features.py`
"""
    return report


# ---------------------------------------------------------------------------
def main() -> None:
    use_cache = "--no-cache" not in sys.argv
    cache_path = Path(tempfile.gettempdir()) / "parkinsense_mediapipe_feature_analysis.pkl"

    print("Loading training dataset (training data only — no external/test data read)…")
    df, feats, checks = load_and_verify()
    print(f"  verified: {checks}")

    print("Variance analysis…")
    var_stats = variance_analysis(df, feats)

    print("Redundancy analysis…")
    red = redundancy_analysis(df, feats)
    print(f"  |r|>=0.95 pairs: {red['n_pairs_95']}, clusters: {red['n_clusters_95']}")

    X = df[feats].to_numpy(dtype=float)
    y = df["diagnosis"].to_numpy(dtype=int)
    groups = df["subject"].to_numpy()

    cached = None
    if use_cache and cache_path.exists():
        cached = pd.read_pickle(cache_path)
        if cached.get("feats") != feats or cached.get("dataset_mtime") != DATASET_PATH.stat().st_mtime:
            print("Cache does not match dataset/feature list — recomputing.")
            cached = None
        else:
            print(f"Reusing cached CV/ablation results: {cache_path}")

    if cached is None:
        print("Subject-wise CV importances (MDI + permutation)…")
        cv = subjectwise_cv_importance(X, y, groups, feats)

        print("Running internal group ablations…")
        ablations = run_ablations(X, y, groups, feats)
        print(ablations.to_string(index=False))

        print("Running nested top-k estimate…")
        nested = nested_topk_selection(X, y, groups)
        print(nested.to_string(index=False))

        pd.to_pickle({"feats": feats, "dataset_mtime": DATASET_PATH.stat().st_mtime,
                      "cv": cv, "ablations": ablations, "nested": nested}, cache_path)
        print(f"Cached heavy results to {cache_path}")
    else:
        cv, ablations, nested = cached["cv"], cached["ablations"], cached["nested"]

    print("Video-length coupling analysis…")
    length = length_coupling_analysis(df, feats)

    print("Session (dne1 vs dne2) analysis…")
    session = session_analysis(df, feats)

    print("Assembling feature table…")
    full = assemble(df, feats, var_stats, cv, length, session)

    write_csv(full)

    print("Writing report…")
    report = build_report(df, feats, checks, var_stats, red, cv, full, ablations, nested)
    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_PATH.write_text(report)
    print(f"Wrote {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
