"""
ParkinSense — Feature Domain Shift Audit Script
==============================================
Calculates standardized mean differences (D_j) and scale ratios (R_j) between the
training dataset features and TULIP features (both before and after domain normalization).
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)


def compute_domain_shift_metrics(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    epsilon: float = 1e-6,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute per-feature standardized mean difference D_j and scale ratio R_j.
    """
    rows = []
    for col in feature_cols:
        mu_train = float(df_train[col].mean())
        sigma_train = float(df_train[col].std())
        mu_test = float(df_test[col].mean())
        sigma_test = float(df_test[col].std())

        D_j = abs(mu_test - mu_train) / (sigma_train + epsilon)
        R_j = sigma_test / (sigma_train + epsilon)

        rows.append({
            "feature_name": col,
            "mu_train": round(mu_train, 6),
            "sigma_train": round(sigma_train, 6),
            "mu_test": round(mu_test, 6),
            "sigma_test": round(sigma_test, 6),
            "standardized_mean_diff_D": round(D_j, 4),
            "scale_ratio_R": round(R_j, 4),
        })

    df_metrics = pd.DataFrame(rows)

    D_vals = df_metrics["standardized_mean_diff_D"].values
    R_vals = df_metrics["scale_ratio_R"].values

    summary = {
        "num_features": len(feature_cols),
        "median_D": round(float(np.median(D_vals)), 4),
        "p90_D": round(float(np.percentile(D_vals, 90)), 4),
        "count_D_gt_2": int(np.sum(D_vals > 2.0)),
        "count_D_gt_3": int(np.sum(D_vals > 3.0)),
        "count_R_gt_5": int(np.sum(R_vals > 5.0)),
        "count_R_lt_0_2": int(np.sum(R_vals < 0.2)),
    }

    return df_metrics, summary


def run_feature_domain_audit():
    import joblib

    train_path = PROJECT_ROOT / "data" / "processed" / "pd_dataset.csv"
    output_dir = PROJECT_ROOT / "data" / "processed" / "TULIP"
    ensure_dir(output_dir)

    model_path = PROJECT_ROOT / "models" / "pd_detector.pkl"
    model = joblib.load(model_path)
    feature_cols = list(model.feature_names_in_)

    df_train = pd.read_csv(train_path)[feature_cols]

    print("Loaded training features:", df_train.shape)
    print("Run this after extracting raw and normalized TULIP features.")


if __name__ == "__main__":
    run_feature_domain_audit()
