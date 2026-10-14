"""
ParkinSense — Common Feature Contract & Validator
=================================================
Defines the strict 290-dimensional feature contract shared between
MediaPipe and 4D-Humans gait-extraction pipelines and required by
the central Parkinson's Disease and UPDRS severity classifiers.

Contract Specifications:
- Exact Dimension: 290 features (1 row DataFrame)
- Global metadata features (2):
    fps (frames per second, positive float)
    num_frames (total analyzed frames, positive int)
- Joint kinematics summary statistics (72 channels × 4 stats = 288):
    Channels 0..2:   Pelvis root rotation (global orientation axis-angles)
    Channels 3..71:  23 SMPL body joints (3 axis-angles per joint)
    Stats per channel: mean, std, min, max
- Data integrity: Finite float64, zero NaN, zero Inf.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

EXPECTED_FEATURE_COUNT = 290

# Build exact 290 feature column list
_FEATURE_COLUMNS: List[str] = ["fps", "num_frames"]
for _i in range(72):
    _FEATURE_COLUMNS.extend([
        f"pose{_i}_mean",
        f"pose{_i}_std",
        f"pose{_i}_min",
        f"pose{_i}_max",
    ])
FEATURE_CONTRACT_COLUMNS: Tuple[str, ...] = tuple(_FEATURE_COLUMNS)
assert len(FEATURE_CONTRACT_COLUMNS) == EXPECTED_FEATURE_COUNT


class FeatureContractError(ValueError):
    """Raised when a feature vector violates the 290-dim feature contract."""
    pass


def validate_feature_vector(feature_df: pd.DataFrame, strict_order: bool = True) -> None:
    """
    Validate that feature_df strictly conforms to the ParkinSense 290-dim contract.

    Args:
        feature_df: DataFrame containing the extracted features (expected shape: (1, 290)).
        strict_order: Whether to enforce the exact feature column ordering.

    Raises:
        FeatureContractError: If any contract requirement is violated.
    """
    if not isinstance(feature_df, pd.DataFrame):
        raise FeatureContractError(
            f"Expected pandas DataFrame for feature vector, got {type(feature_df).__name__}."
        )

    if feature_df.empty:
        raise FeatureContractError("Feature DataFrame is empty.")

    if feature_df.shape[0] != 1:
        raise FeatureContractError(
            f"Expected exactly 1 row for feature inference vector, got {feature_df.shape[0]} rows."
        )

    n_cols = feature_df.shape[1]
    if n_cols != EXPECTED_FEATURE_COUNT:
        raise FeatureContractError(
            f"Feature count mismatch: got {n_cols} columns, expected exactly {EXPECTED_FEATURE_COUNT}."
        )

    # Check for missing columns
    present_cols = set(feature_df.columns)
    expected_cols = set(FEATURE_CONTRACT_COLUMNS)
    missing = expected_cols - present_cols
    if missing:
        sample_missing = sorted(list(missing))[:5]
        raise FeatureContractError(
            f"Missing {len(missing)} required features from contract. Sample missing: {sample_missing}"
        )

    # Check extra columns
    extra = present_cols - expected_cols
    if extra:
        sample_extra = sorted(list(extra))[:5]
        raise FeatureContractError(
            f"Encountered {len(extra)} unexpected extra columns in feature vector. Sample: {sample_extra}"
        )

    # Strict order check
    if strict_order:
        current_cols = tuple(feature_df.columns)
        if current_cols != FEATURE_CONTRACT_COLUMNS:
            mismatches = [
                f"Index {idx}: expected '{exp}', got '{act}'"
                for idx, (exp, act) in enumerate(zip(FEATURE_CONTRACT_COLUMNS, current_cols))
                if exp != act
            ][:3]
            raise FeatureContractError(
                f"Feature ordering mismatch against contract. First mismatches: {mismatches}"
            )

    # Check finite values
    arr = feature_df.values
    n_nan = int(np.sum(np.isnan(arr)))
    n_inf = int(np.sum(np.isinf(arr)))
    if n_nan > 0 or n_inf > 0:
        nan_cols = [col for col in feature_df.columns if feature_df[col].isna().any()]
        raise FeatureContractError(
            f"Feature vector contains non-finite values ({n_nan} NaN, {n_inf} Inf). "
            f"Affected columns: {nan_cols[:5]}"
        )

    # Physical / domain sanity checks
    fps_val = float(feature_df["fps"].iloc[0])
    num_frames_val = int(feature_df["num_frames"].iloc[0])

    if fps_val <= 0 or fps_val > 240:
        raise FeatureContractError(f"Invalid video FPS value {fps_val:.2f} (must be between 1 and 240).")

    if num_frames_val < 1:
        raise FeatureContractError(f"Invalid frame count {num_frames_val} (must be >= 1).")


def conform_to_feature_contract(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Realign and validate a DataFrame to strictly match the 290-dim contract.

    Args:
        feature_df: Input DataFrame containing the required columns.

    Returns:
        DataFrame strictly reordered and validated against FEATURE_CONTRACT_COLUMNS.
    """
    if not isinstance(feature_df, pd.DataFrame):
        raise FeatureContractError(
            f"Expected pandas DataFrame for feature vector, got {type(feature_df).__name__}."
        )

    present_cols = set(feature_df.columns)
    expected_cols = set(FEATURE_CONTRACT_COLUMNS)
    missing = expected_cols - present_cols
    if missing:
        sample_missing = sorted(list(missing))[:5]
        raise FeatureContractError(
            f"Cannot conform to feature contract: missing {len(missing)} features. Sample: {sample_missing}"
        )

    # Align column order exactly
    aligned_df = feature_df[list(FEATURE_CONTRACT_COLUMNS)].copy()
    validate_feature_vector(aligned_df, strict_order=True)
    return aligned_df
