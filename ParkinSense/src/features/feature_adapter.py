"""
ParkinSense — Feature Adapter
==============================
Adapts and validates feature vectors from both MediaPipe and 4D-Humans
pipelines into the standardized, model-ready 290-dimensional representation.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from src.utils import get_logger
from src.features.feature_contract import (
    FEATURE_CONTRACT_COLUMNS,
    EXPECTED_FEATURE_COUNT,
    FeatureContractError,
    validate_feature_vector,
    conform_to_feature_contract,
)

logger = get_logger(__name__)


class FeatureAdapter:
    """
    Standardizes feature outputs from different gait extractors to the shared model contract.
    """

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict

    def adapt(
        self,
        feature_df: pd.DataFrame,
        source_pipeline: str = "unknown",
    ) -> pd.DataFrame:
        """
        Adapt feature dataframe from a given pipeline to the common model contract.

        Args:
            feature_df: Raw extracted feature DataFrame.
            source_pipeline: Name of the originating extractor ('mediapipe' or 'fourd_humans').

        Returns:
            Validated, aligned DataFrame with exactly 290 columns.
        """
        logger.info(
            "FeatureAdapter: adapting feature vector from pipeline '%s' (input shape: %s)...",
            source_pipeline,
            getattr(feature_df, "shape", None),
        )

        try:
            aligned_df = conform_to_feature_contract(feature_df)
        except FeatureContractError as err:
            logger.error(
                "FeatureAdapter validation failure for pipeline '%s': %s",
                source_pipeline,
                err,
            )
            raise

        logger.info(
            "FeatureAdapter: successfully verified 290 features for pipeline '%s'.",
            source_pipeline,
        )
        return aligned_df
