"""
ParkinSense — Backend Adapter Facade
====================================
Exposes FeatureAdapter and FeatureContract utilities to the backend.
"""

from __future__ import annotations

from src.features.feature_contract import (
    FEATURE_CONTRACT_COLUMNS,
    EXPECTED_FEATURE_COUNT,
    FeatureContractError,
    validate_feature_vector,
    conform_to_feature_contract,
)
from src.features.feature_adapter import FeatureAdapter

__all__ = [
    "FEATURE_CONTRACT_COLUMNS",
    "EXPECTED_FEATURE_COUNT",
    "FeatureContractError",
    "validate_feature_vector",
    "conform_to_feature_contract",
    "FeatureAdapter",
]
