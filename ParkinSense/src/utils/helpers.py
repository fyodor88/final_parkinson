"""
ParkiSense — Shared Utilities
==============================
Centralised logging factory, path resolution helpers, and geometric
math used across every module in the pipeline.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants — MediaPipe landmark indices
# ---------------------------------------------------------------------------

LANDMARK_NAMES: List[str] = [
    "NOSE",          # 0
    "LEFT_EYE_INNER",# 1
    "LEFT_EYE",      # 2
    "LEFT_EYE_OUTER",# 3
    "RIGHT_EYE_INNER",# 4
    "RIGHT_EYE",     # 5
    "RIGHT_EYE_OUTER",# 6
    "LEFT_EAR",      # 7
    "RIGHT_EAR",     # 8
    "MOUTH_LEFT",    # 9
    "MOUTH_RIGHT",   # 10
    "LEFT_SHOULDER", # 11
    "RIGHT_SHOULDER",# 12
    "LEFT_ELBOW",    # 13
    "RIGHT_ELBOW",   # 14
    "LEFT_WRIST",    # 15
    "RIGHT_WRIST",   # 16
    "LEFT_PINKY",    # 17
    "RIGHT_PINKY",   # 18
    "LEFT_INDEX",    # 19
    "RIGHT_INDEX",   # 20
    "LEFT_THUMB",    # 21
    "RIGHT_THUMB",   # 22
    "LEFT_HIP",      # 23
    "RIGHT_HIP",     # 24
    "LEFT_KNEE",     # 25
    "RIGHT_KNEE",    # 26
    "LEFT_ANKLE",    # 27
    "RIGHT_ANKLE",   # 28
    "LEFT_HEEL",     # 29
    "RIGHT_HEEL",    # 30
    "LEFT_FOOT_INDEX",# 31
    "RIGHT_FOOT_INDEX",# 32
]

# Convenient integer constants for the joints used in gait feature extraction
LM = {name: idx for idx, name in enumerate(LANDMARK_NAMES)}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger with a consistent format.

    Args:
        name:  Module ``__name__``.
        level: Logging level (default INFO).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def project_root() -> Path:
    """Return the absolute path to the ParkiSense project root."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: str | Path) -> Path:
    """
    Create *path* (including all parents) if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved :class:`Path` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def calculate_angle(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> float:
    """
    Compute the interior angle at vertex *b* formed by the vectors b→a and b→c.

    Args:
        a: 2-D or 3-D coordinate of the first point.
        b: 2-D or 3-D coordinate of the vertex.
        c: 2-D or 3-D coordinate of the third point.

    Returns:
        Angle in degrees [0, 180].
    """
    a, b, c = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-9 or norm_bc < 1e-9:
        return 0.0
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))
