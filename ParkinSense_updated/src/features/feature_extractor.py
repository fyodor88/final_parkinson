"""
ParkiSense — Gait Feature Extractor
=====================================
Reads the per-frame landmark CSV produced by :mod:`src.pose.mediapipe_pose`
and computes 18 clinically-motivated gait features per video.

Features
--------
 1. left_knee_angle           – mean angle at left knee (Hip→Knee→Ankle)
 2. right_knee_angle          – mean angle at right knee
 3. left_hip_angle            – mean angle at left hip (Shoulder→Hip→Knee)
 4. right_hip_angle           – mean angle at right hip
 5. left_elbow_angle          – mean angle at left elbow (Shoulder→Elbow→Wrist)
 6. right_elbow_angle         – mean angle at right elbow
 7. step_width                – mean lateral distance between ankles
 8. stride_length             – mean longitudinal displacement of foot midpoint
 9. walking_speed             – mean frame-to-frame CoM (hip midpoint) displacement
10. cadence                   – estimated steps per second
11. arm_swing                 – mean wrist-to-wrist distance
12. com_motion_variance       – variance of CoM (hip midpoint) trajectory
13. pelvis_tilt               – mean angle between hip line and horizontal
14. gait_symmetry             – ratio of left/right stride length (≈1 → symmetric)
15. average_velocity          – mean displacement of CoM midpoint per frame
16. joint_angle_variance      – variance across all joint angles per frame, averaged
17. frame_count               – number of valid (landmark-detected) frames
18. dominant_stride_frequency – dominant oscillation frequency of hip Y-coordinate (Hz)

Usage (CLI):
    python src/features/feature_extractor.py --input outputs/landmarks.csv

Usage (Python API):
    from src.features.feature_extractor import GaitFeatureExtractor
    extractor = GaitFeatureExtractor()
    features  = extractor.extract_features("outputs/landmarks.csv")
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Resolve project root for standalone execution
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import get_logger, ensure_dir, calculate_angle, euclidean_distance, LM

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_LANDMARKS_CSV = str(_PROJECT_ROOT / "outputs" / "landmarks.csv")
DEFAULT_FEATURES_CSV  = str(_PROJECT_ROOT / "outputs" / "features.csv")


# ---------------------------------------------------------------------------
# GaitFeatureExtractor
# ---------------------------------------------------------------------------

class GaitFeatureExtractor:
    """
    Computes 18 gait features from a MediaPipe landmark CSV.

    The extractor operates in two phases:
      1. **Frame-level** — angles and distances computed per frame.
      2. **Video-level** — statistical aggregations (mean, variance) over
         all valid frames, producing a single feature row per video.

    Args:
        features_csv: Path where the output features CSV will be written.
        fps:          Video frame-rate used for time-domain features.
                      If ``None``, falls back to 30 fps.
        normalize:    Whether to apply :class:`sklearn.preprocessing.StandardScaler`
                      to the output features.  The fitted scaler is stored in
                      ``self.scaler`` for later reuse.
    """

    # Joint landmark indices (re-exported from utils for convenience)
    _L_SHOULDER = LM["LEFT_SHOULDER"]   # 11
    _R_SHOULDER = LM["RIGHT_SHOULDER"]  # 12
    _L_ELBOW    = LM["LEFT_ELBOW"]      # 13
    _R_ELBOW    = LM["RIGHT_ELBOW"]     # 14
    _L_WRIST    = LM["LEFT_WRIST"]      # 15
    _R_WRIST    = LM["RIGHT_WRIST"]     # 16
    _L_HIP      = LM["LEFT_HIP"]        # 23
    _R_HIP      = LM["RIGHT_HIP"]       # 24
    _L_KNEE     = LM["LEFT_KNEE"]       # 25
    _R_KNEE     = LM["RIGHT_KNEE"]      # 26
    _L_ANKLE    = LM["LEFT_ANKLE"]      # 27
    _R_ANKLE    = LM["RIGHT_ANKLE"]     # 28

    FEATURE_COLUMNS = [
        "left_knee_angle",
        "right_knee_angle",
        "left_hip_angle",
        "right_hip_angle",
        "left_elbow_angle",
        "right_elbow_angle",
        "step_width",
        "stride_length",
        "walking_speed",
        "cadence",
        "arm_swing",
        "com_motion_variance",
        "pelvis_tilt",
        "gait_symmetry",
        "average_velocity",
        "joint_angle_variance",
        "frame_count",
        "dominant_stride_frequency",
    ]

    def __init__(
        self,
        features_csv: str = DEFAULT_FEATURES_CSV,
        fps: Optional[float] = None,
        normalize: bool = True,
    ) -> None:
        self.features_csv = features_csv
        self.fps          = fps or 30.0
        self.normalize    = normalize
        self.scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(
        self,
        landmarks_csv: str = DEFAULT_LANDMARKS_CSV,
        label: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Read *landmarks_csv* and compute a single feature-vector for the video.

        Args:
            landmarks_csv: Path to the landmark CSV (output of PoseExtractor).
            label:         Optional class label (0=control, 1=parkinson) to
                           append as a ``label`` column in the output CSV.

        Returns:
            Dictionary mapping feature name → scalar value.

        Raises:
            FileNotFoundError: If *landmarks_csv* does not exist.
            ValueError:        If the CSV has insufficient valid frames.
        """
        if not os.path.exists(landmarks_csv):
            raise FileNotFoundError(f"Landmarks CSV not found: {landmarks_csv}")

        df = pd.read_csv(landmarks_csv)
        logger.info("Loaded %d landmark rows from %s", len(df), landmarks_csv)

        features = self._compute_features(df)

        if label is not None:
            features["label"] = label

        # Write to CSV
        ensure_dir(Path(self.features_csv).parent)
        out_df = pd.DataFrame([features])

        if self.normalize:
            feature_cols = [c for c in self.FEATURE_COLUMNS if c in out_df.columns]
            self.scaler = StandardScaler()
            out_df[feature_cols] = self.scaler.fit_transform(out_df[feature_cols])

        out_df.to_csv(self.features_csv, index=False)
        logger.info("✅ Features saved → %s", self.features_csv)
        return features

    def extract_features_from_df(
        self,
        df: pd.DataFrame,
        label: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Same as :meth:`extract_features` but accepts a pre-loaded DataFrame.
        Does NOT write to CSV (for use in the dataset builder).

        Args:
            df:    Landmark DataFrame (already loaded).
            label: Optional label to include in the returned dict.

        Returns:
            Feature dictionary.
        """
        features = self._compute_features(df)
        if label is not None:
            features["label"] = label
        return features

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute all 18 gait features from a landmark DataFrame."""
        frames = sorted(df["frame"].unique())
        n_frames = len(frames)

        if n_frames < 2:
            raise ValueError(f"Insufficient frames for feature extraction ({n_frames} valid frames).")

        # Per-frame accumulators
        left_knee_angles:  list[float] = []
        right_knee_angles: list[float] = []
        left_hip_angles:   list[float] = []
        right_hip_angles:  list[float] = []
        left_elbow_angles: list[float] = []
        right_elbow_angles: list[float] = []
        step_widths:       list[float] = []
        arm_swings:        list[float] = []
        pelvis_tilts:      list[float] = []
        joint_var_list:    list[float] = []

        com_positions:     list[np.ndarray] = []   # (N, 2) hip midpoints
        l_ankle_x_series:  list[float] = []
        r_ankle_x_series:  list[float] = []

        logger.info("Computing per-frame gait features over %d frames …", n_frames)

        for frame in tqdm(frames, desc="Feature computation", unit="frame"):
            fdata = df[df["frame"] == frame]
            row   = self._frame_to_dict(fdata)

            if row is None:
                continue  # skip frames with missing landmarks

            try:
                # ---------- Joint angles ----------
                lka = calculate_angle(row[self._L_HIP],   row[self._L_KNEE], row[self._L_ANKLE])
                rka = calculate_angle(row[self._R_HIP],   row[self._R_KNEE], row[self._R_ANKLE])
                lha = calculate_angle(row[self._L_SHOULDER], row[self._L_HIP],  row[self._L_KNEE])
                rha = calculate_angle(row[self._R_SHOULDER], row[self._R_HIP],  row[self._R_KNEE])
                lea = calculate_angle(row[self._L_SHOULDER], row[self._L_ELBOW], row[self._L_WRIST])
                rea = calculate_angle(row[self._R_SHOULDER], row[self._R_ELBOW], row[self._R_WRIST])

                left_knee_angles.append(lka)
                right_knee_angles.append(rka)
                left_hip_angles.append(lha)
                right_hip_angles.append(rha)
                left_elbow_angles.append(lea)
                right_elbow_angles.append(rea)

                # Per-frame joint variance
                all_angles = [lka, rka, lha, rha, lea, rea]
                joint_var_list.append(float(np.var(all_angles)))

                # ---------- Step width ----------
                sw = euclidean_distance(row[self._L_ANKLE][:2], row[self._R_ANKLE][:2])
                step_widths.append(sw)

                # ---------- Arm swing ----------
                aw = euclidean_distance(row[self._L_WRIST][:2], row[self._R_WRIST][:2])
                arm_swings.append(aw)

                # ---------- Pelvis tilt ----------
                l_hip = row[self._L_HIP]
                r_hip = row[self._R_HIP]
                dx    = r_hip[0] - l_hip[0]
                dy    = r_hip[1] - l_hip[1]
                tilt  = float(np.degrees(np.arctan2(dy, dx))) if dx != 0 else 0.0
                pelvis_tilts.append(tilt)

                # ---------- CoM (hip midpoint) ----------
                com = (l_hip[:2] + r_hip[:2]) / 2.0
                com_positions.append(com)

                # ---------- Ankle X for cadence ----------
                l_ankle_x_series.append(float(row[self._L_ANKLE][0]))
                r_ankle_x_series.append(float(row[self._R_ANKLE][0]))

            except Exception as exc:
                logger.debug("Frame %s skipped: %s", frame, exc)
                continue

        n_valid = len(com_positions)
        if n_valid < 2:
            raise ValueError(f"Not enough valid frames after landmark filtering ({n_valid}).")

        com_arr = np.array(com_positions)  # shape (n_valid, 2)

        # ---------- Walking speed / average velocity ----------
        frame_displacements = np.linalg.norm(np.diff(com_arr, axis=0), axis=1)
        walking_speed       = float(np.mean(frame_displacements))
        average_velocity    = walking_speed  # same quantity, kept as separate semantic feature

        # ---------- CoM motion variance ----------
        com_motion_variance = float(np.var(com_arr, axis=0).mean())

        # ---------- Stride length ----------
        # Use longitudinal (y-axis in image = vertical) displacement of ankle midpoint
        ankle_mid_x = (np.array(l_ankle_x_series) + np.array(r_ankle_x_series)) / 2.0
        if len(ankle_mid_x) > 1:
            stride_length = float(np.abs(np.diff(ankle_mid_x)).mean())
        else:
            stride_length = 0.0

        # ---------- Gait symmetry ----------
        if len(l_ankle_x_series) > 1 and len(r_ankle_x_series) > 1:
            l_disp = float(np.abs(np.diff(l_ankle_x_series)).mean())
            r_disp = float(np.abs(np.diff(r_ankle_x_series)).mean())
            gait_symmetry = (l_disp / r_disp) if r_disp > 1e-6 else 1.0
        else:
            gait_symmetry = 1.0

        # ---------- Cadence (dominant frequency of ankle oscillation) ----------
        ankle_signal = np.array(l_ankle_x_series) - np.array(r_ankle_x_series)
        if len(ankle_signal) >= 10:
            freqs, psd = signal.welch(ankle_signal, fs=self.fps, nperseg=min(len(ankle_signal), 64))
            dominant_idx       = int(np.argmax(psd[1:])) + 1  # skip DC
            dominant_freq_hz   = float(freqs[dominant_idx])
            cadence            = dominant_freq_hz * 2.0  # steps/sec (bilateral)
        else:
            dominant_freq_hz   = 0.0
            cadence            = 0.0

        features: Dict[str, float] = {
            "left_knee_angle":           float(np.mean(left_knee_angles))  if left_knee_angles  else 0.0,
            "right_knee_angle":          float(np.mean(right_knee_angles)) if right_knee_angles else 0.0,
            "left_hip_angle":            float(np.mean(left_hip_angles))   if left_hip_angles   else 0.0,
            "right_hip_angle":           float(np.mean(right_hip_angles))  if right_hip_angles  else 0.0,
            "left_elbow_angle":          float(np.mean(left_elbow_angles)) if left_elbow_angles else 0.0,
            "right_elbow_angle":         float(np.mean(right_elbow_angles)) if right_elbow_angles else 0.0,
            "step_width":                float(np.mean(step_widths))       if step_widths       else 0.0,
            "stride_length":             stride_length,
            "walking_speed":             walking_speed,
            "cadence":                   cadence,
            "arm_swing":                 float(np.mean(arm_swings))        if arm_swings        else 0.0,
            "com_motion_variance":       com_motion_variance,
            "pelvis_tilt":               float(np.mean(pelvis_tilts))      if pelvis_tilts      else 0.0,
            "gait_symmetry":             gait_symmetry,
            "average_velocity":          average_velocity,
            "joint_angle_variance":      float(np.mean(joint_var_list))    if joint_var_list    else 0.0,
            "frame_count":               float(n_valid),
            "dominant_stride_frequency": dominant_freq_hz,
        }

        return features

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _frame_to_dict(fdata: pd.DataFrame) -> Optional[Dict[int, np.ndarray]]:
        """
        Convert a single-frame subset of the landmark DataFrame into a dict
        mapping landmark_id → np.array([x, y, z]).

        Returns ``None`` if any required landmark is missing.
        """
        required = {
            LM["LEFT_SHOULDER"], LM["RIGHT_SHOULDER"],
            LM["LEFT_ELBOW"],    LM["RIGHT_ELBOW"],
            LM["LEFT_WRIST"],    LM["RIGHT_WRIST"],
            LM["LEFT_HIP"],      LM["RIGHT_HIP"],
            LM["LEFT_KNEE"],     LM["RIGHT_KNEE"],
            LM["LEFT_ANKLE"],    LM["RIGHT_ANKLE"],
        }
        present = set(fdata["landmark_id"].tolist())
        if not required.issubset(present):
            return None

        row: Dict[int, np.ndarray] = {}
        for _, r in fdata.iterrows():
            row[int(r["landmark_id"])] = np.array([r["x"], r["y"], r.get("z", 0.0)], dtype=float)
        return row


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="feature_extractor",
        description="ParkiSense — Gait Feature Extractor",
    )
    p.add_argument("--input",  "-i", default=DEFAULT_LANDMARKS_CSV, help="Input landmarks CSV")
    p.add_argument("--output", "-o", default=DEFAULT_FEATURES_CSV,  help="Output features CSV")
    p.add_argument("--fps",    "-f", type=float, default=30.0,       help="Video FPS (default: 30)")
    p.add_argument("--no-normalize", action="store_true",            help="Disable feature normalization")
    return p


def main() -> None:
    args      = _build_arg_parser().parse_args()
    extractor = GaitFeatureExtractor(
        features_csv=args.output,
        fps=args.fps,
        normalize=not args.no_normalize,
    )
    extractor.extract_features(args.input)


if __name__ == "__main__":
    main()
