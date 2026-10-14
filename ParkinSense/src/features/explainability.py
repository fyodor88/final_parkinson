"""
ParkinSense — Model Explainability & Clinical Gait Interpretation
==================================================================
Maps Random Forest feature importances and pose channel dynamics to clinically
interpretable gait indicators.

Uses SMPL 24-joint anatomical definitions:
- Joints 16, 17, 18, 19, 20, 21: Upper Extremities (Arm Swing, Wrist Dynamics)
- Joints 1, 2, 4, 5, 7, 8: Lower Extremities (Hip Extension, Knee Flexion, Ankle Stride)
- Joints 0, 3, 6, 9: Trunk & Posture (Pelvic Tilt, Spine Rigidity)
"""

from __future__ import annotations

from typing import Dict, Any, List
import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

# SMPL Joint to Clinical Gait Description Mapping
JOINT_CLINICAL_MAP = {
    0: ("Pelvis / Root", "Trunk & Pelvis Base Balance", "Altered center-of-mass stability"),
    1: ("Left Hip", "Left Hip Extension & Motion", "Reduced left hip rotation during gait cycle"),
    2: ("Right Hip", "Right Hip Extension & Motion", "Reduced right hip rotation during gait cycle"),
    3: ("Lower Spine", "Lumbar Spine Rigidity", "Increased trunk stiffness and lower back rigidity"),
    4: ("Left Knee", "Left Knee Flexion Dynamics", "Reduced left knee flexion range during stride"),
    5: ("Right Knee", "Right Knee Flexion Dynamics", "Reduced right knee flexion range during stride"),
    6: ("Mid Spine", "Trunk Posture Alignment", "Forward trunk flexion and postural instability"),
    7: ("Left Ankle", "Left Foot Strike & Ankle Motion", "Altered foot clearance and reduced left ankle motion"),
    8: ("Right Ankle", "Right Foot Strike & Ankle Motion", "Altered foot clearance and reduced right ankle motion"),
    9: ("Chest / Thorax", "Upper Body Rigidity", "Thoracic stiffness during locomotion"),
    12: ("Neck / Head", "Head Posture Stability", "Head/neck posture oscillation"),
    16: ("Left Shoulder", "Left Arm Swing Amplitude", "Asymmetric or reduced left arm swing"),
    17: ("Right Shoulder", "Right Arm Swing Amplitude", "Asymmetric or reduced right arm swing"),
    18: ("Left Elbow", "Left Elbow Flexion Range", "Rigid left elbow posture during walking"),
    19: ("Right Elbow", "Right Elbow Flexion Range", "Rigid right elbow posture during walking"),
    20: ("Left Wrist", "Left Wrist Oscillation", "Micro-tremor or reduced wrist mobility"),
    21: ("Right Wrist", "Right Wrist Oscillation", "Micro-tremor or reduced wrist mobility"),
}


class ModelExplainer:
    """
    Computes explainable clinical gait indicators for ParkinSense model predictions.
    """

    def explain(
        self,
        model: Any,
        feature_df: pd.DataFrame,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Extract top feature importances and convert to clinical gait indicators.

        Args:
            model: Trained RandomForestClassifier model.
            feature_df: Single-row DataFrame containing the 290 features.
            top_k: Number of top features to report.

        Returns:
            List of dicts detailing feature name, joint, clinical indicator, importance score.
        """
        if not hasattr(model, "feature_importances_"):
            logger.warning("Model does not provide feature_importances_. Returning basic explanation.")
            return []

        importances = model.feature_importances_
        feature_names = getattr(model, "feature_names_in_", feature_df.columns.tolist())

        # Sort feature indices by importance descending
        sorted_indices = np.argsort(importances)[::-1]

        explanations = []
        for idx in sorted_indices:
            feat_name = feature_names[idx]
            importance_val = float(importances[idx])

            # Filter out general metadata columns like fps/num_frames unless top
            if feat_name in ["fps", "num_frames"]:
                continue

            # Parse pose channel (e.g. pose16_std -> pose_idx 16)
            if feat_name.startswith("pose"):
                parts = feat_name.replace("pose", "").split("_")
                try:
                    pose_channel = int(parts[0])
                    stat_type = parts[1] if len(parts) > 1 else "value"
                    joint_idx = pose_channel // 3  # 3 axis-angle channels per joint

                    joint_name, clinical_indicator, default_desc = JOINT_CLINICAL_MAP.get(
                        joint_idx, (f"Joint {joint_idx}", "Body Pose Variation", "Observed joint motion deviation")
                    )

                    explanations.append({
                        "feature": feat_name,
                        "joint_index": joint_idx,
                        "joint_name": joint_name,
                        "indicator": clinical_indicator,
                        "importance": round(importance_val, 4),
                        "stat_type": stat_type,
                        "description": default_desc,
                    })

                except (ValueError, IndexError):
                    continue

            if len(explanations) >= top_k:
                break

        logger.info("Generated %d explainable gait indicators.", len(explanations))
        return explanations
