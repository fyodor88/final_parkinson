# Normalized Feature Training Representation Design

## 1. Executive Summary
This document defines the mathematical design and formal invariants of the normalized MediaPipe/SMPL 290-dimensional gait representation used for Parkinson's Disease (PD) classification in ParkinSense.

---

## 2. Mathematical Representation & Invariant Matrix

| Representation Domain | Standard MoCap / Baseline SMPL | Spatial MediaPipe Normalization | Spatiotemporal MediaPipe Normalization |
| :--- | :--- | :--- | :--- |
| **Kinematic Root Position** | Arbitrary room coordinates $\mathbf{p}_0(t) \in \mathbb{R}^3$ | Pelvis centered: $\mathbf{p}'_k(t) = \mathbf{p}_k(t) - \mathbf{p}_{\text{pelvis}}(t)$ | Pelvis centered + frame interpolated |
| **Spatial Scaling** | Absolute meters (subject height dependent) | Unit torso height scale: $\mathbf{p}''_k(t) = \mathbf{p}'_k(t) / s(t)$ where $s(t) = \|\mathbf{p}_{\text{neck}} - \mathbf{p}_{\text{pelvis}}\|$ | Unit torso height scale |
| **Heading / Orientation** | Arbitrary camera/lab orientation | Canonical frame $R_{\text{canonical}}(t) = [\mathbf{x}_{\text{shoulder}}, \mathbf{y}_{\text{torso}}, \mathbf{z}_{\text{normal}}]$ | Canonical frame $R_{\text{canonical}}(t)$ |
| **Joint Rotations ($R_i \in SO(3)$)** | Relative SMPL joint rotations (joints 1–23) + global orient (joint 0) | Relative SMPL joint rotations (joints 1–23) + canonical global orient | Relative SMPL joint rotations + cycle normalized |
| **Axis-Angle Vector ($\mathbf{r}_i \in \mathbb{R}^3$)** | $\mathbf{r}_i = \text{Log}(R_i) \in \mathbb{R}^3$ | $\mathbf{r}_i = \text{Log}(R_i) \in \mathbb{R}^3$ | $\mathbf{r}_i = \text{Log}(R_i) \in \mathbb{R}^3$ |
| **Summary Statistics (290 Feats)** | `fps`, `num_frames`, `pose0_mean` .. `pose71_max` | `fps`, `num_frames`, `pose0_mean` .. `pose71_max` | `fps`, `num_frames`, `pose0_mean` .. `pose71_max` |

---

## 3. Invariance Properties per Feature Subgroup

1. **Joints 1–23 (`pose3` .. `pose71`, 207 features)**:
   * **Semantics**: Relative joint rotations between parent and child segments in the SMPL kinematic tree (e.g. knee flexion, hip abduction/rotation relative to pelvis).
   * **Translation Invariance**: **100% Invariant** (relative vectors cancel translation offset).
   * **Scale Invariance**: **100% Invariant** (rotation angles are dimensionless scalar radians).
   * **Rotation Invariance**: **100% Invariant** (measured in local parent coordinate frames).

2. **Joint 0 (`global_orient`, `pose0` .. `pose2`, 9 features)**:
   * **Semantics**: Root pelvis orientation relative to world frame.
   * **Translation Invariance**: **100% Invariant**.
   * **Scale Invariance**: **100% Invariant**.
   * **Rotation Invariance**: In Spatial/Spatiotemporal mode, root orientation is aligned to the canonical forward heading frame ($R_{\text{canonical}}$), enforcing global heading invariance.

3. **Temporal Metadata (`fps`, `num_frames`, 2 features)**:
   * **Semantics**: Frame rate (fps) and sequence length (num_frames).
   * Kept for 100% backward feature compatibility with the Random Forest signature.

---

## 4. Training vs MediaPipe Compatibility Analysis

* **MoCap (DNE.pkl) Compatibility**:
  * DNE contains 3D motion-capture SMPL joint rotation vectors $\mathbf{r}_i \in \mathbb{R}^3$ for 24 joints across 97 subjects (322 trials).
  * Because relative joint rotations (`pose3`..`pose71`) in DNE are measured in local anatomical bone frames, their mathematical semantics directly match MediaPipe-estimated relative joint rotations.
  * Aligning `global_orient` (`pose0`..`pose2`) to canonical orientation in DNE produces a mathematically aligned, normalized 290-feature training set (`data/processed/pd_dataset_mediapipe_normalized.csv`).
