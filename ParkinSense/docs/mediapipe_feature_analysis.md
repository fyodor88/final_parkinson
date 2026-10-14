# MediaPipe 290-Feature Representation Analysis

> **Scope.** Training data only (`data/processed/pd_dataset_mediapipe_normalized.csv`).
> **No TULIP label, prediction, or metric was read, used, or referenced for feature
> selection or model tuning in this analysis.** No model file was modified, and no
> feature selection was applied to the pipeline. This report makes **no claim that any
> feature subset improves external accuracy** — every number below that comes from
> cross-validation is an *internal, training-domain* estimate only.

---

## 1. Dataset Verification

| Check | Observed | Expected | Status |
| --- | --- | --- | --- |
| Trials | 322 | 322 | PASS |
| Subjects | 58 | 58 | PASS |
| Features | 290 | 290 | PASS |
| Healthy trials | 205 | 205 | PASS |
| Parkinson's trials | 117 | 117 | PASS |
| Healthy subjects | 28 | info | info |
| Parkinson's subjects | 30 | info | info |
| Trials per subject (min-max) | 2-16 | info | info |
| Label constant within subject | True | True | PASS |
| Missing values in features | 0 | 0 | PASS |
| Distinct fps values | 60.0 | info | info |

Structure: **30 Parkinson's + 28 healthy subjects**;
every subject carries a single diagnosis (no subject-level label leakage across folds);
trials per subject range 2–16.

Recording-session structure (relevant to Sections 7–8): subjects come from two collection
blocks — `dne1` (15 subjects, **all healthy**, 124 trials) and `dne2` (43 subjects, 13 healthy / 30 Parkinson's, 198 trials). Session is therefore *partially* confounded
with label: `dne1` never contains a positive case.

---

## 2. Feature Taxonomy (Semantic Groups)

The representation has **no position channels and no gait/dynamics channels** in the
requested sense; it is 2 metadata columns + 72 rotation channels × 4 summary statistics.

| Group | Requested semantic type | Count | Share of importance |
| --- | --- | --- | --- |
| `temporal_video_length` | temporal features + FPS/video-length features | 2 | 0.4% of RF importance, 0.1% of positive permutation importance |
| `pose_rotation_root` | pose/rotation statistics - root (global) orientation | 12 | 0.0% of RF importance, 0.0% of positive permutation importance |
| `pose_rotation_joint` | pose/rotation statistics - relative joint rotations | 276 | 99.6% of RF importance, 99.9% of positive permutation importance |
| `position_statistics` | position statistics | 0 | ABSENT from this representation |
| `gait_features` | gait features | 0 | ABSENT from this representation |

Per-statistic breakdown:

| statistic | n | sum MDI | % of MDI | sum perm. (ΔAUC) | % of positive perm. | mean fold MDI std | median \|rho\| vs num_frames |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fps | 1 | 0.000000 | 0.0% | 0.000000 | 0.0% | 0.000000 | n/a |
| max | 72 | 0.242029 | 24.2% | 0.019296 | 26.8% | 0.001314 | 0.112 |
| mean | 72 | 0.200024 | 20.0% | 0.017156 | 23.8% | 0.001121 | 0.095 |
| min | 72 | 0.190219 | 19.0% | 0.014218 | 19.7% | 0.001225 | 0.136 |
| num_frames | 1 | 0.003714 | 0.4% | 0.000040 | 0.1% | 0.001349 | 1.000 |
| std | 72 | 0.364014 | 36.4% | 0.003951 | 5.5% | 0.001544 | 0.168 |

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

### 3.1 Dead (exactly zero-variance) features — 13

| Feature | Group | Reason |
| --- | --- | --- |
| `fps` | temporal_video_length | constant across all training videos |
| `pose0_mean` | pose_rotation_root | 0 by construction |
| `pose0_std` | pose_rotation_root | 0 by construction |
| `pose0_min` | pose_rotation_root | 0 by construction |
| `pose0_max` | pose_rotation_root | 0 by construction |
| `pose1_mean` | pose_rotation_root | 0 by construction |
| `pose1_std` | pose_rotation_root | 0 by construction |
| `pose1_min` | pose_rotation_root | 0 by construction |
| `pose1_max` | pose_rotation_root | 0 by construction |
| `pose2_mean` | pose_rotation_root | 0 by construction |
| `pose2_std` | pose_rotation_root | 0 by construction |
| `pose2_min` | pose_rotation_root | 0 by construction |
| `pose2_max` | pose_rotation_root | 0 by construction |

* `fps` is **60.0 for all 322 trials** — a constant column that consumes a
  feature slot and contributes exactly nothing (it cannot split, cannot be permuted).
* `pose0..pose2` (all 4 statistics each) are **zeroed by construction** during canonical
  frame alignment, i.e. the root-orientation block is a permanently dead 12-feature block.

These 13 features are pure noise-free constants; they neither help nor
directly hurt (the RF ignores them), but they inflate the advertised "290-dimensional"
representation to only **277 usable dimensions**.

### 3.2 Near-zero-variance features (0 < var < 1e-05 rad²)

Count: **6**
(std < 0.0032 rad ≈ 0.18°).

| Feature | Group | Variance | Std |
| --- | --- | --- | --- |
| `pose70_std` | pose_rotation_joint | 2.067e-06 | 1.438e-03 |
| `pose67_std` | pose_rotation_joint | 2.337e-06 | 1.529e-03 |
| `pose68_std` | pose_rotation_joint | 5.620e-06 | 2.371e-03 |
| `pose66_std` | pose_rotation_joint | 6.503e-06 | 2.550e-03 |
| `pose69_std` | pose_rotation_joint | 6.602e-06 | 2.569e-03 |
| `pose71_std` | pose_rotation_joint | 7.182e-06 | 2.680e-03 |

Additional context: median feature variance is 1.064e-03;
41 of 290 features have variance < 1e-4 rad²
(std < 0.01 rad ≈ 0.57°). These channels are near-static in the training domain and are
the ones most likely to fall on the wrong side of a domain shift.

---

## 4. Redundancy / Correlation Analysis

| Correlation band (pairwise \|r\|) | Redundant pairs |
| --- | --- |
| ≥ 0.95 | 36 |
| 0.90 – 0.95 | 63 |
| 0.80 – 0.90 | 119 |

* Redundancy clusters at |r| ≥ 0.95: **16** (largest cluster has
  **6** features).
* Of the ≥0.95 pairs, **28 are within the same channel**
  (i.e. `poseX_min`/`poseX_max`/`poseX_mean` of one joint) and
  **8 are across different channels**.
* Maximum |r| observed: **0.9830** (no exactly duplicated columns).

Top redundant pairs (|r| ≥ 0.95):

| Feature A | Feature B | \|r\| |
| --- | --- | --- |
| `pose41_mean` | `pose41_min` | 0.9830 |
| `pose44_mean` | `pose44_max` | 0.9825 |
| `pose68_mean` | `pose68_min` | 0.9793 |
| `pose41_mean` | `pose44_mean` | 0.9780 |
| `pose27_mean` | `pose27_min` | 0.9779 |
| `pose65_mean` | `pose65_max` | 0.9766 |
| `pose62_mean` | `pose62_min` | 0.9760 |
| `pose68_mean` | `pose68_max` | 0.9738 |
| `pose9_mean` | `pose9_min` | 0.9733 |
| `pose71_mean` | `pose71_max` | 0.9730 |
| `pose60_mean` | `pose60_max` | 0.9726 |
| `pose62_mean` | `pose62_max` | 0.9717 |
| `pose41_min` | `pose44_max` | 0.9716 |
| `pose69_mean` | `pose69_max` | 0.9704 |
| `pose27_mean` | `pose27_max` | 0.9702 |

Next band (0.90 ≤ |r| < 0.95):

| Feature A | Feature B | \|r\| |
| --- | --- | --- |
| `pose67_mean` | `pose67_max` | 0.9495 |
| `pose34_mean` | `pose34_min` | 0.9478 |
| `pose68_min` | `pose71_max` | 0.9476 |
| `pose67_mean` | `pose70_mean` | 0.9462 |
| `pose68_mean` | `pose71_max` | 0.9459 |
| `pose68_min` | `pose71_mean` | 0.9451 |
| `pose62_max` | `pose65_mean` | 0.9409 |
| `pose35_mean` | `pose35_max` | 0.9404 |
| `pose59_mean` | `pose59_max` | 0.9395 |
| `pose62_min` | `pose65_max` | 0.9375 |

**Interpretation.** Redundancy is dominated by *within-channel* statistics: `min`, `max`
and `mean` of a joint channel are near-collinear because the underlying rotation
oscillates around a fixed offset. `min`/`max` are also mechanically coupled to `mean ±
amplitude`, so the 4-statistic × 72-channel grid contains substantial duplicated
information (~36 high-collinearity pairs). Redundant features split RF
importance arbitrarily across themselves, which is exactly why single-feature importance
ranks must be read at *group* level (Section 5) rather than feature level.

---

## 5. Importance Analysis (training data, subject-wise 5-fold GroupKFold)

Cross-validation uses the same `GroupKFold(n_splits=5)` grouping as
`src/training/train_mediapipe_normalized.py` — folds are split by **subject**, so no
subject's trials appear in both train and validation.

| Fold | Train trials | Val trials | Train subj | Val subj | Train PD trials | Val PD trials | Val baseline AUC | Val accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 256 | 66 | 46 | 12 | 93 | 24 | 0.9177 | 0.8030 |
| 2 | 258 | 64 | 47 | 11 | 91 | 26 | 0.9615 | 0.8906 |
| 3 | 258 | 64 | 47 | 11 | 91 | 26 | 0.9367 | 0.7812 |
| 4 | 258 | 64 | 46 | 12 | 104 | 13 | 0.9698 | 0.9062 |
| 5 | 258 | 64 | 46 | 12 | 89 | 28 | 0.9990 | 0.9531 |

Note: `GroupKFold` balances *sample counts*, not class balance — the validation folds hold
between 13 and 28 Parkinson's trials out of
64–66
validation trials, which adds variance to any per-fold importance estimate. All importance
numbers below are therefore reported as 5-fold means with their across-fold spread.

Global scale: median MDI importance = 0.001195, median permutation importance =
0.000084. Features with permutation importance ≤ 0:
**106 / 290**
(features that, when shuffled on held-out subjects, do not degrade ROC-AUC).

**MDI and held-out permutation importance disagree strongly**
(Spearman between `rf_importance` and `permutation_importance` = **0.348**):
the features the forest splits on most are *not* the features that move held-out AUC.
Most strikingly, the `std` block absorbs 36.4% of all MDI
importance but only 5.5% of positive permutation
importance, and several high-MDI `std` features (`pose6_std`, `pose13_std`, `pose22_std`,
`pose28_std`) have *negative* held-out permutation importance. **Ranking for the next
experiment must use the held-out permutation column, not `rf_importance`.**

### 5.1 Group-level importance shares

| Group | n | % of RF (MDI) importance | % of positive permutation importance | mean MDI | mean perm. |
| --- | --- | --- | --- | --- | --- |
| `temporal_video_length` | 2 | 0.4% | 0.1% | 0.001857 | 0.000020 |
| `pose_rotation_root` | 12 | 0.0% | 0.0% | 0.000000 | 0.000000 |
| `pose_rotation_joint` | 276 | 99.6% | 99.9% | 0.003610 | 0.000198 |

### 5.2 Top 30 features (ranked by held-out permutation importance)

| # | Feature | Group | Joint | RF imp. | RF rank | Perm. imp. (ΔAUC) | fold MDI mean ± std | fold perm. std | univ. CV AUC | \|rho\| vs num_frames | session SMD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `pose9_max` | pose_rotation_joint | Spine1 | 0.025910 | 7 | 0.003752 | 0.028841 ± 0.007197 | 0.003040 | 0.905 | 0.427 | 0.794 |
| 2 | `pose9_mean` | pose_rotation_joint | Spine1 | 0.032570 | 4 | 0.003308 | 0.029994 ± 0.009028 | 0.002980 | 0.919 | 0.285 | 1.077 |
| 3 | `pose45_mean` | pose_rotation_joint | Head | 0.023778 | 10 | 0.002621 | 0.023200 ± 0.007059 | 0.001732 | 0.858 | 0.199 | 0.747 |
| 4 | `pose30_std` | pose_rotation_joint | L_Foot | 0.048276 | 2 | 0.002503 | 0.045339 ± 0.008021 | 0.002893 | 0.908 | 0.290 | 0.840 |
| 5 | `pose13_min` | pose_rotation_joint | L_Knee | 0.029032 | 6 | 0.002073 | 0.025843 ± 0.009294 | 0.003135 | 0.878 | 0.316 | 0.416 |
| 6 | `pose45_max` | pose_rotation_joint | Head | 0.019936 | 14 | 0.001993 | 0.018512 ± 0.004912 | 0.001585 | 0.853 | 0.268 | 0.569 |
| 7 | `pose67_max` | pose_rotation_joint | L_Hand | 0.012761 | 22 | 0.001762 | 0.011286 ± 0.002556 | 0.000970 | 0.810 | 0.091 | 0.469 |
| 8 | `pose5_std` | pose_rotation_joint | L_Hip | 0.005228 | 43 | 0.001672 | 0.005036 ± 0.002401 | 0.001507 | 0.744 | 0.046 | 0.752 |
| 9 | `pose67_mean` | pose_rotation_joint | L_Hand | 0.008896 | 29 | 0.001421 | 0.007089 ± 0.001340 | 0.000741 | 0.808 | 0.177 | 0.415 |
| 10 | `pose67_min` | pose_rotation_joint | L_Hand | 0.005262 | 42 | 0.001378 | 0.006095 ± 0.001969 | 0.001514 | 0.789 | 0.208 | 0.333 |
| 11 | `pose70_min` | pose_rotation_joint | R_Hand | 0.006429 | 37 | 0.001370 | 0.005520 ± 0.001774 | 0.001357 | 0.756 | 0.078 | 0.359 |
| 12 | `pose70_mean` | pose_rotation_joint | R_Hand | 0.005208 | 44 | 0.001360 | 0.006373 ± 0.003552 | 0.001369 | 0.774 | 0.169 | 0.251 |
| 13 | `pose17_mean` | pose_rotation_joint | R_Knee | 0.020895 | 13 | 0.001193 | 0.018719 ± 0.004184 | 0.002502 | 0.850 | 0.207 | 0.657 |
| 14 | `pose4_std` | pose_rotation_joint | L_Hip | 0.017055 | 16 | 0.001180 | 0.018097 ± 0.002108 | 0.001483 | 0.852 | 0.383 | 0.014 |
| 15 | `pose10_min` | pose_rotation_joint | Spine1 | 0.014368 | 19 | 0.001136 | 0.012916 ± 0.004386 | 0.001950 | 0.875 | 0.433 | 0.117 |
| 16 | `pose16_max` | pose_rotation_joint | R_Knee | 0.013360 | 21 | 0.000972 | 0.015606 ± 0.003177 | 0.001730 | 0.852 | 0.366 | 0.395 |
| 17 | `pose9_min` | pose_rotation_joint | Spine1 | 0.008273 | 31 | 0.000970 | 0.009207 ± 0.002203 | 0.001694 | 0.889 | 0.136 | 1.127 |
| 18 | `pose3_max` | pose_rotation_joint | L_Hip | 0.016100 | 18 | 0.000952 | 0.018105 ± 0.006883 | 0.001081 | 0.866 | 0.429 | 0.032 |
| 19 | `pose27_max` | pose_rotation_joint | Spine3 | 0.011334 | 23 | 0.000935 | 0.010472 ± 0.004236 | 0.002991 | 0.839 | 0.263 | 0.617 |
| 20 | `pose58_min` | pose_rotation_joint | R_Elbow | 0.004547 | 47 | 0.000849 | 0.007135 ± 0.004064 | 0.000852 | 0.808 | 0.175 | 0.304 |
| 21 | `pose69_max` | pose_rotation_joint | R_Hand | 0.005617 | 41 | 0.000811 | 0.004359 ± 0.002332 | 0.001452 | 0.750 | 0.051 | 0.591 |
| 22 | `pose47_mean` | pose_rotation_joint | Head | 0.006367 | 38 | 0.000785 | 0.004841 ± 0.003139 | 0.000767 | 0.742 | 0.079 | 0.038 |
| 23 | `pose5_min` | pose_rotation_joint | L_Hip | 0.001299 | 138 | 0.000668 | 0.002326 ± 0.001691 | 0.000918 | 0.755 | 0.029 | 0.759 |
| 24 | `pose71_std` | pose_rotation_joint | R_Hand | 0.001293 | 140 | 0.000609 | 0.001519 ± 0.001230 | 0.000592 | 0.697 | 0.023 | 0.537 |
| 25 | `pose66_min` | pose_rotation_joint | L_Hand | 0.001701 | 98 | 0.000608 | 0.002643 ± 0.002060 | 0.000930 | 0.711 | 0.131 | 0.281 |
| 26 | `pose43_mean` | pose_rotation_joint | R_Collar | 0.001912 | 89 | 0.000597 | 0.002972 ± 0.001269 | 0.000987 | 0.586 | 0.078 | 0.480 |
| 27 | `pose51_min` | pose_rotation_joint | R_Shoulder | 0.002413 | 73 | 0.000575 | 0.001278 ± 0.000963 | 0.000550 | 0.677 | 0.015 | 0.178 |
| 28 | `pose37_max` | pose_rotation_joint | Neck | 0.002137 | 82 | 0.000551 | 0.001510 ± 0.000860 | 0.000546 | 0.648 | 0.320 | 0.095 |
| 29 | `pose61_mean` | pose_rotation_joint | L_Wrist | 0.001529 | 112 | 0.000540 | 0.002000 ± 0.000673 | 0.000657 | 0.725 | 0.200 | 0.317 |
| 30 | `pose64_mean` | pose_rotation_joint | R_Wrist | 0.002342 | 76 | 0.000537 | 0.002511 ± 0.000858 | 0.000585 | 0.751 | 0.232 | 0.219 |

Rank agreement: Spearman(univariate CV AUC, permutation importance) = **0.253**;
Spearman(session SMD, MDI importance) = **0.394**;
Spearman(|rho| vs num_frames, MDI importance) = **0.453**.

### 5.3 Which joints dominate

| Joint | Name | n features | sum MDI | sum perm. | median \|rho\| vs len | median session SMD |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | Spine1 | 12 | 0.134292 | 0.008903 | 0.254 | 0.116 |
| 15 | Head | 12 | 0.070098 | 0.007221 | 0.151 | 0.330 |
| 22 | L_Hand | 12 | 0.039082 | 0.006390 | 0.076 | 0.307 |
| 23 | R_Hand | 12 | 0.033208 | 0.005415 | 0.114 | 0.365 |
| 1 | L_Hip | 12 | 0.076449 | 0.005111 | 0.088 | 0.263 |
| 10 | L_Foot | 12 | 0.061176 | 0.003296 | 0.072 | 0.145 |
| 13 | L_Collar | 12 | 0.016921 | 0.002355 | 0.082 | 0.190 |
| 17 | R_Shoulder | 12 | 0.013046 | 0.002264 | 0.111 | 0.212 |
| 12 | Neck | 12 | 0.015315 | 0.001767 | 0.141 | 0.092 |
| 18 | L_Elbow | 12 | 0.031227 | 0.001664 | 0.175 | 0.306 |
| 21 | R_Wrist | 12 | 0.009835 | 0.001516 | 0.175 | 0.198 |
| 5 | R_Knee | 12 | 0.076023 | 0.001339 | 0.190 | 0.314 |
| 11 | R_Foot | 12 | 0.064348 | 0.001327 | 0.084 | 0.167 |
| 7 | L_Ankle | 12 | 0.048527 | 0.001254 | 0.199 | 0.384 |
| 8 | R_Ankle | 12 | 0.031802 | 0.001015 | 0.179 | 0.283 |

### 5.4 Unstable importance across folds

Definition used: a feature is *unstable* if its mean held-out permutation importance is
positive, yet its across-fold coefficient of variation exceeds 1.0 **or** it has
permutation importance ≤ 0 in ≥ 2 of the 5 folds.

Count of unstable features: **167 / 290**
(36 of them are inside the global top-50 by permutation importance — the
rest are near-zero features whose coefficient of variation is meaningless).

| Feature | mean perm. imp. | fold perm. std | CV | #folds ≤ 0 | fold MDI mean ± std |
| --- | --- | --- | --- | --- | --- |
| `pose11_min` | 0.000000 | 0.000095 | 470.78 | 4 | 0.000796 ± 0.000414 |
| `pose61_min` | 0.000002 | 0.000579 | 269.54 | 3 | 0.001657 ± 0.000827 |
| `pose10_max` | 0.000005 | 0.001029 | 210.75 | 3 | 0.019708 ± 0.006118 |
| `pose64_min` | 0.000019 | 0.000582 | 29.93 | 3 | 0.001586 ± 0.000617 |
| `pose3_std` | 0.000076 | 0.002040 | 26.94 | 2 | 0.018842 ± 0.003245 |
| `pose44_max` | 0.000029 | 0.000719 | 24.61 | 3 | 0.001276 ± 0.000741 |
| `pose27_min` | 0.000084 | 0.001972 | 23.53 | 3 | 0.010784 ± 0.006025 |
| `pose28_max` | 0.000017 | 0.000408 | 23.39 | 4 | 0.001891 ± 0.000883 |
| `pose33_std` | 0.000265 | 0.004458 | 16.84 | 2 | 0.046545 ± 0.008055 |
| `pose8_std` | 0.000019 | 0.000291 | 15.28 | 2 | 0.000899 ± 0.000599 |
| `pose38_min` | 0.000028 | 0.000417 | 15.14 | 4 | 0.001225 ± 0.000550 |
| `pose47_std` | 0.000015 | 0.000219 | 14.52 | 4 | 0.001840 ± 0.001483 |
| `pose61_std` | 0.000043 | 0.000500 | 11.66 | 4 | 0.000878 ± 0.000289 |
| `pose15_min` | 0.000067 | 0.000732 | 10.87 | 2 | 0.010580 ± 0.005642 |
| `pose52_std` | 0.000021 | 0.000231 | 10.79 | 3 | 0.000960 ± 0.000582 |
| `pose41_mean` | 0.000039 | 0.000399 | 10.21 | 3 | 0.000803 ± 0.000278 |
| `num_frames` | 0.000040 | 0.000373 | 9.26 | 3 | 0.002476 ± 0.001349 |
| `pose18_mean` | 0.000025 | 0.000225 | 9.08 | 4 | 0.000609 ± 0.000284 |
| `pose21_std` | 0.000100 | 0.000834 | 8.36 | 3 | 0.002398 ± 0.001018 |
| `pose19_min` | 0.000074 | 0.000557 | 7.50 | 3 | 0.001002 ± 0.000446 |

`importance_std` in the CSV is the across-fold standard deviation of the **fold-level MDI
importance**; the fold-permutation standard deviation shown here is the held-out
counterpart. High `importance_std` relative to `mean_fold_importance` means the feature's
value depends strongly on *which subjects* are in the training fold — the classic
signature of a subject-/session-specific feature rather than a disease-general one.

### 5.5 Features that pass every hygiene flag

Filter applied (all conditions, training data only), cumulatively:

| Hygiene filter | features surviving |
| --- | --- |
| 1. all features | 290 |
| 2. − constant features (`fps`, `pose0..2`) | 277 |
| 3. + permutation importance > 0 in **all 5** held-out folds | 10 |
| 4. + not length-coupled (\|rho\| vs `num_frames` < 0.3) | 9 |
| 5. + not session-shifted (SMD < 0.5) | 6 |

Top survivors (sorted by held-out permutation importance):

| Feature | Joint | statistic | Perm. imp. | RF imp. | univ. CV AUC | \|rho\| vs len | session SMD | fold perm. std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `pose67_max` | L_Hand | max | 0.001762 | 0.012761 | 0.810 | 0.091 | 0.469 | 0.000970 |
| `pose67_mean` | L_Hand | mean | 0.001421 | 0.008896 | 0.808 | 0.177 | 0.415 | 0.000741 |
| `pose67_min` | L_Hand | min | 0.001378 | 0.005262 | 0.789 | 0.208 | 0.333 | 0.001514 |
| `pose58_min` | R_Elbow | min | 0.000849 | 0.004547 | 0.808 | 0.175 | 0.304 | 0.000852 |
| `pose47_mean` | Head | mean | 0.000785 | 0.006367 | 0.742 | 0.079 | 0.038 | 0.000767 |
| `pose70_max` | R_Hand | max | 0.000499 | 0.005870 | 0.780 | 0.243 | 0.137 | 0.000647 |

Only **6 features survive all filters** (out of 290). The filter is deliberately
strict — "positive in all 5 folds" is a high bar for features whose permutation effects
are near zero — so this is a *floor*, not proof that only six features carry signal. It is
nonetheless the natural anchor for a "stable core" arm in the next experiment.

---

## 6. Do fps / num_frames / video-length features dominate?

### 6.1 `fps`

* Constant at 60.0 across all 322 training trials → **zero variance, zero
  importance, zero predictive value**. In this training set `fps` cannot dominate anything;
  on external data it will simply be an uninformative slot (unless the external videos have
  a different fps, in which case the column shifts while carrying no signal).

### 6.2 `num_frames` (video duration proxy)

| Metric | Value |
| --- | --- |
| Permutation importance (rank) | 0.000040 (rank 167/290) |
| RF (MDI) importance (rank) | 0.003714 (rank 56/290) |
| Univariate subject-wise CV AUC | 0.7190 |
| Point-biserial correlation with label | 0.3996 |
| Mean frames — healthy / PD | 127.2 / 187.1 |
| Mean duration @60fps — healthy / PD | 2.12s / 3.12s |
| Session SMD (healthy dne1 vs dne2) | 0.563 |

Video duration is **genuinely label-associated in this training domain** (PD trials are
47% longer on average: 187 vs 127
frames), so it is a real — but *protocol-dependent* — signal. Recording protocol
also moves duration in healthy subjects (session SMD above), confirming it is partly a
collection artifact rather than pure pathophysiology.

### 6.3 Indirect duration dependence inside the pose features

`num_frames` leaks into the rotation statistics themselves: a longer video samples more gait
cycles, which inflates `std` and stretches `min`/`max`.

| Statistic | n | median \|rho\| vs `num_frames` | max \|rho\| | # features with \|rho\| ≥ 0.3 |
| --- | --- | --- | --- | --- |
| `max` | 72 | 0.112 | 0.429 | 9 |
| `mean` | 72 | 0.095 | 0.285 | 0 |
| `min` | 72 | 0.136 | 0.433 | 4 |
| `std` | 72 | 0.168 | 0.545 | 13 |

**26 / 288 non-constant pose statistics**
have |Spearman rho| ≥ 0.3 with `num_frames`. The most length-coupled features:

| Feature | statistic | \|rho\| vs num_frames | MDI imp. | perm. imp. |
| --- | --- | --- | --- | --- |
| `num_frames` | num_frames | 1.000 | 0.003714 | 0.000040 |
| `pose6_std` | std | 0.545 | 0.024715 | -0.000965 |
| `pose10_std` | std | 0.545 | 0.029106 | -0.000518 |
| `pose3_std` | std | 0.494 | 0.024204 | 0.000076 |
| `pose16_std` | std | 0.467 | 0.013875 | -0.000732 |
| `pose13_std` | std | 0.438 | 0.032685 | -0.000953 |
| `pose10_min` | min | 0.433 | 0.014368 | 0.001136 |
| `pose3_max` | max | 0.429 | 0.016100 | 0.000952 |
| `pose9_max` | max | 0.427 | 0.025910 | 0.003752 |
| `pose6_max` | max | 0.424 | 0.021455 | -0.000156 |
| `pose10_max` | max | 0.422 | 0.018144 | 0.000005 |
| `pose28_std` | std | 0.409 | 0.016261 | -0.001196 |
| `pose14_min` | min | 0.407 | 0.003865 | 0.000083 |
| `pose4_std` | std | 0.383 | 0.017055 | 0.001180 |
| `pose17_max` | max | 0.383 | 0.001918 | 0.000148 |

### 6.4 Direct answer

* **`fps` does not dominate** — it is dead.
* **`num_frames` alone does not dominate** — it carries roughly
  0.4% of total RF importance and
  0.1% of positive permutation importance (see Section 5.1), with
  univariate CV AUC 0.719 versus ≈0.90 for the best rotation features.
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

| Feature | Joint | session SMD | MDI imp. | perm. imp. | univ. CV AUC |
| --- | --- | --- | --- | --- | --- |
| `pose9_min` | Spine1 | 1.127 | 0.008273 | 0.000970 | 0.889 |
| `pose9_mean` | Spine1 | 1.077 | 0.032570 | 0.003308 | 0.919 |
| `pose17_min` | R_Knee | 0.846 | 0.009218 | 0.000213 | 0.792 |
| `pose30_std` | L_Foot | 0.840 | 0.048276 | 0.002503 | 0.908 |
| `pose45_min` | Head | 0.826 | 0.006581 | 0.000478 | 0.778 |
| `pose55_mean` | L_Elbow | 0.816 | 0.009276 | 0.000416 | 0.817 |
| `pose9_max` | Spine1 | 0.794 | 0.025910 | 0.003752 | 0.905 |
| `pose27_min` | Spine3 | 0.783 | 0.008837 | 0.000084 | 0.851 |
| `pose5_min` | L_Hip | 0.759 | 0.001299 | 0.000668 | 0.755 |
| `pose5_std` | L_Hip | 0.752 | 0.005228 | 0.001672 | 0.744 |
| `pose18_std` | Spine2 | 0.750 | 0.000612 | 0.000063 | 0.692 |
| `pose45_mean` | Head | 0.747 | 0.023778 | 0.002621 | 0.858 |
| `pose21_mean` | L_Ankle | 0.741 | 0.003117 | 0.000506 | 0.689 |
| `pose27_mean` | Spine3 | 0.738 | 0.006803 | 0.000311 | 0.843 |
| `pose33_min` | R_Foot | 0.737 | 0.000846 | -0.000052 | 0.687 |

* Features with session SMD ≥ 0.5: **49 / 290**.
* Features flagged (constant, length-coupled, or session-shifted) and their rank among the
  most important: **85 features**, of which
  **18 are inside the global top-30**.

Cross-check: univariate AUC computed **within `dne2` only** (both classes, same session —
session effect removed) stays high for the leading rotation features (`pose10_std` ≈ 0.90, `pose3_std` ≈ 0.88, `pose13_std` ≈ 0.88, `pose3_max` ≈ 0.88, `pose6_std` ≈ 0.88), and
`num_frames` ≈ 0.79; i.e. genuine label signal
exists in the pose statistics and is not *only* a session artifact. The risk is that the
model currently mixes label signal with session/protocol signal and cannot tell them apart.

Flagged features (all rows with at least one flag, sorted by permutation importance):

| Feature | Group | flags | perm. imp. | MDI imp. | \|rho\| vs len | session SMD |
| --- | --- | --- | --- | --- | --- | --- |
| `pose9_max` | pose_rotation_joint | length-coupled;session-shifted; | 0.003752 | 0.025910 | 0.427 | 0.794 |
| `pose9_mean` | pose_rotation_joint | session-shifted; | 0.003308 | 0.032570 | 0.285 | 1.077 |
| `pose45_mean` | pose_rotation_joint | session-shifted; | 0.002621 | 0.023778 | 0.199 | 0.747 |
| `pose30_std` | pose_rotation_joint | session-shifted; | 0.002503 | 0.048276 | 0.290 | 0.840 |
| `pose13_min` | pose_rotation_joint | length-coupled; | 0.002073 | 0.029032 | 0.316 | 0.416 |
| `pose45_max` | pose_rotation_joint | session-shifted; | 0.001993 | 0.019936 | 0.268 | 0.569 |
| `pose5_std` | pose_rotation_joint | session-shifted; | 0.001672 | 0.005228 | 0.046 | 0.752 |
| `pose17_mean` | pose_rotation_joint | session-shifted; | 0.001193 | 0.020895 | 0.207 | 0.657 |
| `pose4_std` | pose_rotation_joint | length-coupled; | 0.001180 | 0.017055 | 0.383 | 0.014 |
| `pose10_min` | pose_rotation_joint | length-coupled; | 0.001136 | 0.014368 | 0.433 | 0.117 |
| `pose16_max` | pose_rotation_joint | length-coupled; | 0.000972 | 0.013360 | 0.366 | 0.395 |
| `pose9_min` | pose_rotation_joint | session-shifted; | 0.000970 | 0.008273 | 0.136 | 1.127 |
| `pose3_max` | pose_rotation_joint | length-coupled; | 0.000952 | 0.016100 | 0.429 | 0.032 |
| `pose27_max` | pose_rotation_joint | session-shifted; | 0.000935 | 0.011334 | 0.263 | 0.617 |
| `pose69_max` | pose_rotation_joint | session-shifted; | 0.000811 | 0.005617 | 0.051 | 0.591 |
| `pose5_min` | pose_rotation_joint | session-shifted; | 0.000668 | 0.001299 | 0.029 | 0.759 |
| `pose71_std` | pose_rotation_joint | session-shifted; | 0.000609 | 0.001293 | 0.023 | 0.537 |
| `pose37_max` | pose_rotation_joint | length-coupled; | 0.000551 | 0.002137 | 0.320 | 0.095 |
| `pose55_min` | pose_rotation_joint | length-coupled; | 0.000531 | 0.003671 | 0.337 | 0.138 |
| `pose21_max` | pose_rotation_joint | session-shifted; | 0.000506 | 0.000914 | 0.212 | 0.737 |
| `pose21_mean` | pose_rotation_joint | session-shifted; | 0.000506 | 0.003117 | 0.203 | 0.741 |
| `pose28_min` | pose_rotation_joint | session-shifted; | 0.000502 | 0.004766 | 0.293 | 0.598 |
| `pose59_max` | pose_rotation_joint | session-shifted; | 0.000478 | 0.002697 | 0.080 | 0.624 |
| `pose45_min` | pose_rotation_joint | session-shifted; | 0.000478 | 0.006581 | 0.029 | 0.826 |
| `pose64_max` | pose_rotation_joint | length-coupled; | 0.000460 | 0.000605 | 0.307 | 0.007 |
| `pose66_max` | pose_rotation_joint | session-shifted; | 0.000438 | 0.003238 | 0.017 | 0.512 |
| `pose55_mean` | pose_rotation_joint | session-shifted; | 0.000416 | 0.009276 | 0.231 | 0.816 |
| `pose24_max` | pose_rotation_joint | session-shifted; | 0.000404 | 0.003828 | 0.189 | 0.529 |
| `pose4_min` | pose_rotation_joint | session-shifted; | 0.000389 | 0.004086 | 0.025 | 0.673 |
| `pose5_max` | pose_rotation_joint | session-shifted; | 0.000385 | 0.001455 | 0.061 | 0.593 |
| `pose8_max` | pose_rotation_joint | session-shifted; | 0.000372 | 0.002411 | 0.061 | 0.640 |
| `pose49_min` | pose_rotation_joint | session-shifted; | 0.000331 | 0.002498 | 0.117 | 0.594 |
| `pose27_mean` | pose_rotation_joint | session-shifted; | 0.000311 | 0.006803 | 0.221 | 0.738 |
| `pose59_mean` | pose_rotation_joint | session-shifted; | 0.000257 | 0.002914 | 0.156 | 0.542 |
| `pose40_mean` | pose_rotation_joint | session-shifted; | 0.000245 | 0.004449 | 0.021 | 0.513 |
| `pose22_min` | pose_rotation_joint | session-shifted; | 0.000238 | 0.006911 | 0.196 | 0.605 |
| `pose47_min` | pose_rotation_joint | session-shifted; | 0.000231 | 0.001414 | 0.055 | 0.501 |
| `pose17_min` | pose_rotation_joint | session-shifted; | 0.000213 | 0.009218 | 0.055 | 0.846 |
| `pose37_std` | pose_rotation_joint | length-coupled; | 0.000168 | 0.001509 | 0.364 | 0.086 |
| `pose19_std` | pose_rotation_joint | session-shifted; | 0.000165 | 0.001001 | 0.108 | 0.651 |

---

## 8. Internal ablations of predefined feature groups (descriptive, training-only)

Same 5-fold subject-wise GroupKFold, same RF hyperparameters. **These are internal
estimates on the training distribution. They are NOT external results and must not be
quoted as such.**

| Feature subset | n | ROC-AUC (mean ± std) | Accuracy (mean ± std) |
| --- | --- | --- | --- |
| A. all 290 features (current pipeline) | 290 | 0.9570 ± 0.0280 | 0.8669 ± 0.0647 |
| B. temporal/video-length only (fps, num_frames) | 2 | 0.6079 ± 0.0884 | 0.5933 ± 0.0804 |
| C. all minus temporal (288) | 288 | 0.9562 ± 0.0286 | 0.8668 ± 0.0435 |
| D. all minus constant features (277) | 277 | 0.9586 ± 0.0233 | 0.8698 ± 0.0439 |
| E. joint rotations only, pose3..71 (276) | 276 | 0.9548 ± 0.0311 | 0.8514 ± 0.0552 |
| F. root orientation only, pose0..2 (12) | 12 | 0.5000 ± 0.0000 | 0.5196 ± 0.1589 |
| G. pose means only (72) | 72 | 0.9501 ± 0.0341 | 0.8818 ± 0.0567 |
| H. pose std only (72) | 72 | 0.9376 ± 0.0421 | 0.8671 ± 0.0720 |
| I. pose min+max only (144) | 144 | 0.9613 ± 0.0245 | 0.8509 ± 0.0448 |
| J. pose mean+std only (144) | 144 | 0.9515 ± 0.0299 | 0.8762 ± 0.0664 |

Nested top-k (inner subject-wise permutation importance selects, outer subject-wise fold
scores — no leakage between selection and evaluation):

| Variant | n | ROC-AUC (mean ± std) | Accuracy (mean ± std) |
| --- | --- | --- | --- |
| baseline: all 290 features (same outer folds) | 290 | 0.9570 ± 0.0280 | 0.8669 ± 0.0647 |
| nested selection: top-30 by inner permutation importance | 30 | 0.9321 ± 0.0507 | 0.8482 ± 0.0481 |
| nested selection: top-60 by inner permutation importance | 60 | 0.9380 ± 0.0486 | 0.8576 ± 0.0666 |

Observations (internal only; values are mean subject-wise CV ROC-AUC):

* Removing `fps`/`num_frames` changes internal performance very little
  (0.9570 → 0.9562) — the metadata block is not the main carrier
  of internal accuracy, so dropping it costs nothing internally.
* The root-orientation block alone scores 0.5000 (chance) — it is dead weight;
  the relative joint-rotation block alone reaches 0.9548 vs 0.9570
  for all 290 features.
* **The `min`/`max` block alone scores 0.9613 — the highest of every subset —
  even though it is the most length-coupled group (Section 6.3).** In other words the
  internal metric *rewards* duration-sensitive range statistics. This is precisely the
  pattern that is expected to fail when external videos have different trial durations:
  internal ranking alone is not a safe guide for feature selection here.
* `mean`-only (0.9501) and `mean`+`std` (0.9515) stay close to the
  full-model AUC with half or fewer features, while being the least length-coupled
  statistics (Section 6.3).
* Nested top-k selection results (top-30 → 0.9321, top-60 → 0.9380) fall **below** the full-feature
  baseline (0.9570): aggressive single-shot pruning discards
  redundant-but-complementary signal. Group-level, hypothesis-driven arms (Section 11) are
  therefore the safer next step than top-k pruning.

---

## 9. Findings

1. **13 of 290 features are exactly constant** (`fps` + the entire
   root-orientation block `pose0..pose2`); the effective representation is
   277 dimensions, and the root-orientation group contributes
   0.0% of importance (nothing).
2. **Redundancy is structural**: 36 pairs at |r| ≥ 0.95, almost all
   within-channel `min`/`max`/`mean` collinearity; importance is therefore smeared across
   near-duplicate statistics.
3. **The strongest features are relative joint-rotation statistics**, concentrated in the
   `Spine1`, `Head`, `L_Hand`, `R_Hand` channels (Section 5.3), with univariate
   subject-wise CV AUC up to 0.919 — but **MDI and held-out
   permutation importance rank these features very differently**
   (Spearman = 0.348); held-out permutation importance is the trustworthy
   column.
4. **Video length is a hidden second signal**: explicit `num_frames` is a moderate
   predictor (CV AUC 0.719), and 26 features are
   mechanically length-coupled through their `std`/`min`/`max` statistics.
5. **Session/protocol shift is visible in the training data itself**: 49
   features separate healthy `dne1` from healthy `dne2` by SMD ≥ 0.5, and
   session-SMD correlates with feature importance (rho = 0.39).
6. **Importance is not fully stable**: 167 features have positive mean
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
   (138 features):
   near-full internal AUC (Section 8, subset J) while dropping the duration-sensitive
   `min`/`max` block; `mean` is the least length-coupled statistic of all four
   (median \|rho\| vs `num_frames` ≈ 0.10).
   The `mean`-only variant (69 features,
   Section 8 subset G) is the leaner fallback.
   Caveat: `std` is the *most* length-coupled statistic (median \|rho\| vs `num_frames`
   ≈ 0.17), so run
   `mean`-only and `mean`+`std` as separate arms rather than assuming they are equivalent.
2. **Joint families with the highest summed held-out importance** (`Spine1`, `Head`, `L_Hand`, `R_Hand` and the
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
     session-shifted (healthy `dne1` vs `dne2` SMD < 0.5).
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

| Column | Definition |
| --- | --- |
| `feature` | feature name |
| `feature_group` | `temporal_video_length` / `pose_rotation_root` / `pose_rotation_joint` |
| `variance` | sample variance over the 322 training trials |
| `mean` | mean over the 322 training trials |
| `std` | standard deviation over the 322 training trials |
| `rf_importance` | MDI (Gini) importance, RF trained on all 322 trials, normalized to sum 1 |
| `permutation_importance` | mean held-out permutation importance (ΔROC-AUC), 5 subject-wise folds × 10 repeats |
| `mean_fold_importance` | mean across the 5 folds of the fold-level MDI importance |
| `importance_std` | std across the 5 folds of the fold-level MDI importance (stability) |

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
