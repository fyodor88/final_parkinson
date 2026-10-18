# Phase 3 Feature Selection and Validation Plan

Planning date: 2026-10-15  
Scope: experiment design only. No model was trained or evaluated, no selector was fitted, and no production behavior or existing data/model artifact was changed.

## 1. Inputs Reviewed

The current Phase 2 artifacts were read at these exact paths:

- `docs/phase2_data_integrity_report.md`
- `docs/phase2_feature_compatibility_report.md`
- `data/processed/improved_model/phase2_tulip_clinical_candidate.csv`

The current candidate CSV contains 43 recording rows, 8 participants, 17 healthy and 26 Parkinson's recordings, 288 finite pose-statistic columns, and the retained `fps` / `num_frames` provenance columns. Its recording IDs are unique, participant labels are consistent, and it contains no YouTube rows. The disputed `subject13_subject14_camera3` recording remains excluded; this plan does not infer its identity. The candidate remains a candidate dataset, not ground truth independently re-established by this phase.

The DNE processed table currently contains 322 trials from 58 participants: 205 healthy and 117 Parkinson's trials, with 28 healthy and 30 Parkinson's participants and no mixed-label participant in the current table. The literal DNE `subject` strings do not overlap the candidate's TULIP IDs, but there is no cross-dataset identity registry to establish whether any real person appears in both datasets.

## 2. Feature Definitions and Limits

### 2.1 Contract and calculation

`src/features/feature_contract.py` defines 290 columns: `fps`, `num_frames`, and 288 statistics. For each of 72 channels, the statistics are `mean`, population `std` (`numpy.std`, default `ddof=0`), `min`, and `max` over the rows/frames supplied to the extractor.

`src/features/smpl_feature_extractor.py` converts each `global_orient` / `body_pose` 3x3 matrix to a three-component axis-angle vector with SciPy `Rotation.from_matrix(...).as_rotvec()`. The resulting 72 channel values are in radians under the assumed rotation convention. For channel index `c`, the nominal joint is `floor(c / 3)` and the nominal component is `c mod 3`; feature names record the channel and statistic, not a clinical angle plane.

| Column block                                           | Nominal mapping                                                 | Potential gait relevance                                                               | Interpretation limit                                                                                                                                        |
| ------------------------------------------------------ | --------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pose0`-`pose2`                                        | Pelvis/root global orientation components                       | Trunk/pelvis orientation may relate to walking posture                                 | Root/world reference and camera/estimator conventions are not standardized across domains. The normalized DNE table zeros these channels.                   |
| `pose3`-`pose8`                                        | Left/right hip rotation components                              | Hip-region movement proxy                                                              | Not direct hip flexion/extension or a validated clinical measurement.                                                                                       |
| `pose9`-`pose11`, `pose18`-`pose20`, `pose27`-`pose29` | Spine1, Spine2, Spine3                                          | Trunk movement proxy                                                                   | The MediaPipe spine levels are geometrically interpolated from landmarks; they are not independently observed joints.                                       |
| `pose12`-`pose17`                                      | Left/right knee rotation components                             | Knee-region movement proxy                                                             | Not a measured sagittal-plane knee angle. Axis meanings depend on the pose representation and reference frame.                                              |
| `pose21`-`pose26`, `pose30`-`pose35`                   | Left/right ankle and foot rotation components                   | Distal-leg movement proxy                                                              | Foot/ankle estimates depend on sparse landmarks and fallback rules; not step events or clinical ankle angles.                                               |
| `pose36`-`pose71`                                      | Neck, collar, head, shoulder, elbow, wrist, and hand components | Upper-body movement / arm-swing-related proxies, especially shoulder and elbow regions | Wrist/hand are weakly observed; MediaPipe hand orientation is copied from wrist orientation. These are not direct arm-swing amplitude or asymmetry metrics. |

The ranges above use the nominal SMPL joint ordering in `data/processed/improved_model/tulip_feature_contract.json` and `src/pose/mediapipe_smpl.py`. They support coarse joint-region grouping only. Even in that map, the three axis-angle components cannot safely be renamed flexion, rotation, or abduction without validated coordinate-frame conventions. Individual `poseN_*` features are therefore mostly opaque model coordinates, not named clinical biomarkers.

The representation contains no direct stride length, step time, cadence, gait speed, stance/swing phase, or time-resolved gait-cycle features. `std` is variation in one pose component over the retained frames; it is not step-time variability. Component-wise mean/min/max can also be sensitive to the axis-angle representation near its wrap boundaries. Do not claim that a selected pose component measures a specific clinical sign.

### 2.2 Metadata columns

| Column       | Units and source     | Meaning in the current paths                                                                                                                              | Phase 4 use                                                                                                                                 |
| ------------ | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `fps`        | frames/second        | DNE reads `trial["fps"]`; MediaPipe and 4D-Humans read OpenCV video metadata (fallback 30 FPS); mock 4D-Humans uses 29.75.                                | Exclude from primary predictors. It is acquisition metadata, not gait speed or cadence. Keep only for auditing.                             |
| `num_frames` | count of rows/frames | DNE uses `pose.shape[0]`; MediaPipe counts accepted landmark frames after missed frames are skipped; HMR2 counts frames in the selected longest tracklet. | Exclude from primary predictors. Counts have different sampling/detection semantics and reflect recording duration. Keep only for auditing. |

Thus the historical 290-column model input is not a 290-measurement gait feature set. The primary feature-count experiment should use all 288 pose statistics as its full-feature baseline. Include the historical 290-column baseline only as a separately labeled legacy reproduction, not as the primary gait-only baseline.

### 2.3 Domain generation is not interchangeable

- **DNE:** `src/datasets/build_pd_dataset.py` reads `trial["pose"]` directly and summarizes its 72 columns. That builder does not verify the raw array's coordinate frame, per-channel convention, or joint mapping against either video extractor. `src/datasets/build_pd_dataset_normalized.py` additionally zeros channels 0-2, so it is a distinct representation from the unnormalized DNE table and the standard live application contract.
- **MediaPipe:** `src/pose/mediapipe_smpl.py` creates SMPL-like rotations from sparse MediaPipe world landmarks using geometric bone frames. Missing or invalid joint estimates become identity rotations; the hand estimate copies the wrist. `src/inference/mediapipe_predict.py` counts only frames with accepted landmarks. TULIP's candidate features were extracted through this MediaPipe-derived path/cache.
- **4D-Humans/HMR2:** `backend/fourd_humans.py` consumes reconstructed `global_orient` and `body_pose` matrices from HMR2/PHALP, selects the longest tracklet, and sends those matrices to the shared summary extractor. These are model-inferred SMPL parameters and a different frame population from MediaPipe.

The common helper and column names do not establish equal joint semantics, reference frames, estimator error, frame coverage, or units in practice. The existing compatibility report correctly concludes that cross-domain equivalence has not been established. No DNE+TULIP combination or shared-classifier integration is approved by this plan.

## 3. Feature-Selection Plan

### 3.1 Candidate sets and safeguards

For each eligible dataset independently, compare `k = 10, 20, 30, 50` selected pose statistics against the full **288-pose-statistic** baseline. On DNE only, include an additional 290-column historical reference using `fps` and `num_frames`; do not let it replace the 288-feature primary baseline. TULIP's primary baseline has 288 features because it is intended to exclude acquisition metadata.

Use the same predeclared outer participant folds for all feature counts and selectors. For every outer fold:

1. Fit missing-value handling (if later needed), variance filtering, correlation filtering, scaling, supervised ranking/selection, and any calibration only on the outer training participants. The current audited tables have no missing feature values; do not add an imputer unless a future input requires it.
2. Apply the fitted transformation and selected feature names unchanged to the outer validation participants.
3. Fix the primary Random Forest configuration in advance, initially matching the established `n_estimators=300`, `class_weight="balanced"`, `random_state=42` baseline, so the first comparison isolates feature selection. If hyperparameters or thresholds are tuned later, tune only in inner participant-grouped folds.
4. Save selected feature names and selection frequencies for every outer fold. Never rank/select from the full data table before splitting.

### 3.2 Methods

| Method                             | Recommended role                    | Fold-specific procedure and caution                                                                                                                                                                                                                                                                                                                                                           |
| ---------------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Correlation redundancy filter      | Pre-filter / diagnostic             | Recompute pairwise correlation using only the training fold. A threshold such as absolute Spearman `rho >= 0.95` can define redundancy groups; keep a deterministic representative or let the supervised ranker choose within each group. Also remove exact constants using training data only. Correlation is not evidence of clinical redundancy, especially across distinct pose channels. |
| Random Forest importance           | Primary supervised ranking          | Prefer group-held-out permutation importance within the outer training data, aggregated over inner participant folds. Use impurity/MDA importance as a sensitivity analysis only; correlated features split or bias importance. Do not reuse the existing full-table importance ranking as if it were fold-independent.                                                                       |
| L1 Logistic Regression             | Secondary ranking / stability check | Standardize inside each training fold, fit class-weighted L1 logistic regression, rank by absolute coefficients, and choose top-k. Treat coefficient ranks as unstable when features are correlated or the small TULIP sample is underdetermined. Scaling and coefficient selection remain inside the fold.                                                                                   |
| Sequential Feature Selection (SFS) | Not in primary plan                 | Forward/backward selection over 288 features in nested group CV is computationally expensive and statistically unstable, particularly with 8 TULIP participants. Consider only on a smaller, pre-filtered DNE training-fold subset if later runtime and group counts justify it; it must remain entirely inside nested training folds.                                                        |

The defensible initial experiment is two ranking families (Random Forest permutation and L1 logistic), with a fold-local correlation/constant pre-filter as a predeclared option. Do not run a large search across many correlation cutoffs, selectors, models, and thresholds on TULIP. Report each fixed `k` comparison; do not select the winning `k` from the outer test scores and then report those same scores as an unbiased final estimate. If a winner is required, choose it in inner CV and evaluate that complete selection procedure on outer folds, or reserve a genuinely new cohort.

If fold-local filtering leaves fewer than the requested `k` independent/usable features, report that fact and do not pad the subset with duplicate features merely to reach 50. Report both the selected names and their nominal joint/statistic mapping, but do not infer clinical meaning from the names.

## 4. Leakage-Safe Validation Design

### 4.1 Participant grouping

- **DNE:** Group all trials by `subject`. The audited table has 58 participants (28 healthy, 30 Parkinson's), with 2-16 trials per participant according to the existing feature-analysis report. Use `StratifiedGroupKFold` when feasible to approximately preserve participant-level class balance; all trials from each participant stay together. Use repeated outer 5-fold grouped splits with fixed, recorded seeds (primary seed 42; sensitivity seeds 123 and 2026). Every preprocessing and supervised selection step is fitted separately within each fold.
- **TULIP candidate:** Group by `subject_id`, never `recording_id` or camera. The 43 recordings represent only 8 participants (3 healthy, 5 Parkinson's), each with 5-6 camera recordings. Use outer leave-one-subject-out as a descriptive exploratory design; all recordings from the held-out participant are validation data. Report a participant-level prediction as primary, with recording-level metrics secondary. Aggregate that participant's camera probabilities by a predeclared mean before applying the fixed 0.5 threshold; do not count camera views as independent participants.
- **Cross-dataset identity:** The literal ID sets do not overlap, but no identity crosswalk was found. Do not assert that real people are distinct across datasets without authoritative provenance. If overlap is discovered, collapse aliases to a common group before any split or exclude unresolved overlap.
- **Disputed recording:** Continue excluding `subject13_subject14_camera3` unless an authoritative source resolves its identity. Do not assign it to a new or existing participant by filename inference. The six unverified YouTube examples remain excluded from every supervised experiment.

### 4.2 Historical exposure and what counts as external

The workspace contains prior TULIP use in `scripts/train_improved_mediapipe_model.py`: it combines the cached TULIP rows with DNE, runs grouped CV on the combined rows, uses DNE-to-TULIP leave-one-dataset-out metrics in its model-selection score, trains on the combined table, and then evaluates on those same included TULIP rows. The historical cache also included YouTube rows with reused IDs/contradictory labels. Consequently, TULIP has influenced prior model fitting and selection in this project. The present 8 clinical participants cannot be described as a locked independent test set for this model lineage, even if a new script now holds them out.

The normalized DNE analysis reports no TULIP labels used for that distinct model/analysis. That narrower lineage claim does not make the TULIP collection globally untouched, nor does it make the current TULIP candidate independent of prior project-level model choices. DNE itself has also been examined by the existing full-table feature-analysis script; new fold-contained CV is useful internal reanalysis, but not pristine confirmatory evidence after prior analyst exposure.

An independent external validation claim requires new verified participants/videos whose labels and identities were not used in fitting, feature selection, threshold/calibration choice, or model selection. No such untouched external cohort is established in the inspected artifacts.

### 4.3 Small-sample and confounding limits

TULIP has only 8 independent participants. A reliable holdout would leave too few participants in training, especially only 3 healthy controls overall. LOSO folds each contain a single-class test subject, so per-fold ROC-AUC is undefined. A pooled out-of-fold participant-level AUC is computable only over 8 predictions and must be labeled descriptive and highly uncertain, not clinical validation. Avoid tuning architecture, hyperparameters, thresholds, and many feature-selection variants against these 8 participants.

DNE's participant-level counts are more suitable for grouped internal development, but the existing report identifies a collection/session effect: one collection block is all healthy and the other contains both labels. Subject grouping prevents participant leakage but does not eliminate collection/session confounding. A leave-one-session-out test cannot estimate both-class performance when one session has no Parkinson's cases. Report this as a dataset design limitation; a new cohort with both classes under matched acquisition conditions is needed for stronger generalization evidence.

## 5. Phase 4 Experiment Matrix

| Experiment               | Planned data / comparison                                                                                                                                       | Status and interpretation                                                                                                                                                                                                                                                                                                                         |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A. DNE-only baseline     | Refit the established RF within participant-grouped folds on DNE. Compare 288 pose-only full baseline with a separately labeled 290-column legacy reproduction. | **Proceed as internal, exploratory DNE-domain comparison only.** Do not score the saved production artifact on records it may have seen and call that validation. Keep session confounding and prior DNE analysis exposure explicit.                                                                                                              |
| B. TULIP-only            | Use only the 43-row Phase 2 candidate, subject-grouped LOSO, all 288 baseline and the predeclared reduced sets.                                                 | **Exploratory only.** TULIP has 8 groups, one unresolved excluded clip, and historical project exposure. Not an independent external test or a basis for a clinical performance claim.                                                                                                                                                            |
| C. Combined DNE + TULIP  | Concatenation or transfer learning across datasets.                                                                                                             | **No-go / postponed.** DNE pose semantics and the video-derived MediaPipe/HMR2 estimates are not shown equivalent. Do not reuse the existing combined CSV. Reconsider only after an authoritative channel/frame mapping and a valid matched-domain bridge are established. TULIP's prior exposure still prevents independent external validation. |
| D. Reduced features      | Within each dataset separately, evaluate 10, 20, 30, and 50 features from the fold-local correlation + RF-permutation and L1 ranking procedures.                | **Proceed only within A and exploratory B.** Same grouped folds across counts; selection fitted within each training fold. Report stability and all candidates; do not promise improved performance.                                                                                                                                              |
| E. Full-feature baseline | All 288 pose statistics; separately, DNE's historical 290 columns including metadata for legacy comparison.                                                     | **Required comparator.** The 288 set is the primary gait-feature baseline. The 290 set is a historical compatibility reference with metadata leakage risk, not the primary biological-feature baseline.                                                                                                                                           |

No selected DNE feature subset should be fed to either production extractor on the basis of this comparison. A selected model's extraction contract must be validated separately for MediaPipe and HMR2 before any integration is considered.

## 6. Metrics and Reproducibility

For each dataset, selector, feature count, and outer split, report:

- Primary metrics at the participant level: aggregate each participant's outer-fold trial/recording probabilities by the predeclared mean, then apply the fixed threshold. Report per-trial/per-camera metrics only as secondary, correlated observations.
- Accuracy, precision, PD sensitivity/recall, healthy specificity, F1, confusion matrix, and ROC-AUC only when both classes are represented at the evaluation level.
- Primary unit and counts: number of participants, recordings/trials, per-class participant counts, and per-class recording counts. For TULIP, the participant-level result is primary; camera-level scores are secondary and clustered.
- OOF predictions aggregated from outer validation folds only. Report fold-level metrics plus pooled OOF confusion matrix. Do not use training accuracy as generalization evidence.
- Brier score and log loss for probabilities. Reliability plots/calibration claims only when the number of independent participants is adequate; calibration fitting must be cross-fitted inside training folds. TULIP's 8 participants are insufficient for a meaningful calibration claim.
- Selected feature names per fold, feature-selection frequency/stability, selector settings, correlation cutoff, exact input schema/order, and any exclusions.
- Split strategy and participant IDs per fold, seeds (42 primary; 123 and 2026 sensitivity runs), classifier settings, class/sample weights, threshold (0.5 primary), software/library versions, data-file hashes, and code revision.

Use a fixed 0.5 decision threshold for the primary comparison. Any threshold selected to target sensitivity or specificity must be chosen only using inner training-fold predictions, then frozen before outer validation. If participant-equal weighting is used to avoid allowing people with more trials/camera views to dominate, calculate weights from training participants only and report a sensitivity comparison with unweighted recording-level fits.

## 7. Risks, Questions, and Go/No-Go

### Go for Phase 4, with limits

- A DNE-only grouped internal comparison of 288 full pose statistics against the reduced counts is designable; its interpretation remains internal and subject to session confounding and previous DNE feature-analysis exposure.
- A TULIP-only grouped exploratory analysis on the 43-row candidate is designable if treated as exploratory and not independent confirmation.
- Keep all preprocessing, feature filtering, ranking, scaling, selection, calibration, and any threshold choice within training participants. Use no YouTube examples.

### No-go until resolved

- Do not combine DNE and TULIP, assert feature equivalence, or integrate a reduced model with both live pipelines.
- Do not call the current TULIP cohort a locked external test. Prior TULIP use in model selection/training invalidates that claim.
- Do not use the disputed camera-3 recording until identity provenance is authoritative.
- Do not claim statistically reliable calibration or clinical utility from eight TULIP participants.

### Decisions requiring user approval

1. Approve Phase 4 as two separate exploratory tracks (DNE-only grouped CV plus TULIP-only grouped LOSO), with no combined dataset and no production integration.
2. Confirm that the 43-row TULIP candidate is acceptable for exploratory analysis while permanently excluding the disputed row unless provenance is supplied.
3. Confirm whether the Phase 4 RF settings should remain fixed to the existing 300-tree, balanced-weight baseline for the primary feature-count comparison, with tuning deferred to nested DNE-only analysis.
4. Supply authoritative provenance for the disputed TULIP identity and, if confirmatory external validation is desired, identify/provide a genuinely new verified participant cohort.

## 8. Phase 3 Completion

Inspected the current Phase 2 reports and candidate; feature contract, shared extractor, DNE builders, MediaPipe landmark-to-rotation code, HMR2/PHALP converter, and existing training/feature-analysis scripts. The candidate's row/group/label/finite-value integrity was checked with a read-only CSV parser. No model training, performance evaluation, package installation, or application change was performed.

Created: `docs/phase3_feature_selection_validation_plan.md` only. Existing Phase 2 files and all models/datasets are unchanged by this phase.

**Phase 4 readiness:** ready only for the two separate, exploratory within-domain plans described above, after approval. Not ready for combined DNE+TULIP training, independent TULIP validation, or production integration.
