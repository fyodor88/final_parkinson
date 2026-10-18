# Phase 4 — Repository Audit Report

Audit date: 2026-10-17 (all facts below were verified against files on disk in this session)

## 0. Workspace and location resolution

The project exists in two copies on this machine:

| Path | Git | Content |
| --- | --- | --- |
| `/Users/tarunsingh/final_parkinson/ParkinSense` | tracked inside workspace repo `/Users/tarunsingh/final_parkinson` (branch `main`, commit `b2e3a79` "Initial commit") | The project **plus** all Phase 2/3/4 artifacts from previous sessions |
| `/Users/tarunsingh/ParkinSense` | separate repo, uncommitted working-tree changes | Same source/data snapshot (verified byte-identical) **minus** the Phase 2/3/4 artifacts; has 4 newer `static/gait_frames/job_*` app-run folders |

A recursive diff (excluding `.git`, caches, `uploads/`, `remote_output/`, app outputs) showed the only differences are the prior-phase artifacts present in `final_parkinson/ParkinSense`. **All Phase 4 work in this session was therefore performed in `/Users/tarunsingh/final_parkinson/ParkinSense`**, which is the workspace root handed to this session and the only copy that contains the Phase 2–4 lineage. The other copy was treated as read-only.

Git status of the workspace repo: single initial commit; all Phase 2/3/4 artifacts are untracked. No tracked source/data/model file was modified before this session's work.

## 1. Repository inventory

```
ParkinSense/
├── backend/            Flask API, prediction engine, MediaPipe/4D-Humans pipeline adapters
├── src/                feature extraction, feature contract, training/inference modules
│   ├── features/       feature_contract.py (290-dim), smpl_feature_extractor.py, adapters
│   ├── pose/           mediapipe_smpl.py (landmarks → SMPL-like rotations), normalization
│   ├── inference/      predict.py, mediapipe_predict.py (production inference path)
│   └── training/       legacy training helpers
├── scripts/            dataset building, TULIP extraction, training, evaluation, Phase 4 runners
├── tests/              6 test files (MediaPipe, contract, E2E, Phase 4)
├── data/
│   ├── raw/            DNE.pkl, Tulip/gait/{HT,PD}/..., videos/
│   └── processed/      pd_dataset.csv, TULIP features, improved_model/, phase4_* run dirs
├── models/             production + candidate model pickles
├── docs/               final_* methodology/leakage/external docs + phase2/phase3 reports
├── reports/            model audit + training report; reports/phase4/ created this session
├── configs/, frontend/ React UI, 4D-Humans/ (HMR2/PHALP code copy)
```

## 2. Datasets (verified)

### 2.1 DNE (internal dataset, `data/processed/pd_dataset.csv`)

| Property | Value |
| --- | --- |
| Rows (trials) | 322 |
| Participants (`subject`) | 58 (28 healthy, 30 Parkinson's; no mixed-label subject) |
| Class distribution (rows) | 205 healthy (0), 117 Parkinson's (1) |
| Trials per participant | min 2, median 5, max 16 |
| Pose features | 288 (`pose0_mean` … `pose71_max`), exactly contract-ordered |
| Metadata | `fps` constant 60; `num_frames` min 20 / median 130 / max 571 |
| Identifiers | `subject` (grouping), `trial` (unique recording) |
| Missing / non-finite | 0 |
| Raw source | `data/raw/DNE.pkl`; builder `src/datasets/build_pd_dataset.py` |

### 2.2 TULIP clinical (external, `data/processed/improved_model/phase2_tulip_clinical_candidate.csv`)

| Property | Value |
| --- | --- |
| Rows (recordings) | 43 (of 44 clinical; one row excluded for unresolved identity) |
| Participants (`subject_id`) | 8 (3 healthy, 5 Parkinson's) |
| Class distribution (rows) | 17 healthy (0), 26 Parkinson's (1) |
| Pose features | 288, exactly contract-ordered |
| Metadata | `fps` ≈ 80.0; `num_frames` 859–1395 |
| Identifiers | `subject_id` (grouping), `recording_id` (unique) |
| Missing / non-finite | 0 |
| Excluded row | `subject13_subject14_camera3` (filename/directory identity conflict, Phase 2) |
| Feature source | MediaPipe extraction cache (`tulip_extracted_features.csv`), not re-extracted in this session |

### 2.3 TULIP auxiliary tables

- `tulip_training_dataset.csv`: 44 clinical rows (includes the disputed recording); used only for continuity checks.
- `tulip_youtube_eval_dataset.csv`: 6 rows / 6 subjects, labels 3/3 but marked **unverified** by Phase 2 — excluded from fitting; reportable descriptively only.
- `tulip_extracted_features.csv`: 50 rows (44 clinical + 6 YouTube), `source_dataset='tulip'`.
- Participant ID overlap between DNE and TULIP: **none** (string sets disjoint; no identity crosswalk exists, so real-person overlap cannot be excluded).

## 3. Feature contract — why 288 vs 290

`src/features/feature_contract.py` defines the production input as **290 columns in fixed order**:

1. `fps` (acquisition metadata)
2. `num_frames` (frame count metadata)
3. 288 pose statistics: for each of 72 axis-angle channels → `mean`, `std` (population), `min`, `max`.

The DNE table and all TULIP tables contain all 290 columns plus non-feature columns (`subject/trial/diagnosis/...`). The "288 vs 290" difference is **exactly the two metadata columns**, and nothing else:

- Phase 2/3 concluded `fps`/`num_frames` are acquisition metadata with different semantics per domain (DNE 60 fps, 20–571 frames vs TULIP ≈80 fps, 859–1395 frames) and must be excluded from primary predictors.
- The production baseline `models/pd_detector.pkl` nevertheless consumed all 290 columns (audit Finding A, `reports/mediapipe_model_audit.md`). The 290-feature experiment is therefore kept only as a labeled "legacy production-contract reproduction".
- No feature was added, removed, reordered, or fabricated in any Phase 4 experiment.

The production contract remains intact and is not modified by Phase 4 experiments.

## 4. Extraction pipelines

- **MediaPipe**: `src/inference/mediapipe_predict.py` → `src/pose/mediapipe_smpl.py` (33 world landmarks → 24 SMPL-like rotation matrices; identity fallback for degenerate joints) → `src/features/smpl_feature_extractor.py` (axis-angle + summary stats) → 290 contract. TULIP features were produced through this path/cache.
- **4D-Humans / HMR2**: `backend/fourd_humans.py` (remote GPU service; longest-tracklet HMR2/PHALP SMPL rotations) → shared matrix→axis-angle statistics → 290 contract.
- Both pipelines target the same 290-dim contract, but Phase 2 (`docs/phase2_feature_compatibility_report.md`) established that channel values are **not proven equivalent** across DNE / MediaPipe / HMR2 domains. Phase 4 neither assumes equivalence nor combines domains.

## 5. Model inventory (verified by loading with `joblib`)

| Artifact | Type | Features | Training data | Role |
| --- | --- | --- | --- | --- |
| `models/pd_detector.pkl` | RandomForestClassifier, 300 trees, balanced | 290 | DNE | **Current production classifier** (`backend/prediction.py`, `src/inference/predict.py`) |
| `models/pd_detector_baseline.pkl` | RandomForestClassifier | 290 | DNE | Earlier baseline copy |
| `models/pd_detector_mediapipe_normalized.pkl` | RandomForestClassifier, 300 trees | 290 | normalized DNE | Documented external-eval model |
| `models/pd_detector_improved.pkl` | dict: CalibratedClassifierCV(SVM RBF C=1) + StandardScaler | 288 | **DNE + TULIP combined (372 rows / 69 subjects)** | Historical candidate; TULIP-contaminated for external claims; not the default |
| `models/pd_detector_tulip_candidate.pkl` | dict: RF max_depth=4 | 288 | TULIP-informed | Historical candidate; not default |
| `models/random_forest.pkl` | Severity (UPDRS) model | — | — | Separate task |

Production model loading emits `InconsistentVersionWarning` under the experiment environment (pickled with scikit-learn 1.9.0; experiment env is 1.7.2). This is handled explicitly in the Phase 4 production-model evaluation (cross-checked under scikit-learn 1.9.1).

## 6. Prior results (read before rerunning)

1. `reports/mediapipe_model_audit.md` — Findings A–F (metadata leakage into features; calibration/regularization issues).
2. `data/processed/mediapipe_model_comparison.csv` — trial-level DNE grouped CV: RF baseline acc 0.8822 / AUC 0.9566; best SVM (C=0.5) acc 0.9039 / AUC 0.9638.
3. `reports/improved_model_training_report.md` — combined-data CV (DNE+TULIP): RF 0.841±0.073, SVM 0.816±0.081. **Leave-one-dataset-out DNE→TULIP: accuracy 0.40–0.60, AUC 0.40–0.50, every model collapses to a single class.** Its "External TULIP hold-out 0.9091 accuracy" is contaminated (TULIP rows were in training) and is not usable evidence.
4. `docs/final_normalized_external_evaluation.md` — normalized RF on TULIP: acc 37.5%, sens 0, spec 100%, AUC 0.667 (all-healthy collapse).
5. Phase 4 runs 3/4 (`data/processed/phase4_feature_experiments_run{3,4}/`, RF-only): DNE pooled participant-level accuracy 0.828–0.914, AUC 0.908–0.972 (seed-dependent); TULIP leave-one-subject-out: accuracy 0.500–0.625, **specificity 0.000 in every configuration (all-PD collapse)**, AUC 0.267–0.467. run4 added `full_288` and `legacy_290_metadata` arms. Runs 1/2 are empty directories (failed attempts; no results). Run 3 and run 4 agree exactly on all shared cells, so the pipeline is internally reproducible.

## 7. Audit conclusions: completed / missing / blockers

### Completed before this session

- Phase 2 data-integrity and feature-compatibility audits (docs + candidate CSV).
- Phase 3 feature-selection/validation plan.
- Phase 4 partial: RF-only feature-count experiments (k = 10/20/30/50 + full-288 + legacy-290) for DNE (3 seeds × 5-fold StratifiedGroupKFold) and TULIP (LOSO), pooled participant-level metrics, selected-feature JSONL, run metadata; a Phase 4 test file.
- Historical model audit and comparison artifacts described in §6.

### Missing before this session (targets of this session)

- Multi-model comparison (regularized RF, SVM-RBF, logistic regression) under identical grouped folds.
- Reproduction of the prior run in the same environment against the same numbers (efficiency check).
- Direct evaluation of the current production model on the verified 43-row TULIP candidate.
- Fold-wise metrics with mean ± SD; Brier score / log loss; explicit single-class collapse flags.
- External-generalization diagnostics: distribution shift quantification, domain classifier, metadata comparison, probability distributions, DNE-OOF-based calibration applied to TULIP scores.
- Confusion-matrix and diagnostic plots; selected-feature CSV (only per-fold JSONL existed).
- Separation of clinical vs YouTube TULIP reporting.
- Saved experimental models (none existed; prior Phase 4 intentionally saved no models).
- Repository audit report (this document) and final Phase 4 report.

### Blockers and caveats (cannot be resolved by more code)

1. **TULIP historical exposure**: earlier project models were trained on TULIP rows (`pd_detector_improved.pkl` trained on 372 rows / 69 subjects). No fully untouched external cohort exists, so any TULIP result in this session is *external-domain evaluation of a DNE-only-trained model lineage*, not independent clinical validation.
2. **No identity crosswalk** between DNE `subject` IDs and TULIP `subject_id`s; real-person overlap cannot be excluded, only string-ID overlap (none observed).
3. **Disputed recording** `subject13_subject14_camera3` remains excluded; identity unresolved.
4. **YouTube subset labels are unverified**; kept out of all fitting and selection.
5. **Only 8 TULIP participants (3 controls)** — calibration claims and fine-grained model selection on TULIP are statistically unsupportable.
6. **DNE collection/session confounding** documented in Phase 3: grouped CV removes subject leakage but not session effects (one collection block is all healthy).
7. DNE↔TULIP feature semantics are not established equivalent (Phase 2); this is a property of the data, not a fixable code bug.

## 8. Environment used

- Interpreter: `/opt/anaconda3/bin/python` — Python 3.13.9, scikit-learn 1.7.2, pandas 2.3.3, numpy 2.3.5, scipy 1.16.3, matplotlib 3.10.6, seaborn 0.13.2, pytest 8.4.2, mediapipe and opencv (5.0.0) importable.
- This matches `phase4_run_metadata.json` of run 4 exactly (Python 3.13.9 / scikit-learn 1.7.2), maximizing reproducibility of prior folds.
- Production model (pickled under scikit-learn 1.9.0) additionally cross-checked under the project venv with scikit-learn 1.9.1.
