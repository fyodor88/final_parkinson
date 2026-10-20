# Phase 4 — Dataset Merge Report

Report date: 2026-10-19  
Verified against actual files in `/home/pranav/final_parkinson/ParkinSense/data/`

## 1. Source Datasets

### 1.1 DNE (Internal Dataset)

| Property | Value |
|----------|-------|
| Source file | `data/processed/pd_dataset.csv` |
| Rows (recordings) | 322 |
| Subjects | 58 (28 healthy, 30 PD) |
| Class distribution | HT: 205, PD: 117 |
| Pose features | 288 (`pose0_mean` … `pose71_max`) |
| Metadata | `fps` = 60 (constant), `num_frames` min 20 / max 571 |
| Missing/non-finite | 0 |
| Subject-label consistency | No subject has mixed labels |

### 1.2 TULIP Clinical (External Dataset)

| Property | Value |
|----------|-------|
| Source file | `data/processed/improved_model/phase2_tulip_clinical_candidate.csv` |
| Total recordings | 44 |
| Eligible recordings | 43 (1 excluded) |
| Excluded recording | `subject13_subject14_camera3` (disputed subject identity) |
| Subjects | 8 (3 healthy: subject01, subject07, subject12; 5 PD: subject02, subject05, subject08, subject11, subject13) |
| Class distribution (eligible) | HT: 17, PD: 26 |
| Pose features | 288 (contract-ordered) |
| Metadata | `fps` ≈ 80.0, `num_frames` 859–1395 |
| Missing/non-finite | 0 |

### 1.3 TULIP YouTube (Unverified External)

| Property | Value |
|----------|-------|
| Rows | 6 |
| Subjects | 6 (unique, no overlap with clinical) |
| Class distribution | HT: 3, PD: 3 |
| Label status | **UNVERIFIED** — kept out of all fitting and primary evaluation |

## 2. Merged Expanded Dataset

| Property | Value |
|----------|-------|
| Output file | `data/processed/phase4/expanded_training_dataset.csv` |
| Total rows | 372 |
| Eligible for training | 365 (322 DNE + 43 TULIP Clinical) |
| Not eligible | 7 (1 disputed TULIP Clinical + 6 YouTube) |
| Total subjects | 72 |
| Eligible subjects | 66 (58 DNE + 8 TULIP Clinical) |
| Feature columns | 302 (288 pose + 2 metadata + 12 ID/provenance) |
| Subject-label consistency | All subjects have consistent labels |
| Subject ID overlap DNE↔TULIP | None (string-ID disjoint; no identity crosswalk) |

## 3. Provenance Columns

The expanded dataset includes these provenance columns for traceability:

| Column | Description |
|--------|-------------|
| `sample_id` | Unique row identifier |
| `subject_id` | Participant identifier (groups recordings) |
| `recording_id` | Unique recording identifier |
| `dataset_source` | `DNE`, `TULIP_Clinical`, or `TULIP_YouTube` |
| `cohort` | Clinical cohort or source grouping |
| `label` | 0 (healthy) or 1 (PD) |
| `diagnosis_name` | Human-readable diagnosis label |
| `camera_id` | Camera identifier (TULIP) or N/A (DNE) |
| `extraction_method` | Feature extraction pipeline used |
| `eligible_for_training` | Boolean: whether this row participates in primary experiments |
| `retention_status` | Exclusion reason if applicable |

## 4. Quality Checks Performed

1. ✅ Subject-label consistency (no mixed-label subjects)
2. ✅ No NaN or Inf in 288 pose feature columns
3. ✅ No constant (zero-variance) pose features
4. ✅ Subject IDs unique within each source
5. ✅ No string-level subject ID overlap between DNE and TULIP
6. ✅ Disputed recording correctly excluded
7. ✅ YouTube recordings correctly excluded from eligible set
8. ✅ Feature column order matches contract (`pose0_mean`, `pose0_std`, …)

## 5. Limitations

1. **No identity crosswalk**: DNE and TULIP use different subject ID schemes; real-person overlap cannot be excluded.
2. **Different acquisition conditions**: DNE recorded at 60 fps; TULIP at ~80 fps. `fps` and `num_frames` reflect these differences.
3. **Feature domain shift**: Phase 2 documented that pose feature distributions differ between DNE and TULIP, even for the same gait channels. This is a property of the data, not a merge error.
4. **YouTube labels unverified**: Cannot be used for clinical evaluation.
5. **Small TULIP cohort**: Only 8 subjects (3 controls) — insufficient for reliable cross-domain generalization claims.
