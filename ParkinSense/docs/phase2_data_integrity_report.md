# Phase 2 Data Integrity Report

Audit date: 2026-10-15

## Scope and handling

This was a data audit and preparation phase only. No model training, model evaluation, feature selection, or production/application changes were performed. Existing source tables and artifacts were left unchanged.

The verified clinical source was `data/processed/improved_model/tulip_training_dataset.csv` (44 video rows). Each row was cross-checked by `recording_id` against `tulip_video_inventory.csv` and `tulip_extraction_report.csv`; participant labels were also compared with `tulip_participant_manifest.csv` and `tulip_label_mapping.csv`.

The separate candidate file, `data/processed/improved_model/phase2_tulip_clinical_candidate.csv`, is a row-preserving subset of the verified clinical table. It excludes one recording whose participant identity cannot be established from the available project records. No labels, IDs, or feature values were inferred, repaired, or merged.

## Dataset counts

| Dataset / state                         | Video or sample rows | Unique participants/groups | Healthy (0) | Parkinson's (1) |                      Missing cells |               Exact duplicate rows |
| --------------------------------------- | -------------------: | -------------------------: | ----------: | --------------: | ---------------------------------: | ---------------------------------: |
| DNE processed table                     |                  322 |                         58 |         205 |             117 |                                  0 |                                  0 |
| TULIP verified clinical source          |                   44 |                          8 |          17 |              27 |                                  0 |                                  0 |
| Phase 2 TULIP candidate                 |                   43 |                          8 |          17 |              26 |                                  0 |                                  0 |
| Unverified YouTube collection, excluded |                    6 |           6 as inventoried |           3 |               3 | Not included in clinical candidate | Not included in clinical candidate |

The candidate contains three healthy-control participants and five Parkinson's participants. Every recording from a participant retains the same `subject_id`, so grouped validation must use `subject_id`, never a video-level random split. The 43 rows represent 43 recordings, not 43 independent participants.

## Verified clinical participant counts

| Subject ID  | Verified label  | Source rows | Candidate rows | Status                                         |
| ----------- | --------------- | ----------: | -------------: | ---------------------------------------------- |
| `subject01` | Healthy (0)     |           5 |              5 | Consistent                                     |
| `subject07` | Healthy (0)     |           6 |              6 | Consistent                                     |
| `subject12` | Healthy (0)     |           6 |              6 | Consistent                                     |
| `subject02` | Parkinson's (1) |           6 |              6 | Consistent                                     |
| `subject05` | Parkinson's (1) |           5 |              5 | Consistent                                     |
| `subject08` | Parkinson's (1) |           5 |              5 | Consistent                                     |
| `subject11` | Parkinson's (1) |           5 |              5 | Consistent                                     |
| `subject13` | Parkinson's (1) |           6 |              5 | One recording excluded for unresolved identity |

## Label checks and exclusions

The recorded mapping is `HT` / Healthy Control -> `0`, `PD` / Parkinson's Disease -> `1`. The label mapping describes the clinical labels as verified through the TULIP clinical protocol. In the 44-row clinical table, every `subject_id` has one label, all 44 recording IDs are unique, all IDs match an inventory row, and inventory labels/cohort/subject assignments agree with the feature table. DNE's dataset builder maps raw `healthy-control` to `0` and `parkinson` to `1`; the processed DNE table has no participant with mixed labels.

The six YouTube examples are excluded from this supervised candidate because the project manifest marks their clinical status as unverified. They are not promoted to verified examples based on filenames, video appearance, or directory labels.

One clinical recording is also excluded:

| Recording ID                  | Project-assigned subject | Label | Identity concern                                                                                                                                                                                                                                        | Candidate action                                         |
| ----------------------------- | ------------------------ | ----: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `subject13_subject14_camera3` | `subject13`              |     1 | The video is under the `subject13` directory and the manifest/inventory assign it to `subject13`, but its filename is `subject14_camera3.mp4`. No independent provenance in the inspected files resolves whether the filename or assignment is correct. | Excluded; no reassignment or label correction attempted. |

The separate inventory reports `subject13` with a camera 3 view, which is consistent with the directory assignment but does not resolve the filename conflict. The candidate therefore keeps the other five `subject13` recordings and does not infer that the disputed clip belongs to that participant. The filename `subject5_camrea5.mp4` has a spelling anomaly but is consistently indexed under `subject05` in the participant directory, inventory, and extraction report; it is retained with the original ID and filename unchanged.

## Feature-table integrity

The verified TULIP source has 44 unique recording IDs and 8 unique subject IDs. It has 288 `poseN_{mean,std,min,max}` columns plus `fps` and `num_frames`; those values parse as finite numbers. Clinical TULIP recordings have recorded frame counts from 859 to 1,395 and FPS from approximately 79.991 to 80.058. The candidate preserves the source columns, order, IDs, labels, and values. `fps` and `num_frames` are acquisition/sequence metadata, not part of the 288 pose-statistic feature set proposed for any future experiment.

The source manifest reports 44 verified clinical videos and 6 unverified YouTube videos. The 44-row clinical table has no missing feature cells or exact duplicate rows. These checks establish table consistency, not the clinical truth of the upstream labels or participant identities beyond their supplied provenance.

## Open issues and Phase 3 gate

- Resolve the `subject13` / `subject14_camera3` identity conflict using authoritative dataset provenance before considering the excluded recording.
- The candidate contains only 8 independent participants (3 healthy, 5 Parkinson's). This is too small for a credible independent TULIP subject holdout while also reserving enough subjects for development. Any future results must make this limitation explicit and keep all cameras for each participant in one group.
- The dataset is suitable as a separate TULIP-only analysis candidate, not as evidence that its values are interchangeable with DNE or with both live application extractors. See `docs/phase2_feature_compatibility_report.md`.
- No conclusion about model quality, calibration, or improvement is made in Phase 2.

## Proposed Phase 3 files (not changed in Phase 2)

1. New `src/training/train_tulip_candidate.py`: a reproducible, TULIP-only, subject-grouped experiment using the 43-row candidate; `fps` and `num_frames` excluded from predictors. Do not mix DNE or YouTube data absent a verified feature bridge.
2. New `tests/test_tulip_candidate_data.py`: assert candidate row count, unique video IDs, unique and label-consistent participant groups, no YouTube rows, finite pose features, and stable feature ordering.
3. New `docs/phase3_tulip_model_evaluation.md`: report only actually executed subject-grouped results and limitations.
4. A separately named candidate artifact such as `models/pd_detector_phase3_tulip_candidate.pkl`, only if Phase 3 is authorized and a training protocol is approved. Do not overwrite `models/pd_detector.pkl` or the existing `models/pd_detector_tulip_candidate.pkl`.

No changes to `backend/prediction.py`, `backend/pipeline.py`, Flask endpoints, frontend files, or the production artifact are proposed in Phase 3 on the current evidence. Integration requires resolving feature semantics and validating compatibility for both live pipelines first.
