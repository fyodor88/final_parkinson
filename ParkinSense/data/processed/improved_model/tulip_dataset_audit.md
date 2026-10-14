# TULIP Dataset Audit & Verification Report

**Date:** October 13, 2026  
**Location:** `data/raw/Tulip/gait/`  
**Output Directory:** `data/processed/improved_model/`  

---

## 1. Executive Summary

This audit establishes the provenance, participant grouping, label reliability, and eligibility criteria for all video assets in the TULIP gait dataset located at `data/raw/Tulip/gait/`.

Prior iterations in the project exhibited a critical flaw: video files from YouTube were conflated with clinical subjects (e.g. sharing `subject_yt_1` for both Healthy Control and Parkinson's Disease), and unverified YouTube clips were mixed into clinical training.

This audit establishes a strict boundary:
1. **Clinical TULIP Cohort (44 videos, 8 unique participants):** Clinically verified participants recorded in a controlled laboratory setting across synchronized multi-camera views (5–6 cameras per participant). Eligible for supervised training and participant-grouped validation.
2. **YouTube Web Cohort (6 videos, 6 unique participants):** Unverified online videos lacking UPDRS motor scores, medical history, and clinical documentation. Excluded from supervised training; held strictly for exploratory testing.

---

## 2. Dataset Inventory Summary

| Metric | Clinical TULIP Cohort | YouTube Web Cohort | Total Collection |
| :--- | :---: | :---: | :---: |
| **Total Video Files** | 44 | 6 | 50 |
| **Unique Human Participants** | 8 | 6 | 14 |
| **Healthy Control (HT / 0) Participants** | 3 (`subject01`, `subject07`, `subject12`) | 3 (`yt_ht_1`, `yt_ht_2`, `yt_ht_3`) | 6 |
| **Parkinson's Disease (PD / 1) Participants**| 5 (`subject02`, `subject05`, `subject08`, `subject11`, `subject13`) | 3 (`yt_pd_1`, `yt_pd_2`, `yt_pd_3`) | 8 |
| **Videos: Healthy Control** | 17 | 3 | 20 |
| **Videos: Parkinson's Disease** | 27 | 3 | 30 |
| **Frame Rates** | Constant ~80.01 FPS | Variable (23.98–60.0 FPS) | 23.98–80.05 FPS |
| **Resolution** | Standardized 768 × 480 | Heterogeneous (1280×720, 1920×1080) | 768×480 to 1920×1080 |
| **Multi-Camera Setup** | Synchronized 5–6 cameras per trial | Single uncalibrated web camera | 1–6 cameras |
| **Supervised Training Eligibility** | **Eligible** (Verified clinical labels) | **Excluded** (Unverified clinical labels) | 44 eligible / 6 excluded |

---

## 3. Participant Breakdown (Clinical Cohort)

All clinical recordings feature multi-camera capture of continuous walking bouts. Every camera view of a participant captures the identical underlying individual.

| Participant ID | Clinical Diagnosis | Class Target | Cameras Available | Frame Range | Duration Range (s) | Training Eligibility |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `subject01` | Healthy Control | 0 | Camera 1, 2, 3, 4, 5 (5 views) | 859–1085 | 10.74–13.56 | ✅ Eligible |
| `subject07` | Healthy Control | 0 | Camera 1, 2, 3, 4, 5, 6 (6 views) | 1205–1395 | 15.06–17.43 | ✅ Eligible |
| `subject12` | Healthy Control | 0 | Camera 1, 2, 3, 4, 5, 6 (6 views) | 1184–1288 | 14.80–16.10 | ✅ Eligible |
| `subject02` | Parkinson's Disease | 1 | Camera 1, 2, 3, 4, 5, 6 (6 views) | 1262–1347 | 15.77–16.83 | ✅ Eligible |
| `subject05` | Parkinson's Disease | 1 | Camera 1, 2, 3, 4, 5 (5 views) | 1178–1279 | 14.72–15.98 | ✅ Eligible |
| `subject08` | Parkinson's Disease | 1 | Camera 1, 2, 3, 4, 5 (5 views) | 1146–1295 | 14.32–16.18 | ✅ Eligible |
| `subject11` | Parkinson's Disease | 1 | Camera 2, 3, 4, 5, 6 (5 views) | 1238–1377 | 15.47–17.21 | ✅ Eligible |
| `subject13` | Parkinson's Disease | 1 | Camera 1, 2, 3, 4, 5, 6 (6 views) | 1157–1341 | 14.46–16.76 | ✅ Eligible |

---

## 4. Exclusion Documentation (YouTube Cohort)

The following 6 videos are excluded from supervised model fitting per Phase 2 guidelines:

| Recording ID | Subject ID | Category | Filename | Reason for Exclusion |
| :--- | :--- | :---: | :--- | :--- |
| `yt_pd_1_vidssave.com Parkinson's...` | `yt_pd_1` | PD | `vidssave.com Parkinson's Disease Freezing & Festinating Gait 1080P.mp4` | Unverified web video; no clinical protocol or UPDRS score |
| `yt_pd_2_vidssave.com Parkinsonian...` | `yt_pd_2` | PD | `vidssave.com Parkinsonian Gait 720P.mp4` | Unverified web video; no clinical protocol or UPDRS score |
| `yt_pd_3_vidssave.com Parkinsonian...` | `yt_pd_3` | PD | `vidssave.com Parkinsonian Gait 720P.mp4` | Unverified web video; no clinical protocol or UPDRS score |
| `yt_ht_1_videoplayback (2)` | `yt_ht_1` | HT | `videoplayback (2).mp4` | Unverified web video; no neurological history confirmation |
| `yt_ht_2_videoplayback (1)` | `yt_ht_2` | HT | `videoplayback (1).mp4` | Unverified web video; no neurological history confirmation |
| `yt_ht_3_videoplayback` | `yt_ht_3` | HT | `videoplayback.mp4` | Unverified web video; no neurological history confirmation |

---

## 5. Methodological Guidelines for Training & Validation

1. **Participant-Grouped Splitting:** In any cross-validation split, all recordings belonging to a participant MUST be assigned to the same fold. Splitting camera views of the same individual across train and test folds constitutes direct data leakage.
2. **Class Balance:** The clinical cohort contains 3 Healthy Control participants (17 recordings) and 5 Parkinson's Disease participants (27 recordings). Cross-validation schemes must use `GroupKFold` or `StratifiedGroupKFold` on the 8 participants.
3. **No External Test Tuning:** The candidate model trained on verified clinical TULIP data must be evaluated on its grouped validation folds without hyperparameter tuning on held-out samples.
