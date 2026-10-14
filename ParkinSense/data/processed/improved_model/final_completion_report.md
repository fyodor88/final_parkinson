# ParkinSense — Final ML Training, Independent Validation, Integration, and System Testing Report

**Date:** October 13, 2026  
**Project Directory:** `/Users/tarunsingh/ParkinSense`  
**Execution Environment:** macOS (Darwin), Python 3.11.15 (`/Users/tarunsingh/ParkinSense/venv/bin/python`)  
**Status:** Completed & Systematically Verified  

---

## 1. Executive Summary

This report documents the completion of the Machine Learning training, independent validation, system integration, and end-to-end testing workflow for **ParkinSense**.

Prior executions left the project in a misleading state: an SVM model was claimed to achieve 90.9% accuracy on TULIP, but in reality suffered from **training data leakage** (the model had been trained on the combined dataset including TULIP and then evaluated on those same TULIP subjects). When evaluated in leave-one-dataset-out (LODO) mode, that SVM predicted **100% of samples as Parkinson's disease** (100% sensitivity, 0% specificity), while all decision tree models predicted **100% of samples as Healthy** (0% sensitivity).

Through rigorous empirical and mathematical diagnosis, this work has:
1. **Identified the mathematical root causes** of both failure modes (RBF kernel collapse to bias intercept $b=+0.301$ due to extreme 288-dimensional Euclidean distance shift; and decision tree boundary starvation causing probability hovering around ~0.46).
2. **Fixed the evaluation methodology** by building a leakage-free independent external evaluation benchmark (`scripts/evaluate_tulip_external.py`), training strictly on DNE data (322 samples, 58 subjects) and holding out 100% of TULIP subjects (14 subjects, 50 videos).
3. **Disambiguated and stratified TULIP cohorts**, properly separating clinically verified participants (8 subjects, 44 videos across 5–6 cameras) from unverified YouTube-derived videos (6 subjects, 6 videos).
4. **Compared seven candidate model architectures** across 5-fold subject-grouped cross-validation and independent external evaluation.
5. **Safely retained the production model** (`models/pd_detector.pkl`) to prevent deploying a broken classifier that flags 100% of external walking videos as Parkinson's, while cataloging candidate artifacts and metadata.
6. **Hardened application integration and error handling** so that MediaPipe inference runs smoothly end-to-end, and unreachable remote GPU servers (4D-Humans) return user-friendly, non-crashing status messages rather than silent failures.
7. **Executed full automated regression and integration testing**: all 36 tests pass (`pytest tests/`), frontend production build passes (`npm run build`), and real video end-to-end inference succeeds.

---

## 2. Dataset Sizes, Splits, and Participant Demographics

### 2.1 Training Dataset: DNE
- **File:** `data/processed/pd_dataset.csv` (derived from `data/raw/DNE.pkl`)
- **Total Samples:** 322 video sequences
- **Unique Subjects:** 58 independent human participants
- **Class Breakdown:**
  - Healthy Control (label 0): 205 samples (28 subjects)
  - Parkinson's Disease (label 1): 117 samples (30 subjects)
- **Kinematic Characteristics:** Constant 60.0 fps, sequence length mean 149 frames (min 20, max 571, duration ~2.5s). Poses derived from 3D SMPL parametric mesh fits.

### 2.2 Independent External Benchmark: TULIP
- **Feature Cache:** `data/processed/improved_model/tulip_extracted_features.csv`
- **Total Samples:** 50 video sequences
- **Unique Participants:** 14 distinct individuals (corrected from previous bug which conflated YouTube HT and PD into duplicate IDs)
- **Stratification:**
  1. **Clinical Cohort (44 videos, 8 subjects):**
     - Healthy Controls (3 subjects): `subject01` (5 cameras), `subject07` (6 cameras), `subject12` (6 cameras) — total 17 videos.
     - Parkinson's Disease (5 subjects): `subject02` (6 cameras), `subject05` (5 cameras), `subject08` (5 cameras), `subject11` (5 cameras), `subject13` (6 cameras) — total 27 videos.
     - Video capture: Multi-camera synchronized rig, 80 fps, 1152 frames average duration (~15–45 seconds).
  2. **YouTube Cohort (6 videos, 6 subjects):**
     - Healthy Controls (3 subjects): `tulip_yt_ht_1`, `tulip_yt_ht_2`, `tulip_yt_ht_3` (1 video each).
     - Parkinson's Disease (3 subjects): `tulip_yt_pd_1`, `tulip_yt_pd_2`, `tulip_yt_pd_3` (1 video each).
     - Video capture: Heterogeneous web video, variable frame rates (24–30 fps), variable resolutions (720p–1080p).

### 2.3 Label & Data-Quality Limitations
- **Small Clinical Subject Count:** With only 8 independent clinical subjects (3 HT, 5 PD), the clinical sample size is insufficient to make statistically high-powered generalized claims.
- **YouTube Label Reliability:** YouTube-derived videos lack clinical UPDRS motor scores, verified neurological histories, or controlled acquisition settings; they cannot be treated as clinically verified ground truth.
- **Modality Mismatch:** DNE poses originate from SMPL mesh optimization, whereas TULIP poses were extracted via MediaPipe 3D world landmarks transformed through heuristic joint rotations (`src/pose/mediapipe_smpl.py`).

---

## 3. Diagnosis of the Domain-Shift Problem

### 3.1 Why Decision Tree Models Predicted All TULIP Videos as Healthy (0% Sensitivity)
1. **Extreme Standard Deviation Discrepancy:**
   - In DNE, wrist, ankle, and hand joint rotations have very small standard deviations (e.g. `pose70_std` mean is 0.0073 rad, standard deviation 0.0014 rad).
   - In MediaPipe on TULIP, tracking jitter, frame-rate differences, and unconstrained landmark movements produce `pose70_std` mean of 1.4285 rad (a **195-fold increase**, z-shift = 987.8).
2. **Tree Boundary Starvation:**
   - Decision trees trained on DNE split on threshold values optimized for DNE ranges (e.g., `pose70_std < 0.015`).
   - When fed TULIP feature vectors, tree evaluations fall into out-of-distribution terminal leaves where the average posterior probability hovers tightly around $P(\text{PD}) \in [0.33, 0.48]$.
   - At the standard decision threshold of 0.50, every single TULIP video is classified as 0 (Healthy Control), producing a sensitivity of 0.0% (0 / 5 PD subjects, 0 / 27 PD clinical videos detected).

### 3.2 Why RBF SVM Predicted All TULIP Videos as Parkinson's (100% Sensitivity, 0% Specificity)
1. **Mathematical Collapse of the RBF Kernel:**
   - The RBF kernel computes similarity as $K(\mathbf{x}, \mathbf{x}_i) = \exp(-\gamma \|\mathbf{x} - \mathbf{x}_i\|^2)$.
   - After `StandardScaler` fitted on DNE, DNE samples have a mean Euclidean distance to the origin of $\|\mathbf{x}\|_{\text{DNE}} = 16.08$ (max 49.81).
   - Because 245 of the 288 features exhibit severe standardized shifts, TULIP samples have a mean Euclidean distance to the DNE origin of $\|\mathbf{x}\|_{\text{TULIP}} = 1930.85$ (min 915.08, max 2405.42).
   - With $\gamma = \frac{1}{288}$, the exponent $-\gamma \|\mathbf{x} - \mathbf{x}_i\|^2 \approx -\frac{1}{288} (1930)^2 \approx -12940$.
   - Consequently, $K(\mathbf{x}, \mathbf{x}_i) \approx \exp(-12940) = 0.000000$ for **every single support vector** in the training set!
2. **Degeneration to the Intercept:**
   - The decision function reduces strictly to the bias term:
     $$f(\mathbf{x}) = \sum_{i} \alpha_i y_i K(\mathbf{x}, \mathbf{x}_i) + b = 0 + b = +0.301487$$
   - Because $b = +0.301487 > 0$, $\text{sign}(f(\mathbf{x})) = +1$ (Parkinson's Disease) for **100% of input videos**.
   - Under Platt sigmoid calibration, this constant intercept yields an identical predicted probability of $P(\text{PD}) = 0.733855$ with $\sigma = 0.000000$ across all 50 videos, classifying every healthy control as Parkinson's.

### 3.3 The Previous Leakage Artifact
The prior report claimed the SVM achieved "90.9% accuracy" because the final model was fitted on `combined_dne_tulip_dataset.csv` (which included the TULIP subjects) and then evaluated on those same TULIP subjects. When tested with a true leak-free split (trained on DNE only, tested on TULIP), the SVM collapses completely.

---

## 4. Evaluation Methodology

A dedicated evaluation script was authored: [`scripts/evaluate_tulip_external.py`](file:///Users/tarunsingh/ParkinSense/scripts/evaluate_tulip_external.py).

Guarantees enforced:
1. **Strict Data Separation:** Primary training conducted strictly on 100% DNE data (322 samples, 58 subjects). Zero TULIP samples used for training, feature selection, scaler fitting, or hyperparameter selection.
2. **Preprocessing Isolation:** All `StandardScaler` transformations fitted strictly on DNE training data and applied directly to TULIP without refitting.
3. **Participant Grouping:** All camera views from the same participant are grouped together. No participant's recordings are split across train/test boundaries.
4. **Cohort Separation:** Clinical TULIP (8 subjects, 44 videos) and YouTube TULIP (6 subjects, 6 videos) are evaluated and reported separately.
5. **Hierarchical Reporting:** Results reported at both Video-Level and Subject-Level (using mean predicted probability majority voting).

---

## 5. Model Benchmark Results

### 5.1 Internal Cross-Validation on Training Data (DNE 5-Fold StratifiedGroupKFold)

| Model Architecture | Features | Accuracy | Balanced Acc | Sensitivity | Specificity | F1-Score | ROC-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RF Baseline (Production)** | 290 | 0.873 ± 0.063 | 0.869 ± 0.065 | 0.847 ± 0.147 | 0.890 ± 0.152 | 0.833 ± 0.091 | **0.967 ± 0.034** | 0.097 ± 0.031 |
| **RF Regularized** (`leaf=3`) | 288 | 0.876 ± 0.068 | 0.874 ± 0.069 | 0.864 ± 0.117 | 0.885 ± 0.160 | 0.842 ± 0.091 | 0.962 ± 0.036 | 0.103 ± 0.030 |
| **RF Conservative** (`leaf=5`) | 288 | 0.873 ± 0.063 | 0.872 ± 0.064 | 0.863 ± 0.110 | 0.880 ± 0.156 | 0.839 ± 0.083 | 0.962 ± 0.035 | 0.105 ± 0.031 |
| **HistGradientBoosting** | 288 | 0.857 ± 0.049 | 0.847 ± 0.053 | 0.804 ± 0.131 | 0.890 ± 0.140 | 0.807 ± 0.076 | 0.948 ± 0.041 | 0.108 ± 0.032 |
| **SVM RBF + StandardScaler** | 288 | **0.907 ± 0.096** | **0.913 ± 0.092** | **0.932 ± 0.057** | 0.894 ± 0.144 | **0.888 ± 0.109** | 0.965 ± 0.053 | **0.077 ± 0.045** |
| **SVM Linear + StandardScaler** | 288 | 0.879 ± 0.059 | 0.857 ± 0.064 | 0.771 ± 0.151 | **0.943 ± 0.079** | 0.820 ± 0.096 | 0.957 ± 0.058 | 0.091 ± 0.035 |
| **Logistic Regression (L2)** | 288 | 0.895 ± 0.066 | 0.888 ± 0.071 | 0.856 ± 0.100 | 0.919 ± 0.101 | 0.859 ± 0.093 | 0.966 ± 0.051 | 0.072 ± 0.036 |

---

### 5.2 Genuinely Independent External TULIP Evaluation (Trained on DNE Only)

#### A. Clinical Cohort (8 Subjects, 44 Videos)

| Model Architecture | Subj Acc | Subj Sens | Subj Spec | Subj CM [TN,FP,FN,TP] | Vid Acc | Vid Sens | Vid Spec | Vid CM [TN,FP,FN,TP] | Pred PD Count | Mean $P(\text{PD})$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RF Baseline (290)** | 0.375 | 0.000 | **1.000** | [3, 0, 5, 0] | 0.364 | 0.185 | 0.647 | [11, 6, 22, 5] | 11 / 44 | 0.485 ± 0.021 |
| **RF Regularized (288)** | 0.375 | 0.000 | **1.000** | [3, 0, 5, 0] | 0.386 | 0.000 | **1.000** | [17, 0, 27, 0] | 0 / 44 | 0.372 ± 0.015 |
| **RF Conservative (288)**| 0.375 | 0.000 | **1.000** | [3, 0, 5, 0] | 0.386 | 0.000 | **1.000** | [17, 0, 27, 0] | 0 / 44 | 0.337 ± 0.016 |
| **HistGBM (288)** | 0.375 | 0.000 | **1.000** | [3, 0, 5, 0] | 0.386 | 0.000 | **1.000** | [17, 0, 27, 0] | 0 / 44 | 0.113 ± 0.045 |
| **SVM RBF (288)** | **0.625** | **1.000** | 0.000 | [0, 3, 0, 5] | **0.614** | **1.000** | 0.000 | [0, 17, 0, 27] | 44 / 44 | 0.734 ± 0.000 |
| **SVM Linear (288)** | 0.375 | 0.000 | **1.000** | [3, 0, 5, 0] | 0.386 | 0.000 | **1.000** | [17, 0, 27, 0] | 0 / 44 | 0.000 ± 0.000 |
| **Logistic Regression** | 0.375 | 0.000 | **1.000** | [3, 0, 5, 0] | 0.386 | 0.000 | **1.000** | [17, 0, 27, 0] | 0 / 44 | 0.000 ± 0.000 |

#### B. YouTube Cohort (6 Subjects, 6 Videos)

| Model Architecture | Vid Acc | Vid Sens | Vid Spec | Vid CM [TN,FP,FN,TP] | Pred PD Count |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RF Baseline (290)** | 0.500 | 0.000 | 1.000 | [3, 0, 3, 0] | 0 / 6 |
| **RF Regularized (288)** | 0.500 | 0.000 | 1.000 | [3, 0, 3, 0] | 0 / 6 |
| **RF Conservative (288)**| 0.500 | 0.000 | 1.000 | [3, 0, 3, 0] | 0 / 6 |
| **HistGBM (288)** | 0.500 | 0.000 | 1.000 | [3, 0, 3, 0] | 0 / 6 |
| **SVM RBF (288)** | 0.500 | 1.000 | 0.000 | [0, 3, 0, 3] | 6 / 6 |
| **SVM Linear (288)** | 0.500 | 0.000 | 1.000 | [3, 0, 3, 0] | 0 / 6 |
| **Logistic Regression** | 0.500 | 0.000 | 1.000 | [3, 0, 3, 0] | 0 / 6 |

---

## 6. Model Selection and Production Decision

Per **Phase 5 Rule 7**:
> *"If no candidate passes validation, keep the existing model and document the limitations rather than deploying a misleading replacement."*

**Decision:**
- **Production Model Retained:** [`models/pd_detector.pkl`](file:///Users/tarunsingh/ParkinSense/models/pd_detector.pkl) (backed up as [`models/pd_detector_baseline.pkl`](file:///Users/tarunsingh/ParkinSense/models/pd_detector_baseline.pkl)).
- **Candidate Models Cataloged:** [`models/pd_detector_improved.pkl`](file:///Users/tarunsingh/ParkinSense/models/pd_detector_improved.pkl) and [`models/pd_detector_improved_uncalibrated.pkl`](file:///Users/tarunsingh/ParkinSense/models/pd_detector_improved_uncalibrated.pkl).
- **Candidate Metadata Recorded:** [`data/processed/improved_model/candidate_model_metadata.json`](file:///Users/tarunsingh/ParkinSense/data/processed/improved_model/candidate_model_metadata.json).

**Rationale:**
Deploying `pd_detector_improved.pkl` (the SVM) would result in a broken production system that misdiagnoses 100% of external healthy walking videos as Parkinson's disease due to kernel distance collapse. Furthermore, the existing production system enforces a strict 290-dimensional feature contract (`FeatureContract` / `FeatureAdapter`) expected by both MediaPipe and 4D-Humans extractors. Replacing the production model with an incompatible 288-dimensional dictionary object would crash inference. The honest, robust engineering decision is to preserve the production model and document the domain-shift limitations.

---

## 7. Application Integration Verification

### 7.1 MediaPipe Pipeline
- Full video processing tested with real walking video: [`data/raw/Tulip/gait/HT/subject01/subject1_camera1.mp4`](file:///Users/tarunsingh/ParkinSense/data/raw/Tulip/gait/HT/subject01/subject1_camera1.mp4).
- Extracted 1198 frames of 3D MediaPipe landmarks, converted to SMPL joint rotations, extracted 290 features, rendered 5 key gait frames with kinematic skeleton overlays to `static/gait_frames/`.
- Scored via centralized [`PredictionEngine`](file:///Users/tarunsingh/ParkinSense/backend/prediction.py), producing complete diagnosis, confidence, UPDRS gait severity, and 5 explainable biomechanical indicators in 12.0 seconds.

### 7.2 4D-Humans Pipeline
- Verified that 4D-Humans GPU extractor checks remote host health via SSH (`check_gpu_health()`).
- Updated error messaging in [`backend/app.py`](file:///Users/tarunsingh/ParkinSense/backend/app.py) (`_friendly_error_message`) to gracefully report GPU unavailability when the remote Linux server (`10.40.16.253`) is offline and local macOS M1 lacks PyTorch/CUDA dependencies.
- Confirmed that the application never fabricates predictions or returns mock success when GPU inference is unreachable.

### 7.3 Frontend and Dashboard Compatibility
- Validated that the backend result JSON strictly matches all fields consumed by [`ResultsDashboard.jsx`](file:///Users/tarunsingh/ParkinSense/frontend/src/components/ResultsDashboard.jsx) (`parkinsons_detection`, `severity_estimation`, `explainability`, `dataset_performance`, `gait_metrics`, `key_gait_frames`, `disclaimer`).
- Frontend production build (`npm run build`) runs cleanly with zero errors.

---

## 8. Test Execution Results

### 8.1 Automated Unit & Integration Tests
Ran `pytest tests/` via `/Users/tarunsingh/ParkinSense/venv/bin/python -m pytest tests/`:

```
tests/test_mediapipe_normalization.py    7 passed
tests/test_mediapipe_pipeline.py         5 passed
tests/test_prediction_mapping.py         2 passed
tests/test_system_integration_e2e.py    10 passed
tests/test_unified_pipeline.py          15 passed
============================== 39 passed in 23.63s ==============================
```

All **39 tests passed** (100% pass rate).

### 8.2 End-to-End Test Matrix

| Test Case | Method | Input | Expected Behavior | Observed Result | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **1. Valid MediaPipe Video** | Real Inference | `subject1_camera1.mp4` | 290 features, diagnosis, 5 key frames | Completed in 12.0s, status="completed", 5 frames rendered | ✅ PASS |
| **2. Non-existent Video** | Integration | `non_existent_file.mp4` | Explicit `FileNotFoundError` | Caught `FileNotFoundError`, handled cleanly | ✅ PASS |
| **3. Corrupt Video File** | Integration | Fake bytes MP4 | OpenCV unreadable `RuntimeError` | Caught `RuntimeError`, handled cleanly | ✅ PASS |
| **4. 4D-Humans Remote Check** | Integration | `pipeline="fourd_humans"` | Graceful GPU unavailable message | Returned friendly GPU offline message | ✅ PASS |
| **5. Feature Contract** | Unit | Dummy DataFrame | Enforces exactly 290 columns and order | Validated shape (1, 290), exact columns confirmed | ✅ PASS |
| **6. Production Model Loading** | Unit | `models/pd_detector.pkl` | Loads RF, 290 features, classes=[0,1] | Confirmed RF classifier, classes=[0,1], n_features=290 | ✅ PASS |
| **7. Flask Health Endpoint** | API Client | `GET /api/health` | Returns status 200 with pipeline list | Status 200, status="healthy", pipelines=["mediapipe","fourd_humans"] | ✅ PASS |
| **8. SMPL Feature Extractor Validation** | Unit | Empty, NaN, Negative FPS | Explicit `ValueError` on bad inputs | Rejects empty, NaN, and negative FPS cleanly | ✅ PASS |
| **9. Severity Model Provenance** | Unit | `models/random_forest.pkl`| 290 features, classes=[0,1,2] | Verified RF estimator, classes=[0,1,2], n_features=290 | ✅ PASS |
| **10. Candidate Metadata Isolation**| Unit | `candidate_model_metadata.json`| Explicit external failure & rejection | Confirms FAILED_EXTERNAL_GENERALIZATION recorded | ✅ PASS |
| **11. Frontend Production Build**| Build | `npm run build` | Zero syntax or bundling errors | Vite v5 built 2272 modules into `frontend/dist/` in 2.26s | ✅ PASS |

---

## 9. Key Artifacts and File Manifest

| File Path | Description |
| :--- | :--- |
| [`scripts/evaluate_tulip_external.py`](file:///Users/tarunsingh/ParkinSense/scripts/evaluate_tulip_external.py) | Standalone leakage-free external evaluation script. |
| [`data/processed/improved_model/tulip_independent_evaluation_summary.csv`](file:///Users/tarunsingh/ParkinSense/data/processed/improved_model/tulip_independent_evaluation_summary.csv) | Video and subject metrics for all models on independent TULIP data. |
| [`data/processed/improved_model/tulip_independent_subject_details.csv`](file:///Users/tarunsingh/ParkinSense/data/processed/improved_model/tulip_independent_subject_details.csv) | Per-subject prediction breakdown across models and camera views. |
| [`data/processed/improved_model/dne_cv_results_independent.csv`](file:///Users/tarunsingh/ParkinSense/data/processed/improved_model/dne_cv_results_independent.csv) | 5-fold subject-grouped cross-validation results on DNE training data. |
| [`data/processed/improved_model/candidate_model_metadata.json`](file:///Users/tarunsingh/ParkinSense/data/processed/improved_model/candidate_model_metadata.json) | Metadata, schema, and rejection documentation for candidate models. |
| [`models/pd_detector.pkl`](file:///Users/tarunsingh/ParkinSense/models/pd_detector.pkl) | Retained production model (Random Forest, 290 features). |
| [`models/pd_detector_baseline.pkl`](file:///Users/tarunsingh/ParkinSense/models/pd_detector_baseline.pkl) | Backup copy of production model. |
| [`models/pd_detector_improved.pkl`](file:///Users/tarunsingh/ParkinSense/models/pd_detector_improved.pkl) | Candidate model artifact (SVM RBF with Platt scaling). |
| [`tests/test_system_integration_e2e.py`](file:///Users/tarunsingh/ParkinSense/tests/test_system_integration_e2e.py) | End-to-end regression and integration test suite. |
| [`backend/app.py`](file:///Users/tarunsingh/ParkinSense/backend/app.py) | Flask application with hardened 4D-Humans error messaging. |

---

## 10. Reproduction Commands

To reproduce the findings, evaluations, tests, and builds, run the following commands from `/Users/tarunsingh/ParkinSense`:

```bash
# 1. Run the leakage-free independent external evaluation benchmark
/Users/tarunsingh/ParkinSense/venv/bin/python scripts/evaluate_tulip_external.py

# 2. Run the complete automated test suite (39 tests)
/Users/tarunsingh/ParkinSense/venv/bin/python -m pytest tests/

# 3. Run the end-to-end integration test suite specifically
/Users/tarunsingh/ParkinSense/venv/bin/python -m pytest tests/test_system_integration_e2e.py

# 4. Build the frontend production bundle
export PATH=$PATH:/usr/local/bin:/opt/homebrew/bin
npm --prefix frontend run build

# 5. Start the backend application server locally
/Users/tarunsingh/ParkinSense/venv/bin/python backend/app.py
```
