# ParkinSense — MediaPipe Pose Pipeline Technical Audit

## Executive Summary
This document reports the technical audit of the **MediaPipe Pose-based Parkinson's gait classification pipeline** in ParkinSense. The audit inspected data structures, feature definitions, training code, prediction routines, cross-validation setup, and frontend integration to identify bugs, sources of leakage, calibration issues, and quality vulnerabilities.

---

## 1. End-to-End Data Flow & Pipeline Architecture

```
Walking Video (.mp4 / .avi / .mov)
  │
  ▼
backend/app.py [POST /api/predict]
  │
  ▼
backend/pipeline.py [analyze_video(mode="mediapipe")]
  │
  ▼
src/inference/mediapipe_predict.py [MediaPipePipeline.run()]
  │
  ├── 1. extract_landmarks_from_video()
  │      • Model: mediapipe.solutions.pose (complexity=1, det_conf=0.5, track_conf=0.5)
  │      • Extracts 3D world landmarks (33 joints) and 2D image landmarks
  │      • Check: requires >= 10 detected frames
  │
  ├── 2. build_smpl_dataframe() -> src/pose/mediapipe_smpl.py
  │      • Maps MediaPipe landmarks to 24-joint SMPL rotation topology
  │      • Computes 24 3x3 rotation matrices per frame (216 values)
  │
  ├── 3. SMPLFeatureExtractor.extract_features() -> src/features/smpl_feature_extractor.py
  │      • Converts rotation matrices into 72 3D axis-angle vectors via scipy.spatial.transform.Rotation
  │      • Computes summary statistics (mean, std, min, max) = 288 features
  │      • Appends recording features: fps (1) + num_frames (1) = 290 features total
  │
  ├── 4. ML Inference -> models/pd_detector.pkl
  │      • Model: RandomForestClassifier (300 trees, class_weight="balanced")
  │      • Predicts class 0 (Healthy) or 1 (Parkinson's)
  │      • Computes class probabilities [P(Healthy), P(Parkinson's)]
  │
  ├── 5. Key Gait Frame Extraction & Visualization
  │      • Scores frame kinematic variance, selects 5 key frames using non-maximum suppression
  │      • Renders skeleton overlays and zoom crops saved to static/gait_frames/<job_id>/
  │
  └── 6. Structured Response Return
         • API returns JSON with diagnosis, raw probabilities, risk level, gait metrics, and key frame URLs
```

---

## 2. Audit Findings & Confirmed Bugs

### Finding A: Non-Gait Recording Metadata Included as Model Features (`fps` and `num_frames`)
* **Location**: `src/features/smpl_feature_extractor.py` (lines 76–79)
* **Evidence**:
  ```python
  feature_dict: Dict[str, float] = {
      "fps": float(fps),
      "num_frames": int(poses_arr.shape[0]),
  }
  ```
* **Impact**: `fps` and `num_frames` reflect camera hardware settings and recording length rather than biological gait kinematics. If control and Parkinson's video collections in the dataset have different average durations or frame rates, the model can exploit these non-gait metadata features to achieve artificially inflated validation scores while performing poorly on real-world inputs.
* **Recommendation**: Exclude `fps` and `num_frames` from model features, leaving 288 pure biomechanical features.

### Finding B: Saved Production Model Trained on Incomplete Dataset Split
* **Location**: `src/models/train_pd_detector.py` (lines 58–93, 135)
* **Evidence**:
  ```python
  splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
  train_index, test_index = next(splitter.split(X, y, groups))
  ...
  model.fit(X_train, y_train)
  ...
  joblib.dump(model, MODEL_PATH)
  ```
* **Impact**: `models/pd_detector.pkl` was saved after fitting on only 80% of the subjects in `pd_dataset.csv`. 20% of the dataset subjects were completely left out of model training. Furthermore, training on a single split rather than cross-validation can overfit to specific subject splits.
* **Recommendation**: Evaluate cross-validation performance across all subjects and train final production candidates using hyperparameter-regularized cross-validation protocols.

### Finding C: Fully Grown Trees Without Leaf Regularization Causing Probability Ambiguity
* **Location**: `src/models/train_pd_detector.py` (lines 86–91)
* **Evidence**: `RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")` leaves `min_samples_leaf=1` and `max_depth=None` at default values.
* **Impact**: Unconstrained trees split until every leaf is pure, creating noisy decision boundaries. When evaluating borderline video samples, different trees in the forest make conflicting predictions, causing the averaged probability output (`predict_proba()`) to cluster around 45%–55%. This is raw tree disagreement, not a calibrated posterior probability.
* **Recommendation**: Apply leaf regularization (`min_samples_leaf=2..5`, `max_depth=10..15`), evaluate alternative algorithms (e.g. Regularized RBF SVM, HistGradientBoosting), and perform probability calibration (e.g. Platt scaling / Sigmoid).

### Finding D: Minimal Frame Threshold and Lack of Pose Quality Checks
* **Location**: `src/inference/mediapipe_predict.py` (lines 65, 813–814)
* **Evidence**:
  ```python
  _MIN_FRAMES_REQUIRED = 10
  ...
  if detected_count < _MIN_FRAMES_REQUIRED:
      raise RuntimeError(f"MediaPipe detected landmarks in only {detected_count} frames...")
  ```
* **Impact**: 10 frames correspond to 0.33 seconds of video at 30 fps. This is insufficient to capture even a single step cycle. Furthermore, there is no check for lower-body landmark visibility (hip, knee, ankle) or subject tracking stability. Poorly lit videos or partial body views produce unreliable landmarks that get fed into the classifier.
* **Recommendation**: Implement minimum sequence duration (e.g. >= 45–60 frames), lower-body visibility thresholds, and return an explicit `inconclusive` status for unusable videos.

### Finding E: Target Class & Probability Mapping Verification
* **Location**: `data/processed/pd_dataset.csv`, `src/models/train_pd_detector.py`, `src/inference/mediapipe_predict.py`
* **Evidence**:
  - `pd_dataset.csv` contains `diagnosis = 0` (Healthy Control, 205 rows, 28 subjects) and `diagnosis = 1` (Parkinson's, 117 rows, 30 subjects).
  - `pd_detector.pkl` model `classes_` is `[0, 1]`.
  - `predict_proba()[:, 1]` extracts class 1 probability.
  - `src/inference/mediapipe_predict.py`:
    ```python
    pd_probability = float(pd_probs[1]) if len(pd_probs) > 1 else ...
    diagnosis_name = "Parkinson's Likely" if pd_pred_class == 1 else "Healthy / Control"
    ```
* **Status**: **VERIFIED CORRECT**. Index 1 corresponds to Parkinson's disease. Class probabilities for Healthy and Parkinson's sum to 100.0%.

---

## 3. Recommended Remediation Plan

1. **Establish Baseline & Comparison (Phase B)**:
   - Perform 5-fold subject-grouped cross-validation (`GroupKFold` on `subject`).
   - Benchmark: Current RF, Regularized RF (`min_samples_leaf=2, 3, 5`), RBF SVM with `StandardScaler`, and `HistGradientBoostingClassifier`.
   - Evaluate with and without `fps` / `num_frames`.
2. **Landmark Quality Checks & Inconclusive Status (Phase C & G)**:
   - Require minimum 45 detected frames (1.5 seconds at 30 fps).
   - Check mean visibility of lower body joints (hips, knees, ankles) >= 0.5.
   - Return `status = "inconclusive"` when quality checks fail or when probability is near 50%.
3. **Probability Calibration (Phase F)**:
   - Assess Brier scores and apply Platt scaling (`CalibratedClassifierCV`) within nested cross-validation.
4. **Model Selection & Safety (Phase E & H)**:
   - Select best candidate based on ROC-AUC, Brier score, and balanced sensitivity/specificity.
   - Save to `models/pd_detector_mediapipe_candidate.pkl` for review before replacing production model.
