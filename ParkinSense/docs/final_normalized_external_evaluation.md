# Final Normalized MediaPipe Model External TULIP Evaluation

## 1. Executive Summary
This document presents the final locked external evaluation of the newly trained normalized Random Forest model (`pd_detector_mediapipe_normalized.pkl`) on the 8-subject TULIP benchmark dataset (44 videos).

---

## 2. Subject-Level External Evaluation Metrics

* **Accuracy**: **37.50%**
* **Precision**: **0.00%**
* **Sensitivity / Recall**: **0.00%**
* **Specificity**: **100.00%**
* **F1-Score**: **0.0000**
* **ROC-AUC**: **0.6667**
* **Confusion Matrix [TN, FP, FN, TP]**: [3, 0, 5, 0]

---

## 3. Video-Level External Evaluation Metrics

* **Video Accuracy**: **38.64%**
* **Video Precision**: **0.00%**
* **Video Sensitivity**: **0.00%**
* **Video Specificity**: **100.00%**
* **Video F1-Score**: **0.0000**
* **Video ROC-AUC**: **0.5621**

---

## 4. Subject-by-Subject Prediction Breakdown

| Subject ID | Category | Ground Truth | Num Cameras | Mean $P(\text{PD})$ | Predicted Label | Correct? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **subject01** | HT | 0 | 5 | 0.3840 | 0 (Healthy) | ✅ True |
| **subject07** | HT | 0 | 6 | 0.3717 | 0 (Healthy) | ✅ True |
| **subject12** | HT | 0 | 6 | 0.3933 | 0 (Healthy) | ✅ True |
| **subject02** | PD | 1 | 6 | 0.3950 | 0 (Healthy) | ❌ False |
| **subject05** | PD | 1 | 5 | 0.3847 | 0 (Healthy) | ❌ False |
| **subject08** | PD | 1 | 5 | 0.3787 | 0 (Healthy) | ❌ False |
| **subject11** | PD | 1 | 5 | 0.3807 | 0 (Healthy) | ❌ False |
| **subject13** | PD | 1 | 6 | 0.3945 | 0 (Healthy) | ❌ False |

---

## 5. Controlled Three-Way Model Architecture Comparison

| Experiment | Accuracy | Sensitivity | Specificity | Precision | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Baseline (Old RF + Unnormalized)** | **37.50%** | **0.00%** | **100.00%** | **0.00%** | **0.0000** | **0.8333** |
| **Old RF + Normalized Features** | **37.50%** | **0.00%** | **100.00%** | **0.00%** | **0.0000** | **0.5333** |
| **New Normalized RF + Normalized Features** | **37.50%** | **0.00%** | **100.00%** | **0.00%** | **0.0000** | **0.6667** |
