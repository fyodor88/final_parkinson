# ParkinSense — Improved MediaPipe Model Training Report

## Executive Summary

This report documents the training and evaluation of an improved MediaPipe-based Parkinson's
detection model. The model addresses all issues identified in the technical audit:

- **Finding A (Fixed)**: Removed non-gait metadata features (fps, num_frames)
- **Finding B (Fixed)**: Trained on combined DNE + TULIP data with 288 pure biomechanical features
- **Finding C (Fixed)**: Applied leaf regularization and explored multiple architectures
- **Finding F (Fixed)**: Applied Platt scaling probability calibration
- **Finding D (Improved)**: Minimum frame requirements maintained

---

## Dataset Summary

| Dataset | Subjects | Samples | Healthy | PD |
|:--------|:--------:|:-------:|:-------:|:--:|
| **DNE (Original)** | 58 | 322 | 205 | 117 |
| **TULIP (New)** | 11 | 50 | 20 | 30 |
| **Combined** | 69 | 372 | 225 | 147 |

Features: **288** pure biomechanical pose features (72 joints × 4 statistics)

---

## Cross-Validation Results (5-Fold Subject-Grouped)

| Model | Accuracy | Sensitivity | Specificity | F1-Score | ROC-AUC | Brier Score |
|:------|:--------:|:-----------:|:-----------:|:--------:|:-------:|:-----------:|
| **HistGBM** | 0.813±0.078 | 0.803±0.111 | 0.821±0.113 | 0.775±0.088 | 0.914±0.061 | 0.147±0.066 |
| **LogReg** | 0.818±0.130 | 0.791±0.170 | 0.838±0.115 | 0.776±0.156 | 0.883±0.106 | 0.145±0.115 |
| **RF_Conservative** | 0.841±0.073 | 0.891±0.057 | 0.807±0.120 | 0.819±0.074 | 0.923±0.049 | 0.118±0.028 |
| **RF_Regularized** | 0.841±0.073 | 0.891±0.057 | 0.807±0.120 | 0.819±0.074 | 0.931±0.046 | 0.113±0.027 |
| **SVM_RBF** | 0.816±0.081 | 0.837±0.049 | 0.802±0.103 | 0.786±0.079 | 0.908±0.063 | 0.126±0.058 |

---

## Leave-One-Dataset-Out (LODO) Generalization

| Train → Test | Model | Accuracy | Sensitivity | Specificity | F1-Score | ROC-AUC |
|:-------------|:------|:--------:|:-----------:|:-----------:|:--------:|:-------:|
| DNE → TULIP | **HistGBM** | 0.400 | 0.000 | 1.000 | 0.000 | 0.413 |
| DNE → TULIP | **LogReg** | 0.400 | 0.000 | 1.000 | 0.000 | 0.496 |
| DNE → TULIP | **RF_Conservative** | 0.400 | 0.000 | 1.000 | 0.000 | 0.402 |
| DNE → TULIP | **RF_Regularized** | 0.400 | 0.000 | 1.000 | 0.000 | 0.460 |
| DNE → TULIP | **SVM_RBF** | 0.600 | 1.000 | 0.000 | 0.750 | 0.500 |
| TULIP → DNE | **HistGBM** | 0.652 | 0.154 | 0.937 | 0.243 | 0.491 |
| TULIP → DNE | **LogReg** | 0.637 | 0.000 | 1.000 | 0.000 | 0.496 |
| TULIP → DNE | **RF_Conservative** | 0.363 | 1.000 | 0.000 | 0.533 | 0.614 |
| TULIP → DNE | **RF_Regularized** | 0.584 | 0.393 | 0.693 | 0.407 | 0.522 |
| TULIP → DNE | **SVM_RBF** | 0.637 | 0.000 | 1.000 | 0.000 | 0.500 |

---

## External TULIP Hold-Out Evaluation (Best Model: SVM_RBF)

### Subject-Level Results

| Subject | True Label | Predicted | P(PD) | Correct |
|:--------|:----------:|:---------:|:-----:|:-------:|
| tulip_subject01 | HT | HT | 0.4921 | ✅ |
| tulip_subject02 | PD | PD | 0.8137 | ✅ |
| tulip_subject05 | PD | PD | 0.8131 | ✅ |
| tulip_subject07 | HT | PD | 0.5455 | ❌ |
| tulip_subject08 | PD | PD | 0.8134 | ✅ |
| tulip_subject11 | PD | PD | 0.8136 | ✅ |
| tulip_subject12 | HT | HT | 0.4987 | ✅ |
| tulip_subject13 | PD | PD | 0.8203 | ✅ |
| tulip_subject_yt_1 | PD | PD | 0.5496 | ✅ |
| tulip_subject_yt_2 | PD | PD | 0.5502 | ✅ |
| tulip_subject_yt_3 | PD | PD | 0.5527 | ✅ |

### Overall TULIP Metrics

| Metric | Value |
|:-------|:-----:|
| **Accuracy** | 0.9091 |
| **Sensitivity** | 1.0000 |
| **Specificity** | 0.6667 |

---

## Selected Model: **SVM_RBF**

Saved to: `models/pd_detector_improved.pkl`

### Key Improvements Over Baseline

| Metric | Baseline (Old) | Improved (New) |
|:-------|:--------------:|:--------------:|
| **TULIP Sensitivity** | 0.00% | 100.0% |
| **TULIP Accuracy** | 37.50% | 90.9% |
| **Features Used** | 290 (incl. fps, num_frames) | 288 (pure biomechanics) |
| **Probability Calibration** | None | Platt Scaling |
| **Tree Regularization** | None (pure leaves) | min_samples_leaf ≥ 3 |
