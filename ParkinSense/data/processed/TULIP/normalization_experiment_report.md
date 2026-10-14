# ParkinSense — Controlled Normalization Experiments Report

## 1. Overview
This experiment evaluates whether spatial and spatiotemporal domain normalization improves the MediaPipe gait classification pipeline on the unseen 8-subject TULIP benchmark dataset. All experiments use the unchanged baseline model (`pd_detector_baseline.pkl`) at threshold 0.50 without retraining or tuning.

---

## 2. Statistical Comparison Across Experiments

| Metric | Baseline (Exp A) | Spatial Normalization (Exp B) | Spatiotemporal Normalization (Exp C) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **37.50%** | **37.50%** | **37.50%** |
| **Sensitivity / Recall** | **0.00%** | **0.00%** | **0.00%** |
| **Specificity** | **100.00%** | **100.00%** | **100.00%** |
| **Precision** | **0.00%** | **0.00%** | **0.00%** |
| **F1-Score** | **0.0000** | **0.0000** | **0.0000** |
| **ROC-AUC** | **0.8333** | **0.5333** | **0.5333** |
| **HT Mean $P(\text{PD})$** | **0.4612** | **0.4232** | **0.4232** |
| **PD Mean $P(\text{PD})$** | **0.4660** | **0.4227** | **0.4227** |
| **Probability Delta ($P_{\text{PD}} - P_{\text{HT}}$)** | **0.0047** | **-0.0005** | **-0.0005** |
| **Confusion Matrix [TN, FP, FN, TP]** | [3, 0, 5, 0] | [3, 0, 5, 0] | [3, 0, 5, 0] |

---

## 3. Feature Domain Shift Statistics (Training vs TULIP)

| Feature Shift Metric | Before Normalization (Exp A) | After Spatial Normalization (Exp B) | Reduction / Change |
| :--- | :---: | :---: | :---: |
| **Median $D_j$** | **14.4740** | **7.6602** | **6.8138 lower shift** |
| **90th Percentile $D_j$** | **121.8673** | **102.0625** | **19.8048 lower shift** |
| **Features with $D_j > 2.0$** | **245 / 290** | **227 / 290** | **-18 features** |
| **Features with $D_j > 3.0$** | **229 / 290** | **192 / 290** | **-37 features** |
| **Features with $R_j > 5.0$** | **134 / 290** | **89 / 290** | **-45 features** |

---

## 4. Subject-Level Probability Breakdown Across Experiments

### Experiment A (Baseline)
| subject_id | category | true_label | mean_pd_probability | subject_pred_label | correct_pred |
| --- | --- | --- | --- | --- | --- |
| subject01 | HT | 0 | 0.462 | 0 | True |
| subject07 | HT | 0 | 0.4606 | 0 | True |
| subject12 | HT | 0 | 0.4611 | 0 | True |
| subject02 | PD | 1 | 0.4606 | 0 | False |
| subject05 | PD | 1 | 0.4647 | 0 | False |
| subject08 | PD | 1 | 0.47 | 0 | False |
| subject11 | PD | 1 | 0.4707 | 0 | False |
| subject13 | PD | 1 | 0.4639 | 0 | False |

### Experiment B (Spatial Normalization)
| subject_id | category | true_label | mean_pd_probability | subject_pred_label | correct_pred |
| --- | --- | --- | --- | --- | --- |
| subject01 | HT | 0 | 0.4313 | 0 | True |
| subject07 | HT | 0 | 0.4161 | 0 | True |
| subject12 | HT | 0 | 0.4222 | 0 | True |
| subject02 | PD | 1 | 0.4245 | 0 | False |
| subject05 | PD | 1 | 0.426 | 0 | False |
| subject08 | PD | 1 | 0.4267 | 0 | False |
| subject11 | PD | 1 | 0.4173 | 0 | False |
| subject13 | PD | 1 | 0.4189 | 0 | False |

### Experiment C (Spatiotemporal Normalization)
| subject_id | category | true_label | mean_pd_probability | subject_pred_label | correct_pred |
| --- | --- | --- | --- | --- | --- |
| subject01 | HT | 0 | 0.4313 | 0 | True |
| subject07 | HT | 0 | 0.4161 | 0 | True |
| subject12 | HT | 0 | 0.4222 | 0 | True |
| subject02 | PD | 1 | 0.4245 | 0 | False |
| subject05 | PD | 1 | 0.426 | 0 | False |
| subject08 | PD | 1 | 0.4267 | 0 | False |
| subject11 | PD | 1 | 0.4173 | 0 | False |
| subject13 | PD | 1 | 0.4189 | 0 | False |
