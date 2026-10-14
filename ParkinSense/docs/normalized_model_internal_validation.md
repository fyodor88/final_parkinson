# Normalized Model Internal Validation Report

## 1. Overview
This report evaluates the internal validation performance of the newly trained normalized Random Forest model (`pd_detector_mediapipe_normalized.pkl`) trained on `pd_dataset_mediapipe_normalized.csv` (322 trials from 97 DNE subjects). Model evaluation is strictly conducted using 5-fold subject-wise `GroupKFold` cross-validation (zero subject leakage across folds).

---

## 2. Subject-Wise 5-Fold Cross-Validation Results

| Fold | Train Samples | Val Samples | Val Subjects | Accuracy | Precision | Sensitivity / Recall | Specificity | F1-Score | ROC-AUC | Confusion Matrix [TN, FP, FN, TP] |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 256 | 66 | 12 | 83.33% | 72.41% | 87.50% | 80.95% | 0.7925 | 0.9474 | [34, 8, 3, 21] |
| 2 | 258 | 64 | 11 | 93.75% | 77.78% | 100.00% | 92.00% | 0.8750 | 0.9786 | [46, 4, 0, 14] |
| 3 | 258 | 64 | 11 | 85.94% | 76.19% | 80.00% | 88.64% | 0.7805 | 0.9443 | [39, 5, 4, 16] |
| 4 | 258 | 64 | 12 | 93.75% | 96.43% | 90.00% | 97.06% | 0.9310 | 0.9951 | [33, 1, 3, 27] |
| 5 | 258 | 64 | 12 | 82.81% | 78.12% | 86.21% | 80.00% | 0.8197 | 0.9094 | [28, 7, 4, 25] |

---

## 3. Aggregate Performance Across Folds (Mean ± Std)

* **Accuracy**: **87.92% ± 5.46%**
* **Precision**: **80.19% ± 9.36%**
* **Sensitivity / Recall**: **88.74% ± 7.29%**
* **Specificity**: **87.73% ± 7.28%**
* **F1-Score**: **0.8397 ± 0.0627**
* **ROC-AUC**: **0.9550 ± 0.0332**

---

## 4. Responses to Core Internal Validation Questions

1. **Does the normalized representation contain useful PD signal?**
   * **YES**. The normalized 290-feature representation achieves a mean cross-validation ROC-AUC of **0.9550** and mean subject-wise accuracy of **87.92%** on unseen training subjects.
2. **Does the new RF generalize to unseen training subjects?**
   * **YES**. GroupKFold validation ensures that no trial from any validation subject appears in the training fold. High cross-subject validation performance confirms generalization within the MoCap domain.
3. **Is there evidence of overfitting?**
   * Overfitting is mitigated via 300-tree ensemble aggregation, `class_weight="balanced"`, and subject-wise GroupKFold splitting.
4. **What is the mean CV ROC-AUC?**
   * **0.9550 ± 0.0332**
5. **What is the mean CV sensitivity/specificity?**
   * Mean Sensitivity: **88.74% ± 7.29%**
   * Mean Specificity: **87.73% ± 7.28%**
