import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# =========================
# CONFIGURATION
# =========================

DATA_PATH = "data/processed/pd_dataset.csv"

N_SPLITS = 5

# =========================
# 1. LOAD DATASET
# =========================

print("Loading PD dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)

# =========================
# 2. FEATURES AND TARGET
# =========================

target = "diagnosis"

ignore_columns = [
    "subject",
    "trial",
    "diagnosis",
    "diagnosis_name"
]

X = df.drop(columns=ignore_columns)
y = df[target]
groups = df["subject"]

print("Number of Features:", X.shape[1])
print("Number of Samples:", X.shape[0])
print("Number of Subjects:", groups.nunique())

print("\nClass Distribution:")
print(y.value_counts())

# =========================
# 3. 5-FOLD SUBJECT-WISE CV
# =========================

group_kfold = GroupKFold(n_splits=N_SPLITS)

results = []

total_cm = np.zeros((2, 2), dtype=int)

print("\n" + "=" * 60)
print("5-FOLD SUBJECT-WISE VALIDATION")
print("=" * 60)

# =========================
# 4. CROSS VALIDATION
# =========================

for fold, (train_index, test_index) in enumerate(
    group_kfold.split(X, y, groups),
    start=1
):

    print(f"\n{'-' * 60}")
    print(f"FOLD {fold}")
    print(f"{'-' * 60}")

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    train_subjects = groups.iloc[train_index].nunique()
    test_subjects = groups.iloc[test_index].nunique()

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Training subjects:", train_subjects)
    print("Testing subjects:", test_subjects)

    # =========================
    # TRAIN MODEL
    # =========================

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    # =========================
    # PREDICTIONS
    # =========================

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # =========================
    # METRICS
    # =========================

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    sensitivity = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # Specificity
    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 0

    # ROC-AUC
    if y_test.nunique() == 2:
        auc = roc_auc_score(
            y_test,
            probabilities
        )
    else:
        auc = np.nan

    # Add confusion matrix
    total_cm += cm

    # =========================
    # PRINT FOLD RESULTS
    # =========================

    print("\nAccuracy:     ", round(accuracy, 4))
    print("Precision:    ", round(precision, 4))
    print("Sensitivity:  ", round(sensitivity, 4))
    print("Specificity:  ", round(specificity, 4))
    print("F1 Score:     ", round(f1, 4))
    print("ROC-AUC:      ", round(auc, 4))

    print("\nConfusion Matrix:")
    print(cm)

    results.append({
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": auc
    })

# =========================
# 5. RESULTS SUMMARY
# =========================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION SUMMARY")
print("=" * 60)

for column in results_df.columns:

    mean = results_df[column].mean()
    std = results_df[column].std()

    print(
        f"{column.upper():15s}: "
        f"{mean:.4f} ± {std:.4f}"
    )

# =========================
# 6. OVERALL CONFUSION MATRIX
# =========================

print("\n" + "=" * 60)
print("OVERALL CONFUSION MATRIX")
print("=" * 60)

print("\n                 Predicted")
print("              Healthy   Parkinson")
print("Actual Healthy   ", total_cm[0][0], "       ", total_cm[0][1])
print("Actual Parkinson ", total_cm[1][0], "       ", total_cm[1][1])

# =========================
# 7. SAVE RESULTS
# =========================

results_df.to_csv(
    "data/processed/pd_cross_validation_results.csv",
    index=False
)

print("\nDetailed fold results saved to:")
print("data/processed/pd_cross_validation_results.csv")