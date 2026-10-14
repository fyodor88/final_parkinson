import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# =========================
# CONFIGURATION
# =========================

DATA_PATH = "data/processed/pd_dataset.csv"
MODEL_PATH = "models/pd_detector.pkl"

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

# Columns that must NOT be used as ML features
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

print("\nClass Distribution:")
print(y.value_counts())

# =========================
# 3. SUBJECT-WISE SPLIT
# =========================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

print("\nTraining Subjects:", groups.iloc[train_index].nunique())
print("Testing Subjects:", groups.iloc[test_index].nunique())

# =========================
# 4. TRAIN MODEL
# =========================

print("\nTraining Parkinson's Detector...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# =========================
# 5. PREDICTIONS
# =========================

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# =========================
# 6. EVALUATION
# =========================

accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 50)
print("PARKINSON'S DETECTOR RESULTS")
print("=" * 50)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Healthy", "Parkinson"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# ROC-AUC requires both classes in the test set
if y_test.nunique() == 2:
    auc = roc_auc_score(y_test, probabilities)
    print("\nROC-AUC:", round(auc, 4))

# =========================
# 7. SAVE MODEL
# =========================

print("\nSaving model...")

joblib.dump(model, MODEL_PATH)

print("Model saved successfully!")
print("Location:", MODEL_PATH)