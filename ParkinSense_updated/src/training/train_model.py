# import pandas as pd
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report


# # =========================
# # 1. LOAD DATASET
# # =========================

# DATA_PATH = "data/processed/dataset.csv"
# MODEL_PATH = "models/random_forest.pkl"

# print("Loading dataset...")

# df = pd.read_csv(DATA_PATH)

# print("Dataset Shape:", df.shape)


# # =========================
# # 2. SEPARATE FEATURES
# #    AND TARGET
# # =========================

# target = "updrs_gait"

# # These columns are NOT ML features
# ignore_columns = [
#     "subject",
#     "trial",
#     "medication",
#     "updrs_gait"
# ]

# X = df.drop(columns=ignore_columns)
# y = df[target]

# print("Number of Features:", X.shape[1])
# print("Number of Samples:", X.shape[0])


# # =========================
# # 3. TRAIN / TEST SPLIT
# # =========================

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.20,
#     random_state=42,
#     stratify=y
# )

# print("Training Samples:", X_train.shape[0])
# print("Testing Samples:", X_test.shape[0])


# # =========================
# # 4. TRAIN RANDOM FOREST
# # =========================

# print("\nTraining Random Forest...")

# model = RandomForestClassifier(
#     n_estimators=200,
#     random_state=42,
#     n_jobs=-1
# )

# model.fit(X_train, y_train)


# # =========================
# # 5. EVALUATE
# # =========================

# predictions = model.predict(X_test)

# accuracy = accuracy_score(y_test, predictions)

# print("\nAccuracy:", accuracy)

# print("\nClassification Report:")
# print(classification_report(y_test, predictions))


# # =========================
# # 6. SAVE MODEL
# # =========================

# print("\nSaving model...")

# joblib.dump(model, MODEL_PATH)

# print("Model saved successfully!")
# print("Location:", MODEL_PATH)