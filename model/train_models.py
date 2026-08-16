"""
ML Assignment 2 - Model Training Script
Dataset: Telco Customer Churn (IBM sample dataset, widely distributed via
Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Trains 5 classification models to predict customer churn (Yes/No) and
computes evaluation metrics for each. Saves trained models, the fitted
scaler, the final (post-encoding) feature column list, and a label
encoder for the target, plus a ready-to-upload test_data.csv that
matches the app's expected schema exactly.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)
import joblib

RANDOM_STATE = 42

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
df = pd.read_csv("../raw_data/Telco-Customer-Churn.csv")
print(f"Raw dataset shape: {df.shape}")

# Drop the identifier column — not a predictive feature
df = df.drop(columns=["customerID"])

# TotalCharges is read as text because a few rows have blank values
# (new customers with 0 tenure). Coerce to numeric and impute the median.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
n_missing = df["TotalCharges"].isna().sum()
print(f"TotalCharges: {n_missing} blank values found, imputing with median")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# ---------------------------------------------------------
# 2. Encode target
# ---------------------------------------------------------
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df["Churn"])  # No=0, Yes=1
print(f"Target classes: {list(target_encoder.classes_)} -> {list(range(len(target_encoder.classes_)))}")

X_raw = df.drop(columns=["Churn"])

# ---------------------------------------------------------
# 3. Encode categorical features (one-hot)
# ---------------------------------------------------------
categorical_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
print(f"Categorical columns one-hot encoded: {categorical_cols}")
X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)
X = X.astype(float)
feature_names = X.columns.tolist()
print(f"Final feature count after encoding: {len(feature_names)}")

# ---------------------------------------------------------
# 4. Train/test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Save the test split as test_data.csv AT THE PROJECT ROOT (required
# repo structure), in the SAME already-encoded format the app expects
# — ready to upload as-is.
test_df = X_test.copy()
test_df["Churn"] = target_encoder.inverse_transform(y_test)
test_df.to_csv("../test_data.csv", index=False)
print(f"Saved test_data.csv with shape {test_df.shape}")

# ---------------------------------------------------------
# 5. Scale features
# ---------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_names, "feature_names.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

# ---------------------------------------------------------
# 6. Define and train models
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes": GaussianNB(),
    # n_estimators/max_depth kept modest to keep the pickled model small
    # for Streamlit Community Cloud's free-tier deployment limits.
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=RANDOM_STATE
    ),
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "AUC": round(roc_auc_score(y_test, proba), 4),
        "Precision": round(precision_score(y_test, preds), 4),
        "Recall": round(recall_score(y_test, preds), 4),
        "F1": round(f1_score(y_test, preds), 4),
        "MCC": round(matthews_corrcoef(y_test, preds), 4),
    }
    results.append(metrics)

    fname = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, fname)
    print(f"Saved {fname}")

# ---------------------------------------------------------
# 7. Save results summary
# ---------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("metrics_summary.csv", index=False)
print("\n=== Evaluation Metrics ===")
print(results_df.to_string(index=False))
