"""
ML Assignment 2 - Streamlit App
Classification Model Explorer: Telco Customer Churn Dataset

Uses 5 PRE-TRAINED models (trained by model/train_models.py on the Telco
Customer Churn dataset). Upload a test CSV to evaluate against them —
the app tolerates column order, extra columns, minor header naming
differences, and missing/blank values, as long as the required feature
columns are present.
"""

import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
)

st.set_page_config(page_title="Telco Churn — Model Explorer", layout="wide")

st.title("📞 Telco Customer Churn — Classification Model Explorer")
st.markdown(
    "This app demonstrates **5 classification models** trained on the "
    "[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) "
    "dataset to predict whether a customer will churn. Upload a **test CSV** "
    "with the required feature columns to see live evaluation results — "
    "column order, extra columns, and minor naming differences are handled "
    "automatically."
)

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest.pkl",
}

TARGET_COL_ALIASES = ["churn", "target", "class", "label", "y", "outcome"]


@st.cache_resource
def load_artifacts():
    scaler = joblib.load("model/scaler.pkl")
    feature_names = joblib.load("model/feature_names.pkl")
    target_encoder = joblib.load("model/target_encoder.pkl")
    return scaler, feature_names, target_encoder


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def normalize_col(col: str) -> str:
    """Lowercase, strip, collapse whitespace/underscores for fuzzy matching."""
    return re.sub(r"[\s_]+", " ", str(col).strip().lower())


def find_target_column(columns):
    normalized = {normalize_col(c): c for c in columns}
    for alias in TARGET_COL_ALIASES:
        if alias in normalized:
            return normalized[alias]
    return None


def map_feature_columns(data_columns, required_features):
    """Match required feature names to the uploaded file's columns,
    tolerating case/whitespace differences. Returns (mapping, missing)."""
    normalized_data = {normalize_col(c): c for c in data_columns}
    mapping = {}
    missing = []
    for feat in required_features:
        key = normalize_col(feat)
        if key in normalized_data:
            mapping[feat] = normalized_data[key]
        else:
            missing.append(feat)
    return mapping, missing


def coerce_target(y_series, target_encoder):
    """Best-effort conversion of a target column to encoded integers,
    tolerating text labels (matched case-insensitively against the
    classes the model was trained on) or already-encoded integers."""
    y = y_series.copy()
    class_names = list(target_encoder.classes_)
    lower_class_map = {str(c).strip().lower(): i for i, c in enumerate(class_names)}

    numeric = pd.to_numeric(y, errors="coerce")
    if numeric.notna().all():
        unique_vals = set(pd.unique(numeric.dropna()).astype(int))
        valid_vals = set(range(len(class_names)))
        if unique_vals <= valid_vals:
            return numeric.astype(int), None
        return None, f"Target column has unexpected numeric values: {sorted(unique_vals)}"

    text = y.astype(str).str.strip().str.lower()
    mapped = text.map(lower_class_map)
    if mapped.isna().any():
        return None, (
            f"Could not interpret some target values. Expected one of: {class_names}"
        )
    return mapped.astype(int), None


scaler, feature_names, target_encoder = load_artifacts()
CLASS_NAMES = list(target_encoder.classes_)


def label_for(code):
    return target_encoder.inverse_transform([code])[0]


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])
model_choice = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Note:** The uploaded CSV must contain the {len(feature_names)} required "
    "feature columns (column order and extra columns don't matter — see "
    "`model/feature_names.pkl` or the README for the exact list, matching "
    "`data/test_data.csv`). A target column is optional — include one "
    f"(named `Churn`, `target`, `class`, `label`, etc., with values from "
    f"{CLASS_NAMES}) to see evaluation metrics; without it you'll still "
    "get predictions."
)

# ---------------------------------------------------------
# Main logic
# ---------------------------------------------------------
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        data.columns = [str(c).strip() for c in data.columns]

        if data.empty:
            st.error("The uploaded CSV is empty.")
            st.stop()

        # ---- Match feature columns (tolerant of order/casing) ----
        feature_map, missing_cols = map_feature_columns(data.columns, feature_names)
        if missing_cols:
            st.error(
                f"Uploaded CSV is missing {len(missing_cols)} required feature "
                f"column(s): {missing_cols}"
            )
            st.stop()

        X = data[[feature_map[f] for f in feature_names]].copy()
        X.columns = feature_names  # restore expected order/names

        # ---- Coerce features to numeric, handle blanks ----
        X = X.apply(pd.to_numeric, errors="coerce")
        n_missing = int(X.isna().sum().sum())
        if n_missing > 0:
            st.warning(
                f"{n_missing} missing/non-numeric feature value(s) found — "
                "filled with each column's mean for prediction."
            )
            X = X.fillna(X.mean(numeric_only=True))

        X_scaled = scaler.transform(X)

        model = load_model(MODEL_FILES[model_choice])
        preds = model.predict(X_scaled)
        proba = model.predict_proba(X_scaled)[:, 1]

        # ---- Optional target column for evaluation ----
        target_col = find_target_column(data.columns)
        y = None
        if target_col is not None:
            y, err = coerce_target(data[target_col], target_encoder)
            if err:
                st.warning(f"Found a target column ('{target_col}') but {err} "
                            "Showing predictions only, without evaluation metrics.")
                y = None

        if y is not None:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader(f"📊 Evaluation Metrics — {model_choice}")
                metrics = {
                    "Accuracy": accuracy_score(y, preds),
                    "AUC Score": roc_auc_score(y, proba),
                    "Precision": precision_score(y, preds),
                    "Recall": recall_score(y, preds),
                    "F1 Score": f1_score(y, preds),
                    "MCC Score": matthews_corrcoef(y, preds),
                }
                metrics_df = pd.DataFrame(
                    {"Metric": list(metrics.keys()), "Value": [round(v, 4) for v in metrics.values()]}
                )
                st.dataframe(metrics_df, hide_index=True, use_container_width=True)

                st.subheader("📋 Classification Report")
                report = classification_report(
                    y, preds, target_names=[str(c) for c in CLASS_NAMES], output_dict=True
                )
                st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

            with col2:
                st.subheader("🔲 Confusion Matrix")
                cm = confusion_matrix(y, preds)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=[str(c) for c in CLASS_NAMES],
                    yticklabels=[str(c) for c in CLASS_NAMES],
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

                st.subheader("🔍 Sample Predictions")
                preview = data.copy()
                preview["Predicted"] = [label_for(p) for p in preds]
                preview["Actual"] = [label_for(a) for a in y]
                st.dataframe(preview[["Predicted", "Actual"]].head(15), use_container_width=True)

            st.markdown("---")
            st.subheader("📈 Compare All Models on This Data")
            if st.checkbox("Show comparison across all 5 models"):
                comparison_rows = []
                for name, path in MODEL_FILES.items():
                    m = load_model(path)
                    p = m.predict(X_scaled)
                    pr = m.predict_proba(X_scaled)[:, 1]
                    comparison_rows.append({
                        "Model": name,
                        "Accuracy": round(accuracy_score(y, p), 4),
                        "AUC": round(roc_auc_score(y, pr), 4),
                        "Precision": round(precision_score(y, p), 4),
                        "Recall": round(recall_score(y, p), 4),
                        "F1": round(f1_score(y, p), 4),
                        "MCC": round(matthews_corrcoef(y, p), 4),
                    })
                st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, use_container_width=True)

        else:
            st.info(
                "No usable target/label column found in this file, so only "
                "predictions are shown below (no accuracy/metrics)."
            )
            st.subheader(f"🔍 Predictions — {model_choice}")
            preview = data.copy()
            preview["Predicted"] = [label_for(p) for p in preds]
            preview["Confidence (Churn=Yes)"] = np.round(proba, 4)
            st.dataframe(preview, use_container_width=True)

            st.markdown("---")
            st.subheader("📈 Compare Predictions Across All Models")
            if st.checkbox("Show predictions from all 5 models"):
                comparison = data.copy()
                for name, path in MODEL_FILES.items():
                    m = load_model(path)
                    p = m.predict(X_scaled)
                    comparison[name] = [label_for(v) for v in p]
                st.dataframe(comparison, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("👈 Upload a test CSV from the sidebar to begin.")
    st.markdown("### About the dataset")
    st.markdown(
        "- **Source:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) "
        "(IBM sample dataset, distributed via Kaggle)\n"
        "- **Instances:** 7,043 customers (1,409 in the test split provided)\n"
        "- **Features:** 19 raw attributes (demographics, account info, "
        f"services subscribed) → {len(feature_names)} features after "
        "one-hot encoding categorical columns\n"
        "- **Target:** Binary — whether the customer churned (`Yes` / `No`)"
    )
    st.markdown("### Flexible test data")
    st.markdown(
        f"Any CSV works as long as it contains the {len(feature_names)} "
        "required (already-encoded) feature columns — in any order, "
        "alongside any extra columns. A target/label column is optional; "
        "if present under a common name (`Churn`, `target`, `class`, "
        "`label`, or `y`) it's used to compute evaluation metrics, "
        "otherwise you'll just get predictions."
    )
