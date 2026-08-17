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

# ===========================================================
# Page config & global styling
# ===========================================================
st.set_page_config(
    page_title="Telco Churn — Model Explorer",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* ---- overall spacing ---- */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---- hero header ---- */
    .hero {
        background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 55%, #2563EB 100%);
        padding: 2rem 2.25rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
    }
    .hero h1 {
        margin: 0 0 0.4rem 0;
        font-size: 1.9rem;
        font-weight: 700;
    }
    .hero p {
        margin: 0;
        opacity: 0.92;
        font-size: 0.98rem;
        line-height: 1.5;
    }

    /* ---- section headers ---- */
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0.25rem 0 0.75rem 0;
        color: #1F2937;
    }

    /* ---- info / status pills ---- */
    .pill {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .pill-green { background: #DCFCE7; color: #15803D; }
    .pill-blue  { background: #DBEAFE; color: #1D4ED8; }

    /* ---- metric cards ---- */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 0.9rem 1rem 0.6rem 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* ---- dataframes ---- */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ---- sidebar ---- */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #E5E7EB;
    }

    /* ---- footer ---- */
    .footer-note {
        text-align: center;
        color: #9CA3AF;
        font-size: 0.8rem;
        margin-top: 2.5rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest.pkl",
}

MODEL_ICONS = {
    "Logistic Regression": "📈",
    "Decision Tree": "🌳",
    "KNN": "📍",
    "Naive Bayes": "🎲",
    "Random Forest (Ensemble)": "🌲",
}

TARGET_COL_ALIASES = ["churn", "target", "class", "label", "y", "outcome"]

CONFUSION_MATRIX_FIGSIZE = (2.6, 2.2)  # small, fixed-size confusion matrix


# ===========================================================
# Cached loaders
# ===========================================================
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("model/scaler.pkl")
    feature_names = joblib.load("model/feature_names.pkl")
    target_encoder = joblib.load("model/target_encoder.pkl")
    return scaler, feature_names, target_encoder


@st.cache_resource
def load_model(path):
    return joblib.load(path)


# ===========================================================
# Helpers
# ===========================================================
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


def section_title(text: str):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "blue"):
    st.markdown(f'<span class="pill pill-{kind}">{text}</span>', unsafe_allow_html=True)


def render_confusion_matrix(y_true, y_pred, class_names):
    """Small, fixed-size confusion matrix heatmap, centered in a narrow column."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=CONFUSION_MATRIX_FIGSIZE)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples", ax=ax,
        xticklabels=[str(c) for c in class_names],
        yticklabels=[str(c) for c in class_names],
        cbar=False, linewidths=0.5, linecolor="white",
        annot_kws={"size": 9},
    )
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    left, mid, right = st.columns([1, 1, 1])
    with mid:
        st.pyplot(fig, use_container_width=False)


scaler, feature_names, target_encoder = load_artifacts()
CLASS_NAMES = list(target_encoder.classes_)


def label_for(code):
    return target_encoder.inverse_transform([code])[0]


# ===========================================================
# Hero header
# ===========================================================
st.markdown(
    """
    <div class="hero">
        <h1>📞 Telco Customer Churn — Model Explorer</h1>
        <p>
            Compare <b>5 classification models</b> trained on the
            <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
               style="color:#E0E7FF; text-decoration:underline;" target="_blank">
               Telco Customer Churn</a> dataset. Upload a test CSV to get live
            predictions and evaluation metrics — column order, extra columns, and
            minor naming differences are handled automatically.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===========================================================
# Sidebar controls
# ===========================================================
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])

    view_mode = st.radio(
        "View",
        ["Single Model", "Compare All Models"],
        help="Single Model: explore one model in depth. "
             "Compare All Models: run every model on the same data side by side.",
    )

    model_choice = None
    if view_mode == "Single Model":
        model_choice = st.selectbox(
            "Select a model",
            list(MODEL_FILES.keys()),
            format_func=lambda m: f"{MODEL_ICONS.get(m, '🔹')}  {m}",
        )

    st.markdown("---")
    with st.expander("ℹ️  Data requirements", expanded=not uploaded_file):
        st.markdown(
            f"- Must contain the **{len(feature_names)} required feature columns** "
            "(order doesn't matter, extra columns are ignored).\n"
            "- See `model/feature_names.pkl` or the README for the exact list "
            "(matches `data/test_data.csv`).\n"
            "- A **target column is optional** — include one named `Churn`, "
            f"`target`, `class`, `label`, etc. with values from `{CLASS_NAMES}` "
            "to unlock evaluation metrics; without it you'll still get predictions."
        )

# ===========================================================
# Main logic
# ===========================================================
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
                f"⚠️ {n_missing} missing/non-numeric feature value(s) found — "
                "filled with each column's mean for prediction."
            )
            X = X.fillna(X.mean(numeric_only=True))

        X_scaled = scaler.transform(X)

        # ---- Optional target column for evaluation ----
        target_col = find_target_column(data.columns)
        y = None
        if target_col is not None:
            y, err = coerce_target(data[target_col], target_encoder)
            if err:
                st.warning(
                    f"Found a target column ('{target_col}') but {err} "
                    "Showing predictions only, without evaluation metrics."
                )
                y = None

        pill(f"{len(data):,} rows loaded", "green")
        st.write("")

        # ===================================================
        # MODE 1 — Single model deep dive
        # ===================================================
        if view_mode == "Single Model":
            with st.spinner(f"Running {model_choice}…"):
                model = load_model(MODEL_FILES[model_choice])
                preds = model.predict(X_scaled)
                proba = model.predict_proba(X_scaled)[:, 1]

            if y is not None:
                metrics = {
                    "Accuracy": accuracy_score(y, preds),
                    "AUC": roc_auc_score(y, proba),
                    "Precision": precision_score(y, preds),
                    "Recall": recall_score(y, preds),
                    "F1 Score": f1_score(y, preds),
                    "MCC": matthews_corrcoef(y, preds),
                }

                section_title(f"{MODEL_ICONS.get(model_choice, '🔹')} {model_choice} — Evaluation")

                m_cols = st.columns(6)
                for col, (name, val) in zip(m_cols, metrics.items()):
                    col.metric(name, f"{val:.3f}")

                st.write("")
                tab_report, tab_matrix, tab_samples = st.tabs(
                    ["📋 Classification Report", "🔲 Confusion Matrix", "🔍 Sample Predictions"]
                )

                with tab_report:
                    report = classification_report(
                        y, preds, target_names=[str(c) for c in CLASS_NAMES], output_dict=True
                    )
                    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

                with tab_matrix:
                    render_confusion_matrix(y, preds, CLASS_NAMES)

                with tab_samples:
                    preview = data.copy()
                    preview["Predicted"] = [label_for(p) for p in preds]
                    preview["Actual"] = [label_for(a) for a in y]
                    preview["Confidence (Churn=Yes)"] = np.round(proba, 4)
                    st.dataframe(
                        preview[["Predicted", "Actual", "Confidence (Churn=Yes)"]].head(15),
                        use_container_width=True,
                    )

            else:
                pill("No target column detected — predictions only", "blue")
                section_title(f"{MODEL_ICONS.get(model_choice, '🔹')} {model_choice} — Predictions")

                preview = data.copy()
                preview["Predicted"] = [label_for(p) for p in preds]
                preview["Confidence (Churn=Yes)"] = np.round(proba, 4)

                churn_rate = float(np.mean(preds == 1)) if 1 in set(np.unique(preds)) else float(
                    np.mean([label_for(p) == CLASS_NAMES[1] for p in preds])
                )
                k1, k2, k3 = st.columns(3)
                k1.metric("Rows scored", f"{len(preview):,}")
                k2.metric("Predicted churn rate", f"{churn_rate:.1%}")
                k3.metric("Avg. churn confidence", f"{proba.mean():.3f}")

                st.write("")
                st.dataframe(preview, use_container_width=True)

        # ===================================================
        # MODE 2 — Compare all models
        # ===================================================
        else:
            section_title("📈 Compare All Models")

            if y is not None:
                st.caption("Metrics computed against the target column found in your file.")
                comparison_rows = []
                with st.spinner("Scoring all 5 models…"):
                    for name, path in MODEL_FILES.items():
                        m = load_model(path)
                        p = m.predict(X_scaled)
                        pr = m.predict_proba(X_scaled)[:, 1]
                        comparison_rows.append({
                            "Model": f"{MODEL_ICONS.get(name, '🔹')} {name}",
                            "Accuracy": round(accuracy_score(y, p), 4),
                            "AUC": round(roc_auc_score(y, pr), 4),
                            "Precision": round(precision_score(y, p), 4),
                            "Recall": round(recall_score(y, p), 4),
                            "F1": round(f1_score(y, p), 4),
                            "MCC": round(matthews_corrcoef(y, p), 4),
                        })
                comp_df = pd.DataFrame(comparison_rows)
                st.dataframe(
                    comp_df.style.highlight_max(
                        subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
                        color="#DCFCE7",
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
                st.write("")
                st.bar_chart(comp_df.set_index("Model")[["Accuracy", "AUC", "F1"]])

            else:
                pill("No target column detected — predictions only", "blue")
                st.caption("No evaluation metrics available; showing each model's prediction per row.")
                comparison = data.copy()
                with st.spinner("Running all 5 models…"):
                    for name, path in MODEL_FILES.items():
                        m = load_model(path)
                        p = m.predict(X_scaled)
                        comparison[name] = [label_for(v) for v in p]
                st.dataframe(comparison, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    # ---------------------------------------------------------
    # Landing state — no file uploaded yet
    # ---------------------------------------------------------
    st.info("👈 Upload a test CSV from the sidebar to begin.")

    col_about, col_flex = st.columns(2)

    with col_about:
        section_title("📊 About the dataset")
        st.markdown(
            "- **Source:** [Telco Customer Churn]"
            "(https://www.kaggle.com/datasets/blastchar/telco-customer-churn) "
            "(IBM sample dataset, distributed via Kaggle)\n"
            "- **Instances:** 7,043 customers (1,409 in the test split provided)\n"
            "- **Features:** 19 raw attributes (demographics, account info, "
            f"services subscribed) → **{len(feature_names)}** features after "
            "one-hot encoding categorical columns\n"
            "- **Target:** Binary — whether the customer churned (`Yes` / `No`)"
        )

    with col_flex:
        section_title("🧩 Flexible test data")
        st.markdown(
            f"- Any CSV works as long as it contains the **{len(feature_names)}** "
            "required (already-encoded) feature columns — in any order, "
            "alongside any extra columns.\n"
            "- A target/label column is **optional**; if present under a common "
            "name (`Churn`, `target`, `class`, `label`, or `y`) it's used to "
            "compute evaluation metrics, otherwise you'll just get predictions.\n"
            "- Missing or non-numeric feature values are automatically filled "
            "with the column mean."
        )

    st.write("")
    section_title("🧠 Models available")
    icon_cols = st.columns(len(MODEL_FILES))
    for col, name in zip(icon_cols, MODEL_FILES.keys()):
        with col:
            st.markdown(
                f"""
                <div style="text-align:center; padding:1rem 0.5rem;
                            border:1px solid #E5E7EB; border-radius:12px;">
                    <div style="font-size:1.6rem;">{MODEL_ICONS.get(name, '🔹')}</div>
                    <div style="font-size:0.85rem; font-weight:600; margin-top:0.3rem;">
                        {name}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.caption(
        "Tip: choose **Compare All Models** in the sidebar to run all 5 models "
        "on your data side by side, instead of exploring them one at a time."
    )

st.markdown(
    '<div class="footer-note">Built with Streamlit · ML Assignment 2 — '
    'Telco Customer Churn Model Explorer</div>',
    unsafe_allow_html=True,
)
