# Telco Customer Churn — Classification Model Explorer

## a. Problem Statement

Customer churn — when a subscriber stops using a company's service —
is one of the most expensive problems in subscription-based businesses,
since acquiring a new customer typically costs far more than retaining
an existing one. This project builds and compares five supervised
classification models to predict, from a customer's account and
service-usage attributes, **whether that customer will churn (leave
the service) or not**. This is a **binary classification** problem
(`Churn` = `Yes` / `No`). An accurate, interpretable model lets a
telecom provider proactively target at-risk customers with retention
offers before they leave.

## b. Dataset Description

- **Name:** Telco Customer Churn
- **Source:** IBM sample dataset, publicly distributed via Kaggle:
  https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Instances:** 7,043 customers
  - Train/test split: 80% / 20% → **1,409 rows** in the provided
    `data/test_data.csv`
- **Raw features:** 19 attributes (after dropping the `customerID`
  identifier column), covering:
  - **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - **Account info:** `tenure` (months), `Contract`, `PaperlessBilling`,
    `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
  - **Services subscribed:** `PhoneService`, `MultipleLines`,
    `InternetService`, `OnlineSecurity`, `OnlineBackup`,
    `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Final feature count:** 30 (after one-hot encoding the 15
  categorical columns; numeric columns are kept as-is and everything
  is standardized with `StandardScaler`)
- **Target:** Binary — `Churn` = `Yes` (customer left) or `No`
  (customer stayed)
- **Class balance:** 5,174 `No` (73.5%) vs. 1,869 `Yes` (26.5%) — a
  **moderately imbalanced** dataset, which is why accuracy alone is
  not a sufficient metric here and AUC, F1, and MCC are reported
  alongside it.
- **Data cleaning:** 11 rows had a blank `TotalCharges` value (new
  customers with 0 months of tenure); these were imputed with the
  column median.

## c. GitHub Repository Link

https://github.com/JeyaniveditaJayakumar/telco-churn-app


## d. Models Used

All 5 models were trained on the same 80/20 train/test split
(`random_state=42`, stratified by target) with standardized features.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8070 | 0.8416 | 0.6584 | 0.5668 | 0.6092 | 0.4843 |
| Decision Tree | 0.7800 | 0.7947 | 0.5899 | 0.5615 | 0.5753 | 0.4272 |
| KNN | 0.7700 | 0.8084 | 0.5706 | 0.5401 | 0.5549 | 0.4004 |
| Naive Bayes | 0.6558 | 0.8093 | 0.4269 | 0.8663 | 0.5719 | 0.3951 |
| Random Forest (Ensemble) | 0.8048 | 0.8330 | 0.6645 | 0.5348 | 0.5926 | 0.4710 |

*(Reproducible via `model/train_models.py`; raw values also saved in
`model/metrics_summary.csv`.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer — highest accuracy (0.807) and highest AUC (0.842), with the best balance across precision, recall, F1, and MCC. Churn appears largely linearly separable by features like contract type, tenure, and monthly charges, which favors a linear model. |
| Decision Tree | A single tree (max depth 8) generalizes reasonably (0.78 accuracy) but trails Logistic Regression on every metric, especially AUC (0.795) — consistent with a single tree's tendency to overfit local splits and produce less calibrated probability estimates. |
| KNN | Weakest accuracy among the strong performers (0.77) despite a respectable AUC (0.808). With 30 features after one-hot encoding, distance-based similarity becomes less meaningful (curse of dimensionality), which likely hurts its precision/recall balance. |
| Naive Bayes | Lowest accuracy (0.656) but by far the **highest recall (0.866)** — it flags most true churners, at the cost of many false positives (lowest precision, 0.427). This happens because Naive Bayes assumes feature independence, which is clearly violated here (e.g. streaming/security add-ons are correlated), skewing its decision boundary. Still useful if the business cost of *missing* a churner outweighs the cost of a false alarm. |
| Random Forest (Ensemble) | Very close second overall — second-best accuracy (0.805), second-best AUC (0.833), and the highest precision (0.665) of any model. The ensemble clearly reduces the overfitting/variance problem seen in the single Decision Tree (its accuracy jumped from 0.78 to 0.805 and MCC from 0.427 to 0.471), but its recall (0.535) still trails Logistic Regression's, meaning it misses more actual churners despite being more "confident" on the ones it does flag. |
| **Overall Winner for your dataset?** | **Logistic Regression** — it leads on Accuracy, AUC, F1, and MCC simultaneously, with Random Forest a close second (higher precision, lower recall). Logistic Regression is the most balanced and reliable model here. *(Caveat: if the business priority is catching as many churners as possible even at the cost of more false alarms, Naive Bayes' much higher recall — 0.866 vs. 0.567 — would be the better operational choice; the "winner" depends on the retention team's tolerance for false positives.)* |

## Project Structure

```
ML_Assignment2_Project/
├── app.py                          # Streamlit app .py
├── requirements.txt
├── README.md
├── test_data.csv                   # held-out test split (upload this to the app)
├── raw_data/
│   └── Telco-Customer-Churn.csv    # full raw dataset (source for training)
└── model/
    ├── train_models.py             # training script
    ├── scaler.pkl
    ├── feature_names.pkl
    ├── target_encoder.pkl
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── metrics_summary.csv
```

## Setup

```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## App Features

- **Test CSV upload** (Streamlit free tier friendly — only the test
  split needs uploading, models are pre-trained and bundled)
- **Model selection dropdown** — choose any of the 5 trained models
- **Live evaluation metrics** — Accuracy, AUC, Precision, Recall, F1,
  MCC, plus a full classification report
- **Confusion matrix** heatmap
- **Model comparison** — view all 5 models' metrics side by side on
  the same uploaded data
- **Flexible test-data handling** — the uploaded CSV's column order
  doesn't matter, extra columns are ignored, header casing/whitespace
  differences are tolerated, and a target/label column is optional
  (predictions are shown either way; metrics only when a usable target
  is present)

To try it: upload `test_data.csv` from the root of this repo, pick a
model, and explore the results. Retrain from scratch with
`cd model && python train_models.py` if needed.

## Live Streamlit App

https://telco-churn-app-yadejhff5z6sor4x8hu9ne.streamlit.app/
