# 🏦 RetainPulse: Bank Customer Churn Prediction & Triage System

An end-to-end Machine Learning and Streamlit web application designed to forecast retail banking customer attrition, evaluate churn risk tiers, and generate proactive retention strategies.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

##  Executive Overview

Customer attrition directly impacts bank profitability. This project develops an end-to-end data pipeline addressing:
- **Class Imbalance Handling**: Optimized for high churn recall (catching flight-risk customers before they depart).
- **Leakage-Free Feature Engineering**: Custom transformers calculating solvency indicators and behavioral ratios.
- **Enterprise Web Interface**: An interactive Streamlit dashboard supporting single-customer risk scoring and batch CSV evaluations.

---

##  Dataset & Feature Architecture

The model trains on bank demographic and transaction records:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `credit_score` | Integer | Customer credit bureau rating (350–850) |
| `country` | Categorical | Country of account origin (France, Germany, Spain) |
| `gender` | Categorical | Customer gender (Male, Female) |
| `age` | Integer | Customer age (18–92) |
| `tenure` | Integer | Number of years as an account holder |
| `balance` | Float | Current account balance |
| `products_number` | Integer | Active products held (1–4) |
| `credit_card` | Binary | Credit card possession (1 = Yes, 0 = No) |
| `active_member` | Binary | Activity status index (1 = Active, 0 = Inactive) |
| `estimated_salary` | Float | Annual income estimate |
| `churn` | Binary Target | Retained (0) vs. Churned (1) |

### Engineered Features
1. `zero_balance`: Binary indicator capturing zero-balance status (36.17% of accounts).
2. `balance_salary_ratio`: Measures relative liquidity (`balance / (estimated_salary + 1)`).
3. `tenure_age_ratio`: Contextualizes loyalty duration against customer life stage.
4. `credit_score_age_ratio`: Ratio of credit rating to customer age.

---

##  Model Performance & Validation

The final classifier is a tuned **Random Forest** with cost-sensitive weighting (`balanced_subsample`) optimized via 5-Fold Stratified Cross-Validation:

- **ROC-AUC Score**: `0.8579`
- **Overall Accuracy**: `85.0%`
- **Churn Recall (Class 1)**: `59.0%` (catches nearly 60% of at-risk customers)
- **Churn Precision (Class 1)**: `63.0%`
- **Majority Class F1 (Class 0)**: `0.91`

---

##  Application Features

- **Single Customer Scoring**: Input profile variables to generate real-time churn probability, risk badges (Low, Medium, Critical), and mitigation actions.
- **Batch CSV Analysis**: Upload customer records, run vectorized predictions, and export ranked CSV files with risk classifications.
- **Model Governance Panel**: View validation metrics, operational boundaries, and pipeline hyperparameters.

---

## 🛠️ Installation & Quickstart

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/Mysterious-Magici/bank-customer-churn-intelligence-system/)<Mysterious-Magici>/bank-customer-churn-prediction.git
cd bank-customer-churn-prediction
