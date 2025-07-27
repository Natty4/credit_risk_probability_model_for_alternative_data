# 📊 Credit Risk Probability Model 🚀

> Predicting Customer Credit Risk from Alternative Data using FastAPI + MLflow + GitHub Actions

---

## Overview

This project builds an end-to-end credit risk scoring system using **alternative behavioral data**. It leverages Recency-Frequency-Monetary (RFM) features from transaction logs and uses machine learning to predict whether a customer is **high risk** or **low risk**.

The system supports:

- 🔄 Clean, reproducible **ETL pipelines**
- 🧪 Model training with **MLflow tracking & model registry**
- 🚀 Serving predictions via **FastAPI**
- ✅ **Unit-tested** with automated **CI/CD on GitHub Actions**

---

## Project Structure

```plaintext
├── api/                    # FastAPI app
│   ├── main.py             # API logic
│   └── pydantic_models.py  # Input/output schema
├── config/                 # Config files (optional)
├── data/                   # Raw + processed CSV files
│   ├── raw/
│   └── processed/
├── notebooks/              # Exploration & prototyping
├── src/                    # Source code for pipeline
│   ├── data_processing.py  # Feature engineering pipeline
│   ├── proxy_label_engineering.py  # RFM clustering logic
│   └── train.py            # ML training script with MLflow
├── tests/                  # Unit tests for pipeline + API
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI workflow
├── requirements.txt        # Dependencies
├── README.md               # You’re here!
````

---

## Data Pipeline

* **Source**: Transaction logs (`transactions.csv`)
* **Engineered Features**:

  * Recency, Transaction Count, Total/Avg/Std Amounts
  * Positive Amounts, Refund Totals
* **Labeling**:

  * RFM-based clustering using KMeans
  * Customers in low-frequency + high-recency clusters are labeled `is_high_risk = 1`

---

## Model Training

Three models are trained and evaluated using `MLflow`:

* `Logistic Regression`
* `Random Forest` ✅ (Best Model)
* `Gradient Boosting`

**Metrics Tracked**:

* Accuracy, F1 Score, ROC-AUC

✅ Best model: **Random Forest** with **ROC-AUC = 0.999**

---

## API: Credit Risk Prediction

After training, the best model is served using **FastAPI**:

### Run Locally

```bash
uvicorn api.main:app --reload
```

### Example Request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Total_Amount": 0.02,
    "Avg_Amount": 0.01,
    "Std_Amount": 0.005,
    "Max_Amount": 0.03,
    "Min_Amount": 0.005,
    "Avg_Pos_Amount": 0.02,
    "Total_Refunds": 0.001,
    "Transaction_Count": 0.1,
    "Recency": 10
}'
```

### Response

```json
{
  "risk_probability": 0.92,
  "risk_label": "high_risk"
}
```

---

## 🧪 Testing

```bash
pytest tests/
```

Includes:

* ✅ Feature pipeline test
* ✅ API response test

---

## ♻️ CI/CD with GitHub Actions

Every push to `main` triggers:

* ✅ Dependency install
* ✅ Test suite run
* ✅ Report via GitHub Actions

Defined in `.github/workflows/ci.yml`

---

## 📦 Future Improvements

* 🔐 Add authentication to API
* 🐳 Dockerize & deploy on Azure / AWS / Heroku
* 📈 Integrate dashboards for risk monitoring
* 🧠 Use real loan default data for supervised learning

---


## 📊 Credit Scoring Business Understanding

### 1. Basel II and the Importance of Interpretability

The Basel II Accord mandates that financial institutions manage credit risk using internal models that estimate parameters such as Probability of Default (PD). These models directly affect capital reserve requirements, which makes transparency a regulatory obligation. 

Interpretable models enable auditors, regulators, and risk officers to understand how credit decisions are made and verify their fairness. In this context, interpretability is not just a modeling preference—it is a compliance requirement.

---

### 2. Why Use a Proxy Variable?

Since no explicit default labels exist in the dataset, we must create a proxy variable using behavioral signals such as Recency, Frequency, and Monetary value (RFM). This enables supervised learning by labeling customers as high or low risk based on patterns of disengagement.

However, proxy-based labeling introduces uncertainty. It assumes that behavioral disengagement correlates with credit risk, which may not always hold. Poor proxy definitions can result in business risks such as rejecting good customers or approving bad ones. Therefore, careful proxy engineering and validation are essential.

---

### 3. Interpretable vs. Complex Models

There is a trade-off between transparency and predictive performance. Simple models like Logistic Regression with WoE are easy to interpret and justify in regulatory and legal settings. Complex models like Gradient Boosting Machines (GBMs) may achieve higher accuracy but are harder to explain.

In regulated contexts, financial institutions often favor simpler, interpretable models to maintain compliance and public trust. Where advanced models are used, they must be supplemented with post-hoc explainability tools like SHAP to support transparency.

---
## Credits

Inspired by alternative data applications in fintech and micro-lending in emerging markets.
Big thanks 10academy 💙
