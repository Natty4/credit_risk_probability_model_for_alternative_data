# Credit Risk Probability Model for Alternative Data
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
