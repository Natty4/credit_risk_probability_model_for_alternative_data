import mlflow
from fastapi import FastAPI
from api.pydantic_models import CustomerFeatures, RiskPrediction
import pandas as pd
from functools import lru_cache

# Initialize API
app = FastAPI(title="Credit Risk Prediction API")

@lru_cache()
def get_model():
    model_name = 'credit-risk-rf-best'
    logged_model_uri = f"models:/{model_name}/latest"
    return mlflow.sklearn.load_model(logged_model_uri)

@app.get("/")
def home():
    return {"status": "API running ✅"}

@app.post("/predict", response_model=RiskPrediction)
def predict_risk(customer: CustomerFeatures):
    input_df = pd.DataFrame([customer.model_dump()])
    model = get_model()
    risk_proba = model.predict_proba(input_df)[0][1]
    label = "high_risk" if risk_proba >= 0.5 else "low_risk"
    return RiskPrediction(risk_probability=round(risk_proba, 4), risk_label=label)