import mlflow
from fastapi import FastAPI
from api.pydantic_models import CustomerFeatures, RiskPrediction
import pandas as pd

# Initialize API
app = FastAPI(title="Credit Risk Prediction API")

# Load the best model from MLflow
# logged_model_uri = "models:/credit-risk-rf-best/Production"
# model = mlflow.sklearn.load_model(logged_model_uri)
model_name = 'credit-risk-rf-best'
logged_model_uri = f"models:/{model_name}/latest"
model = mlflow.sklearn.load_model(logged_model_uri)

@app.get("/")
def home():
    return {"status": "API running ✅"}

@app.post("/predict", response_model=RiskPrediction)
def predict_risk(customer: CustomerFeatures):
    # Convert input to DataFrame
    input_df = pd.DataFrame([customer.dict()])

    # Predict risk
    risk_proba = model.predict_proba(input_df)[0][1]
    label = "high_risk" if risk_proba >= 0.5 else "low_risk"

    return RiskPrediction(risk_probability=round(risk_proba, 4), risk_label=label)
