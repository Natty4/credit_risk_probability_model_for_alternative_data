from pydantic import BaseModel
from typing import List

# Input schema
class CustomerFeatures(BaseModel):
    Total_Amount: float
    Avg_Amount: float
    Std_Amount: float = 0.0
    Max_Amount: float
    Min_Amount: float
    Avg_Pos_Amount: float
    Total_Refunds: float
    Transaction_Count: float
    Recency: float

# Output schema
class RiskPrediction(BaseModel):
    risk_probability: float
    risk_label: str  # "high_risk" or "low_risk"
