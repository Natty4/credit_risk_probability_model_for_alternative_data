from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import patch
import numpy as np

client = TestClient(app)

@patch("api.main.model.predict_proba")
def test_predict(mock_predict):
    # Simulate output of a real binary classifier (low risk vs high risk)
    mock_predict.return_value = np.array([[0.08, 0.85]])

    response = client.post("/predict", json={
        "Total_Amount": 500.0,
        "Avg_Amount": 100.0,
        "Std_Amount": 20.0,
        "Max_Amount": 200.0,
        "Min_Amount": 50.0,
        "Avg_Pos_Amount": 90.0,
        "Total_Refunds": 2.0,
        "Transaction_Count": 10,
        "Recency": 15
    })

    assert response.status_code == 200
    data = response.json()
    assert "risk_probability" in data
    assert "risk_label" in data
    assert data["risk_label"] == "high_risk"