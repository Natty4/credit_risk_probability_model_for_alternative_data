from unittest.mock import patch, MagicMock
from api.main import app
from fastapi.testclient import TestClient
import numpy as np

client = TestClient(app)

@patch("api.main.get_model")
def test_predict(mock_get_model):
    mock_model = MagicMock()
    # Return [prob_low_risk, prob_high_risk]
    mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])
    mock_get_model.return_value = mock_model

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
    assert response.json()["risk_label"] == "high_risk"
    assert 0.8 < response.json()["risk_probability"] < 0.9