from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_home():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == {"status": "API running ✅"}

def test_predict_low_risk():
    payload = {
        "Total_Amount": 1000.0,
        "Avg_Amount": 250.0,
        "Std_Amount": 10.0,
        "Max_Amount": 300.0,
        "Min_Amount": 200.0,
        "Avg_Pos_Amount": 250.0,
        "Total_Refunds": 0.0,
        "Transaction_Count": 5,
        "Recency": 5
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    assert "risk_probability" in res.json()
    assert "risk_label" in res.json()
    
    
    