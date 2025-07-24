import pandas as pd
from src.data_processing import build_feature_pipeline

def test_pipeline_runs():
    sample = pd.DataFrame({
        "TransactionId": ["T1", "T2", "T3"],
        "Amount": [100, -50, 200],
        "Value": [100, 50, 200],
        "CustomerId": ["C1", "C1", "C2"],
        "TransactionStartTime": ["2022-01-01", "2022-01-03", "2022-01-02"]
    })

    pipe = build_feature_pipeline()
    out = pipe.fit_transform(sample)
    
    # Assert it's a DataFrame and has rows
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] > 0
    assert 'CustomerId' in out.columns