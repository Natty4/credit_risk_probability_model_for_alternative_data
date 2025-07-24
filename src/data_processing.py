from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class TransactionAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Amount_Pos'] = X['Amount'].apply(lambda x: x if x > 0 else 0)
        X['Amount_Neg'] = X['Amount'].apply(lambda x: -x if x < 0 else 0)

        agg_df = X.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'std', 'max', 'min'],
            'Amount_Pos': 'mean',
            'Amount_Neg': 'sum',
            'TransactionId': 'count',
        }).reset_index()

        agg_df.columns = ['CustomerId', 'Total_Amount', 'Avg_Amount', 'Std_Amount',
                          'Max_Amount', 'Min_Amount', 'Avg_Pos_Amount',
                          'Total_Refunds', 'Transaction_Count']
        return agg_df


def build_feature_pipeline():
    return Pipeline([
        ('aggregator', TransactionAggregator())
        # no scaler here
    ])