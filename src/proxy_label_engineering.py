
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def compute_recency(df: pd.DataFrame, snapshot_date: str = "2019-01-01") -> pd.DataFrame:
    # Parse transaction timestamps and remove timezone info
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True)
    df['TransactionStartTime'] = df['TransactionStartTime'].dt.tz_localize(None)

    # Parse snapshot date as tz-naive as well
    snapshot_dt = pd.to_datetime(snapshot_date)

    # Compute Recency (in days)
    df['Recency'] = (snapshot_dt - df['TransactionStartTime']).dt.days
    recency_df = df.groupby('CustomerId')['Recency'].min().reset_index()
    return recency_df

def assign_rfm_clusters(customer_df: pd.DataFrame, raw_df: pd.DataFrame, snapshot_date: str = "2019-01-01") -> pd.DataFrame:
    recency_df = compute_recency(raw_df, snapshot_date=snapshot_date)
    
    # Merge Recency
    df = pd.merge(customer_df, recency_df, on='CustomerId', how='left')
    
    # Handle NaNs (if any)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Select RFM features
    rfm_df = df[['Recency', 'Transaction_Count', 'Total_Amount']].copy()
    rfm_df.columns = ['Recency', 'Frequency', 'Monetary']

    # Scale
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df)

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(rfm_scaled)

    cluster_stats = df.groupby('cluster')[['Recency', 'Transaction_Count', 'Total_Amount']].mean()
    high_risk_cluster = cluster_stats.sort_values(
            by=['Transaction_Count', 'Total_Amount', 'Recency'],
            ascending=[True, True, False]
        ).index[0]

    df['is_high_risk'] = (df['cluster'] == high_risk_cluster).astype(int)

    return df.drop(columns=['cluster'])