# src/regime_detector.py
from sklearn.cluster import KMeans
import numpy as np

def detect_regimes(df, n_clusters=3):
    # Example feature: rolling volatility
    df['volatility'] = df['Close'].pct_change().rolling(20).std()
    df.dropna(inplace=True)

    kmeans = KMeans(n_clusters=n_clusters)
    df['regime'] = kmeans.fit_predict(df[['volatility']])
    return df
