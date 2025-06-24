from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

def detect_regimes(df, n_clusters=3):
    """
    Perform KMeans clustering on already prepared 'volatility' and 'momentum' features.

    Assumes that the input df already contains 'volatility' and 'momentum' columns.
    """

    #Check if required features exist
    required_features = ['volatility', 'momentum']
    for feature in required_features:
        if feature not in df.columns:
            raise ValueError(f"Missing required feature column: {feature}")

    # Apply KMeans clustering
    features = df[['volatility', 'momentum']].copy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['regime'] = kmeans.fit_predict(features)

    return df
