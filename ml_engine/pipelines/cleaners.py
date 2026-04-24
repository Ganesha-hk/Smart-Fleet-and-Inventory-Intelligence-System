import pandas as pd
import numpy as np

def clean_missing_values(df):
    """Handles missing values by imputing with mean for numerical and mode for categorical."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    return df

def encode_labels(df):
    """Encodes categorical strings like risk levels into integers."""
    risk_mapping = {
        'Low Risk': 0,
        'Moderate Risk': 1,
        'High Risk': 2,
        'Critical Risk': 3
    }
    
    if 'risk_classification' in df.columns:
        # If it's already numerical, skip
        if df['risk_classification'].dtype == object:
            df['risk_classification'] = df['risk_classification'].map(risk_mapping).fillna(1)
            
    return df

def process_timestamps(df):
    """Converts timestamp to datetime objects and extracts time features."""
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp').reset_index(drop=True)
    return df
