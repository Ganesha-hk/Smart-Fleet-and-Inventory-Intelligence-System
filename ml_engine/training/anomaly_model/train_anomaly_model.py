import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Add project root to sys.path for internal imports if needed
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from ml_engine.pipelines.engineering import calculate_derived_metrics

class AnomalyEnsemble:
    """Ensemble of Isolation Forest and Local Outlier Factor."""
    def __init__(self, contamination=0.03):
        self.iso_forest = IsolationForest(contamination=contamination, random_state=42)
        self.lof = LocalOutlierFactor(contamination=contamination, novelty=True)
        self.scaler = StandardScaler()
        self.contamination = contamination

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.iso_forest.fit(X_scaled)
        self.lof.fit(X_scaled)
        return self

    def predict_score(self, X):
        """Combines and normalizes scores from both models."""
        X_scaled = self.scaler.transform(X)
        
        # Isolation Forest: decision_function (higher is more normal)
        iso_scores = self.iso_forest.decision_function(X_scaled)
        # Normalize to 0-1 (higher = more anomalous)
        iso_min, iso_max = iso_scores.min(), iso_scores.max()
        iso_norm = 1 - ((iso_scores - iso_min) / (iso_max - iso_min + 1e-9))
        
        # LOF: score_samples (higher is more normal)
        lof_scores = self.lof.score_samples(X_scaled)
        lof_min, lof_max = lof_scores.min(), lof_scores.max()
        lof_norm = 1 - ((lof_scores - lof_min) / (lof_max - lof_min + 1e-9))
        
        return (iso_norm + lof_norm) / 2

def feature_engineering(df):
    print("Running centralized feature engineering...")
    return calculate_derived_metrics(df)

def preprocess(df):
    print("Selecting features for anomaly detection...")
    # These must match InferenceService.ANOMALY_FEATURES exactly
    features = [
        'fuel_consumption_rate', 'eta_variation_hours', 'traffic_congestion_level', 
        'weather_condition_severity', 'iot_temperature', 'cargo_condition_status', 
        'route_risk_level', 'customs_clearance_time', 'driver_behavior_score', 
        'fatigue_monitoring_score', 'movement_distance_km', 'speed_kmph', 
        'behavior_risk_score', 'environmental_risk', 'traffic_fatigue_interaction'
    ]
    
    # Ensure all features exist
    for f in features:
        if f not in df.columns:
            df[f] = 0.0 # Fill missing with 0
            
    return df[features]

def train_models(X):
    # Dynamic contamination based on logical target
    target_rate = 0.05
    print(f"Training Anomaly Ensemble with contamination={target_rate}...")
    ensemble = AnomalyEnsemble(contamination=target_rate)
    ensemble.fit(X)
    return ensemble

def compute_anomaly_score(ensemble, X):
    print("Computing anomaly scores...")
    scores = ensemble.predict_score(X)
    # Goal: ~3-8% anomaly rate
    # We use the 94th percentile to get ~6% rate
    threshold = np.percentile(scores, 94)
    flags = (scores >= threshold).astype(int)
    return scores, flags

def save_model(ensemble, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(ensemble, path)
    print(f"Model saved to {path}")

def main():
    # Paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'processed_dataset_v2.csv')
    ARTIFACTS_PATH = os.path.join(BASE_DIR, 'ml_engine', 'artifacts', 'anomaly_model.joblib')
    
    try:
        # 1. Load
        print(f"Loading data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        
        # 2. Feature Engineering
        df_feat = feature_engineering(df)
        
        # 3. Preprocess
        X = preprocess(df_feat)
        
        # 4. Train
        ensemble = train_models(X)
        
        # 5. Score
        scores, flags = compute_anomaly_score(ensemble, X)
        df_feat['anomaly_score'] = scores
        df_feat['anomaly_flag'] = flags
        
        # 6. Print Results
        anomaly_pct = (flags.sum() / len(flags)) * 100
        print(f"\nAnomaly Detection Results:")
        print(f"Total Records: {len(df)}")
        print(f"Anomalies Found: {flags.sum()} ({anomaly_pct:.2f}%)")
        
        # 7. Save
        save_model(ensemble, ARTIFACTS_PATH)
        
    except Exception as e:
        print(f"Error in anomaly training pipeline: {e}")

if __name__ == "__main__":
    main()
