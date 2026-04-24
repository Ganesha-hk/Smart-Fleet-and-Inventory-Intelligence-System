import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

class AnomalyFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for anomaly-specific feature engineering."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # 1. movement_distance (Delta lat/long approximation)
        X['prev_lat'] = X['vehicle_gps_latitude'].shift(1).fillna(X['vehicle_gps_latitude'])
        X['prev_lon'] = X['vehicle_gps_longitude'].shift(1).fillna(X['vehicle_gps_longitude'])
        
        X['movement_distance'] = np.sqrt(
            (X['vehicle_gps_latitude'] - X['prev_lat'])**2 + 
            (X['vehicle_gps_longitude'] - X['prev_lon'])**2
        )
        # 2. behavior_risk_score = driver_behavior_score + fatigue_monitoring_score
        X['behavior_risk_score'] = X['driver_behavior_score'] + X['fatigue_monitoring_score']
        # 3. environmental_risk = traffic_congestion_level + route_risk_level
        X['environmental_risk'] = X['traffic_congestion_level'] + X['route_risk_level']
        return X.drop(columns=['prev_lat', 'prev_lon'])

class AnomalyEnsemble:
    """Ensemble of Isolation Forest and Local Outlier Factor."""
    def __init__(self, contamination=0.05):
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
        iso_scores = self.iso_forest.decision_function(X_scaled)
        iso_norm = 1 - ((iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9))
        lof_scores = self.lof.score_samples(X_scaled)
        lof_norm = 1 - ((lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-9))
        combined_score = (iso_norm + lof_norm) / 2
        return combined_score
