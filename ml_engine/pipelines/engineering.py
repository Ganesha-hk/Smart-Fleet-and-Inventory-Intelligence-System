import numpy as np
import pandas as pd

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """NumPy vectorized haversine calculation for performance."""
    R = 6371  # Earth radius in km
    
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2)**2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def calculate_derived_metrics(df):
    """Generates optimized derived features for the intelligence pipeline."""
    df = df.copy()
    
    # 1. Physics: Distance & Speed
    # Shift lat/long to get previous position - Handle vehicle_id grouping if available (Issue FIX 2)
    if 'vehicle_id' in df.columns:
        prev_lat = df.groupby('vehicle_id')['vehicle_gps_latitude'].shift(1).fillna(df['vehicle_gps_latitude'])
        prev_lon = df.groupby('vehicle_id')['vehicle_gps_longitude'].shift(1).fillna(df['vehicle_gps_longitude'])
    else:
        prev_lat = df['vehicle_gps_latitude'].shift(1).fillna(df['vehicle_gps_latitude'])
        prev_lon = df['vehicle_gps_longitude'].shift(1).fillna(df['vehicle_gps_longitude'])
    
    # Vectorized Haversine
    df['movement_distance_km'] = haversine_vectorized(
        prev_lat.values, prev_lon.values,
        df['vehicle_gps_latitude'].values, df['vehicle_gps_longitude'].values
    )
    
    # Speed estimation with time delta (Only if missing or invalid)
    if 'speed_kmph' not in df.columns or df['speed_kmph'].nunique() <= 1:
        if 'timestamp' in df.columns:
            try:
                df['ts_dt'] = pd.to_datetime(df['timestamp'])
                time_diff = df['ts_dt'].diff().dt.total_seconds() / 3600.0
                time_diff = time_diff.abs().replace(0, 1.0).fillna(1.0)
                raw_speed = df['movement_distance_km'] / time_diff
                # REMOVED: / 15.0 scaling factor (Issue FIX 3)
                df['speed_kmph'] = np.clip(raw_speed, 0, 120) 
                df = df.drop(columns=['ts_dt'])
            except:
                df['speed_kmph'] = np.clip(df['movement_distance_km'] / 1.0, 0, 120)
        else:
            df['speed_kmph'] = np.clip(df['movement_distance_km'] / 1.0, 0, 120)
        
    # 2. Risk Indices: Balanced and Normalized
    # Apply standard scaling logic to reach realistic distributions
    # Driver Score is usually 0.3-0.9 where 0.9 is GOOD.
    # Fatigue is 0.1-1.0 where 1.0 is BAD.
    
    # behavior_norm: 0 (Good behavior) -> 1 (Bad behavior)
    behavior_raw = df['driver_behavior_score']
    behavior_norm = np.clip((0.9 - behavior_raw) / 0.6, 0, 1)
    
    # fatigue_norm: 0 (Fresh) -> 1 (Tired)
    fatigue_raw = df['fatigue_monitoring_score']
    fatigue_norm = np.clip((fatigue_raw - 0.1) / 0.9, 0, 1)
    
    # Balanced behavior risk: 60% Behavior, 40% Fatigue (Industry Standard)
    df['behavior_risk_score'] = (0.6 * behavior_norm + 0.4 * fatigue_norm)
    
    # Combined congestion: traffic + port
    df['congestion_index'] = np.clip(df['traffic_congestion_level'] + df.get('port_congestion_level', 0), 0, 20) / 20.0
    
    # Environmental risk: weather + route
    df['environmental_risk'] = (df['weather_condition_severity'] + df['route_risk_level']) / 20.0
    
    # Interaction features (Issue 5)
    df['traffic_fatigue_interaction'] = df['traffic_congestion_level'] * df['fatigue_monitoring_score']
    
    # 3. Logistics friction
    df['logistics_delay_factors'] = (
        df['eta_variation_hours'] + 
        df['customs_clearance_time'] + 
        df['loading_unloading_time']
    )
    
    return df
