import pandas as pd
import numpy as np
import os

def generate_mock_data(n_samples=1000):
    """Generates a synthetic fleet logistics dataset with 26 columns."""
    np.random.seed(42)
    
    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='H'),
        'vehicle_gps_latitude': np.random.uniform(24.9, 25.5, n_samples),
        'vehicle_gps_longitude': np.random.uniform(54.9, 55.6, n_samples),
        'fuel_consumption_rate': np.random.uniform(5, 20, n_samples),
        'eta_variation_hours': np.random.uniform(-2, 5, n_samples),
        'traffic_congestion_level': np.random.randint(1, 10, n_samples),
        'warehouse_inventory_level': np.random.randint(10, 100, n_samples),
        'loading_unloading_time': np.random.uniform(0.5, 4, n_samples),
        'handling_equipment_availability': np.random.randint(1, 5, n_samples),
        'order_fulfillment_status': np.random.randint(0, 2, n_samples),
        'weather_condition_severity': np.random.randint(1, 10, n_samples),
        'port_congestion_level': np.random.randint(1, 10, n_samples),
        'shipping_costs': np.random.uniform(100, 1000, n_samples),
        'supplier_reliability_score': np.random.uniform(0.5, 1, n_samples),
        'lead_time_days': np.random.randint(1, 30, n_samples),
        'historical_demand': np.random.randint(50, 500, n_samples),
        'iot_temperature': np.random.uniform(-10, 30, n_samples),
        'cargo_condition_status': np.random.randint(0, 3, n_samples),
        'route_risk_level': np.random.randint(1, 5, n_samples),
        'customs_clearance_time': np.random.uniform(1, 12, n_samples),
        'driver_behavior_score': np.random.uniform(60, 100, n_samples),
        'fatigue_monitoring_score': np.random.uniform(0, 100, n_samples),
        'disruption_likelihood_score': np.random.uniform(0, 1, n_samples),
        
        # Target variables with some correlations
    }
    
    # Calculate delay_probability based on some features
    delay_base = (
        data['traffic_congestion_level'] * 0.1 +
        data['weather_condition_severity'] * 0.05 +
        (100 - data['driver_behavior_score']) * 0.01 +
        data['eta_variation_hours'] * 0.1
    )
    data['delay_probability'] = np.clip(delay_base / 2 + np.random.normal(0, 0.05, n_samples), 0, 1)
    
    # Risk classification
    data['risk_classification'] = (data['delay_probability'] > 0.6).astype(int)
    
    # Delivery time deviation
    data['delivery_time_deviation'] = data['delay_probability'] * 10 + np.random.normal(0, 1, n_samples)
    
    df = pd.DataFrame(data)
    return df

def main():
    # Define paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    FILE_PATH = os.path.join(PROCESSED_DATA_DIR, 'processed_dataset.csv')
    
    print(f"Generating synthetic data...")
    df = generate_mock_data(n_samples=2000)
    
    print(f"Saving dataset to {FILE_PATH}...")
    df.to_csv(FILE_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
