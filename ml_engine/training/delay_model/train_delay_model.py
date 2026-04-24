import os
import json
import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from ml_engine.pipelines.engineering import calculate_derived_metrics

def load_data(file_path):
    """Loads dataset from the specified path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def preprocess_data(df, target_col='delay_probability'):
    """Preprocesses the dataset: handles missing values and splits into X and y."""
    print("Running centralized feature engineering...")
    df = calculate_derived_metrics(df)
    
    print("Preprocessing data with strict feature selection...")
    # These must match InferenceService.DELAY_FEATURES exactly
    features = [
        'vehicle_gps_latitude', 'vehicle_gps_longitude', 'fuel_consumption_rate', 
        'eta_variation_hours', 'traffic_congestion_level', 'warehouse_inventory_level', 
        'loading_unloading_time', 'handling_equipment_availability', 'order_fulfillment_status', 
        'weather_condition_severity', 'port_congestion_level', 'shipping_costs', 
        'supplier_reliability_score', 'lead_time_days', 'historical_demand', 
        'iot_temperature', 'cargo_condition_status', 'route_risk_level', 
        'customs_clearance_time', 'driver_behavior_score', 'fatigue_monitoring_score', 
        'movement_distance_km', 'speed_kmph', 'behavior_risk_score', 
        'environmental_risk', 'traffic_fatigue_interaction'
    ]
    
    # Ensure all features exist
    for f in features:
        if f not in df.columns:
            df[f] = 0.0 # Fill missing with 0
            
    X = df[features]
    y = df[target_col]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """Trains hybrid models using expanded GridSearchCV."""
    print("Starting model training with expanded GridSearchCV...")
    
    models = {
        'RandomForest': {
            'model': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
            'params': {}
        }
    }
    
    best_overall_model = None
    best_mae = float('inf')
    best_model_name = ""

    for name, config in models.items():
        print(f"Tuning {name}...")
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', config['model'])
        ])
        
        grid_search = GridSearchCV(
            pipeline, 
            config['params'], 
            cv=3, 
            scoring='neg_mean_absolute_error', 
            n_jobs=1
        )
        grid_search.fit(X_train, y_train)
        
        current_mae = -grid_search.best_score_
        print(f"{name} best CV MAE: {current_mae:.4f}")
        
        if current_mae < best_mae:
            best_mae = current_mae
            best_overall_model = grid_search.best_estimator_
            best_model_name = name

    print(f"Training Complete. Best Model: {best_model_name}")
    return best_overall_model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints performance metrics."""
    print("Evaluating model on test set...")
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print("-" * 30)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    print("-" * 30)
    
    return mae, rmse, r2

def explain_model(model, X_train, artifacts_dir):
    """Generates SHAP explainability artifacts."""
    print("Generating SHAP explainability...")
    
    # We use the actual model from the pipeline
    regressor = model.named_steps['model']
    transform_pipe = Pipeline(model.steps[:-1])
    # Use a small sample for SHAP for performance (Issue 5 - speed)
    X_explain = X_train_transformed[:100]
    explainer = shap.Explainer(regressor, X_explain)
    shap_values = explainer(X_explain, check_additivity=False)
    
    # Save explainer object
    explainer_path = os.path.join(artifacts_dir, 'delay_explainer.joblib')
    joblib.dump(explainer, explainer_path)
    print(f"Explainer saved to {explainer_path}")
    
    return shap_values

def save_artifacts(model, X_train, artifacts_dir):
    """Saves the best model and feature importance metadata."""
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(artifacts_dir, 'delay_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Feature importance
    regressor = model.named_steps['model']
    feature_names = X_train.columns.tolist()
    importances = regressor.feature_importances_
    
    feature_importance_map = sorted(
        zip(feature_names, importances.tolist()), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    importance_path = os.path.join(artifacts_dir, 'delay_features.json')
    with open(importance_path, 'w') as f:
        json.dump({'top_features': feature_importance_map[:10]}, f, indent=4)
    print(f"Feature importance saved to {importance_path}")

def main():
    # Define paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    DATA_PATH = os.path.join(BASE_DIR, "data/processed/processed_dataset_v2.csv")
    ARTIFACTS_DIR = os.path.join(BASE_DIR, 'ml_engine', 'artifacts')
    
    try:
        # 1. Load Data
        df = load_data(DATA_PATH)
        
        # Take a representative sample for the final demo artifact to ensure speed
        df = df.sample(n=min(5000, len(df)), random_state=42)
        
        # 2. Preprocess
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # 3. Train
        best_model = train_models(X_train, y_train)
        
        # 4. Evaluate
        evaluate_model(best_model, X_test, y_test)
        
        # 4. Evaluate
        evaluate_model(best_model, X_test, y_test)
        
        # 5. Save Artifacts (Simplified)
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        joblib.dump(best_model, os.path.join(ARTIFACTS_DIR, 'delay_model.joblib'))
        print(f"Model saved to artifacts/delay_model.joblib")
        
    except Exception as e:
        print(f"Error in training pipeline: {e}")

if __name__ == "__main__":
    main()
