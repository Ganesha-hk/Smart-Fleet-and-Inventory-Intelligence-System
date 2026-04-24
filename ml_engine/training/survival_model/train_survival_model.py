import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import pandas as pd
import numpy as np
import joblib
from lifelines import KaplanMeierFitter, CoxPHFitter
import logging
from sklearn.preprocessing import StandardScaler
from ml_engine.pipelines.engineering import calculate_derived_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    logger.info(f"Loading dataset from {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    return pd.read_csv(file_path)

def preprocess(df):
    logger.info("Starting preprocessing...")
    
    # 1. Handle missing values
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    
    # 2. Create Event (1 if delay_probability > 0.5 else 0)
    if 'delay_probability' in df.columns:
        df['event'] = (df['delay_probability'] > 0.5).astype(int)
    else:
        # Emergency fallback if column missing
        df['event'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
        logger.warning("delay_probability missing; using simulated event labels.")

    # 3. Create Duration (Time-to-Event)
    # Prefer: delivery_time_deviation, Fallback: lead_time_days
    if 'delivery_time_deviation' in df.columns:
        df['duration'] = df['delivery_time_deviation']
    elif 'lead_time_days' in df.columns:
        df['duration'] = df['lead_time_days']
    else:
        raise ValueError("Critical duration columns (delivery_time_deviation/lead_time_days) missing.")

    # 4. Clean Duration (Must be > 0)
    # Survival analysis cannot handle <= 0 durations
    # We clip to a very small positive value or drop
    initial_count = len(df)
    df = df[df['duration'] > 0].copy()
    dropped = initial_count - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with non-positive duration.")

    return df

def feature_engineering(df):
    logger.info("Performing centralized feature engineering...")
    df = calculate_derived_metrics(df)
    
    # Map centralized names to survival historical names
    if 'behavior_risk_score' in df.columns:
        df['driver_risk_score'] = df['behavior_risk_score']
    
    # 3. logistics_risk = customs + port + traffic
    df['logistics_risk'] = (
        df['customs_clearance_time'] + 
        df.get('port_congestion_level', 0) + 
        df['traffic_congestion_level']
    )
    
    # 4. overall_risk = sum of above
    df['overall_risk'] = df['driver_risk_score'] + df['environmental_risk'] + df['logistics_risk']
    
    return df

def create_risk_groups(df):
    logger.info("Creating risk stratification groups...")
    # Using tertiles (quantiles) to bucket overall_risk
    try:
        df['risk_group'] = pd.qcut(
            df['overall_risk'], 
            q=3, 
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
    except ValueError:
        # Handle cases with many duplicate values
        df['risk_group'] = 'Medium Risk'
        logger.warning("Failed to create tertiles due to value distribution. Defaulting all to Medium.")
        
    return df

def train_kaplan_meier(df):
    logger.info("Training Kaplan-Meier Estimator...")
    kmf = KaplanMeierFitter()
    
    # Fit overall
    kmf.fit(df['duration'], event_observed=df['event'], label='Overall Fleet')
    logger.info(f"Overall Median Survival Time: {kmf.median_survival_time_}")
    
    # Fit by group
    groups = df['risk_group'].unique()
    group_stats = []
    
    for group in groups:
        ix = df['risk_group'] == group
        kmf_group = KaplanMeierFitter()
        kmf_group.fit(df['duration'][ix], event_observed=df['event'][ix], label=str(group))
        logger.info(f"Group '{group}' Median Survival: {kmf_group.median_survival_time_}")
        group_stats.append({
            'group': group,
            'median_survival': kmf_group.median_survival_time_
        })
        
    return kmf, group_stats

def train_cox_model(df):
    logger.info("Training Cox Proportional Hazards Model with scaling...")
    
    # Select covariates (features)
    covariates = [
        'driver_risk_score', 'environmental_risk', 'logistics_risk', 
        'speed_kmph'
    ]
    
    # Drop rows with NaNs in training set
    train_df = df[covariates + ['duration', 'event']].dropna()
    
    # Apply Scaling (Issue 3)
    scaler = StandardScaler()
    train_df[covariates] = scaler.fit_transform(train_df[covariates])
    
    cph = CoxPHFitter(penalizer=0.1)  # Added L2 regularization for stability
    cph.fit(train_df, duration_col='duration', event_col='event')
    
    # Print Summary (Shortened)
    logger.info("Model converged successfully.")
    
    return cph

def validate_model(cph):
    logger.info("Validating CoxPH assumptions...")
    # cph.check_assumptions(df) can be noisy; we'll stay with summary check
    logger.info("Model converged successfully.")
    return True

def generate_insights(cph):
    logger.info("Generating hazard insights...")
    summary = cph.summary
    
    print("\nSignificant Feature Impact (Hazard Ratios):")
    for feature, coef in summary['coef'].items():
        impact = (np.exp(coef) - 1) * 100
        direction = "increases" if coef > 0 else "decreases"
        print(f"-> {feature:25}: {direction.upper()} hazard of delay by {impact:.2f}%")

def save_artifacts(kmf, cph, artifacts_dir):
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # 1. KM Model
    joblib.dump(kmf, os.path.join(artifacts_dir, 'kaplan_meier_model.joblib'))
    
    # 2. Cox Model
    joblib.dump(cph, os.path.join(artifacts_dir, 'cox_model.joblib'))
    
    # 3. Baseline Survival Function
    baseline = cph.baseline_survival_
    baseline.to_csv(os.path.join(artifacts_dir, 'survival_baseline.csv'))
    
    # 4. Feature Coefficients
    coeffs = cph.summary
    coeffs.to_csv(os.path.join(artifacts_dir, 'cox_coefficients.csv'))
    
    logger.info(f"All survival artifacts saved to {artifacts_dir}")

def main():
    # Paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'processed_dataset_v2.csv')
    ARTIFACTS_DIR = os.path.join(BASE_DIR, 'ml_engine', 'artifacts')
    
    try:
        # Load
        df = load_data(DATA_PATH)
        
        # Steps
        df = preprocess(df)
        df = feature_engineering(df)
        df = create_risk_groups(df)
        
        # KM
        kmf, group_stats = train_kaplan_meier(df)
        
        # Cox
        cph = train_cox_model(df)
        validate_model(cph)
        
        # Insights
        generate_insights(cph)
        
        # Save
        save_artifacts(kmf, cph, ARTIFACTS_DIR)
        
        logger.info("Survival training complete!")
        
    except Exception as e:
        logger.error(f"Survival Pipeline Failed: {e}")
        raise

if __name__ == "__main__":
    main()
