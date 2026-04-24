import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[3]
FLEET_DATASET = ROOT_DIR / "data" / "processed" / "processed_dataset_v2.csv"
INVENTORY_DATASET = ROOT_DIR / "data" / "processed" / "inventory_dataset.csv"
ARTIFACTS_DIR = ROOT_DIR / "ml_engine" / "artifacts"

FEATURE_COLUMNS = [
    "inventory_level",
    "historical_demand",
    "lead_time",
    "supplier_reliability",
    "linked_delay_risk",
    "hour",
    "day_of_week",
    "is_weekend",
]


def build_inventory_dataset(target_rows: int = 25000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    fleet_df = pd.read_csv(FLEET_DATASET)
    fleet_df["timestamp"] = pd.to_datetime(fleet_df["timestamp"])

    warehouses = [
        "Dubai Central Hub",
        "Jebel Ali Port Hub",
        "Sharjah East Depot",
        "Abu Dhabi Distribution Center",
        "Al Quoz Service Yard",
        "Ruwais Logistics Park",
        "Khalifa Industrial Hub",
        "Fujairah Coastal Warehouse",
    ]
    warehouse_bias = {
        "Dubai Central Hub": 0.04,
        "Jebel Ali Port Hub": 0.09,
        "Sharjah East Depot": 0.03,
        "Abu Dhabi Distribution Center": 0.05,
        "Al Quoz Service Yard": 0.02,
        "Ruwais Logistics Park": 0.07,
        "Khalifa Industrial Hub": 0.06,
        "Fujairah Coastal Warehouse": 0.08,
    }
    class_ranges = {
        "A": (80, 210),
        "B": (180, 420),
        "C": (320, 760),
    }

    fleet_delay_samples = fleet_df["delay_probability"].to_numpy()
    fleet_timestamps = fleet_df["timestamp"].sort_values().to_numpy()

    rows = []
    for idx in range(1, target_rows + 1):
        warehouse = warehouses[(idx - 1) % len(warehouses)]
        item_class = rng.choice(["A", "B", "C"], p=[0.25, 0.45, 0.30])
        low, high = class_ranges[item_class]

        base_timestamp = pd.Timestamp(fleet_timestamps[(idx * 17) % len(fleet_timestamps)])
        timestamp = base_timestamp + pd.Timedelta(minutes=int(rng.integers(-90, 91)))

        base_demand = rng.integers(low, high)
        seasonal_multiplier = (
            1.0
            + (timestamp.hour / 24.0) * 0.22
            + (timestamp.dayofweek in [3, 4]) * 0.08
            + (timestamp.day in [1, 15, 30]) * 0.05
        )
        historical_demand = int(round(base_demand * seasonal_multiplier + rng.normal(0, 14)))
        historical_demand = max(historical_demand, 20)

        lead_time = int(np.clip(rng.normal(5.5 if item_class == "A" else 7.0, 2.0), 1, 16))
        supplier_reliability = float(np.clip(rng.normal(0.88 - warehouse_bias[warehouse], 0.09), 0.5, 0.99))
        linked_delay_risk = float(
            np.clip(rng.choice(fleet_delay_samples) + warehouse_bias[warehouse] + rng.normal(0, 0.07), 0.02, 0.98)
        )

        inventory_anchor = historical_demand * rng.uniform(0.9, 2.3)
        delay_pressure = 1.0 - linked_delay_risk * rng.uniform(0.2, 0.5)
        reliability_buffer = 0.82 + supplier_reliability * 0.45
        inventory_level = int(np.clip(inventory_anchor * delay_pressure * reliability_buffer + rng.normal(0, 20), 15, 1500))

        rows.append(
            {
                "item_id": f"ITEM-{idx:05d}",
                "warehouse_location": warehouse,
                "inventory_level": inventory_level,
                "historical_demand": historical_demand,
                "lead_time": lead_time,
                "supplier_reliability": round(supplier_reliability, 4),
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "linked_delay_risk": round(linked_delay_risk, 4),
            }
        )

    inventory_df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    INVENTORY_DATASET.parent.mkdir(parents=True, exist_ok=True)
    inventory_df.to_csv(INVENTORY_DATASET, index=False)
    return inventory_df


def build_training_frame(inventory_df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    frame = inventory_df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["hour"] = frame["timestamp"].dt.hour
    frame["day_of_week"] = frame["timestamp"].dt.dayofweek
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)

    frame["demand_target"] = (
        frame["historical_demand"] * (1.02 + 0.1 * frame["linked_delay_risk"])
        + frame["lead_time"] * 5.0
        - frame["inventory_level"] * 0.045
        + (1 - frame["supplier_reliability"]) * 60
        + frame["hour"] * 1.5
        + rng.normal(0, 10, len(frame))
    ).clip(lower=15)

    projected_cover_days = frame["inventory_level"] / np.maximum(frame["demand_target"], 1)
    cover_pressure = np.clip(1 - (projected_cover_days / np.maximum(frame["lead_time"] * 0.28, 1.0)), 0, 1)
    delay_pressure = np.clip((frame["linked_delay_risk"] - 0.38) / 0.42, 0, 1)
    supplier_pressure = np.clip((0.9 - frame["supplier_reliability"]) / 0.25, 0, 1)
    demand_pressure = np.clip(
        (frame["demand_target"] - frame["inventory_level"]) / np.maximum(frame["demand_target"], 1),
        0,
        1,
    )

    frame["stockout_pressure"] = (
        0.42 * cover_pressure
        + 0.26 * delay_pressure
        + 0.16 * supplier_pressure
        + 0.16 * demand_pressure
        + rng.normal(0, 0.035, len(frame))
    ).clip(0, 1)

    threshold = float(frame["stockout_pressure"].quantile(0.68))
    frame["stockout_target"] = (frame["stockout_pressure"] >= threshold).astype(int)
    return frame


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURE_COLUMNS,
            )
        ]
    )


def train_models(frame: pd.DataFrame, random_state: int = 42):
    X = frame[FEATURE_COLUMNS]
    y_demand = frame["demand_target"]
    y_stockout = frame["stockout_target"]

    X_train, X_test, y_demand_train, y_demand_test, y_stockout_train, y_stockout_test = train_test_split(
        X,
        y_demand,
        y_stockout,
        test_size=0.2,
        random_state=random_state,
        stratify=y_stockout,
    )

    demand_model = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestRegressor(n_estimators=240, max_depth=12, random_state=random_state, n_jobs=-1)),
        ]
    )
    demand_model.fit(X_train, y_demand_train)

    stockout_model = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(max_iter=2000, random_state=random_state, class_weight="balanced", C=0.6)),
        ]
    )
    stockout_model.fit(X_train, y_stockout_train)

    demand_predictions = demand_model.predict(X_test)
    stockout_predictions = stockout_model.predict(X_test)

    metrics = {
        "dataset_rows": int(len(frame)),
        "warehouse_count": int(frame["warehouse_location"].nunique()),
        "demand_mae": round(float(mean_absolute_error(y_demand_test, demand_predictions)), 4),
        "demand_r2": round(float(r2_score(y_demand_test, demand_predictions)), 4),
        "stockout_accuracy": round(float(accuracy_score(y_stockout_test, stockout_predictions)), 4),
        "stockout_f1": round(float(f1_score(y_stockout_test, stockout_predictions)), 4),
    }
    return demand_model, stockout_model, metrics


def save_artifacts(demand_model, stockout_model, metrics: dict) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(demand_model, ARTIFACTS_DIR / "demand_model.joblib")
    joblib.dump(stockout_model, ARTIFACTS_DIR / "stockout_model.joblib")
    with open(ARTIFACTS_DIR / "inventory_training_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def main() -> None:
    inventory_df = build_inventory_dataset()
    training_frame = build_training_frame(inventory_df)
    demand_model, stockout_model, metrics = train_models(training_frame)
    save_artifacts(demand_model, stockout_model, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
