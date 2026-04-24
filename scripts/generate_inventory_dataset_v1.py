import math
import os
import random
from collections import Counter

import pandas as pd


SEED = 20260423

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "inventory_dataset_v1.csv")

KARNATAKA_BOUNDS = {"lat": (11.5, 16.5), "lng": (74.0, 78.5)}
DUBAI_BOUNDS = {"lat": (25.05, 25.40), "lng": (55.10, 55.45)}

PRODUCT_CATALOG = {
    "Fuel": [
        {"product_name": "Petrol (ULP 91)", "unit_type": "liters", "supplier_region": "karnataka"},
        {"product_name": "Diesel (B7)", "unit_type": "liters", "supplier_region": "karnataka"},
        {"product_name": "Marine Fuel (IFO 380)", "unit_type": "liters", "supplier_region": "karnataka"},
    ],
    "Food Supply": [
        {"product_name": "Rice (50kg bags)", "unit_type": "kg", "supplier_region": "karnataka"},
        {"product_name": "Wheat (bulk tons)", "unit_type": "kg", "supplier_region": "karnataka"},
        {"product_name": "Packaged Meals (MRE kits)", "unit_type": "units", "supplier_region": "karnataka"},
    ],
    "Medical": [
        {"product_name": "Antibiotics (Amoxicillin)", "unit_type": "units", "supplier_region": "karnataka"},
        {"product_name": "First Aid Kits", "unit_type": "units", "supplier_region": "karnataka"},
    ],
}

KARNATAKA_CLUSTERS = [
    {"code": "BLR", "lat": 12.9716, "lng": 77.5946, "sigma_lat": 0.22, "sigma_lng": 0.24, "weight": 1.45},
    {"code": "MYS", "lat": 12.2958, "lng": 76.6394, "sigma_lat": 0.17, "sigma_lng": 0.18, "weight": 0.92},
    {"code": "MLR", "lat": 12.9141, "lng": 74.8560, "sigma_lat": 0.16, "sigma_lng": 0.17, "weight": 0.85},
    {"code": "UBL", "lat": 15.3647, "lng": 75.1240, "sigma_lat": 0.18, "sigma_lng": 0.21, "weight": 0.98},
    {"code": "BLG", "lat": 15.8497, "lng": 74.4977, "sigma_lat": 0.15, "sigma_lng": 0.16, "weight": 0.74},
    {"code": "KLR", "lat": 13.3409, "lng": 74.7421, "sigma_lat": 0.15, "sigma_lng": 0.17, "weight": 0.66},
    {"code": "DVG", "lat": 14.4644, "lng": 75.9218, "sigma_lat": 0.16, "sigma_lng": 0.18, "weight": 0.70},
    {"code": "BGM", "lat": 14.2294, "lng": 76.3980, "sigma_lat": 0.20, "sigma_lng": 0.20, "weight": 0.64},
    {"code": "SMG", "lat": 13.9299, "lng": 75.5681, "sigma_lat": 0.18, "sigma_lng": 0.19, "weight": 0.68},
    {"code": "BDR", "lat": 15.1394, "lng": 76.9214, "sigma_lat": 0.17, "sigma_lng": 0.18, "weight": 0.65},
    {"code": "KLB", "lat": 16.2302, "lng": 75.7100, "sigma_lat": 0.16, "sigma_lng": 0.17, "weight": 0.62},
]

DUBAI_CLUSTERS = [
    {"code": "DWT", "lat": 25.2048, "lng": 55.2708, "sigma_lat": 0.028, "sigma_lng": 0.030, "weight": 1.28},
    {"code": "DER", "lat": 25.2760, "lng": 55.3300, "sigma_lat": 0.025, "sigma_lng": 0.028, "weight": 1.05},
    {"code": "JBL", "lat": 25.0657, "lng": 55.1713, "sigma_lat": 0.024, "sigma_lng": 0.026, "weight": 0.95},
    {"code": "QUS", "lat": 25.2400, "lng": 55.3800, "sigma_lat": 0.028, "sigma_lng": 0.029, "weight": 0.82},
    {"code": "SHJ", "lat": 25.1450, "lng": 55.2200, "sigma_lat": 0.022, "sigma_lng": 0.025, "weight": 0.76},
]

CATEGORY_WEIGHTS = {
    "karnataka": {"Fuel": 0.34, "Food Supply": 0.42, "Medical": 0.24},
    "dubai": {"Fuel": 0.40, "Food Supply": 0.35, "Medical": 0.25},
}

BASE_DEMAND = {
    "Fuel": 2200.0,
    "Food Supply": 1450.0,
    "Medical": 180.0,
}


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def gaussian_point(rng, center, bounds):
    lat = rng.gauss(center["lat"], center["sigma_lat"])
    lng = rng.gauss(center["lng"], center["sigma_lng"])
    lat = clamp(lat, bounds["lat"][0], bounds["lat"][1])
    lng = clamp(lng, bounds["lng"][0], bounds["lng"][1])
    return round(lat, 6), round(lng, 6)


def weighted_choice(rng, items, weights):
    index = rng.choices(range(len(items)), weights=weights, k=1)[0]
    return items[index]


def choose_category_family(rng, region):
    mapping = CATEGORY_WEIGHTS[region]
    return weighted_choice(rng, list(mapping.keys()), list(mapping.values()))


def choose_product_name(rng, category_family):
    return weighted_choice(rng, PRODUCT_CATALOG[category_family], [1] * len(PRODUCT_CATALOG[category_family]))


def choose_risk_bucket(rng, region, category):
    if region == "karnataka":
        weights = {"LOW": 0.51, "MID": 0.29, "HIGH": 0.13, "CRITICAL": 0.07}
    else:
        weights = {"LOW": 0.30, "MID": 0.34, "HIGH": 0.20, "CRITICAL": 0.16}
    if category in {"Fuel", "Food Supply"}:
        weights["MID"] += 0.02
        weights["LOW"] -= 0.01
        weights["HIGH"] -= 0.01
    if category == "Medical":
        weights["CRITICAL"] += 0.01
        weights["LOW"] -= 0.01
    labels = list(weights.keys())
    values = [weights[label] for label in labels]
    return weighted_choice(rng, labels, values)


def target_days_for_bucket(rng, bucket):
    ranges = {
        "LOW": (11.5, 22.0),
        "MID": (5.2, 9.6),
        "HIGH": (2.1, 4.9),
        "CRITICAL": (0.8, 1.9),
    }
    lower, upper = ranges[bucket]
    mean = (lower + upper) / 2
    sigma = (upper - lower) / 6
    return clamp(rng.gauss(mean, sigma), lower, upper)


def lead_time_for(region, category, rng):
    if region == "karnataka":
        baseline = {"Fuel": 1.8, "Food Supply": 1.5, "Medical": 2.4}[category]
        return round(clamp(rng.gauss(baseline, 0.4), 1.0, 3.0), 2)
    baseline = {"Fuel": 14.5, "Food Supply": 12.5, "Medical": 11.2}[category]
    return round(clamp(rng.gauss(baseline, 1.8), 10.0, 20.0), 2)


def anomaly_for(region, category, bucket, rng):
    base = 0.10 if region == "karnataka" else 0.18
    category_offset = {"Fuel": 0.05, "Food Supply": 0.04, "Medical": 0.10}[category]
    bucket_offset = {"LOW": -0.06, "MID": 0.0, "HIGH": 0.08, "CRITICAL": 0.16}[bucket]
    anomaly = base + category_offset + bucket_offset + rng.gauss(0, 0.03)
    return round(clamp(anomaly, 0.02, 0.92), 4)


def demand_rate_for(region, category, tier_factor, rng):
    regional_factor = 1.04 if region == "karnataka" else 1.18
    demand = BASE_DEMAND[category] * regional_factor * tier_factor
    demand *= clamp(rng.gauss(1.0, 0.12), 0.72, 1.32)
    return round(max(demand, 3.0), 2)


def supply_rate_for(region, demand_rate, bucket, rng):
    if region == "karnataka":
        multiplier = {"LOW": 1.08, "MID": 1.00, "HIGH": 0.94, "CRITICAL": 0.86}[bucket]
    else:
        multiplier = {"LOW": 0.84, "MID": 0.77, "HIGH": 0.69, "CRITICAL": 0.60}[bucket]
    return round(max(demand_rate * clamp(rng.gauss(multiplier, 0.04), 0.58, 1.06), 1.0), 2)


def consumption_rate_for(demand_rate, category, rng):
    category_factor = {"Fuel": 1.05, "Food Supply": 1.03, "Medical": 0.96}[category]
    value = demand_rate * clamp(rng.gauss(category_factor, 0.04), 0.88, 1.12)
    return round(max(value, 1.0), 2)


def build_warehouses(rng, region, total, clusters, bounds):
    weights = [cluster["weight"] for cluster in clusters]
    warehouses = []
    for index in range(1, total + 1):
        cluster = weighted_choice(rng, clusters, weights)
        lat, lng = gaussian_point(rng, cluster, bounds)
        tier = clamp(cluster["weight"] / max(weights), 0.55, 1.0)
        size_signal = clamp(rng.gauss(tier, 0.10), 0.52, 1.08)
        item_count = int(round(clamp(54 + size_signal * 70 + rng.gauss(0, 7), 50, 150)))
        prefix = "KA" if region == "karnataka" else "DU"
        warehouses.append(
            {
                "warehouse_id": f"{prefix}-{cluster['code']}-{index:03d}",
                "region": region,
                "lat": lat,
                "lng": lng,
                "cluster_code": cluster["code"],
                "tier_factor": size_signal,
                "item_count": item_count,
            }
        )
    return warehouses


def generate_dataset():
    rng = random.Random(SEED)
    warehouses = []
    warehouses.extend(build_warehouses(rng, "karnataka", 91, KARNATAKA_CLUSTERS, KARNATAKA_BOUNDS))
    warehouses.extend(build_warehouses(rng, "dubai", 49, DUBAI_CLUSTERS, DUBAI_BOUNDS))

    rows = []
    item_counter = 1
    for warehouse in warehouses:
        for _ in range(warehouse["item_count"]):
            category_family = choose_category_family(rng, warehouse["region"])
            product = choose_product_name(rng, category_family)
            bucket = choose_risk_bucket(rng, warehouse["region"], category_family)
            demand_rate = demand_rate_for(warehouse["region"], category_family, warehouse["tier_factor"], rng)
            target_days = target_days_for_bucket(rng, bucket)
            stock_level = int(round(max(demand_rate * target_days * clamp(rng.gauss(1.0, 0.05), 0.88, 1.14), 8)))
            lead_time = lead_time_for(warehouse["region"], category_family, rng)
            supply_rate = supply_rate_for(warehouse["region"], demand_rate, bucket, rng)
            consumption_rate = consumption_rate_for(demand_rate, category_family, rng)
            anomaly_score = anomaly_for(warehouse["region"], category_family, bucket, rng)
            supplier_region = product["supplier_region"]
            linked_ship_id = ""
            if warehouse["region"] == "dubai":
                linked_ship_id = f"V-{rng.randint(13001, 14000):05d}"
            rows.append(
                {
                    "warehouse_id": warehouse["warehouse_id"],
                    "region": warehouse["region"],
                    "lat": warehouse["lat"],
                    "lng": warehouse["lng"],
                    "sku_id": f"SKU-{item_counter:06d}",
                    "product_name": product["product_name"],
                    "category": category_family,
                    "stock_units": stock_level,
                    "unit_type": product["unit_type"],
                    "daily_demand": demand_rate,
                    "incoming_supply": supply_rate,
                    "lead_time_days": lead_time,
                    "supplier_region": supplier_region,
                    "linked_ship_id": linked_ship_id,
                    "consumption_rate": consumption_rate,
                    "anomaly_score": anomaly_score,
                }
            )
            item_counter += 1

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_PATH, index=False)
    return frame, warehouses


def validate_dataset(frame, warehouses):
    by_region = frame.groupby("region")["warehouse_id"].nunique().to_dict()
    days_to_stockout = frame["stock_units"] / frame["daily_demand"].clip(lower=0.1)
    risk_buckets = pd.cut(
        days_to_stockout,
        bins=[-math.inf, 2, 5, 10, math.inf],
        labels=["CRITICAL", "HIGH", "MID", "LOW"],
    ).value_counts().to_dict()
    risk_buckets = {label: int(risk_buckets.get(label, 0)) for label in ["LOW", "MID", "HIGH", "CRITICAL"]}

    coord_validity = {
        "karnataka_valid": bool(
            frame.loc[frame["region"] == "karnataka", "lat"].between(*KARNATAKA_BOUNDS["lat"]).all()
            and frame.loc[frame["region"] == "karnataka", "lng"].between(*KARNATAKA_BOUNDS["lng"]).all()
        ),
        "dubai_valid": bool(
            frame.loc[frame["region"] == "dubai", "lat"].between(*DUBAI_BOUNDS["lat"]).all()
            and frame.loc[frame["region"] == "dubai", "lng"].between(*DUBAI_BOUNDS["lng"]).all()
        ),
    }

    print(f"dataset_rows={len(frame)}")
    print(f"warehouses_total={len(warehouses)}")
    print(f"region_split={by_region}")
    print(f"risk_distribution={risk_buckets}")
    print(f"null_cells={int(frame.isnull().sum().sum())}")
    print(f"duplicate_rows={int(frame.duplicated().sum())}")
    print(f"valid_coords={coord_validity}")
    print("sample_rows=")
    print(frame.head(10).to_csv(index=False).strip())


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    dataframe, warehouse_rows = generate_dataset()
    validate_dataset(dataframe, warehouse_rows)
