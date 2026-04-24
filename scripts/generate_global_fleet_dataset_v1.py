from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "global_fleet_dataset_v1.csv"

KARNATAKA_COUNT = 8_000
DUBAI_COUNT = 5_000
SHIP_COUNT = 1_000
TOTAL_ROWS = KARNATAKA_COUNT + DUBAI_COUNT + SHIP_COUNT

RISK_TARGETS = {
    "LOW": 4_900,
    "MID": 4_200,
    "HIGH": 2_100,
    "CRITICAL": 2_800,
}

REGION_CATEGORY_COUNTS = {
    "karnataka": {"LOW": 2800, "MID": 2400, "HIGH": 1100, "CRITICAL": 1700},
    "dubai": {"LOW": 1850, "MID": 1500, "HIGH": 850, "CRITICAL": 800},
    "ship": {"LOW": 250, "MID": 300, "HIGH": 150, "CRITICAL": 300},
}

KARNATAKA_BOUNDS = {"lat": (11.5, 16.5), "lng": (74.0, 78.5)}
DUBAI_BOUNDS = {"lat": (25.05, 25.40), "lng": (55.10, 55.45)}
SEA_BOUNDS = {"lat": (12.0, 25.0), "lng": (55.0, 72.2)}

KARNATAKA_CLUSTERS = [
    {"name": "Bengaluru", "center": (12.9716, 77.5946), "sigma": (0.24, 0.30), "weight": 0.19},
    {"name": "Mysuru", "center": (12.2958, 76.6394), "sigma": (0.20, 0.24), "weight": 0.11},
    {"name": "Mangaluru", "center": (12.9141, 74.8560), "sigma": (0.22, 0.20), "weight": 0.14},
    {"name": "Hubballi", "center": (15.3647, 75.1240), "sigma": (0.24, 0.26), "weight": 0.13},
    {"name": "Belagavi", "center": (15.8497, 74.4977), "sigma": (0.18, 0.18), "weight": 0.11},
    {"name": "Davanagere", "center": (14.4644, 75.9218), "sigma": (0.24, 0.24), "weight": 0.11},
    {"name": "Shivamogga", "center": (13.9299, 75.5681), "sigma": (0.18, 0.22), "weight": 0.10},
    {"name": "Ballari", "center": (15.1394, 76.9214), "sigma": (0.18, 0.22), "weight": 0.11},
]

DUBAI_CLUSTERS = [
    {"name": "Downtown", "center": (25.1972, 55.2744), "sigma": (0.024, 0.026), "weight": 0.27},
    {"name": "Dubai Marina", "center": (25.0825, 55.1450), "sigma": (0.020, 0.020), "weight": 0.24},
    {"name": "Deira", "center": (25.2736, 55.3167), "sigma": (0.022, 0.022), "weight": 0.20},
    {"name": "Jebel Ali", "center": (25.0650, 55.1250), "sigma": (0.018, 0.018), "weight": 0.12},
    {"name": "Silicon Oasis", "center": (25.1194, 55.3867), "sigma": (0.020, 0.022), "weight": 0.17},
]

SHIP_CORRIDORS = {
    "A": {
        "name": "Karnataka-Oman-Dubai",
        "weight": 0.40,
        "coastal_bias": 1.45,
        "points": [(12.8, 71.4), (15.1, 68.9), (18.4, 64.7), (21.5, 59.7), (24.3, 56.2)],
    },
    "B": {
        "name": "Karnataka-MidSea-Dubai",
        "weight": 0.34,
        "coastal_bias": 1.30,
        "points": [(13.1, 71.8), (16.0, 68.1), (19.0, 64.2), (21.6, 60.0), (24.1, 56.0)],
    },
    "C": {
        "name": "Karnataka-SouthernCurve-Dubai",
        "weight": 0.26,
        "coastal_bias": 1.20,
        "points": [(12.4, 71.1), (14.0, 68.3), (16.4, 64.8), (19.8, 60.6), (23.7, 56.1)],
    },
}

SHIP_SPEED_PROFILES = [
    {"name": "slow_cargo", "range": (10.0, 18.0), "weight": 0.40},
    {"name": "medium_cargo", "range": (18.0, 25.0), "weight": 0.38},
    {"name": "fast_transit", "range": (22.0, 25.0), "weight": 0.22},
]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def category_from_final_risk(final_risk: float) -> str:
    if final_risk < 0.3:
        return "LOW"
    if final_risk < 0.6:
        return "MID"
    if final_risk < 0.75:
        return "HIGH"
    return "CRITICAL"


def bezier_point(points: list[tuple[float, float]], t: float) -> tuple[float, float]:
    working = list(points)
    while len(working) > 1:
        working = [
            (
                (1.0 - t) * p0[0] + t * p1[0],
                (1.0 - t) * p0[1] + t * p1[1],
            )
            for p0, p1 in zip(working[:-1], working[1:])
        ]
    return working[0]


def sample_cluster_point(
    clusters: list[dict],
    bounds: dict[str, tuple[float, float]],
    corridor_mix: float,
    ambient_noise: tuple[float, float],
) -> tuple[float, float]:
    cluster = random.choices(clusters, weights=[entry["weight"] for entry in clusters], k=1)[0]
    lat = random.gauss(cluster["center"][0], cluster["sigma"][0])
    lng = random.gauss(cluster["center"][1], cluster["sigma"][1])

    if random.random() < corridor_mix:
        anchor = random.choice(clusters)["center"]
        lat = (lat * 0.68) + (anchor[0] * 0.32)
        lng = (lng * 0.68) + (anchor[1] * 0.32)

    lat += random.uniform(-ambient_noise[0], ambient_noise[0])
    lng += random.uniform(-ambient_noise[1], ambient_noise[1])
    return (
        clamp(lat, bounds["lat"][0], bounds["lat"][1]),
        clamp(lng, bounds["lng"][0], bounds["lng"][1]),
    )


def coastal_density_t(route_t: float, coastal_bias: float) -> float:
    edge_pull = 0.5 * (random.betavariate(coastal_bias, 2.2) + (1.0 - random.betavariate(coastal_bias, 2.2)))
    return clamp((route_t * 0.35) + (edge_pull * 0.65), 0.0, 1.0)


def build_ship_anchor_points() -> list[dict[str, float | str]]:
    anchors: list[dict[str, float | str]] = []
    corridor_ids = list(SHIP_CORRIDORS.keys())
    corridor_weights = [SHIP_CORRIDORS[corridor_id]["weight"] for corridor_id in corridor_ids]
    convoy_index = 1

    while len(anchors) < SHIP_COUNT:
        corridor_id = random.choices(corridor_ids, weights=corridor_weights, k=1)[0]
        corridor = SHIP_CORRIDORS[corridor_id]
        convoy_size = min(SHIP_COUNT - len(anchors), random.randint(2, 5))
        route_t = coastal_density_t(random.random(), corridor["coastal_bias"])
        base_lat, base_lng = bezier_point(corridor["points"], route_t)
        convoy_heading_bias = random.uniform(-0.10, 0.10)
        sea_condition = random.choices(
            ["calm", "moderate", "rough"],
            weights=[0.44, 0.38, 0.18],
            k=1,
        )[0]
        for convoy_rank in range(convoy_size):
            spacing_lat = (convoy_rank - (convoy_size - 1) / 2) * random.uniform(0.018, 0.04)
            spacing_lng = (convoy_rank - (convoy_size - 1) / 2) * random.uniform(0.024, 0.055)
            lat = base_lat + random.uniform(-0.04, 0.04) + convoy_heading_bias + spacing_lat
            lng = base_lng + random.uniform(-0.05, 0.05) - convoy_heading_bias + spacing_lng
            anchors.append(
                {
                    "corridor_id": corridor_id,
                    "route_name": corridor["name"],
                    "convoy_id": f"CONVOY-{convoy_index:03d}",
                    "convoy_size": convoy_size,
                    "convoy_rank": convoy_rank + 1,
                    "convoy_spacing_nm": round(random.uniform(0.8, 3.4), 2),
                    "route_t": round(route_t, 4),
                    "sea_condition": sea_condition,
                    "lat": round(clamp(lat, SEA_BOUNDS["lat"][0], SEA_BOUNDS["lat"][1]), 6),
                    "lng": round(clamp(lng, SEA_BOUNDS["lng"][0], SEA_BOUNDS["lng"][1]), 6),
                    "coastal_band": "coastal" if route_t < 0.18 or route_t > 0.82 else ("inner_coastal" if route_t < 0.32 or route_t > 0.68 else "mid_sea"),
                }
            )
            if len(anchors) >= SHIP_COUNT:
                break
        convoy_index += 1
    return anchors


def sample_base_risk_for_bucket(category: str, traffic_factor: float, anomaly_score: float) -> float | None:
    category_targets = {
        "LOW": (0.05, 0.2999),
        "MID": (0.3, 0.5999),
        "HIGH": (0.6, 0.7499),
        "CRITICAL": (0.75, 0.9999),
    }
    preferred_base_ranges = {
        "LOW": (0.08, 0.35),
        "MID": (0.28, 0.62),
        "HIGH": (0.50, 0.82),
        "CRITICAL": (0.68, 0.99),
    }
    target_min, target_max = category_targets[category]
    pref_min, pref_max = preferred_base_ranges[category]
    risk_offset = (0.30 * traffic_factor) + (0.20 * anomaly_score)
    lower = max(pref_min, (target_min - risk_offset) / 0.5)
    upper = min(pref_max, (target_max - risk_offset) / 0.5)
    if lower > upper:
        return None
    return round(random.uniform(lower, upper), 4)


def sample_ship_features(category: str, coastal_band: str, sea_condition: str) -> dict[str, float]:
    congestion_boost = {
        "coastal": 0.10,
        "inner_coastal": 0.05,
        "mid_sea": 0.0,
    }[coastal_band]
    sea_condition_boost = {
        "calm": -0.02,
        "moderate": 0.03,
        "rough": 0.09,
    }[sea_condition]
    traffic_band = {
        "LOW": (0.10, 0.24),
        "MID": (0.18, 0.36),
        "HIGH": (0.30, 0.48),
        "CRITICAL": (0.40, 0.62),
    }[category]
    anomaly_band = {
        "LOW": (0.04, 0.18),
        "MID": (0.10, 0.28),
        "HIGH": (0.20, 0.42),
        "CRITICAL": (0.30, 0.56),
    }[category]
    profile = random.choices(
        SHIP_SPEED_PROFILES,
        weights=[profile["weight"] for profile in SHIP_SPEED_PROFILES],
        k=1,
    )[0]
    sea_congestion = clamp(random.uniform(*traffic_band) + congestion_boost + sea_condition_boost, 0.05, 0.82)
    anomaly = clamp(
        random.uniform(*anomaly_band)
        + (0.05 if coastal_band == "coastal" and category in {"HIGH", "CRITICAL"} else 0.0)
        + max(0.0, sea_condition_boost * 0.55),
        0.0,
        0.72,
    )
    speed = random.uniform(*profile["range"]) * (1.0 - max(0.0, sea_condition_boost * 0.24))
    fatigue = clamp(random.gauss(0.05 + (sea_congestion * 0.18) + ((25.0 - min(speed, 25.0)) / 220.0), 0.04), 0.0, 0.35)
    fuel = random.uniform(35.0, 100.0)
    return {
        "traffic_factor": round(sea_congestion, 4),
        "sea_congestion": round(sea_congestion, 4),
        "anomaly_score": round(anomaly, 4),
        "driver_fatigue": round(fatigue, 4),
        "speed": round(speed, 2),
        "fuel_level": round(fuel, 2),
    }


def sample_vehicle_features(region: str, category: str) -> dict[str, float]:
    if region == "karnataka":
        traffic_band = {
            "LOW": (0.28, 0.58),
            "MID": (0.46, 0.76),
            "HIGH": (0.64, 0.90),
            "CRITICAL": (0.80, 1.0),
        }[category]
        anomaly_band = {
            "LOW": (0.06, 0.24),
            "MID": (0.16, 0.40),
            "HIGH": (0.34, 0.70),
            "CRITICAL": (0.52, 1.0),
        }[category]
        speed_band = {
            "LOW": (48.0, 80.0),
            "MID": (40.0, 68.0),
            "HIGH": (34.0, 58.0),
            "CRITICAL": (30.0, 50.0),
        }[category]
        fatigue_mu = 0.25
        traffic_sensitivity = 0.22
        speed_damping = 165.0
    else:
        traffic_band = {
            "LOW": (0.12, 0.34),
            "MID": (0.24, 0.50),
            "HIGH": (0.40, 0.66),
            "CRITICAL": (0.54, 0.82),
        }[category]
        anomaly_band = {
            "LOW": (0.01, 0.10),
            "MID": (0.05, 0.18),
            "HIGH": (0.12, 0.32),
            "CRITICAL": (0.22, 0.58),
        }[category]
        speed_band = {
            "LOW": (52.0, 80.0),
            "MID": (44.0, 70.0),
            "HIGH": (36.0, 60.0),
            "CRITICAL": (30.0, 52.0),
        }[category]
        fatigue_mu = 0.18
        traffic_sensitivity = 0.15
        speed_damping = 220.0

    traffic = random.uniform(*traffic_band)
    anomaly = random.uniform(*anomaly_band)
    speed = random.uniform(*speed_band)
    fatigue = clamp(random.gauss(fatigue_mu + ((100.0 - speed) / speed_damping) + (traffic * traffic_sensitivity), 0.06 if region == "dubai" else 0.09), 0.0, 1.0)
    fuel = random.uniform(20.0, 100.0)
    return {
        "traffic_factor": round(traffic, 4),
        "anomaly_score": round(anomaly, 4),
        "driver_fatigue": round(fatigue, 4),
        "speed": round(speed, 2),
        "fuel_level": round(fuel, 2),
    }


def build_record(index: int, entity_type: str, region: str, category: str, ship_anchor: dict | None = None) -> dict[str, object]:
    for _ in range(600):
        if entity_type == "ship":
            assert ship_anchor is not None
            lat = ship_anchor["lat"]
            lng = ship_anchor["lng"]
            features = sample_ship_features(category, str(ship_anchor["coastal_band"]), str(ship_anchor["sea_condition"]))
        elif region == "karnataka":
            lat, lng = sample_cluster_point(KARNATAKA_CLUSTERS, KARNATAKA_BOUNDS, corridor_mix=0.38, ambient_noise=(0.04, 0.05))
            features = sample_vehicle_features(region, category)
        else:
            lat, lng = sample_cluster_point(DUBAI_CLUSTERS, DUBAI_BOUNDS, corridor_mix=0.44, ambient_noise=(0.02, 0.025))
            features = sample_vehicle_features(region, category)

        base_risk = sample_base_risk_for_bucket(category, features["traffic_factor"], features["anomaly_score"])
        if base_risk is None:
            continue
        final_risk = (0.5 * base_risk) + (0.3 * features["traffic_factor"]) + (0.2 * features["anomaly_score"])
        if category_from_final_risk(final_risk) == category:
            record = {
                "vehicle_id": f"V-{index:05d}",
                "type": entity_type,
                "lat": round(lat, 6),
                "lng": round(lng, 6),
                "base_risk": round(base_risk, 4),
                **features,
                "route_name": "",
                "convoy_id": "",
                "convoy_size": 0,
                "convoy_rank": 0,
                "convoy_spacing_nm": 0.0,
                "sea_condition": "",
                "sea_congestion": 0.0,
                "route_progress": 0.0,
            }
            if entity_type == "ship":
                record.update(
                    {
                        "route_name": ship_anchor["route_name"],
                        "convoy_id": ship_anchor["convoy_id"],
                        "convoy_size": int(ship_anchor["convoy_size"]),
                        "convoy_rank": int(ship_anchor["convoy_rank"]),
                        "convoy_spacing_nm": float(ship_anchor["convoy_spacing_nm"]),
                        "sea_condition": ship_anchor["sea_condition"],
                        "route_progress": float(ship_anchor["route_t"]),
                    }
                )
            return record
    raise RuntimeError(f"Could not satisfy risk bucket for {entity_type}/{region}/{category}")


def build_region_records(start_index: int, entity_type: str, region: str, counts: dict[str, int], ship_anchors: list[dict] | None = None) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    next_index = start_index
    anchor_index = 0
    for category, count in counts.items():
        for _ in range(count):
            anchor = None if ship_anchors is None else ship_anchors[anchor_index]
            records.append(build_record(next_index, entity_type, region, category, ship_anchor=anchor))
            next_index += 1
            if ship_anchors is not None:
                anchor_index += 1
    return records


def assign_ship_categories(ship_anchors: list[dict]) -> list[tuple[str, dict]]:
    coastal_anchors = [anchor for anchor in ship_anchors if anchor["coastal_band"] == "coastal"]
    remaining_anchors = [anchor for anchor in ship_anchors if anchor["coastal_band"] != "coastal"]
    ordered_anchors = coastal_anchors + remaining_anchors

    critical_count = REGION_CATEGORY_COUNTS["ship"]["CRITICAL"]
    high_count = REGION_CATEGORY_COUNTS["ship"]["HIGH"]
    assigned_categories = (
        ["CRITICAL"] * critical_count +
        ["HIGH"] * high_count +
        ["MID"] * REGION_CATEGORY_COUNTS["ship"]["MID"] +
        ["LOW"] * REGION_CATEGORY_COUNTS["ship"]["LOW"]
    )
    return list(zip(assigned_categories, ordered_anchors))


def validate_dataset(df: pd.DataFrame) -> dict[str, object]:
    final_risk = (
        (0.5 * df["base_risk"]) +
        (0.3 * df["traffic_factor"]) +
        (0.2 * df["anomaly_score"])
    )
    categories = final_risk.apply(category_from_final_risk)
    risk_counts = categories.value_counts().reindex(["LOW", "MID", "HIGH", "CRITICAL"], fill_value=0)
    percentages = (risk_counts / len(df) * 100.0).round(2)

    vehicle_mask = df["type"] == "vehicle"
    ship_mask = df["type"] == "ship"
    in_karnataka = (
        df["lat"].between(*KARNATAKA_BOUNDS["lat"]) &
        df["lng"].between(*KARNATAKA_BOUNDS["lng"])
    )
    in_dubai = (
        df["lat"].between(*DUBAI_BOUNDS["lat"]) &
        df["lng"].between(*DUBAI_BOUNDS["lng"])
    )
    in_sea = (
        df["lat"].between(*SEA_BOUNDS["lat"]) &
        df["lng"].between(*SEA_BOUNDS["lng"])
    )

    ship_corridor_counts = {}
    ship_df = df[ship_mask]
    corridor_samples = {
        corridor_id: [bezier_point(corridor["points"], i / 24) for i in range(25)]
        for corridor_id, corridor in SHIP_CORRIDORS.items()
    }
    ship_corridor_counts = {corridor_id: 0 for corridor_id in SHIP_CORRIDORS}
    for _, row in ship_df.iterrows():
        best_corridor = min(
            SHIP_CORRIDORS.keys(),
            key=lambda corridor_id: min(
                ((row["lat"] - sample[0]) ** 2 + (row["lng"] - sample[1]) ** 2) ** 0.5
                for sample in corridor_samples[corridor_id]
            ),
        )
        ship_corridor_counts[best_corridor] += 1

    average_speeds = {
        "karnataka_vehicles": float(round(df[(df["type"] == "vehicle") & in_karnataka]["speed"].mean(), 2)),
        "dubai_vehicles": float(round(df[(df["type"] == "vehicle") & in_dubai]["speed"].mean(), 2)),
        "ships": float(round(ship_df["speed"].mean(), 2)),
    }

    return {
        "total_rows": int(len(df)),
        "karnataka_count": int((vehicle_mask & in_karnataka).sum()),
        "dubai_count": int((vehicle_mask & in_dubai).sum()),
        "ship_count": int(ship_mask.sum()),
        "risk_counts": {k: int(v) for k, v in risk_counts.items()},
        "risk_percentages": {k: float(v) for k, v in percentages.items()},
        "vehicle_points_valid": bool((in_karnataka[vehicle_mask] | in_dubai[vehicle_mask]).all()),
        "ship_points_valid": bool((in_sea[ship_mask] & ~in_karnataka[ship_mask] & ~in_dubai[ship_mask]).all()),
        "duplicate_ids": int(df["vehicle_id"].duplicated().sum()),
        "null_cells": int(df.isna().sum().sum()),
        "ship_corridor_counts": ship_corridor_counts,
        "average_speeds": average_speeds,
    }


def main() -> None:
    random.seed(20260423)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    next_index = 1

    records.extend(build_region_records(next_index, "vehicle", "karnataka", REGION_CATEGORY_COUNTS["karnataka"]))
    next_index += KARNATAKA_COUNT
    records.extend(build_region_records(next_index, "vehicle", "dubai", REGION_CATEGORY_COUNTS["dubai"]))
    next_index += DUBAI_COUNT

    ship_anchors = build_ship_anchor_points()
    ship_records: list[dict[str, object]] = []
    ship_index = next_index
    for category, anchor in assign_ship_categories(ship_anchors):
        ship_records.append(build_record(ship_index, "ship", "ship", category, ship_anchor=anchor))
        ship_index += 1
    records.extend(ship_records)

    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=20260423).reset_index(drop=True)
    df["vehicle_id"] = [f"V-{idx:05d}" for idx in range(1, len(df) + 1)]
    df.to_csv(OUTPUT_FILE, index=False)

    stats = validate_dataset(df)
    print(f"total rows: {stats['total_rows']}")
    print(f"karnataka count: {stats['karnataka_count']}")
    print(f"dubai count: {stats['dubai_count']}")
    print(f"ship count: {stats['ship_count']}")
    print(f"risk distribution counts: {stats['risk_counts']}")
    print(f"risk distribution percentages: {stats['risk_percentages']}")
    print(f"ships per corridor: {stats['ship_corridor_counts']}")
    print(f"average speed by type: {stats['average_speeds']}")
    print(f"vehicle lat/lng valid: {stats['vehicle_points_valid']}")
    print(f"ship lat/lng valid: {stats['ship_points_valid']}")
    print(f"duplicate ids: {stats['duplicate_ids']}")
    print(f"null cells: {stats['null_cells']}")


if __name__ == "__main__":
    main()
