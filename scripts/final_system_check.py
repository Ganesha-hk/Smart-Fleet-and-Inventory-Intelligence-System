#!/usr/bin/env python3
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "backend"))

from app.inventory.services.inventory_service import inventory_service  # noqa: E402
from app.services.inference_service import inference_service  # noqa: E402


def parse_map_config() -> dict:
    fleet_view = (ROOT_DIR / "frontend" / "src" / "pages" / "FleetView.jsx").read_text(encoding="utf-8")
    counts_block = fleet_view.split("const TARGET_COUNTS = {", 1)[1].split("};", 1)[0]
    zoom_block = fleet_view.split("const ZOOM_RISK_VISIBILITY = {", 1)[1].split("};", 1)[0]
    counts = {
        match.group(1): int(match.group(2))
        for match in re.finditer(r"(LOW|MID|HIGH|CRITICAL):\s*(\d+)", counts_block)
    }
    zoom_rules = {
        match.group(1): float(match.group(2))
        for match in re.finditer(r"(LOW|MID|HIGH|CRITICAL):\s*([0-9.]+)", zoom_block)
    }
    return {
        "counts": counts,
        "max_render": sum(counts.values()),
        "zoom_rules": zoom_rules,
    }


def main() -> int:
    dataset_path = ROOT_DIR / "data" / "processed" / "global_fleet_dataset_v1.csv"
    dataset = pd.read_csv(dataset_path)
    fleet_duplicates = int(dataset["vehicle_id"].duplicated().sum()) if "vehicle_id" in dataset.columns else -1
    fleet_nulls = int(dataset[["vehicle_id", "type", "lat", "lng"]].isnull().sum().sum())

    batch_payload = inference_service.get_sample_batch()
    vehicles = batch_payload["vehicles"]

    required_fields = {"vehicle_id", "type", "lat", "lng", "final_risk", "risk_level"}
    missing_fields = sorted(required_fields - set(vehicles[0].keys())) if vehicles else sorted(required_fields)
    live_risk = Counter(vehicle["risk_level"] for vehicle in vehicles)

    events_active = False
    for _ in range(6):
        next_payload = inference_service.get_sample_batch()
        if next_payload.get("events"):
            events_active = True
            batch_payload = next_payload
            break

    inventory_one = inventory_service.summary()
    inventory_two = inventory_service.summary()

    inventory_dynamic = inventory_one["demand_trend"][-1] != inventory_two["demand_trend"][-1]
    inventory_has_fields = all(
        key in inventory_one["stock_alerts"][0]
        for key in ["product_name", "stock_units", "daily_demand", "days_to_stockout", "incoming_ship_id", "eta_days", "shipment_status"]
    ) if inventory_one["stock_alerts"] else False

    map_config = parse_map_config()
    map_ok = (
        map_config["max_render"] <= 400
        and map_config["counts"] == {"LOW": 140, "MID": 120, "HIGH": 60, "CRITICAL": 80}
        and map_config["zoom_rules"].get("MID") == 5.0
        and map_config["zoom_rules"].get("LOW", 0) > 6.0
    )

    fleet_ok = (
        len(dataset) == 14000
        and int((dataset["type"] == "ship").sum()) == 1000
        and int(dataset["lat"].between(11.5, 16.5).sum()) > int(dataset["lat"].between(25.05, 25.40).sum())
        and fleet_nulls == 0
        and fleet_duplicates == 0
        and batch_payload["totalCount"] == 14000
        and len(vehicles) == 14000
        and not missing_fields
    )

    inventory_ok = inventory_dynamic and inventory_has_fields and bool(inventory_two["restock_recommendations"])
    feed_ok = events_active and bool(batch_payload.get("events"))

    final_status = {
        "status": "PASS" if all([fleet_ok, inventory_ok, map_ok, feed_ok]) else "FAIL",
        "fleet": "OK" if fleet_ok else "FAIL",
        "inventory": "OK" if inventory_ok else "FAIL",
        "map": "OK" if map_ok else "FAIL",
        "feed": "OK" if feed_ok else "FAIL",
    }

    print(json.dumps(final_status, indent=2))
    print(f"system status: {final_status['status']}")
    print(f"total vehicles: {batch_payload['totalCount']}")
    print(f"risk distribution: {dict(live_risk)}")
    print(
        "inventory summary: "
        f"{len(inventory_two['stock_alerts'])} alerts, "
        f"{len(inventory_two['restock_recommendations'])} recommendations, "
        f"{len(inventory_two['warehouse_snapshots'])} warehouse snapshots"
    )
    print("SYSTEM READY" if final_status["status"] == "PASS" else "SYSTEM NOT READY")
    return 0 if final_status["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
