import math
import os
import logging
import time
import joblib
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InventoryService:
    _instance = None

    DATASET_COLUMNS = [
        "warehouse_id",
        "region",
        "lat",
        "lng",
        "sku_id",
        "product_name",
        "category",
        "stock_units",
        "unit_type",
        "daily_demand",
        "incoming_supply",
        "lead_time_days",
        "supplier_region",
        "linked_ship_id",
        "consumption_rate",
        "anomaly_score",
    ]

    CATEGORY_PHASE = {
        "Fuel": 0.25,
        "Food Supply": 1.75,
        "Medical": 2.10,
    }

    CATEGORY_DELAY_BIAS = {
        "Fuel": 0.14,
        "Food Supply": 0.10,
        "Medical": 0.14,
    }
    PRODUCT_FAMILY = {
        "Petrol (ULP 91)": "Fuel",
        "Diesel (B7)": "Fuel",
        "Marine Fuel (IFO 380)": "Fuel",
        "Rice (50kg bags)": "Food Supply",
        "Wheat (bulk tons)": "Food Supply",
        "Packaged Meals (MRE kits)": "Food Supply",
        "Antibiotics (Amoxicillin)": "Medical",
        "First Aid Kits": "Medical",
    }

    REGION_DELAY_BIAS = {
        "karnataka": 0.12,
        "dubai": 0.42,
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self):
        if not self._loaded:
            self._load_assets()
            self._loaded = True

    def _load_assets(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        data_path = os.path.join(base_dir, "data", "processed", "inventory_dataset_v1.csv")
        artifacts_dir = os.path.join(base_dir, "ml_engine", "artifacts")

        try:
            frame = pd.read_csv(data_path)
            missing_columns = [column for column in self.DATASET_COLUMNS if column not in frame.columns]
            if missing_columns:
                raise ValueError(f"Inventory dataset missing required columns: {missing_columns}")

            self.dataset = frame[self.DATASET_COLUMNS].copy()
        except Exception as e:
            logger.error(f"Error loading inventory dataset: {e}")
            self.dataset = pd.DataFrame(columns=self.DATASET_COLUMNS)

        # Load ML Models
        try:
            start_time = time.time()
            logger.info("Loading Inventory ML models (optimized with mmap)...")
            self.demand_model = joblib.load(os.path.join(artifacts_dir, "demand_model.joblib"), mmap_mode='r')
            self.stockout_model = joblib.load(os.path.join(artifacts_dir, "stockout_model.joblib"), mmap_mode='r')
            load_time = time.time() - start_time
            logger.info(f"Inventory models loaded successfully in {load_time:.2f} seconds.")
        except Exception as e:
            logger.warning(f"Failed to load inventory ML models: {e}. Falling back to rule-based logic.")
            self.demand_model = None
            self.stockout_model = None

        self.step = 0
        self.history = deque(maxlen=6)
        self.states = {}
        self.maritime_signal = {
            "ships_total": 0,
            "delayed_ships": 0,
            "arrived_ships": 0,
            "avg_sea_congestion": 0.0,
        }

        for row in self.dataset.to_dict(orient="records"):
            days_to_stockout = self._days_to_stockout(row["stock_units"], row["daily_demand"])
            delay_risk = self._linked_delay_risk(row["region"], row["category"], row["lead_time_days"], 0.0, self.step)
            
            # Use ML model if available for initial risk
            if self.stockout_model:
                ml_inputs = self._get_ml_inputs(row, delay_risk)
                try:
                    final_risk = float(self.stockout_model.predict_proba(ml_inputs)[0][1])
                except:
                    final_risk = self._compute_final_risk(days_to_stockout, row["lead_time_days"], row["anomaly_score"])
            else:
                final_risk = self._compute_final_risk(days_to_stockout, row["lead_time_days"], row["anomaly_score"])
                
            risk_level = self._classify_with_streak(days_to_stockout, 0, row["lead_time_days"])
            self.states[row["sku_id"]] = {
                **row,
                "base_stock_units": float(row["stock_units"]),
                "base_daily_demand": float(row["daily_demand"]),
                "previous_risk": round(final_risk, 4),
                "final_risk": round(final_risk, 4),
                "risk_level": risk_level,
                "critical_streak": 0,
                "days_to_stockout": round(days_to_stockout, 2),
                "linked_delay_risk": round(delay_risk, 4),
                "predicted_demand": round(float(row["daily_demand"]), 2),
                "pending_shipments": [],
                "warehouse_location": row["warehouse_id"],
                "reorder_count": 0,
            }
            if days_to_stockout < row["lead_time_days"] * (1.25 if row["region"] == "dubai" else 1.05):
                self.states[row["sku_id"]]["pending_shipments"].append(
                    {
                        "eta_days": max(1, int(round(row["lead_time_days"] * (0.75 if row["region"] == "dubai" else 0.55)))),
                        "quantity": int(round(max(row["incoming_supply"] * row["lead_time_days"] * 1.6, row["daily_demand"] * 2.0))),
                        "delay_locked": row["region"] == "dubai",
                        "linked_ship_id": row["linked_ship_id"],
                    }
                )

        self._seed_history()

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def _normalize_days_to_stockout(self, days_to_stockout: float) -> float:
        return self._clamp(days_to_stockout / 15.0, 0.0, 1.0)

    def _lead_time_factor(self, lead_time: float) -> float:
        return self._clamp(lead_time / 12.0, 0.0, 1.0)

    def _days_to_stockout(self, stock_level: float, demand_rate: float) -> float:
        return float(stock_level) / max(float(demand_rate), 0.1)

    def _category_family(self, category: str) -> str:
        return self.PRODUCT_FAMILY.get(category, category)

    def _compute_final_risk(self, days_to_stockout: float, lead_time: float, anomaly_score: float) -> float:
        normalized_cover = self._normalize_days_to_stockout(days_to_stockout)
        final_risk = (
            0.5 * (1.0 - normalized_cover)
            + 0.3 * self._lead_time_factor(lead_time)
            + 0.2 * self._clamp(anomaly_score, 0.0, 1.0)
        )
        return round(self._clamp(final_risk, 0.0, 1.0), 4)

    def _get_ml_inputs(self, item: Dict[str, Any], linked_delay_risk: float) -> pd.DataFrame:
        """Prepares feature frame for ML models."""
        now = datetime.now()
        
        # Mapping to feature columns used in training
        data = {
            "inventory_level": [float(item.get("stock_units", 0))],
            "historical_demand": [float(item.get("daily_demand", 0))],
            "lead_time": [float(item.get("lead_time_days", 4))],
            "supplier_reliability": [0.88], # Default as not in dataset
            "linked_delay_risk": [float(linked_delay_risk)],
            "hour": [now.hour],
            "day_of_week": [now.weekday()],
            "is_weekend": [1 if now.weekday() >= 5 else 0]
        }
        return pd.DataFrame(data)

    def _refresh_maritime_signal(self) -> None:
        try:
            from app.services.inference_service import inference_service

            self.maritime_signal = inference_service.get_maritime_supply_signal()
        except Exception:
            self.maritime_signal = {
                "ships_total": 0,
                "delayed_ships": 0,
                "arrived_ships": 0,
                "avg_sea_congestion": 0.0,
            }

    def _linked_delay_risk(
        self,
        region: str,
        category: str,
        lead_time: float,
        stock_gap: float,
        step: int,
    ) -> float:
        category_family = self._category_family(category)
        phase = self.CATEGORY_PHASE[category_family]
        maritime_pressure = min(
            0.32,
            (self.maritime_signal["delayed_ships"] / max(self.maritime_signal["ships_total"], 1)) * 0.45
            + (self.maritime_signal["avg_sea_congestion"] * 0.22),
        )
        sea_lane_pressure = 0.10 + 0.08 * (1 + math.sin(step / 2.8 + phase)) / 2
        convoy_pressure = 0.06 + 0.05 * (1 + math.cos(step / 3.5 + phase * 0.6)) / 2
        stock_pressure = self._clamp(stock_gap / 12.0, 0.0, 0.26)
        delay = (
            self.REGION_DELAY_BIAS[region]
            + self.CATEGORY_DELAY_BIAS[category_family]
            + 0.18 * self._lead_time_factor(lead_time)
            + sea_lane_pressure
            + convoy_pressure
            + stock_pressure
        )
        if region == "karnataka":
            delay -= 0.12
        else:
            delay += maritime_pressure
        return round(self._clamp(delay, 0.05, 0.98), 4)

    def _classify_days(self, days_to_stockout: float) -> str:
        if days_to_stockout < 2.0:
            return "CRITICAL"
        if days_to_stockout < 5.0:
            return "HIGH"
        if days_to_stockout <= 10.0:
            return "MID"
        return "LOW"

    def _classify_with_streak(self, days_to_stockout: float, critical_streak: int, lead_time_days: float | None = None) -> str:
        provisional = self._classify_days(days_to_stockout)
        if lead_time_days is not None and days_to_stockout < lead_time_days:
            return "CRITICAL" if critical_streak >= 2 else "HIGH"
        if provisional == "CRITICAL" and critical_streak < 2:
            return "HIGH"
        return provisional

    def _route_priority(self, urgency: str, linked_delay_risk: float) -> str:
        if urgency == "CRITICAL" or linked_delay_risk >= 0.7:
            return "Immediate maritime slot and local dispatch priority"
        if urgency == "HIGH":
            return "Prioritize next dispatch wave"
        return "Standard replenishment cycle"

    def _shipment_status(self, row: Dict[str, Any]) -> Dict[str, Any]:
        eta_days = min(
            [shipment["eta_days"] for shipment in row["pending_shipments"]],
            default=int(round(float(row["lead_time_days"]))),
        )
        incoming_ship_id = str(row["linked_ship_id"] or "")
        if row["region"] == "dubai" and incoming_ship_id:
            return {
                "incoming_ship_id": incoming_ship_id,
                "eta_days": int(eta_days),
                "shipment_status": f"Vessel {incoming_ship_id} inbound in {int(eta_days)} days",
                "incoming_shipment": f"Vessel {incoming_ship_id} (ETA: {int(eta_days)} days)",
            }
        return {
            "incoming_ship_id": "",
            "eta_days": int(eta_days),
            "shipment_status": f"Local replenishment in {int(eta_days)} days",
            "incoming_shipment": f"Local replenishment (ETA: {int(eta_days)} days)",
        }

    def _format_units(self, value: float, unit_type: str) -> str:
        return f"{int(round(value)):,} {unit_type}"

    def _phase_value(self, sku_id: str) -> float:
        numeric = int(sku_id.split("-")[-1])
        return (numeric % 17) / 17.0

    def _seed_history(self):
        total_base_demand = sum(float(state["base_daily_demand"]) for state in self.states.values())
        for offset in range(-5, 1):
            phase = offset / 2.4
            realized = total_base_demand * (0.96 + 0.04 * math.sin(phase))
            predicted = total_base_demand * (1.0 + 0.03 * math.cos(phase / 1.4))
            self.history.append(
                {
                    "name": f"T{offset}",
                    "demand": round(realized, 2),
                    "predicted": round(predicted, 2),
                    "days_to_stockout": round(8.0 + math.sin(phase) * 0.7, 2),
                    "karnataka_stock": round(realized * 0.58, 2),
                    "dubai_stock": round(realized * 0.42, 2),
                }
            )

    def _maybe_schedule_shipment(self, state: Dict[str, Any]):
        projected_cover = self._days_to_stockout(state["stock_units"], state["predicted_demand"])
        maritime_buffer = self.maritime_signal["delayed_ships"] * 0.35 if state["region"] == "dubai" else 0.0
        trigger_cover = max(state["lead_time_days"] * 1.7, (6.0 + maritime_buffer) if state["region"] == "dubai" else 4.0)
        if projected_cover > trigger_cover or state["pending_shipments"]:
            return

        self.step = max(self.step, 1)
        lead_variation = 0.4 * math.sin(self.step / 2.0 + self._phase_value(state["sku_id"]))
        congestion_delay = 0.0
        category_family = self._category_family(state["category"])
        if state["region"] == "dubai":
            congestion_delay += 0.7 + 0.8 * (1 + math.sin(self.step / 2.6 + self.CATEGORY_PHASE[category_family])) / 2
            congestion_delay += self.maritime_signal["avg_sea_congestion"] * 1.8
        else:
            congestion_delay += 0.15 + 0.25 * (1 + math.cos(self.step / 3.2 + self.CATEGORY_PHASE[category_family])) / 2

        eta_days = max(1, int(round(state["lead_time_days"] * (0.9 if state["region"] == "dubai" else 0.7) + lead_variation + congestion_delay)))
        target_cover = max(
            state["lead_time_days"] * (2.8 if state["region"] == "dubai" else 2.0),
            (9.5 + maritime_buffer) if state["region"] == "dubai" else 6.5,
        )
        target_stock = target_cover * state["predicted_demand"]
        shipment_quantity = max(
            int(round(target_stock - state["stock_units"])),
            int(round(state["incoming_supply"] * state["lead_time_days"] * (2.1 if state["region"] == "dubai" else 1.6))),
        )

        state["pending_shipments"].append(
            {
                "eta_days": eta_days,
                "quantity": shipment_quantity,
                "delay_locked": state["region"] == "dubai",
                "linked_ship_id": state["linked_ship_id"],
            }
        )
        state["reorder_count"] += 1

    def _advance_step(self):
        self.step += 1
        self._refresh_maritime_signal()
        total_demand = 0.0
        total_predicted = 0.0

        for state in self.states.values():
            category_family = self._category_family(state["category"])
            phase = self._phase_value(state["sku_id"]) + self.CATEGORY_PHASE[category_family]
            regional_wave = 0.06 * math.sin(self.step / 2.5 + phase)
            stop_go = 0.10 * math.sin(self.step / 1.8 + phase * 1.3) if state["region"] == "karnataka" else 0.04 * math.sin(self.step / 3.0 + phase)
            demand_multiplier = 1.0 + regional_wave + stop_go
            if state["region"] == "karnataka":
                demand_multiplier += 0.05
            else:
                demand_multiplier += 0.08 + (self.maritime_signal["avg_sea_congestion"] * 0.10)
            demand_multiplier = self._clamp(demand_multiplier, 0.72, 1.35)

            predicted_demand = max(state["base_daily_demand"] * demand_multiplier, 1.0)
            consumption_multiplier = 0.96 + 0.05 * math.cos(self.step / 2.2 + phase * 0.8)
            if category_family in {"Fuel", "Food Supply"}:
                consumption_multiplier += 0.03
            actual_consumption = max(state["consumption_rate"] * consumption_multiplier * 0.88, 0.8)

            for shipment in state["pending_shipments"]:
                if shipment["delay_locked"] and state["region"] == "dubai":
                    congestion_hit = max(
                        (1 + math.sin(self.step / 2.7 + phase)) / 2,
                        self.maritime_signal["avg_sea_congestion"],
                    )
                    if (
                        congestion_hit > 0.72
                        or self.maritime_signal["delayed_ships"] > max(2, self.maritime_signal["ships_total"] * 0.12)
                    ) and shipment["eta_days"] > 1:
                        shipment["eta_days"] += 1
                        shipment["delay_locked"] = False
                shipment["eta_days"] -= 1

            arrivals = [shipment for shipment in state["pending_shipments"] if shipment["eta_days"] <= 0]
            if state["region"] == "dubai" and self.maritime_signal["arrived_ships"] > 0:
                arrivals.append(
                    {
                        "eta_days": 0,
                        "quantity": int(round(state["incoming_supply"] * self.maritime_signal["arrived_ships"] * 0.8)),
                        "linked_ship_id": state["linked_ship_id"],
                    }
                )
            if arrivals:
                state["stock_units"] += sum(float(shipment["quantity"]) for shipment in arrivals)
            state["pending_shipments"] = [shipment for shipment in state["pending_shipments"] if shipment["eta_days"] > 0]

            state["stock_units"] = max(state["stock_units"] - actual_consumption, 0.0)
            state["predicted_demand"] = round(predicted_demand, 2)

            self._maybe_schedule_shipment(state)

            stock_gap = max(state["lead_time_days"] - self._days_to_stockout(state["stock_units"], predicted_demand), 0.0)
            anomaly_wave = 0.05 * math.sin(self.step / 3.4 + phase * 1.1)
            if state["region"] == "karnataka":
                anomaly_wave -= 0.01
                state["lead_time_days"] = round(self._clamp(float(state["lead_time_days"]) * 0.985, 1.0, 3.0), 2)
            else:
                anomaly_wave += self.maritime_signal["avg_sea_congestion"] * 0.06
                if self.maritime_signal["delayed_ships"] > 0:
                    state["lead_time_days"] = round(
                        self._clamp(float(state["lead_time_days"]) + 2.0 + ((self.maritime_signal["delayed_ships"] - 1) * 0.55), 10.0, 20.0),
                        2,
                    )
                    state["incoming_supply"] = round(max(float(state["incoming_supply"]) * 0.6, 1.0), 2)
                elif self.maritime_signal["arrived_ships"] > 0:
                    state["lead_time_days"] = round(
                        self._clamp(float(state["lead_time_days"]) - (0.35 * self.maritime_signal["arrived_ships"]), 10.0, 20.0),
                        2,
                    )
                if self.maritime_signal["delayed_ships"] >= 2:
                    state["anomaly_score"] = round(self._clamp(float(state["anomaly_score"]) + 0.25, 0.02, 0.95), 4)
            state["anomaly_score"] = round(
                self._clamp(state["anomaly_score"] * 0.84 + anomaly_wave + 0.14, 0.02, 0.95),
                4,
            )
            state["linked_delay_risk"] = self._linked_delay_risk(
                state["region"],
                state["category"],
                state["lead_time_days"],
                stock_gap,
                self.step,
            )

            days_to_stockout = self._days_to_stockout(state["stock_units"], predicted_demand)
            
            # Use ML model if available
            if self.stockout_model:
                ml_inputs = self._get_ml_inputs(state, state["linked_delay_risk"])
                try:
                    new_risk = float(self.stockout_model.predict_proba(ml_inputs)[0][1])
                except:
                    new_risk = self._compute_final_risk(days_to_stockout, state["lead_time_days"], state["anomaly_score"])
            else:
                new_risk = self._compute_final_risk(days_to_stockout, state["lead_time_days"], state["anomaly_score"])

            if abs(new_risk - state["previous_risk"]) < 0.05:
                final_risk = state["previous_risk"]
            else:
                delta = self._clamp(new_risk - state["previous_risk"], -0.08, 0.08)
                final_risk = round(self._clamp(state["previous_risk"] + delta, 0.0, 1.0), 4)

            state["days_to_stockout"] = round(days_to_stockout, 2)
            state["critical_streak"] = state["critical_streak"] + 1 if days_to_stockout < state["lead_time_days"] else 0
            state["risk_level"] = self._classify_with_streak(days_to_stockout, state["critical_streak"], state["lead_time_days"])
            state["final_risk"] = final_risk
            state["previous_risk"] = final_risk

            total_demand += actual_consumption
            total_predicted += predicted_demand

        self.history.append(
            {
                "name": f"T+{self.step}",
                "demand": round(total_demand, 2),
                "predicted": round(total_predicted, 2),
                "days_to_stockout": round(float(sum(state["days_to_stockout"] for state in self.states.values()) / max(len(self.states), 1)), 2),
                "karnataka_stock": round(float(sum(state["stock_units"] for state in self.states.values() if state["region"] == "karnataka")), 2),
                "dubai_stock": round(float(sum(state["stock_units"] for state in self.states.values() if state["region"] == "dubai")), 2),
            }
        )

    def _coerce_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        warehouse_id = payload.get("warehouse_id") or payload.get("warehouse_location") or "ADHOC-WH"
        region = str(payload.get("region") or ("dubai" if warehouse_id.startswith("DU-") else "karnataka")).lower()
        category = payload.get("category") or "Fuel"
        product_name = payload.get("product_name") or "Diesel (B7)"
        stock_units = float(payload.get("stock_units", payload.get("stock_level", payload.get("inventory_level", 0))))
        daily_demand = float(payload.get("daily_demand", payload.get("demand_rate", payload.get("historical_demand", 0))))
        incoming_supply = float(payload.get("incoming_supply", payload.get("supply_rate", max(daily_demand * 0.9, 1.0))))
        lead_time_days = float(payload.get("lead_time_days", payload.get("lead_time", 4.0)))
        consumption_rate = float(payload.get("consumption_rate", max(daily_demand, 1.0)))
        anomaly_score = float(payload.get("anomaly_score", payload.get("linked_delay_risk", 0.15)))
        timestamp = payload.get("timestamp") or datetime.now(timezone.utc).isoformat()
        sku_id = payload.get("sku_id") or payload.get("item_id") or "SKU-ADHOC"
        return {
            "sku_id": sku_id,
            "warehouse_location": warehouse_id,
            "region": region,
            "category": category,
            "product_name": product_name,
            "stock_units": stock_units,
            "daily_demand": max(daily_demand, 0.1),
            "incoming_supply": max(incoming_supply, 0.1),
            "lead_time_days": max(lead_time_days, 0.1),
            "consumption_rate": max(consumption_rate, 0.1),
            "anomaly_score": self._clamp(anomaly_score, 0.0, 1.0),
            "unit_type": payload.get("unit_type", "units"),
            "linked_ship_id": payload.get("linked_ship_id", ""),
            "timestamp": timestamp,
        }

    def predict_demand(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        item = self._coerce_payload(payload)
        linked_delay_risk = self._linked_delay_risk(item["region"], item["category"], item["lead_time_days"], 0.0, self.step)
        
        # Use ML model if available
        if self.demand_model:
            ml_inputs = self._get_ml_inputs(item, linked_delay_risk)
            try:
                predicted_demand = round(float(self.demand_model.predict(ml_inputs)[0]), 2)
            except:
                phase = self._phase_value(item["sku_id"]) + self.CATEGORY_PHASE[self._category_family(item["category"])]
                regional_variation = 1.0 + 0.05 * math.sin(self.step / 2.5 + phase)
                if item["region"] == "karnataka":
                    regional_variation += 0.04
                predicted_demand = round(max(item["daily_demand"] * regional_variation, 0.1), 2)
        else:
            phase = self._phase_value(item["sku_id"]) + self.CATEGORY_PHASE[self._category_family(item["category"])]
            regional_variation = 1.0 + 0.05 * math.sin(self.step / 2.5 + phase)
            if item["region"] == "karnataka":
                regional_variation += 0.04
            predicted_demand = round(max(item["daily_demand"] * regional_variation, 0.1), 2)
            
        inventory_level = int(round(item["stock_units"]))
        demand_pressure = "LOW"
        if predicted_demand / max(inventory_level, 1) >= 1.05:
            demand_pressure = "HIGH"
        elif predicted_demand / max(inventory_level, 1) >= 0.7:
            demand_pressure = "MEDIUM"
        return {
            "item_id": item["sku_id"],
            "warehouse_location": item["warehouse_location"],
            "timestamp": item["timestamp"],
            "predicted_demand": predicted_demand,
            "linked_delay_risk": linked_delay_risk,
            "demand_pressure": demand_pressure,
        }

    def stockout_risk(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        item = self._coerce_payload(payload)
        days_of_cover = round(self._days_to_stockout(item["stock_units"], item["daily_demand"]), 2)
        linked_delay_risk = self._linked_delay_risk(item["region"], item["category"], item["lead_time_days"], 0.0, self.step)
        
        # Use ML model if available
        if self.stockout_model:
            ml_inputs = self._get_ml_inputs(item, linked_delay_risk)
            try:
                stockout_probability = round(float(self.stockout_model.predict_proba(ml_inputs)[0][1]), 4)
            except:
                stockout_probability = self._compute_final_risk(days_of_cover, item["lead_time_days"], item["anomaly_score"])
        else:
            stockout_probability = self._compute_final_risk(days_of_cover, item["lead_time_days"], item["anomaly_score"])
            
        risk_level = self._classify_with_streak(days_of_cover, 2 if days_of_cover < item["lead_time_days"] else 0, item["lead_time_days"])
        return {
            "item_id": item["sku_id"],
            "warehouse_location": item["warehouse_location"],
            "timestamp": item["timestamp"],
            "stockout_probability": stockout_probability,
            "risk_level": risk_level,
            "days_of_cover": days_of_cover,
            "linked_delay_risk": linked_delay_risk,
        }

    def restock_recommendation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        item = self._coerce_payload(payload)
        predicted_demand = round(max(item["daily_demand"] * 1.05, 0.1), 2)
        linked_delay_risk = self._linked_delay_risk(item["region"], item["category"], item["lead_time_days"], 0.0, self.step)
        days_of_cover = self._days_to_stockout(item["stock_units"], predicted_demand)
        stockout_probability = self._compute_final_risk(days_of_cover, item["lead_time_days"], item["anomaly_score"])
        urgency = self._classify_with_streak(days_of_cover, 2 if days_of_cover < item["lead_time_days"] else 0, item["lead_time_days"])
        target_cover = max(item["lead_time_days"] * (2.0 if item["region"] == "dubai" else 1.6), 6.0)
        recommended_qty = int(max(round(predicted_demand * target_cover - item["stock_units"]), 0))
        rationale = (
            f"{item['warehouse_location']} has {days_of_cover:.1f} days of cover while route-linked delay exposure is "
            f"{linked_delay_risk * 100:.0f}%."
        )
        return {
            "item_id": item["sku_id"],
            "warehouse_location": item["warehouse_location"],
            "timestamp": item["timestamp"],
            "predicted_demand": predicted_demand,
            "stockout_probability": stockout_probability,
            "recommended_restock_quantity": recommended_qty,
            "urgency": urgency,
            "suggested_route_priority": self._route_priority(urgency, linked_delay_risk),
            "rationale": rationale,
        }

    def summary(self) -> Dict[str, Any]:
        self._advance_step()
        prepared = pd.DataFrame(self.states.values()).copy()

        alerts = (
            prepared.sort_values(["risk_level", "final_risk", "linked_delay_risk"], ascending=[True, False, False])
            .query("risk_level in ['HIGH', 'CRITICAL']")
            .head(6)
            .apply(
                lambda row: {
                    "item_id": row["sku_id"],
                    "warehouse_location": row["warehouse_location"],
                    "stockout_probability": round(float(row["final_risk"]), 4),
                    "linked_delay_risk": round(float(row["linked_delay_risk"]), 4),
                    "inventory_level": int(round(float(row["stock_units"]))),
                    "product_name": row["product_name"],
                    "category": row["category"],
                    "stock_units": self._format_units(float(row["stock_units"]), str(row["unit_type"])),
                    "daily_demand": self._format_units(float(row["predicted_demand"]), str(row["unit_type"])),
                    "days_to_stockout": round(float(row["days_to_stockout"]), 1),
                    **self._shipment_status(row),
                    "risk": row["risk_level"],
                    "message": f"{row['product_name']} at {row['warehouse_location']}",
                },
                axis=1,
            )
            .tolist()
        )

        recommendations = (
            prepared.assign(
                recommended_restock_quantity=lambda df: (
                    (df["predicted_demand"] * (df["lead_time_days"] * 1.9).clip(lower=6.0) - df["stock_units"])
                    .round()
                    .clip(lower=0)
                    .astype(int)
                )
            )
            .sort_values(["recommended_restock_quantity", "final_risk"], ascending=False)
            .head(6)
            .apply(
                lambda row: {
                    "item_id": row["sku_id"],
                    "warehouse_location": row["warehouse_location"],
                    "product_name": row["product_name"],
                    "category": row["category"],
                    "stock_units": self._format_units(float(row["stock_units"]), str(row["unit_type"])),
                    "daily_demand": self._format_units(float(row["predicted_demand"]), str(row["unit_type"])),
                    "days_to_stockout": round(float(row["days_to_stockout"]), 1),
                    **self._shipment_status(row),
                    "recommended_restock_quantity": int(row["recommended_restock_quantity"]),
                    "urgency": row["risk_level"],
                    "suggested_route_priority": self._route_priority(row["risk_level"], float(row["linked_delay_risk"])),
                    "rationale": (
                        f"{row['product_name']} at {row['warehouse_location']} is running at "
                        f"{float(row['days_to_stockout']):.1f} days of cover versus {float(row['lead_time_days']):.1f}-day lead time."
                    ),
                },
                axis=1,
            )
            .tolist()
        )

        snapshot_frame = (
            prepared.groupby(["warehouse_location", "region"], as_index=False)
            .agg(
                avg_inventory_level=("stock_units", "mean"),
                avg_demand_forecast=("predicted_demand", "mean"),
                avg_days_to_stockout=("days_to_stockout", "mean"),
                avg_stockout_probability=("final_risk", "mean"),
                avg_linked_delay_risk=("linked_delay_risk", "mean"),
            )
            .round(2)
        )
        snapshots = pd.concat(
            [
                snapshot_frame.loc[snapshot_frame["region"] == "karnataka"]
                .sort_values("avg_stockout_probability", ascending=False)
                .head(5),
                snapshot_frame.loc[snapshot_frame["region"] == "dubai"]
                .sort_values("avg_stockout_probability", ascending=False)
                .head(5),
            ],
            ignore_index=True,
        ).drop(columns=["region"]).to_dict(orient="records")

        warehouse_count = int(prepared["warehouse_id"].nunique())
        urgent_count = int(prepared["risk_level"].isin(["HIGH", "CRITICAL"]).sum())
        avg_days = float(prepared["days_to_stockout"].mean())
        metrics = [
            {
                "title": "Tracked SKUs",
                "value": str(int(prepared["sku_id"].nunique())),
                "subtitle": f"{warehouse_count} warehouses across Karnataka and Dubai linked to active logistics routes",
            },
            {
                "title": "Avg Days of Cover",
                "value": f"{avg_days:.1f} days",
                "subtitle": "Coverage accounts for demand drawdown, replenishment latency, and route-linked supply pressure",
            },
            {
                "title": "Urgent Restocks",
                "value": str(urgent_count),
                "subtitle": "Items now in HIGH or sustained CRITICAL status after stable backend risk checks",
            },
            {
                "title": "Route Delay Exposure",
                "value": f"{prepared['linked_delay_risk'].mean() * 100:.1f}%",
                "subtitle": "Inbound delay signal weighted by maritime dependency for Dubai and dispatch pressure in Karnataka",
            },
        ]

        return {
            "metrics": metrics,
            "demand_trend": list(self.history),
            "stock_alerts": alerts,
            "restock_recommendations": recommendations,
            "warehouse_snapshots": snapshots,
        }


inventory_service = InventoryService()
