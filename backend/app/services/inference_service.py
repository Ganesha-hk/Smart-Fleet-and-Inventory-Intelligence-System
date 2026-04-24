import os
import sys
import joblib
import pandas as pd
import numpy as np
import logging
import random
import math
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List

# Fix for unpickling custom classes
import app.models.ml_models as ml_models
sys.modules['__main__'].AnomalyEnsemble = ml_models.AnomalyEnsemble
sys.modules['__main__'].AnomalyFeatureEngineer = ml_models.AnomalyFeatureEngineer
from ml_engine.pipelines.engineering import calculate_derived_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceService:
    _instance = None
    DATASET_FILENAME = "global_fleet_dataset_v1.csv"
    TICK_SECONDS = 2.5
    EVENT_COOLDOWN_SECONDS = 12.5
    ANOMALY_SPIKE_THRESHOLD = 0.2
    FATIGUE_SPIKE_THRESHOLD = 0.14
    FATIGUE_ALERT_THRESHOLD = 0.72
    EARTH_RADIUS_M = 6_371_000.0
    KARNATAKA_BOUNDS = {
        "lat": (11.5, 16.5),
        "lng": (74.0, 78.5),
    }
    DUBAI_BOUNDS = {
        "lat": (25.05, 25.40),
        "lng": (55.10, 55.45),
    }
    SEA_BOUNDS = {
        "lat": (12.0, 25.0),
        "lng": (55.0, 72.2),
    }
    GLOBAL_BOUNDS = {
        "lat": (11.5, 25.4),
        "lng": (55.0, 78.5),
    }
    CORRIDOR_PULL_THRESHOLD = 0.010
    CORRIDOR_SNAP_THRESHOLD = 0.032
    MAX_RISK_STEP_DELTA = 0.08
    DUBAI_CORRIDORS = [
        ((25.05, 55.12), (25.40, 55.20)),
        ((25.20, 55.00), (25.20, 55.50)),
        ((25.00, 55.10), (25.40, 55.45)),
        ((25.05, 55.05), (25.35, 55.15)),
        ((25.08, 55.28), (25.42, 55.36)),
        ((25.12, 55.42), (25.40, 55.48)),
    ]
    KARNATAKA_CORRIDORS = [
        ((12.9716, 77.5946), (12.9352, 77.6245)),
        ((12.9716, 77.5946), (13.3409, 74.7421)),
        ((12.9716, 77.5946), (15.3647, 75.1240)),
        ((12.9716, 77.5946), (15.8497, 74.4977)),
        ((12.2958, 76.6394), (14.2251, 76.3983)),
        ((12.2958, 76.6394), (12.9141, 74.8560)),
        ((15.3647, 75.1240), (15.3173, 75.7139)),
        ((15.3173, 75.7139), (16.8302, 75.7100)),
    ]
    SHIP_ROUTE_POLYLINES = [
        [
            (13.4, 72.0),
            (15.8, 68.8),
            (18.9, 65.4),
            (21.4, 61.0),
            (24.4, 56.2),
        ],
        [
            (12.7, 71.4),
            (15.1, 67.9),
            (17.8, 64.3),
            (20.7, 60.2),
            (24.0, 55.8),
        ],
        [
            (14.1, 71.7),
            (16.6, 68.2),
            (19.6, 64.9),
            (22.2, 60.9),
            (24.7, 56.0),
        ],
    ]
    SHIP_ROUTE_NAME_MAP = {
        "Karnataka-Oman-Dubai": SHIP_ROUTE_POLYLINES[1],
        "Karnataka-MidSea-Dubai": SHIP_ROUTE_POLYLINES[0],
        "Karnataka-SouthernCurve-Dubai": SHIP_ROUTE_POLYLINES[2],
    }
    
    # Dataset means for default filling
    MEANS = {
        "vehicle_gps_latitude": 25.2,
        "vehicle_gps_longitude": 55.3,
        "fuel_consumption_rate": 8.0117349835,
        "eta_variation_hours": 2.8930680101,
        "traffic_congestion_level": 4.9914932089,
        "warehouse_inventory_level": 299.2547323421,
        "loading_unloading_time": 2.2916691621,
        "handling_equipment_availability": 0.3026953974,
        "order_fulfillment_status": 0.6007395524,
        "weather_condition_severity": 0.4976081693,
        "port_congestion_level": 6.9784138098,
        "shipping_costs": 459.3744517975,
        "supplier_reliability_score": 0.5008498908,
        "lead_time_days": 5.2275020564,
        "historical_demand": 6022.0012856954,
        "iot_temperature": 0.0447916701,
        "cargo_condition_status": 0.297281603,
        "route_risk_level": 7.0011443574,
        "customs_clearance_time": 2.2964477943,
        "driver_behavior_score": 0.4983912731,
        "fatigue_monitoring_score": 0.6008722994,
        "disruption_likelihood_score": 0.803655567
    }

    # Feature sets
    DELAY_FEATURES = [
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
    
    ANOMALY_FEATURES = [
        'fuel_consumption_rate', 'eta_variation_hours', 'traffic_congestion_level', 
        'weather_condition_severity', 'iot_temperature', 'cargo_condition_status', 
        'route_risk_level', 'customs_clearance_time', 'driver_behavior_score', 
        'fatigue_monitoring_score', 'movement_distance_km', 'speed_kmph', 
        'behavior_risk_score', 'environmental_risk', 'traffic_fatigue_interaction'
    ]
    
    SURVIVAL_FEATURES = [
        'driver_risk_score', 'environmental_risk', 'logistics_risk', 
        'speed_kmph'
    ]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InferenceService, cls).__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self):
        if not self._loaded:
            self._load_models()
            self._load_dataset()
            self._fleet_state: Dict[str, Dict[str, Any]] = {}
            self._fleet_order: List[str] = []
            self._simulation_step = 0
            self._dashboard_history = deque(maxlen=6)
            self._event_feed = deque(maxlen=120)
            self.last_risk_level: Dict[str, str] = {}
            self.last_event_step: Dict[str, float] = {}
            self._loaded = True

    def _load_models(self):
        """Loads models once at startup with performance optimization."""
        import time
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml_engine", "artifacts"))
        
        try:
            start_time = time.time()
            logger.info("Loading ML models from artifacts (optimized with mmap)...")
            
            # Use mmap_mode='r' for faster loading of large models and reduced memory footprint
            self.delay_model = joblib.load(os.path.join(base_path, "delay_model.joblib"), mmap_mode='r')
            self.anomaly_model = joblib.load(os.path.join(base_path, "anomaly_model.joblib"), mmap_mode='r')
            self.cox_model = joblib.load(os.path.join(base_path, "cox_model.joblib"), mmap_mode='r')
            self.km_model = joblib.load(os.path.join(base_path, "kaplan_meier_model.joblib"), mmap_mode='r')
            
            # Load explainer if exists
            try:
                explainer_path = os.path.join(base_path, "delay_explainer.joblib")
                if os.path.exists(explainer_path):
                    self.explainer = joblib.load(explainer_path, mmap_mode='r')
                else:
                    self.explainer = None
            except Exception as e:
                logger.warning(f"Failed to load SHAP explainer: {e}")
                self.explainer = None
            
            load_time = time.time() - start_time
            logger.info(f"All models loaded successfully in {load_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def _load_dataset(self):
        """Loads the processed dataset for sampling."""
        data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "processed", self.DATASET_FILENAME)
        )
        try:
            logger.info(f"Loading dataset from {data_path}...")
            self.dataset = pd.read_csv(data_path)
            numeric_columns = [
                "lat",
                "lng",
                "base_risk",
                "traffic_factor",
                "anomaly_score",
                "driver_fatigue",
                "speed",
                "fuel_level",
            ]
            for column in numeric_columns:
                self.dataset[column] = pd.to_numeric(self.dataset[column], errors="coerce")
            self.dataset = self.dataset.dropna(subset=numeric_columns).reset_index(drop=True)
            logger.info("Dataset loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            self.dataset = None

    def _clamp(self, value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def _reflect_within_bounds(self, value: float, delta: float, lower: float, upper: float) -> Tuple[float, float]:
        next_value = value + delta
        next_delta = delta
        if next_value < lower or next_value > upper:
            next_delta = -delta
            next_value = self._clamp(value + next_delta, lower, upper)
        return next_value, next_delta

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        if math.isnan(numeric) or math.isinf(numeric):
            return default
        return numeric

    def _build_segments_from_points(
        self,
        points: List[Tuple[float, float]],
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        return list(zip(points[:-1], points[1:]))

    def _route_points_for_ship(self, row: Dict[str, Any]) -> List[Tuple[float, float]]:
        route_name = str(row.get("route_name", "Karnataka-MidSea-Dubai"))
        return self.SHIP_ROUTE_NAME_MAP.get(route_name, self.SHIP_ROUTE_POLYLINES[0])

    def _interpolate_polyline(self, points: List[Tuple[float, float]], progress: float) -> Tuple[float, float]:
        if len(points) == 1:
            return points[0]
        clamped_progress = self._clamp(progress, 0.0, 1.0)
        segments = len(points) - 1
        scaled = clamped_progress * segments
        index = min(int(scaled), segments - 1)
        local_t = scaled - index
        start_lat, start_lng = points[index]
        end_lat, end_lng = points[index + 1]
        return (
            start_lat + ((end_lat - start_lat) * local_t),
            start_lng + ((end_lng - start_lng) * local_t),
        )

    def _route_heading(self, points: List[Tuple[float, float]], progress: float) -> float:
        segments = max(len(points) - 1, 1)
        scaled = self._clamp(progress, 0.0, 0.9999) * segments
        index = min(int(scaled), segments - 1)
        start_lat, start_lng = points[index]
        end_lat, end_lng = points[index + 1]
        return math.atan2(end_lng - start_lng, end_lat - start_lat)

    def _corridors_for_region(
        self,
        entity_type: str,
        region: str,
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        if entity_type == "ship":
            segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
            for polyline in self.SHIP_ROUTE_POLYLINES:
                segments.extend(self._build_segments_from_points(polyline))
            return segments
        if region == "karnataka":
            return self.KARNATAKA_CORRIDORS
        return self.DUBAI_CORRIDORS

    def _bounds_for_entity(self, entity_type: str, region: str) -> Dict[str, Tuple[float, float]]:
        if entity_type == "ship":
            return self.SEA_BOUNDS
        if region == "karnataka":
            return self.KARNATAKA_BOUNDS
        return self.DUBAI_BOUNDS

    def _infer_entity_context(self, row: Dict[str, Any]) -> Dict[str, Any]:
        entity_type = str(row.get("type", "vehicle")).lower()
        lat = self._safe_float(row.get("lat"), self.MEANS["vehicle_gps_latitude"])
        lng = self._safe_float(row.get("lng"), self.MEANS["vehicle_gps_longitude"])
        if entity_type == "ship":
            region = "sea"
        elif self.KARNATAKA_BOUNDS["lat"][0] <= lat <= self.KARNATAKA_BOUNDS["lat"][1] and self.KARNATAKA_BOUNDS["lng"][0] <= lng <= self.KARNATAKA_BOUNDS["lng"][1]:
            region = "karnataka"
        else:
            region = "dubai"
        return {
            "entity_type": entity_type,
            "region": region,
            "bounds": self._bounds_for_entity(entity_type, region),
            "corridors": self._corridors_for_region(entity_type, region),
        }

    def _normalize_coordinate(
        self,
        lat: Any,
        lng: Any,
        bounds: Dict[str, Tuple[float, float]],
        corridors: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    ) -> Tuple[float, float]:
        lat_value = self._safe_float(lat, self.MEANS["vehicle_gps_latitude"])
        lng_value = self._safe_float(lng, self.MEANS["vehicle_gps_longitude"])
        if lat_value > 50 and lng_value < 30:
            lat_value, lng_value = lng_value, lat_value
        clamped_lat = self._clamp(lat_value, *bounds["lat"])
        clamped_lng = self._clamp(lng_value, *bounds["lng"])
        _, projected_point, corridor_distance, _ = self._nearest_corridor(clamped_lat, clamped_lng, corridors)
        if corridor_distance > self.CORRIDOR_SNAP_THRESHOLD:
            clamped_lat = (clamped_lat * 0.2) + (projected_point[0] * 0.8)
            clamped_lng = (clamped_lng * 0.2) + (projected_point[1] * 0.8)
        return (
            self._clamp(clamped_lat, *bounds["lat"]),
            self._clamp(clamped_lng, *bounds["lng"]),
        )

    def _project_to_segment(
        self,
        lat: float,
        lng: float,
        seg_start: Tuple[float, float],
        seg_end: Tuple[float, float],
    ) -> Tuple[float, float, float]:
        start_lat, start_lng = seg_start
        end_lat, end_lng = seg_end
        segment_lat = end_lat - start_lat
        segment_lng = end_lng - start_lng
        segment_length_sq = (segment_lat * segment_lat) + (segment_lng * segment_lng)
        if segment_length_sq == 0:
            return start_lat, start_lng, 0.0

        projection = (
            ((lat - start_lat) * segment_lat) +
            ((lng - start_lng) * segment_lng)
        ) / segment_length_sq
        clamped_projection = self._clamp(projection, 0.0, 1.0)
        return (
            start_lat + (segment_lat * clamped_projection),
            start_lng + (segment_lng * clamped_projection),
            clamped_projection,
        )

    def distance_to_segment(
        self,
        lat: float,
        lng: float,
        seg_start: Tuple[float, float],
        seg_end: Tuple[float, float],
    ) -> float:
        projected_lat, projected_lng, _ = self._project_to_segment(lat, lng, seg_start, seg_end)
        return math.hypot(lat - projected_lat, lng - projected_lng)

    def _nearest_corridor(
        self,
        lat: float,
        lng: float,
        corridors: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    ) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float]], Tuple[float, float], float, float]:
        best_corridor = corridors[0]
        best_projected = best_corridor[0]
        best_distance = float("inf")
        best_heading = 0.0

        for seg_start, seg_end in corridors:
            projected_lat, projected_lng, _ = self._project_to_segment(lat, lng, seg_start, seg_end)
            distance = math.hypot(lat - projected_lat, lng - projected_lng)
            if distance < best_distance:
                best_distance = distance
                best_corridor = (seg_start, seg_end)
                best_projected = (projected_lat, projected_lng)
                best_heading = math.atan2(seg_end[1] - seg_start[1], seg_end[0] - seg_start[0])

        return best_corridor, best_projected, best_distance, best_heading

    def _blend_heading(self, current_heading: float, target_heading: float, weight: float) -> float:
        angle_delta = math.atan2(
            math.sin(target_heading - current_heading),
            math.cos(target_heading - current_heading),
        )
        return (current_heading + (angle_delta * weight)) % math.tau

    def _random_movement_profile(self, base_speed: float) -> Tuple[str, float]:
        movement_roll = random.random()
        if movement_roll < 0.12:
            return "stopped", self._clamp(base_speed * random.uniform(0.0, 0.08), 0.0, 6.0)
        if movement_roll < 0.47:
            return "slow", self._clamp(base_speed * random.uniform(0.35, 0.65), 12.0, 38.0)
        return "fast", self._clamp(base_speed * random.uniform(0.78, 1.12), 36.0, 92.0)

    def _movement_profile_for_entity(self, entity_type: str, base_speed: float) -> Tuple[str, float]:
        if entity_type == "ship":
            movement_roll = random.random()
            if movement_roll < 0.16:
                return "slow", self._clamp(base_speed * random.uniform(0.58, 0.82), 10.0, 18.0)
            if movement_roll < 0.78:
                return "cruise", self._clamp(base_speed * random.uniform(0.86, 1.0), 18.0, 25.0)
            return "fast", self._clamp(base_speed * random.uniform(1.0, 1.12), 25.0, 40.0)
        return self._random_movement_profile(base_speed)

    def _movement_profile_for_vehicle_region(self, region: str, base_speed: float) -> Tuple[str, float]:
        if region == "karnataka":
            movement_roll = random.random()
            if movement_roll < 0.20:
                return "stopped", self._clamp(base_speed * random.uniform(0.0, 0.10), 0.0, 8.0)
            if movement_roll < 0.66:
                return "slow", self._clamp(base_speed * random.uniform(0.32, 0.58), 18.0, 42.0)
            return "fast", self._clamp(base_speed * random.uniform(0.72, 1.02), 32.0, 82.0)
        movement_roll = random.random()
        if movement_roll < 0.06:
            return "slow", self._clamp(base_speed * random.uniform(0.55, 0.72), 24.0, 48.0)
        return "fast", self._clamp(base_speed * random.uniform(0.86, 1.06), 42.0, 96.0)

    def _meters_to_lat_lng_delta(self, lat: float, heading: float, meters: float) -> Tuple[float, float]:
        lat_delta = (meters * math.cos(heading)) / 111_320.0
        lng_scale = max(math.cos(math.radians(lat)), 0.2)
        lng_delta = (meters * math.sin(heading)) / (111_320.0 * lng_scale)
        return lat_delta, lng_delta

    def _step_from_heading(self, lat: float, heading: float, speed_kmph: float, tick_seconds: float) -> Tuple[float, float]:
        meters = max(speed_kmph, 0.0) * 1000.0 / 3600.0 * tick_seconds * 0.2
        return self._meters_to_lat_lng_delta(lat, heading, meters)

    def _ship_step_from_knots(self, lat: float, heading: float, speed_knots: float, tick_seconds: float) -> Tuple[float, float]:
        meters = max(speed_knots, 0.0) * 0.514 * tick_seconds * 0.15
        return self._meters_to_lat_lng_delta(lat, heading, meters)

    def _lateral_drift_delta(self, lat: float, heading: float, meters: float) -> Tuple[float, float]:
        return self._meters_to_lat_lng_delta(lat, heading + (math.pi / 2.0), meters)

    def _vehicle_distance_km(self, state: Dict[str, Any]) -> float:
        regional_hubs = {
            "karnataka": (12.9716, 77.5946),
            "dubai": (25.2048, 55.2708),
        }
        target_lat, target_lng = regional_hubs.get(state["region"], (25.2048, 55.2708))
        return float(self.haversine_distance(state["lat"], state["lng"], target_lat, target_lng))

    def _ship_distance_nm(self, state: Dict[str, Any]) -> float:
        remaining_progress = max(0.0, 1.0 - float(state.get("route_progress", 0.0)))
        return round(remaining_progress * float(state.get("route_length_nm", 880.0)), 2)

    def _live_survival_curve(self, fleet: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not fleet:
            return []
        sorted_times = sorted(max(0.05, 1.0 / (float(vehicle.get("final_risk", 0.0)) + 0.01)) for vehicle in fleet)
        total = len(sorted_times)
        curve = [{"time": 0.0, "probability": 100.0}]
        for index, time_to_delay in enumerate(sorted_times):
            remaining = total - index
            curve.append(
                {
                    "time": round(float(time_to_delay), 2),
                    "probability": round((remaining / total) * 100.0, 1),
                }
            )
        if len(curve) > 40:
            step = max(1, len(curve) // 40)
            sampled = curve[::step]
            if sampled[-1]["time"] != curve[-1]["time"]:
                sampled.append(curve[-1])
            return sampled
        return curve

    def _speed_bounds_for_entity(self, entity_type: str) -> Tuple[float, float]:
        if entity_type == "ship":
            return (10.0, 25.0)
        return (30.0, 80.0)

    def _constrain_to_network(
        self,
        state: Dict[str, Any],
        next_lat: float,
        next_lng: float,
    ) -> Tuple[float, float, float]:
        bounds = state["bounds"]
        corridors = state["corridors"]
        if next_lat < bounds["lat"][0] or next_lat > bounds["lat"][1]:
            state["heading"] = (-state["heading"]) % math.tau
        if next_lng < bounds["lng"][0] or next_lng > bounds["lng"][1]:
            state["heading"] = (math.pi - state["heading"]) % math.tau

        bounded_lat = self._clamp(next_lat, *bounds["lat"])
        bounded_lng = self._clamp(next_lng, *bounds["lng"])
        was_out_of_bounds = (bounded_lat != next_lat) or (bounded_lng != next_lng)
        _, projected_point, corridor_distance, corridor_heading = self._nearest_corridor(bounded_lat, bounded_lng, corridors)

        bias_increment = 0.026 if state["entity_type"] == "ship" else 0.045
        max_bias = 0.30 if state["entity_type"] == "ship" else 0.42

        if corridor_distance > self.CORRIDOR_PULL_THRESHOLD:
            state["corridor_bias"] = self._clamp(state["corridor_bias"] + bias_increment, 0.10, max_bias)
        else:
            state["corridor_bias"] = self._clamp(state["corridor_bias"] * 0.985, 0.10, 0.32)

        if was_out_of_bounds or corridor_distance > self.CORRIDOR_SNAP_THRESHOLD:
            snap_strength = 0.68 if state["entity_type"] == "ship" else (0.78 if was_out_of_bounds else 0.46)
            bounded_lat += (projected_point[0] - bounded_lat) * snap_strength
            bounded_lng += (projected_point[1] - bounded_lng) * snap_strength
            state["heading"] = self._blend_heading(state["heading"], corridor_heading, 0.34)
        elif corridor_distance > self.CORRIDOR_PULL_THRESHOLD:
            stronger_pull = min(0.24 if state["entity_type"] == "ship" else 0.34, state["corridor_bias"] + 0.08)
            bounded_lat += (projected_point[0] - bounded_lat) * stronger_pull
            bounded_lng += (projected_point[1] - bounded_lng) * stronger_pull

        return (
            self._clamp(bounded_lat, *bounds["lat"]),
            self._clamp(bounded_lng, *bounds["lng"]),
            corridor_distance,
        )

    def _compute_base_risk(self, row: Dict[str, Any]) -> float:
        return self._clamp(self._safe_float(row.get("base_risk"), 0.35), 0.0, 1.0)

    def _row_to_model_input(self, row: Dict[str, Any]) -> Dict[str, Any]:
        base_risk = self._clamp(self._safe_float(row.get("base_risk"), 0.35), 0.0, 1.0)
        traffic_factor = self._clamp(self._safe_float(row.get("traffic_factor"), 0.5), 0.0, 1.0)
        anomaly_score = self._clamp(self._safe_float(row.get("anomaly_score"), 0.25), 0.0, 1.0)
        fatigue = self._clamp(self._safe_float(row.get("driver_fatigue"), 0.4), 0.0, 1.0)
        entity_type = str(row.get("type", "vehicle")).lower()
        min_speed, max_speed = self._speed_bounds_for_entity(entity_type)
        default_speed = 18.0 if entity_type == "ship" else 48.0
        speed = self._clamp(self._safe_float(row.get("speed"), default_speed), min_speed, max_speed)
        fuel_level = self._clamp(self._safe_float(row.get("fuel_level"), 65.0), 10.0, 100.0)
        driver_behavior = self._clamp(1.0 - ((base_risk * 0.35) + (fatigue * 0.45)), 0.05, 0.95)
        delay_probability = self._clamp(
            (base_risk * 0.50) +
            (traffic_factor * 0.30) +
            (anomaly_score * 0.20),
            0.02,
            0.98,
        )
        return {
            "vehicle_gps_latitude": self._safe_float(row.get("lat"), self.MEANS["vehicle_gps_latitude"]),
            "vehicle_gps_longitude": self._safe_float(row.get("lng"), self.MEANS["vehicle_gps_longitude"]),
            "traffic_congestion_level": round(traffic_factor * 10.0, 4),
            "driver_behavior_score": round(driver_behavior, 4),
            "fatigue_monitoring_score": round(fatigue, 4),
            "speed_kmph": round(speed, 2),
            "fuel_consumption_rate": round(4.0 + ((100.0 - fuel_level) * 0.06), 4),
            "eta_variation_hours": round(0.6 + (base_risk * 3.5), 4),
            "warehouse_inventory_level": 300.0,
            "loading_unloading_time": round(1.8 + (traffic_factor * 0.8), 4),
            "weather_condition_severity": round(0.2 + (anomaly_score * 0.35), 4),
            "route_risk_level": round(4.0 + (traffic_factor * 4.0), 4),
            "disruption_likelihood_score": round(anomaly_score, 4),
            "delay_probability": round(delay_probability, 4),
        }

    def _derive_risk_level(self, final_risk: float) -> str:
        if final_risk > 0.75:
            return "CRITICAL"
        if final_risk >= 0.6:
            return "HIGH"
        if final_risk >= 0.3:
            return "MID"
        return "LOW"

    def _update_vehicle_risk_state(
        self,
        state: Dict[str, Any],
        traffic_factor: float,
        anomaly_score: float,
    ) -> Dict[str, Any]:
        if state.get("last_risk_step") == self._simulation_step:
            return {
                "final_risk": state["final_risk"],
                "risk_level": state["risk_level"],
                "previous_risk_level": state.get("previous_risk_level", state["risk_level"]),
                "critical_streak": state["critical_streak"],
                "alert_active": state["alert_active"],
            }

        previous_risk_level = state.get("risk_level", "LOW")
        final_risk = self._clamp(
            (0.50 * state["base_risk"]) +
            (0.30 * traffic_factor) +
            (0.20 * anomaly_score),
            0.0,
            1.0,
        )
        previous_risk = state.get("previous_risk", final_risk)
        if abs(final_risk - previous_risk) < 0.05:
            final_risk = previous_risk
        else:
            risk_delta = final_risk - previous_risk
            capped_delta = self._clamp(risk_delta, -self.MAX_RISK_STEP_DELTA, self.MAX_RISK_STEP_DELTA)
            final_risk = self._clamp(previous_risk + capped_delta, 0.0, 1.0)

        dynamic_risk = self._clamp(
            (0.6 * traffic_factor) +
            (0.4 * anomaly_score),
            0.0,
            1.0,
        )
        risk_level = self._derive_risk_level(final_risk)
        critical_streak = (state.get("critical_streak", 0) + 1) if risk_level == "CRITICAL" else 0

        state["dynamic_risk"] = dynamic_risk
        state["final_risk"] = final_risk
        state["previous_risk"] = final_risk
        state["previous_risk_level"] = previous_risk_level
        state["risk_level"] = risk_level
        state["critical_streak"] = critical_streak
        state["alert_active"] = critical_streak >= 2
        state["last_risk_step"] = self._simulation_step

        return {
            "final_risk": final_risk,
            "risk_level": risk_level,
            "previous_risk_level": previous_risk_level,
            "critical_streak": critical_streak,
            "alert_active": state["alert_active"],
        }

    def _ensure_fleet_state(self) -> None:
        if self.dataset is None or self.dataset.empty:
            self._fleet_state = {}
            self._fleet_order = []
            return

        if len(self._fleet_order) == len(self.dataset):
            return

        self._fleet_state = {}
        self._fleet_order = []
        self._event_feed = deque(maxlen=120)
        self.last_risk_level = {}
        self.last_event_step = {}
        for dataset_index in range(len(self.dataset)):
            row = self.dataset.iloc[dataset_index].to_dict()
            context = self._infer_entity_context(row)
            lat, lng = self._normalize_coordinate(
                row.get("lat"),
                row.get("lng"),
                context["bounds"],
                context["corridors"],
            )
            min_speed, max_speed = self._speed_bounds_for_entity(context["entity_type"])
            default_speed = 18.0 if context["entity_type"] == "ship" else 52.0
            base_speed = self._clamp(self._safe_float(row.get("speed"), default_speed), min_speed, max_speed)
            phase = random.uniform(0, math.tau)
            heading = random.uniform(0.0, math.tau)
            if context["entity_type"] == "ship":
                movement_state, cruising_speed = self._movement_profile_for_entity(context["entity_type"], base_speed)
            else:
                movement_state, cruising_speed = self._movement_profile_for_vehicle_region(context["region"], base_speed)
            lat_step, lng_step = self._step_from_heading(lat, heading, cruising_speed, self.TICK_SECONDS)
            vehicle_id = str(row.get("vehicle_id", f"V-{dataset_index + 1:05d}"))
            base_risk = self._compute_base_risk(row)
            initial_traffic = self._clamp(self._safe_float(row.get("traffic_factor"), 0.5), 0.0, 1.0)
            initial_anomaly = self._clamp(self._safe_float(row.get("anomaly_score"), 0.2), 0.0, 1.0)
            ship_route_points = self._route_points_for_ship(row) if context["entity_type"] == "ship" else []
            route_progress = self._clamp(self._safe_float(row.get("route_progress"), 0.0), 0.0, 1.0)
            route_heading = self._route_heading(ship_route_points, route_progress) if ship_route_points else heading
            convoy_spacing_nm = self._safe_float(row.get("convoy_spacing_nm"), 1.4)
            initial_final_risk = self._clamp(
                (0.50 * base_risk) +
                (0.30 * initial_traffic) +
                (0.20 * initial_anomaly),
                0.0,
                1.0,
            )
            initial_dynamic_risk = self._clamp(
                (0.60 * initial_traffic) +
                (0.40 * initial_anomaly),
                0.0,
                1.0,
            )
            initial_risk_level = self._derive_risk_level(initial_final_risk)
            initial_fatigue = self._clamp(self._safe_float(row.get("driver_fatigue"), 0.4), 0.0, 1.0)
            self._fleet_state[vehicle_id] = {
                "id": vehicle_id,
                "vehicle_id": vehicle_id,
                "dataset_index": dataset_index,
                "row": row,
                "entity_type": context["entity_type"],
                "region": context["region"],
                "route_name": str(row.get("route_name", "Karnataka-MidSea-Dubai")) if context["entity_type"] == "ship" else "",
                "convoy_id": str(row.get("convoy_id", "")) if context["entity_type"] == "ship" else "",
                "convoy_size": int(self._safe_float(row.get("convoy_size"), 1)) if context["entity_type"] == "ship" else 1,
                "convoy_rank": int(self._safe_float(row.get("convoy_rank"), 1)) if context["entity_type"] == "ship" else 1,
                "convoy_spacing_nm": convoy_spacing_nm,
                "sea_condition": str(row.get("sea_condition", "moderate")) if context["entity_type"] == "ship" else "",
                "sea_congestion": initial_traffic if context["entity_type"] == "ship" else 0.0,
                "bounds": context["bounds"],
                "corridors": context["corridors"],
                "lat": lat,
                "lng": lng,
                "lat_step": lat_step,
                "lng_step": lng_step,
                "phase": phase,
                "heading": route_heading if context["entity_type"] == "ship" else heading,
                "movement_state": movement_state,
                "cruise_speed": cruising_speed,
                "speed": cruising_speed,
                "route_progress": route_progress,
                "route_direction": 1,
                "route_length_nm": 880.0 + (convoy_spacing_nm * 4.5) if context["entity_type"] == "ship" else 0.0,
                "route_points": ship_route_points,
                "last_arrival_step": -999,
                "corridor_bias": random.uniform(0.10, 0.22),
                "heading_alignment": random.uniform(0.12, 0.24),
                "corridor_noise": random.uniform(0.000006, 0.00002),
                "curve_phase": random.uniform(0.0, math.tau),
                "curve_scale": random.uniform(0.012, 0.040) if context["entity_type"] == "ship" else random.uniform(0.032, 0.085),
                "drift_phase": random.uniform(0.0, math.tau),
                "drift_scale_m": random.uniform(35.0, 110.0) if context["entity_type"] == "ship" else random.uniform(12.0, 40.0),
                "battery": self._clamp(self._safe_float(row.get("fuel_level"), 72.0), 10.0, 100.0),
                "base_risk": base_risk,
                "dynamic_risk": initial_dynamic_risk,
                "final_risk": initial_final_risk,
                "previous_risk": initial_final_risk,
                "risk_level": initial_risk_level,
                "previous_risk_level": initial_risk_level,
                "critical_streak": 1 if initial_risk_level == "CRITICAL" else 0,
                "alert_active": False,
                "last_risk_step": None,
                "previous_anomaly_score": initial_anomaly,
                "previous_driver_fatigue": initial_fatigue,
            }
            self.last_risk_level[vehicle_id] = initial_risk_level
            self.last_event_step[vehicle_id] = -9999.0
            self._fleet_order.append(vehicle_id)

    def _emit_vehicle_event(
        self,
        state: Dict[str, Any],
        risk_state: Dict[str, Any],
        anomaly_score: float,
        fatigue: float,
        reasons: List[str],
    ) -> None:
        vehicle_id = state["vehicle_id"]
        previous_level = self.last_risk_level.get(vehicle_id, risk_state.get("previous_risk_level", risk_state["risk_level"]))
        anomaly_spike = anomaly_score - float(state.get("previous_anomaly_score", anomaly_score))
        fatigue_spike = fatigue - float(state.get("previous_driver_fatigue", fatigue))
        fatigue_trigger = fatigue >= self.FATIGUE_ALERT_THRESHOLD and fatigue_spike >= self.FATIGUE_SPIKE_THRESHOLD
        risk_changed = risk_state["risk_level"] != previous_level
        anomaly_trigger = anomaly_spike > self.ANOMALY_SPIKE_THRESHOLD
        event_due = risk_changed or anomaly_trigger or fatigue_trigger
        current_step_seconds = self._simulation_step * self.TICK_SECONDS
        cooldown_ready = (current_step_seconds - self.last_event_step.get(vehicle_id, -9999.0)) >= self.EVENT_COOLDOWN_SECONDS

        if event_due and cooldown_ready:
            if risk_changed:
                event_type = "risk_change"
                title = f"{vehicle_id} changed from {previous_level} to {risk_state['risk_level']}"
                message = f"Risk level escalated to {risk_state['risk_level']}"
            elif anomaly_trigger:
                event_type = "anomaly_spike"
                title = f"{vehicle_id} anomaly spike detected"
                message = f"Anomaly score spiked by {anomaly_spike:.2f}"
            else:
                event_type = "fatigue_spike"
                title = f"{vehicle_id} driver fatigue spike detected"
                message = f"Driver fatigue increased to {fatigue:.2f}"

            self._event_feed.appendleft(
                {
                    "id": f"{vehicle_id}:{self._simulation_step}:{event_type}",
                    "vehicle_id": vehicle_id,
                    "vehicleId": vehicle_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "type": event_type,
                    "title": title,
                    "message": message,
                    "risk_level": risk_state["risk_level"],
                    "previous_risk_level": previous_level,
                    "anomaly_score": round(anomaly_score, 4),
                    "driver_fatigue": round(fatigue, 4),
                    "final_risk": round(risk_state["final_risk"], 4),
                    "reasons": reasons[:3],
                    "solution": state.get("solution", "Maintain standard monitoring."),
                }
            )
            self.last_event_step[vehicle_id] = current_step_seconds

        self.last_risk_level[vehicle_id] = risk_state["risk_level"]
        state["previous_anomaly_score"] = anomaly_score
        state["previous_driver_fatigue"] = fatigue

    def _advance_fleet_state(self) -> None:
        self._simulation_step += 1
        for vehicle_id in self._fleet_order:
            state = self._fleet_state[vehicle_id]
            previous_lat = state["lat"]
            previous_lng = state["lng"]
            if random.random() < 0.012:
                default_speed = 18.0 if state["entity_type"] == "ship" else state["cruise_speed"]
                if state["entity_type"] == "ship":
                    state["movement_state"], state["cruise_speed"] = self._movement_profile_for_entity(
                        state["entity_type"],
                        self._safe_float(state["row"].get("speed"), default_speed),
                    )
                else:
                    state["movement_state"], state["cruise_speed"] = self._movement_profile_for_vehicle_region(
                        state["region"],
                        self._safe_float(state["row"].get("speed"), default_speed),
                    )

            if state["entity_type"] == "ship":
                sea_condition = state["sea_condition"]
                sea_condition_drag = {
                    "calm": 0.04,
                    "moderate": 0.10,
                    "rough": 0.18,
                }.get(sea_condition, 0.10)
                base_congestion = self._clamp(self._safe_float(state["row"].get("sea_congestion"), state["sea_congestion"]), 0.05, 0.95)
                sea_congestion = self._clamp(
                    base_congestion +
                    math.sin((self._simulation_step / 5.6) + state["phase"]) * 0.08 +
                    (0.04 if sea_condition == "rough" else 0.0),
                    0.05,
                    0.95,
                )
                state["sea_congestion"] = sea_congestion
                min_speed, max_speed = self._speed_bounds_for_entity(state["entity_type"])
                target_speed = self._clamp(
                    state["cruise_speed"] * (1.0 - (sea_congestion * 0.32) - sea_condition_drag),
                    min_speed,
                    max_speed,
                )
                state["speed"] += (target_speed - state["speed"]) * 0.08
                state["speed"] = self._clamp(state["speed"], min_speed, max_speed)

                progress_step = max(
                    0.00004,
                    (((state["speed"] * 0.514 * self.TICK_SECONDS) * 0.15) / 1852.0) / max(state["route_length_nm"], 1.0),
                )
                next_progress = state["route_progress"] + (progress_step * state["route_direction"])
                if next_progress >= 1.0:
                    next_progress = 1.0
                    state["route_direction"] = -1
                    state["last_arrival_step"] = self._simulation_step
                elif next_progress <= 0.0:
                    next_progress = 0.0
                    state["route_direction"] = 1
                state["route_progress"] = next_progress

                state["heading"] = self._route_heading(state["route_points"], state["route_progress"])
                base_lat, base_lng = self._interpolate_polyline(state["route_points"], state["route_progress"])
                convoy_offset = (state["convoy_rank"] - ((state["convoy_size"] + 1) / 2.0))
                spacing_scale = state["convoy_spacing_nm"] * 0.0022
                perp_heading = state["heading"] + (math.pi / 2.0)
                lat_offset = math.cos(perp_heading) * convoy_offset * spacing_scale
                lng_offset = math.sin(perp_heading) * convoy_offset * spacing_scale
                curvature = 0.010 + (state["convoy_size"] * 0.0008)
                progress_curve = state["route_progress"] * math.pi
                curve_lat = math.sin(progress_curve) * curvature
                curve_lng = math.cos(progress_curve) * curvature * 0.55
                wave_offset = math.sin((self._simulation_step / 6.4) + state["drift_phase"]) * 0.0011
                swell_offset = math.cos((self._simulation_step / 9.2) + state["curve_phase"]) * 0.00045
                wave_lat = math.cos(perp_heading) * (wave_offset + swell_offset)
                wave_lng = math.sin(perp_heading) * ((wave_offset * 1.15) + (swell_offset * 0.6))
                jitter_lat, jitter_lng = self._ship_step_from_knots(
                    base_lat,
                    perp_heading,
                    random.uniform(-0.12, 0.12),
                    self.TICK_SECONDS,
                )
                state["lat"] = self._clamp(base_lat + curve_lat + lat_offset + wave_lat + jitter_lat, *state["bounds"]["lat"])
                state["lng"] = self._clamp(base_lng + curve_lng + lng_offset + wave_lng + jitter_lng, *state["bounds"]["lng"])
                state["lat_step"] = state["lat"] - previous_lat
                state["lng_step"] = state["lng"] - previous_lng
                state["battery"] = self._clamp(state["battery"] - random.uniform(0.03, 0.09), 18.0, 100.0)
                continue

            _, projected_point, corridor_distance, corridor_heading = self._nearest_corridor(
                state["lat"],
                state["lng"],
                state["corridors"],
            )
            heading_weight = 0.55 if state["entity_type"] == "ship" else state["heading_alignment"]
            state["heading"] = self._blend_heading(
                state["heading"],
                corridor_heading,
                heading_weight * 0.72,
            )
            if state["entity_type"] == "ship":
                heading_noise = random.uniform(-0.012, 0.012)
            elif state["region"] == "karnataka":
                heading_noise = random.uniform(-0.05, 0.05)
            else:
                heading_noise = random.uniform(-0.026, 0.026)
            if corridor_distance < 0.025:
                heading_noise *= 0.55 if state["entity_type"] == "ship" else 0.72
            curvature_heading = math.sin((self._simulation_step / (10.5 if state["region"] == "karnataka" else 13.5)) + state["curve_phase"]) * state["curve_scale"]
            state["heading"] = (state["heading"] + heading_noise + curvature_heading) % math.tau
            min_speed, max_speed = self._speed_bounds_for_entity(state["entity_type"])
            if state["entity_type"] == "ship":
                speed_wave_scale = 1.2
            elif state["region"] == "karnataka":
                speed_wave_scale = 4.4
            else:
                speed_wave_scale = 2.2
            speed_wave = math.sin((self._simulation_step / 3.8) + state["phase"]) * speed_wave_scale
            target_speed = self._clamp(state["cruise_speed"] + speed_wave, min_speed, max_speed)
            if state["region"] == "karnataka" and random.random() < 0.08:
                target_speed = self._clamp(target_speed * random.uniform(0.35, 0.70), min_speed, max_speed)
            elif state["region"] == "dubai" and random.random() < 0.04:
                target_speed = self._clamp(target_speed * random.uniform(0.72, 0.88), min_speed, max_speed)
            state["speed"] += (target_speed - state["speed"]) * 0.22
            state["speed"] = self._clamp(state["speed"], min_speed, max_speed)

            lat_delta, lng_delta = self._step_from_heading(state["lat"], state["heading"], state["speed"], self.TICK_SECONDS)
            bias_strength = state["corridor_bias"]
            bias_lat = self._clamp(
                (projected_point[0] - state["lat"]) * bias_strength,
                -0.00006,
                0.00006,
            )
            bias_lng = self._clamp(
                (projected_point[1] - state["lng"]) * bias_strength,
                -0.00006,
                0.00006,
            )
            drift_meters = math.sin((self._simulation_step / (8.0 if state["region"] == "karnataka" else 10.5)) + state["drift_phase"]) * state["drift_scale_m"]
            drift_lat, drift_lng = self._lateral_drift_delta(state["lat"], corridor_heading, drift_meters)
            noise_scale = state["corridor_noise"] * (1.1 if state["region"] == "karnataka" else 0.8)
            lat_delta += bias_lat + (drift_lat * 0.55) + random.uniform(-0.00018, 0.00018) + random.uniform(-noise_scale, noise_scale)
            lng_delta += bias_lng + (drift_lng * 0.55) + random.uniform(-0.00018, 0.00018) + random.uniform(-noise_scale, noise_scale)
            next_lat = state["lat"] + lat_delta
            next_lng = state["lng"] + lng_delta

            state["lat"], state["lng"], corridor_distance = self._constrain_to_network(state, next_lat, next_lng)
            if corridor_distance > self.CORRIDOR_SNAP_THRESHOLD:
                state["heading"] = self._blend_heading(state["heading"], corridor_heading, 0.22)

            state["lat_step"] = state["lat"] - previous_lat
            state["lng_step"] = state["lng"] - previous_lng
            state["battery"] = self._clamp(state["battery"] - random.uniform(0.05, 0.18), 18.0, 100.0)

    def _build_vehicle_payload(self, state: Dict[str, Any]) -> Dict[str, Any]:
        row = state["row"]
        is_ship = state["entity_type"] == "ship"
        region = state["region"]
        traffic_scale = 10.0
        if is_ship:
            sea_congestion = self._clamp(float(state.get("sea_congestion", self._safe_float(row.get("sea_congestion"), 0.22))), 0.05, 0.95)
            traffic = round(sea_congestion * 10.0, 1)
        else:
            baseline_traffic = self._clamp(self._safe_float(row.get("traffic_factor"), 0.5) * traffic_scale, 0.0, 10.0)
            traffic = self._clamp(
                baseline_traffic +
                math.sin((self._simulation_step / (2.1 if region == "karnataka" else 3.2)) + state["phase"]) *
                (1.15 if region == "karnataka" else 0.68),
                0.0,
                10.0,
            )
        baseline_fatigue = self._clamp(self._safe_float(row.get("driver_fatigue"), 0.45), 0.0, 1.0)
        fatigue = self._clamp(
            baseline_fatigue +
            math.cos((self._simulation_step / (12.0 if is_ship else (4.5 if region == "karnataka" else 6.5))) + state["phase"]) *
            (0.02 if is_ship else (0.10 if region == "karnataka" else 0.05)) +
            ((traffic / 10.0) * (0.04 if region == "karnataka" else 0.02)),
            0.0,
            1.0,
        )
        base_delay = self._clamp(
            (state["base_risk"] * 0.50) +
            (self._safe_float(row.get("traffic_factor"), 0.5) * 0.30) +
            (self._safe_float(row.get("anomaly_score"), 0.25) * 0.20),
            0.02,
            0.98,
        )
        delay_probability = self._clamp(
            base_delay +
            ((traffic - 5.0) * (0.05 if not is_ship else 0.03)) +
            ((fatigue - 0.5) * (0.26 if not is_ship else 0.14)),
            0.02,
            0.98,
        )
        base_anomaly_seed = self._safe_float(row.get("anomaly_score"), 0.25)
        anomaly_baseline = (
            (base_anomaly_seed * (0.78 if is_ship else 0.84)) +
            (0.04 if is_ship else (0.08 if region == "karnataka" else 0.05))
        )
        anomaly_score = self._clamp(
            anomaly_baseline +
            (math.sin((self._simulation_step / (9.0 if is_ship else (5.5 if region == "karnataka" else 7.5))) + state["phase"]) *
             (0.015 if is_ship else (0.04 if region == "karnataka" else 0.025))) +
            ((traffic / 10.0) * (0.010 if is_ship else (0.04 if region == "karnataka" else 0.02))),
            0.0,
            0.72 if is_ship else 0.92,
        )
        baseline_traffic_factor = self._clamp(self._safe_float(row.get("traffic_factor"), 0.5), 0.0, 1.0)
        traffic_factor = self._clamp((baseline_traffic_factor * 0.72) + ((traffic / 10.0) * 0.28), 0.0, 1.0)
        anomaly_score = self._clamp((base_anomaly_seed * 0.72) + (anomaly_score * 0.28), 0.0, 1.0)
        risk_state = self._update_vehicle_risk_state(state, traffic_factor, anomaly_score)
        anomaly_flag = anomaly_score >= 0.72
        driver_score = self._clamp(
            (1.0 - ((state["base_risk"] * 0.35) + (fatigue * 0.45))) * 100.0,
            0.0,
            100.0,
        )
        irregular_speed = state["speed"] >= (34.0 if is_ship else 92.0) or state["speed"] <= (12.0 if is_ship else 18.0)
        traffic_penalty = max(0.0, (traffic_factor - 0.5) * 0.3)
        anomaly_penalty = anomaly_score * 0.2
        distance_to_destination = self._ship_distance_nm(state) if is_ship else round(self._vehicle_distance_km(state), 2)
        time_to_delay = max(0.05, 1.0 / (risk_state["final_risk"] + 0.01))
        reasons = []
        if delay_probability >= 0.7 or traffic >= 7.0:
            reasons.append(f"High {'sea congestion' if is_ship else 'traffic congestion'} (+{round(traffic_penalty * 100)}% delay)")
        if anomaly_score >= 0.7:
            reasons.append(f"{'Route' if is_ship else 'Anomaly'} detected (score {anomaly_score:.2f})")
        if driver_score <= 45.0 or fatigue >= 0.72:
            reasons.append(f"{'Crew strain' if is_ship else 'Driver fatigue'} detected (score {fatigue:.2f})")
        if irregular_speed:
            reasons.append("Irregular speed pattern")
        if not reasons:
            reasons.append("Operational parameters within expected range")

        if risk_state["risk_level"] == "CRITICAL":
            explanation = (
                f"{'Ship' if is_ship else 'Vehicle'} {state['vehicle_id']} is critical at {risk_state['final_risk'] * 100:.0f}% "
                "delay risk due to sustained traffic and anomaly pressure."
            )
        elif risk_state["risk_level"] in {"HIGH", "MID"}:
            explanation = (
                f"{'Ship' if is_ship else 'Vehicle'} {state['vehicle_id']} is elevated at {risk_state['final_risk'] * 100:.0f}% "
                "delay risk with moderate operational pressure."
            )
        else:
            explanation = (
                f"{'Ship' if is_ship else 'Vehicle'} {state['vehicle_id']} remains stable at {risk_state['final_risk'] * 100:.0f}% "
                "delay risk."
            )

        if risk_state["risk_level"] == "CRITICAL" or anomaly_flag:
            status = "critical"
        elif risk_state["risk_level"] in {"HIGH", "MID"}:
            status = "warning"
        else:
            status = "normal"

        # Survival Solution calculation
        try:
            # Prepare individual features for survival expectation
            behavior_raw = 1.0 - ((state["base_risk"] * 0.35) + (fatigue * 0.45))
            behavior_norm = self._clamp((0.9 - behavior_raw) / 0.6, 0.0, 1.0)
            fatigue_norm = self._clamp((fatigue - 0.1) / 0.9, 0.0, 1.0)
            
            X_survival = pd.DataFrame([{
                'driver_risk_score': (0.6 * behavior_norm + 0.4 * fatigue_norm),
                'environmental_risk': (self._safe_float(row.get("weather_condition_severity"), 0.5) + self._safe_float(row.get("route_risk_level"), 7.0)) / 20.0,
                'logistics_risk': self._safe_float(row.get("customs_clearance_time"), 2.3) + self._safe_float(row.get("port_congestion_level"), 7.0) + (traffic_factor * 10.0),
                'speed_kmph': state["speed"]
            }])
            survival_time = float(self.cox_model.predict_expectation(X_survival).iloc[0])
        except Exception:
            survival_time = time_to_delay

        solution = self._get_survival_solution(survival_time, risk_state["risk_level"])
        state["solution"] = solution

        self._emit_vehicle_event(state, risk_state, anomaly_score, fatigue, reasons)

        return {
            "id": state["id"],
            "vehicle_id": state["vehicle_id"],
            "type": state["entity_type"],
            "region": state["region"],
            "lat": round(state["lat"], 6),
            "lng": round(state["lng"], 6),
            "lat_step": round(state["lat_step"], 6),
            "lng_step": round(state["lng_step"], 6),
            "speed": round(state["speed"], 1),
            "battery": round(state["battery"], 1),
            "driverScore": round(driver_score, 1),
            "driver_score": round(driver_score, 1),
            "delay_probability": round(delay_probability, 4),
            "delayRisk": round(risk_state["final_risk"] * 100.0, 1),
            "delay_risk": round(risk_state["final_risk"], 4),
            "anomaly_flag": anomaly_flag,
            "anomaly_score": round(anomaly_score, 4),
            "traffic_factor": round(traffic_factor, 4),
            "sea_congestion": round(float(state.get("sea_congestion", traffic_factor)), 4) if is_ship else None,
            "driver_fatigue": round(fatigue, 4),
            "estimated_time_to_delay": round(survival_time, 2),
            "base_risk": round(state["base_risk"], 4),
            "dynamic_risk": round(state["dynamic_risk"], 4),
            "final_risk": round(risk_state["final_risk"], 4),
            "risk_level": risk_state["risk_level"],
            "alert_active": risk_state["alert_active"],
            "critical_streak": risk_state["critical_streak"],
            "status": status,
            "trafficLevel": round(traffic, 1),
            "traffic_level": round(traffic, 1),
            "fatigueLevel": round(fatigue, 2),
            "fatigue_level": round(fatigue, 2),
            "etaDeviation": round((delay_probability - state["base_risk"]) * 10.0, 1),
            "route_name": state["route_name"] if is_ship else None,
            "convoy_size": state["convoy_size"] if is_ship else None,
            "sea_condition": state["sea_condition"] if is_ship else None,
            "distance_to_destination": distance_to_destination,
            "explanation": explanation,
            "reasons": reasons,
            "solution": solution,
            "confidence_score": 0.92,
        }

    def _get_survival_solution(self, tte: float, risk_level: str) -> str:
        """Generates a predictive solution based on survival model expectations."""
        if risk_level == "CRITICAL" or tte < 2.5:
            return "Immediate Intervention Required: Reroute to secondary logistics hub and initiate driver swap protocol."
        if risk_level == "HIGH" or tte < 6.0:
            return "Preventive Action: Schedule 45-minute mandatory cooling break and optimize remaining route for traffic avoidance."
        if risk_level == "MID" or tte < 15.0:
            return "Enhanced Monitoring: Verify cargo stability sensors and monitor driver biometric fatigue every 15 minutes."
        return "Nominal Operations: Continue with standard scheduled safety checkpoints and fuel efficiency cruising."

    def get_recent_events(self, limit: int = 30) -> List[Dict[str, Any]]:
        return list(self._event_feed)[:limit]

    def _get_fleet_snapshot(self, size: int | None = None) -> List[Dict[str, Any]]:
        vehicles = [self._build_vehicle_payload(self._fleet_state[vehicle_id]) for vehicle_id in self._fleet_order]
        if size is None or size <= 0:
            return vehicles
        return vehicles[: min(size, len(vehicles))]

    def advance_full_fleet(self) -> None:
        self._ensure_fleet_state()
        self._advance_fleet_state()

    def get_random_sample(self) -> Dict[str, Any]:
        """Returns a random row from the dataset as a dictionary."""
        if self.dataset is not None and not self.dataset.empty:
            sample = self.dataset.sample(n=1).iloc[0].to_dict()
            return self._row_to_model_input(sample)
        return {}

    def get_sample_batch(self, size: int | None = None) -> Dict[str, Any]:
        """Updates the full fleet, then returns either the full fleet or a stable slice."""
        if self.dataset is None or self.dataset.empty:
            return {
                "totalCount": 0,
                "vehicles": [],
                "events": [],
            }

        self.advance_full_fleet()
        batch_results = self._get_fleet_snapshot(size)
        print("DATASET SIZE:", len(batch_results if size else self.dataset))

        return {
            "totalCount": int(len(self.dataset)),
            "vehicles": batch_results,
            "events": self.get_recent_events(),
        }

    def get_maritime_supply_signal(self) -> Dict[str, Any]:
        self._ensure_fleet_state()
        fleet = self._get_fleet_snapshot()
        ships = [vehicle for vehicle in fleet if vehicle.get("type") == "ship"]
        delayed_ships = [
            ship for ship in ships
            if float(ship.get("final_risk", 0.0)) >= 0.6 or float(ship.get("sea_congestion") or 0.0) >= 0.58
        ]
        arrived_ships = [
            state for state in self._fleet_state.values()
            if state["entity_type"] == "ship" and (self._simulation_step - int(state.get("last_arrival_step", -999))) <= 3
        ]
        return {
            "ships_total": len(ships),
            "delayed_ships": len(delayed_ships),
            "arrived_ships": len(arrived_ships),
            "avg_sea_congestion": round(
                sum(float(ship.get("sea_congestion") or 0.0) for ship in ships) / max(len(ships), 1),
                4,
            ),
        }

    def _correlation(self, values: List[float], targets: List[float]) -> float:
        if len(values) < 2 or len(targets) < 2:
            return 0.0
        values_array = np.asarray(values, dtype=float)
        targets_array = np.asarray(targets, dtype=float)
        if np.std(values_array) < 1e-9 or np.std(targets_array) < 1e-9:
            return 0.0
        return float(np.corrcoef(values_array, targets_array)[0, 1])

    def _get_feature_importance(self, fleet: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        feature_map = {
            "Traffic Factor": [float(vehicle.get("traffic_factor", 0.0)) for vehicle in fleet],
            "Anomaly Score": [float(vehicle.get("anomaly_score", 0.0)) for vehicle in fleet],
            "Driver Fatigue": [float(vehicle.get("driver_fatigue", 0.0)) for vehicle in fleet],
            "Speed": [float(vehicle.get("speed", 0.0)) for vehicle in fleet],
        }
        targets = [float(vehicle.get("final_risk", 0.0)) for vehicle in fleet]
        feature_importance = [
            {
                "name": name,
                "score": round(abs(self._correlation(values, targets)), 4),
            }
            for name, values in feature_map.items()
        ]
        return sorted(feature_importance, key=lambda item: item["score"], reverse=True)

    def _build_delay_trend(self, fleet: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        avg_delay_risk = sum(vehicle["final_risk"] for vehicle in fleet) / max(len(fleet), 1)
        self._dashboard_history.append(
            {
                "name": f"T+{self._simulation_step}",
                "value": round(avg_delay_risk * 100.0, 1),
            }
        )
        if len(self._dashboard_history) < 6:
            baseline = avg_delay_risk * 100.0
            seed = []
            for index in range(6 - len(self._dashboard_history)):
                seed.append(
                    {
                        "name": f"T-{6 - len(self._dashboard_history) - index}",
                        "value": round(self._clamp(baseline * (0.96 + (index * 0.01)), 0.0, 100.0), 1),
                    }
                )
            return seed + list(self._dashboard_history)
        return list(self._dashboard_history)

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Calculates global metrics using the entire dataset."""
        if self.dataset is None or self.dataset.empty:
            return {}

        self._ensure_fleet_state()
        fleet = self._get_fleet_snapshot()
        total_vehicles = len(fleet)
        if total_vehicles == 0:
            return {}
        avg_delay_risk = sum(vehicle['final_risk'] for vehicle in fleet) / total_vehicles
        low_risk = sum(1 for vehicle in fleet if vehicle['risk_level'] == "LOW")
        mid_risk = sum(1 for vehicle in fleet if vehicle['risk_level'] == "MID")
        high_risk = sum(1 for vehicle in fleet if vehicle['risk_level'] == "HIGH")
        critical_risk = sum(1 for vehicle in fleet if vehicle['risk_level'] == "CRITICAL")
        anomaly_count = sum(1 for vehicle in fleet if vehicle['anomaly_flag'])
        avg_ttd = sum(vehicle['estimated_time_to_delay'] for vehicle in fleet[: min(50, total_vehicles)]) / min(50, total_vehicles)
        fleet_efficiency = round((1.0 - avg_delay_risk) * 100, 1)
        mitigation_confidence = 92.4
        failures_averted = int(max(1, anomaly_count) * 0.42)
        
        return {
            "delayRisk": round(avg_delay_risk * 100, 1),
            "activeAnomalies": anomaly_count,
            "activeVehicles": total_vehicles,
            "avgTimeToDelay": f"{round(avg_ttd, 1)}h",
            "fleetPerformance": fleet_efficiency,
            "mitigationConfidence": mitigation_confidence,
            "failuresAverted": failures_averted,
            "featureImportance": self._get_feature_importance(fleet),
            "delayTrend": self._build_delay_trend(fleet),
            "riskDistribution": [
                {"name": "LOW", "value": int(low_risk)},
                {"name": "MID", "value": int(mid_risk)},
                {"name": "HIGH", "value": int(high_risk)},
                {"name": "CRITICAL", "value": int(critical_risk)},
            ],
            "survivalCurve": self._live_survival_curve(fleet),
        }

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculates distance between points."""
        R = 6371
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        d_phi = np.radians(lat2 - lat1)
        d_lambda = np.radians(lon2 - lon1)
        a = np.sin(d_phi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(d_lambda/2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    def _preprocess(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Processes input JSON into a valid DataFrame with all features."""
        # 1. Start with provided data or means
        full_data = {k: data.get(k, self.MEANS.get(k)) for k in self.MEANS.keys()}
        
        # Convert to DataFrame for centralized engineering
        df = pd.DataFrame([full_data])
        
        # 2. Run Centralized Feature Engineering (Issue 1, 2, 5)
        df_feat = calculate_derived_metrics(df)
        
        # 3. Handle historical aliases for compatibility
        if 'behavior_risk_score' in df_feat.columns:
            df_feat['driver_risk_score'] = df_feat['behavior_risk_score']
            df_feat['behavior_risk_score'] = df_feat['behavior_risk_score'] # redundant but explicit
            
        # Ensure logistics_risk exists for survival model
        df_feat['logistics_risk'] = (
            df_feat['customs_clearance_time'] + 
            df_feat.get('port_congestion_level', 6.9) + 
            df_feat['traffic_congestion_level']
        )
        
        return df_feat

    def _get_explanation(self, df: pd.DataFrame, delay_prob: float) -> str:
        """Generates hybrid SHAP + Rule-based explanation."""
        explanations = []
        
        # Rule-based insights
        if df.iloc[0]['traffic_congestion_level'] > 7:
            explanations.append("High traffic congestion detected on route.")
        if df.iloc[0]['fatigue_monitoring_score'] > 0.7:
            explanations.append("Driver fatigue score is elevated.")
        if df.iloc[0]['weather_condition_severity'] > 0.6:
            explanations.append("Severe weather conditions impacting safety.")
        if df.iloc[0]['driver_behavior_score'] < 0.4:
            explanations.append("Erratic driver behavior detected.")
            
        if not explanations:
            if delay_prob > 0.6:
                explanations.append("Multiple minor logistics delays contributing to risk.")
            else:
                explanations.append("Operational parameters within normal bounds.")
                
        return " | ".join(explanations)

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Unified inference entry point."""
        df = self._preprocess(data)
        
        try:
            # 1. Delay Probability - Explicit feature slicing
            X_delay = df[self.DELAY_FEATURES]
            delay_prob = self.delay_model.predict(X_delay)[0]
            delay_prob = float(np.clip(delay_prob, 0, 1))

            # 2. Anomaly Detection - Explicit feature slicing
            X_anomaly = df[self.ANOMALY_FEATURES]
            anomaly_score = float(self.anomaly_model.predict_score(X_anomaly)[0])
            anomaly_flag = bool(anomaly_score > 0.6) 

            # 3. Survival Analysis - Explicit feature slicing
            X_survival = df[self.SURVIVAL_FEATURES]
            estimated_time = float(self.cox_model.predict_expectation(X_survival).iloc[0])

            traffic_factor = self._clamp(float(df.iloc[0]['traffic_congestion_level']) / 10.0, 0.0, 1.0)
            base_risk = self._clamp(
                (0.5 * delay_prob) +
                (0.3 * traffic_factor) +
                (0.2 * (1.0 - self._clamp(float(df.iloc[0]['driver_behavior_score']), 0.0, 1.0))),
                0.0,
                1.0,
            )
            dynamic_risk = self._clamp((0.6 * traffic_factor) + (0.4 * anomaly_score), 0.0, 1.0)
            final_risk = self._clamp(
                (0.5 * base_risk) +
                (0.3 * traffic_factor) +
                (0.2 * anomaly_score),
                0.0,
                1.0,
            )
            risk_level = self._derive_risk_level(final_risk)

            # 5. Explanation
            explanation = (
                f"Vehicle is {risk_level.lower()} at {final_risk * 100:.0f}% delay risk. "
                f"{self._get_explanation(df, delay_prob)}"
            )

            return {
                "delay_probability": delay_prob,
                "anomaly_flag": anomaly_flag,
                "anomaly_score": anomaly_score,
                "base_risk": round(base_risk, 4),
                "dynamic_risk": round(dynamic_risk, 4),
                "final_risk": round(final_risk, 4),
                "estimated_time_to_delay": round(estimated_time, 2),
                "risk_level": risk_level,
                "explanation": explanation,
                "solution": self._get_survival_solution(estimated_time, risk_level),
                "confidence_score": 0.92 # Static for now
            }
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise

# Singleton access
inference_service = InferenceService()
