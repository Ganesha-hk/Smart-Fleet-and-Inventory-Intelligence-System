from typing import List, Optional

from pydantic import BaseModel, Field


class InventoryRequest(BaseModel):
    item_id: str = Field(..., description="Inventory item identifier")
    warehouse_id: str = Field(..., description="Warehouse identifier")
    region: str = Field(..., description="Warehouse region")
    category: str = Field(..., description="Inventory category")
    product_name: Optional[str] = Field(None, description="Real product name")
    stock_level: float = Field(..., ge=0, description="Current units on hand")
    demand_rate: float = Field(..., ge=0, description="Daily demand rate")
    supply_rate: float = Field(..., ge=0, description="Daily replenishment rate")
    lead_time: float = Field(..., ge=0, description="Inbound lead time in days")
    consumption_rate: float = Field(..., ge=0, description="Observed daily consumption rate")
    anomaly_score: float = Field(..., ge=0, le=1, description="Operational anomaly score")
    unit_type: Optional[str] = None
    linked_ship_id: Optional[str] = None
    timestamp: Optional[str] = Field(None, description="Optional event timestamp aligned with fleet operations")


class DemandPredictionResponse(BaseModel):
    item_id: str
    warehouse_location: str
    timestamp: str
    predicted_demand: float
    linked_delay_risk: float
    demand_pressure: str


class StockoutRiskResponse(BaseModel):
    item_id: str
    warehouse_location: str
    timestamp: str
    stockout_probability: float
    risk_level: str
    days_of_cover: float
    linked_delay_risk: float


class RestockRecommendationResponse(BaseModel):
    item_id: str
    warehouse_location: str
    timestamp: str
    predicted_demand: float
    stockout_probability: float
    recommended_restock_quantity: int
    urgency: str
    suggested_route_priority: str
    rationale: str


class SummaryMetric(BaseModel):
    title: str
    value: str
    subtitle: str


class DemandTrendPoint(BaseModel):
    name: str
    demand: float
    predicted: float


class StockAlertItem(BaseModel):
    item_id: str
    warehouse_location: str
    stockout_probability: float
    linked_delay_risk: float
    inventory_level: int
    product_name: str
    category: str
    stock_units: str
    daily_demand: str
    days_to_stockout: float
    incoming_ship_id: str
    eta_days: int
    shipment_status: str
    incoming_shipment: str
    risk: str
    message: str


class RestockRecommendationItem(BaseModel):
    item_id: str
    warehouse_location: str
    product_name: str
    category: str
    stock_units: str
    daily_demand: str
    days_to_stockout: float
    incoming_ship_id: str
    eta_days: int
    shipment_status: str
    incoming_shipment: str
    recommended_restock_quantity: int
    urgency: str
    suggested_route_priority: str
    rationale: str


class WarehouseSnapshot(BaseModel):
    warehouse_location: str
    avg_inventory_level: float
    avg_demand_forecast: float
    avg_days_to_stockout: float
    avg_stockout_probability: float
    avg_linked_delay_risk: float


class InventorySummaryResponse(BaseModel):
    metrics: List[SummaryMetric]
    demand_trend: List[DemandTrendPoint]
    stock_alerts: List[StockAlertItem]
    restock_recommendations: List[RestockRecommendationItem]
    warehouse_snapshots: List[WarehouseSnapshot]
