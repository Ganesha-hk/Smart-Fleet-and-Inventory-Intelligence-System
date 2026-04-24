from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.services.inference_service import inference_service

router = APIRouter()

class PredictRequest(BaseModel):
    vehicle_gps_latitude: float = Field(..., description="Current latitude of the vehicle")
    vehicle_gps_longitude: float = Field(..., description="Current longitude of the vehicle")
    traffic_congestion_level: float = Field(..., ge=0, le=10, description="Local traffic congestion (0-10)")
    driver_behavior_score: float = Field(..., ge=0, le=1, description="Driver safety/performance score (0-1)")
    fatigue_monitoring_score: float = Field(..., ge=0, le=1, description="Driver fatigue level (0-1)")
    
    # Optional fields with defaults or handled by inference service
    fuel_consumption_rate: Optional[float] = None
    eta_variation_hours: Optional[float] = None
    warehouse_inventory_level: Optional[float] = None
    loading_unloading_time: Optional[float] = None
    weather_condition_severity: Optional[float] = None
    route_risk_level: Optional[float] = None
    disruption_likelihood_score: Optional[float] = None

class PredictResponse(BaseModel):
    delay_probability: float
    anomaly_flag: bool
    anomaly_score: float
    base_risk: float
    dynamic_risk: float
    final_risk: float
    estimated_time_to_delay: float
    risk_level: str
    explanation: str
    confidence_score: float

class SampleBatchResponse(BaseModel):
    totalCount: int
    vehicles: list[Dict[str, Any]]
    events: list[Dict[str, Any]]

@router.post("/", response_model=PredictResponse)
async def predict_fleet_metrics(request: PredictRequest):
    """
    Generate unified fleet predictions including delay risk, anomaly detection, 
    and estimated time to failure.
    """
    try:
        data = request.dict(exclude_none=True)
        prediction = inference_service.predict(data)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sample-predict", response_model=PredictResponse)
async def sample_predict():
    """
    Pick a random row from the real dataset and run the full inference pipeline.
    """
    try:
        sample_data = inference_service.get_random_sample()
        if not sample_data:
            raise HTTPException(status_code=404, detail="Dataset not loaded or empty")
        
        prediction = inference_service.predict(sample_data)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sample-batch", response_model=SampleBatchResponse)
async def sample_batch():
    """
    Update the live fleet first, then return the full fleet.
    """
    try:
        return inference_service.get_sample_batch()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard-metrics", response_model=Dict[str, Any])
async def dashboard_metrics():
    """
    Return aggregated dashboard metrics based on real-time rolling samples.
    """
    try:
        return inference_service.get_dashboard_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
