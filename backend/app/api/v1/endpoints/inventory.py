from fastapi import APIRouter, HTTPException

from app.inventory.schemas.inventory import (
    DemandPredictionResponse,
    InventoryRequest,
    InventorySummaryResponse,
    RestockRecommendationResponse,
    StockoutRiskResponse,
)
from app.inventory.services.inventory_service import inventory_service


router = APIRouter()


@router.post("/predict-demand", response_model=DemandPredictionResponse)
async def predict_inventory_demand(request: InventoryRequest):
    try:
        return inventory_service.predict_demand(request.dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/stockout-risk", response_model=StockoutRiskResponse)
async def stockout_risk(request: InventoryRequest):
    try:
        return inventory_service.stockout_risk(request.dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/restock-recommendation", response_model=RestockRecommendationResponse)
async def restock_recommendation(request: InventoryRequest):
    try:
        return inventory_service.restock_recommendation(request.dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/summary", response_model=InventorySummaryResponse)
async def inventory_summary():
    try:
        return inventory_service.summary()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
