from fastapi import APIRouter
from app.api.v1.endpoints import inventory, predict

api_router = APIRouter()
api_router.include_router(predict.router, prefix="/predict", tags=["prediction"])
api_router.include_router(inventory.router, prefix="/inventory", tags=["inventory"])
