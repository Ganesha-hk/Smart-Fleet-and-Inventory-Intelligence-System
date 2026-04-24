from fastapi import APIRouter
from app.api.v1.endpoints import predict

api_router = APIRouter()
api_router.include_router(predict.router, prefix="/predict", tags=["prediction"])
