from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.api import api_router
import uvicorn

app = FastAPI(
    title="Smart Fleet Intelligence API",
    description="Backend integration for predictive fleet logistics",
    version="1.0.0"
)

# CORS Configuration
# Allow all origins for development; in production, this should be restricted to the frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Smart Fleet Intelligence API is running", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
