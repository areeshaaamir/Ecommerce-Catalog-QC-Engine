from fastapi import FastAPI
from app.db import init_db
from app.config import settings

app = FastAPI(
    title="E-Commerce Catalog QC Engine",
    description="AI-Powered Product Photo & Listing Mismatch Guard",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    init_db()

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "database": settings.DATABASE_PATH,
        "vision_model": settings.VISION_MODEL
    }