from fastapi import APIRouter
from app.models.schemas import HealthResponse
from datetime import datetime

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns the current status of the API.
    """
    return HealthResponse(status="ok", timestamp=datetime.utcnow())
