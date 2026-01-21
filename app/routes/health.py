from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.controllers.health_controller import health_controller

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns the current status of the API.
    """
    return await health_controller.check_health()
