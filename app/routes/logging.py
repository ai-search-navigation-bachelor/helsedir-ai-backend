from fastapi import APIRouter, HTTPException
from app.models.schemas import LogRequest, LogResponse
from app.controllers.logging_controller import logging_controller

router = APIRouter(prefix="/log", tags=["logging"])


@router.post("", response_model=LogResponse)
async def log_event(request: LogRequest):
    """
    Log a user interaction event.

    Args:
        request: Log request with event_type, optional query, content_id, role, timestamp,
                 and optional position/results_shown for learning-to-rank data collection.

    Returns:
        LogResponse indicating success or failure
    """
    try:
        return await logging_controller.log_event(
            event_type=request.event_type,
            query=request.query,
            content_id=request.content_id,
            role=request.role,
            timestamp=request.timestamp,
            search_id=request.search_id,
            position=request.position,
            results_shown=request.results_shown,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logging failed: {str(e)}")
