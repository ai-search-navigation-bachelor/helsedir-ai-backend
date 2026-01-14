from fastapi import APIRouter, HTTPException
from app.models.schemas import LogRequest, LogResponse
from app.services.logging_service import logging_service

router = APIRouter(prefix="/log", tags=["logging"])


@router.post("", response_model=LogResponse)
async def log_event(request: LogRequest):
    """
    Log a user interaction event.

    Args:
        request: Log request with event_type, optional query, content_id, role, and timestamp

    Returns:
        LogResponse indicating success or failure
    """
    try:
        success = logging_service.log_event(
            event_type=request.event_type,
            query=request.query,
            content_id=request.content_id,
            role=request.role,
            timestamp=request.timestamp,
        )

        return LogResponse(success=success)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logging failed: {str(e)}")
