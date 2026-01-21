from fastapi import APIRouter, HTTPException
from app.models.schemas import LogRequest, LogResponse
from app.services.logging_service import logging_service

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
        # Convert results_shown to list of dicts if provided
        results_shown = None
        if request.results_shown:
            results_shown = [
                {
                    "content_id": r.content_id,
                    "position": r.position,
                    "score": r.score,
                }
                for r in request.results_shown
            ]

        success = logging_service.log_event(
            event_type=request.event_type,
            query=request.query,
            content_id=request.content_id,
            role=request.role,
            timestamp=request.timestamp,
            search_id=request.search_id,
            position=request.position,
            results_shown=results_shown,
        )

        return LogResponse(success=success)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logging failed: {str(e)}")
