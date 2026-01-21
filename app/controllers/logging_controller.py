"""
Logging controller.

Handles business logic for event logging operations.
"""

from typing import Optional, List
from datetime import datetime
from app.dto.response.logging import LogResponse
from app.dto.request.logging import SearchResultLog
from app.services.analytics.logging_service import logging_service


class LoggingController:
    """Controller for logging operations."""

    def __init__(self):
        self.logging_service = logging_service

    async def log_event(
        self,
        event_type: str,
        query: Optional[str] = None,
        content_id: Optional[str] = None,
        role: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        search_id: Optional[str] = None,
        position: Optional[int] = None,
        results_shown: Optional[List[SearchResultLog]] = None,
    ) -> LogResponse:
        """
        Log a user interaction event.

        Args:
            event_type: Type of event ('search', 'click', etc.)
            query: Search query (for search events)
            content_id: Content ID (for click events)
            role: User role
            timestamp: Event timestamp
            search_id: Search session ID
            position: Position in results (for click events)
            results_shown: List of results shown (for search events)

        Returns:
            LogResponse indicating success or failure

        Raises:
            Exception: If logging fails
        """
        # Convert results_shown to list of dicts if provided
        results_dict = None
        if results_shown:
            results_dict = [
                {
                    "content_id": r.content_id,
                    "position": r.position,
                    "score": r.score,
                }
                for r in results_shown
            ]

        success = self.logging_service.log_event(
            event_type=event_type,
            query=query,
            content_id=content_id,
            role=role,
            timestamp=timestamp,
            search_id=search_id,
            position=position,
            results_shown=results_dict,
        )

        return LogResponse(success=success)


# Global instance
logging_controller = LoggingController()
