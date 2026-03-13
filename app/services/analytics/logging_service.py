"""
Logging service for user interactions.

Logs search and click events to MySQL database for ML training.
"""

from typing import Optional, List
from app.services.data.database_service import database_service


class LoggingService:
    """Service for logging user interactions to database."""

    def log_event(
        self,
        event_type: str,
        query: Optional[str] = None,
        content_id: Optional[str] = None,
        role: Optional[str] = None,
        search_id: Optional[str] = None,
    ) -> bool:
        """
        Log an event to the database.

        Args:
            event_type: Type of event (search, click)
            query: Search query (for search events)
            content_id: Content ID (for click events)
            role: User role
            search_id: Unique ID linking search and click events

        Returns:
            True if logging was successful
        """
        if event_type == "search" and search_id and query:
            return database_service.log_search(
                search_id=search_id,
                query=query,
                role=role,
            )
        elif event_type == "click" and search_id and content_id:
            return database_service.log_click(
                search_id=search_id,
                content_id=content_id,
            )

        return False

    def get_training_data(self) -> List[dict]:
        """Get training data for learning-to-rank model."""
        return database_service.get_training_data()


# Global instance
logging_service = LoggingService()
