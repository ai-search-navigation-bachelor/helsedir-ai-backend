"""
Logging service for user interactions.

Logs search and click events to MySQL database for ML training.
"""

from datetime import datetime
from typing import Optional, List
from app.services.database_service import database_service


class LoggingService:
    """Service for logging user interactions to database."""

    def log_event(
        self,
        event_type: str,
        query: Optional[str] = None,
        content_id: Optional[str] = None,
        role: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        search_id: Optional[str] = None,
        position: Optional[int] = None,
        results_shown: Optional[List[dict]] = None,
    ) -> bool:
        """
        Log an event to the database.

        Args:
            event_type: Type of event (search, click, role_change)
            query: Search query (for search and click events)
            content_id: Content ID (for click events)
            role: User role
            timestamp: Event timestamp (defaults to now)
            search_id: Unique ID linking search and click events (for learning-to-rank)
            position: Position of clicked result (for click events, learning-to-rank)
            results_shown: Results shown to user (for search events, learning-to-rank)

        Returns:
            True if logging was successful
        """
        if event_type == "search" and search_id and query:
            return database_service.log_search(
                search_id=search_id,
                query=query,
                results=results_shown or [],
                role=role,
            )
        elif event_type == "click" and search_id and content_id:
            return database_service.log_click(
                search_id=search_id,
                content_id=content_id,
                position=position,
                query=query,
                role=role,
            )

        # role_change events are not stored in database
        return True

    def log_search_event(
        self,
        search_id: str,
        query: str,
        results: List[dict],
        role: Optional[str] = None,
    ) -> bool:
        """
        Log a search event with result information for learning-to-rank training.

        Args:
            search_id: Unique ID for this search
            query: Search query
            results: List of search results with content_id, position, and score
            role: User role

        Returns:
            True if logging was successful
        """
        return database_service.log_search(
            search_id=search_id,
            query=query,
            results=results,
            role=role,
        )

    def log_click_event(
        self,
        search_id: str,
        content_id: str,
        position: Optional[int] = None,
        query: Optional[str] = None,
        role: Optional[str] = None,
    ) -> bool:
        """
        Log a click event with position for learning-to-rank training.

        Args:
            search_id: The search_id this click belongs to
            content_id: ID of the clicked content
            position: Position of the clicked result in the search results
            query: The original search query
            role: User role

        Returns:
            True if logging was successful
        """
        return database_service.log_click(
            search_id=search_id,
            content_id=content_id,
            position=position,
            query=query,
            role=role,
        )

    def get_training_data(self) -> List[dict]:
        """
        Get training data for learning-to-rank model.

        Returns:
            List of training examples from database
        """
        return database_service.get_training_data()


# Global instance
logging_service = LoggingService()
