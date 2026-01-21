import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from app.config import settings


class LoggingService:
    """Service for logging user interactions."""

    def __init__(self):
        self.log_file = Path(settings.logs_file)
        self._ensure_log_file_exists()

    def _ensure_log_file_exists(self):
        """Ensure the log file and directory exist."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            self.log_file.touch()

    def _write(self, log_entry: dict) -> bool:
        """Write a log entry to the JSONL file."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            return True
        except Exception as e:
            print(f"Error logging event: {e}")
            return False

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
        Log an event to the JSONL file.

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
        if timestamp is None:
            timestamp = datetime.utcnow()

        log_entry = {
            "event_type": event_type,
            "timestamp": timestamp.isoformat(),
            "query": query,
            "content_id": content_id,
            "role": role,
        }

        # Add learning-to-rank fields if provided
        if search_id is not None:
            log_entry["search_id"] = search_id
        if position is not None:
            log_entry["position"] = position
        if results_shown is not None:
            log_entry["results_shown"] = results_shown

        return self._write(log_entry)

    def log_search_event(
        self,
        query: str,
        results: List[dict],
        role: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Log a search event with result information for learning-to-rank training.

        Args:
            query: Search query
            results: List of search results with id, position, and score
            role: User role
            timestamp: Event timestamp (defaults to now)

        Returns:
            True if logging was successful
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        log_entry = {
            "event_type": "search",
            "timestamp": timestamp.isoformat(),
            "query": query,
            "role": role,
            "results_shown": results,
        }

        return self._write(log_entry)

    def log_click_event(
        self,
        query: str,
        content_id: str,
        position: int,
        role: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Log a click event with position for learning-to-rank training.

        Args:
            query: Search query that led to this click
            content_id: ID of the clicked content
            position: Position of the clicked result in the search results
            role: User role
            timestamp: Event timestamp (defaults to now)

        Returns:
            True if logging was successful
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        log_entry = {
            "event_type": "click",
            "timestamp": timestamp.isoformat(),
            "query": query,
            "content_id": content_id,
            "position": position,
            "role": role,
        }

        return self._write(log_entry)

    def read_logs(self, limit: Optional[int] = None) -> list:
        """
        Read logs from file (useful for analytics).

        Args:
            limit: Maximum number of logs to return (most recent)

        Returns:
            List of log entries
        """
        if not self.log_file.exists():
            return []

        logs = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))

        if limit:
            logs = logs[-limit:]

        return logs


# Global instance
logging_service = LoggingService()
