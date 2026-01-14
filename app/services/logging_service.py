import json
from datetime import datetime
from pathlib import Path
from typing import Optional
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

    def log_event(
        self,
        event_type: str,
        query: Optional[str] = None,
        content_id: Optional[str] = None,
        role: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Log an event to the JSONL file.

        Args:
            event_type: Type of event (search, click, role_change)
            query: Search query (for search events)
            content_id: Content ID (for click events)
            role: User role
            timestamp: Event timestamp (defaults to now)

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

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            return True
        except Exception as e:
            print(f"Error logging event: {e}")
            return False

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
