"""
Health controller.

Handles business logic for health check operations.
"""

from datetime import datetime
from app.models.schemas import HealthResponse


class HealthController:
    """Controller for health check operations."""

    async def check_health(self) -> HealthResponse:
        """
        Perform health check.

        Returns:
            HealthResponse with current status and timestamp
        """
        return HealthResponse(
            status="ok",
            timestamp=datetime.utcnow()
        )


# Global instance
health_controller = HealthController()
