"""
Controllers package.

Controllers handle business logic and orchestrate service calls.
They act as an intermediary between routes (HTTP layer) and services (data layer).
"""

from app.controllers.search_controller import search_controller
from app.controllers.health_controller import health_controller
from app.controllers.logging_controller import logging_controller
from app.controllers.helsedir_controller import helsedir_controller

__all__ = [
    "search_controller",
    "health_controller",
    "logging_controller",
    "helsedir_controller",
]
