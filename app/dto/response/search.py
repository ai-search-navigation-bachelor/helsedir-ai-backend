"""
Search response DTOs.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class SearchResult(BaseModel):
    """Single search result."""
    id: str
    title: str
    info_type: str
    score: float  # Normalized 0-1
    explanation: str


class SearchResponse(BaseModel):
    """Response model for search endpoint with pagination."""
    results: List[SearchResult]
    query: str
    total: int
    search_id: str  # For linking clicks to searches

    # Pagination info
    offset: int
    limit: int
    has_next: bool
    has_prev: bool


class TagResponse(BaseModel):
    """Response model for tagging endpoint."""
    tags: List[str]
    metadata: dict = {}


class ChatSource(BaseModel):
    """Source document for chat response."""
    id: str
    title: str
    snippet: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str
    sources: List[ChatSource]
