"""
Search response DTOs.
"""

from pydantic import BaseModel, Field
from typing import List


class SearchResult(BaseModel):
    """Single search result."""
    id: str
    title: str
    url: str
    snippet: str
    score: float
    explanation: str = Field(..., description="Short explanation of why this result matches")


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    results: List[SearchResult]
    query: str
    total: int


class TagResponse(BaseModel):
    """Response model for tagging endpoint."""
    tags: List[str]
    metadata: dict = {}


class ChatSource(BaseModel):
    """Source document for chat response."""
    id: str
    title: str
    url: str
    snippet: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str
    sources: List[ChatSource]
