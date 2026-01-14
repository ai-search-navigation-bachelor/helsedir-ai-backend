from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# Content models
class ContentItem(BaseModel):
    """Content item from the dataset."""
    id: str
    title: str
    body: str
    url: str
    content_type: str
    published_at: str
    target_groups: List[str]
    tags: Optional[List[str]] = []


# Search models
class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., min_length=1, description="Search query")
    role: Optional[str] = Field(None, description="User role for filtering")
    k: int = Field(10, ge=1, le=100, description="Number of results to return")


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


# Logging models
class LogRequest(BaseModel):
    """Request model for logging endpoint."""
    event_type: str = Field(..., pattern="^(search|click|role_change)$")
    query: Optional[str] = None
    content_id: Optional[str] = None
    role: Optional[str] = None
    timestamp: Optional[datetime] = None


class LogResponse(BaseModel):
    """Response model for logging endpoint."""
    success: bool


# Health model
class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Optional: Tag models (for future features)
class TagRequest(BaseModel):
    """Request model for tagging endpoint."""
    content_id: Optional[str] = None
    text: Optional[str] = None


class TagResponse(BaseModel):
    """Response model for tagging endpoint."""
    tags: List[str]
    metadata: dict = {}


# Optional: Chat models (for future features)
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    question: str = Field(..., min_length=1)
    role: Optional[str] = None


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
