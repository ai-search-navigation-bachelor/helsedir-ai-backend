"""
Search response DTOs.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class GroupedContent(BaseModel):
    """Content grouped by info_type under a theme page."""
    info_type: str
    display_name: str
    items: List['SearchResult']


class SearchResult(BaseModel):
    """Single search result."""
    id: str
    title: str
    info_type: str
    path: Optional[str] = None
    score: float  # Normalized 0-1
    explanation: str
    children: Optional[List[GroupedContent]] = None  # For theme pages with linked content

    # Internal pipeline scores — carried through for logging, excluded from API response
    bm25_score: Optional[float] = Field(default=None, exclude=True)
    semantic_score: Optional[float] = Field(default=None, exclude=True)
    rrf_score: Optional[float] = Field(default=None, exclude=True)


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

    # Category distribution across all fetched results (not just current page)
    category_counts: Dict[str, int] = Field(default_factory=dict)


# =============================================================================
# Categorized Search Response
# =============================================================================

class CategoryResults(BaseModel):
    """Results for a single category."""
    category: str  # info_type key
    display_name: str  # Norwegian display name
    count: int  # Total results in this category
    is_priority: bool  # If true, all results are included
    results: List[SearchResult]  # All results (priority) or top N (preview)


class CategorizedSearchResponse(BaseModel):
    """Response model for categorized search endpoint."""
    query: str
    total: int  # Total results across all categories
    min_score: float  # Score threshold used
    search_id: str

    # Priority categories (e.g., retningslinje) - show all results
    priority_categories: List[CategoryResults]

    # Other categories - show count and top N preview
    other_categories: List[CategoryResults]


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


class ThemePageResult(BaseModel):
    """Single theme page result."""
    id: str
    title: str
    info_type: str
    path: str


class ThemePageResponse(BaseModel):
    """Response model for theme page listing endpoint."""
    results: List[ThemePageResult]
    total: int


class Suggestion(BaseModel):
    """Single autocomplete suggestion."""
    id: str
    title: str


class SuggestionResponse(BaseModel):
    """Response model for autocomplete suggestions endpoint."""
    suggestions: List[Suggestion]
