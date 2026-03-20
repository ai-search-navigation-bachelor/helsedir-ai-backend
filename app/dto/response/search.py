"""
Search response DTOs.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from app.dto.response.content import ContentSummaryResponse


class GroupedContent(BaseModel):
    """Content grouped by info_type under a theme page."""
    info_type: str
    display_name: str
    items: List['SearchResult']


class RerankInfo(BaseModel):
    """ML reranking details for a search result."""
    score: float
    rank_change: int  # Positive = moved up, negative = moved down
    features: Dict[str, float]  # Raw feature values used by the model
    contributions: Dict[str, float]  # Per-feature SHAP contributions


class PipelineScores(BaseModel):
    """Retrieval pipeline scores for developer tooling."""
    bm25: float
    semantic: float
    rrf: float
    role_boost: float  # Multiplier (1.15=match, 0.85=mismatch, 1.0=neutral)
    rerank: Optional[RerankInfo] = None


class SearchResult(BaseModel):
    """Single search result."""
    id: str
    title: str
    short_title: Optional[str] = None
    display_title: str
    info_type: str
    path: Optional[str] = None
    has_text_content: bool
    document_url: Optional[str]
    is_pdf_only: bool
    score: float  # Normalized 0-1
    parent: Optional[ContentSummaryResponse] = None
    root_publication: Optional[ContentSummaryResponse] = None
    children: Optional[List[GroupedContent]] = None  # For theme pages with linked content
    pipeline: Optional[PipelineScores] = None


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
    short_title: Optional[str] = None
    display_title: str
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
    short_title: Optional[str] = None
    display_title: str
    info_type: Optional[str] = None
    path: Optional[str] = None


class SuggestionResponse(BaseModel):
    """Response model for autocomplete suggestions endpoint."""
    suggestions: List[Suggestion]
