"""
Search request DTOs.
"""

from pydantic import BaseModel, Field
from typing import Optional


class SearchRequest(BaseModel):
    """Request model for search endpoint with pagination."""
    query: str = Field(..., min_length=1, description="Search query")
    role: Optional[str] = Field(None, description="User role for filtering")
    method: str = Field("hybrid", description="Search method: 'keyword', 'semantic', or 'hybrid'")

    # Pagination
    offset: int = Field(0, ge=0, description="Number of results to skip")
    limit: int = Field(10, ge=1, le=50, description="Number of results per page")
    search_id: Optional[str] = Field(None, description="Existing search_id for pagination")


class CategorizedSearchRequest(BaseModel):
    """Request model for categorized search endpoint."""
    query: str = Field(..., min_length=1, description="Search query")
    role: Optional[str] = Field(None, description="User role for filtering")
    method: str = Field("hybrid", description="Search method: 'keyword', 'semantic', or 'hybrid'")


class CategorySearchRequest(BaseModel):
    """Request model for searching within a specific category."""
    query: str = Field(..., min_length=1, description="Search query")
    category: str = Field(..., min_length=1, description="Category (info_type) to filter by")
    role: Optional[str] = Field(None, description="User role for filtering")
    method: str = Field("hybrid", description="Search method: 'keyword', 'semantic', or 'hybrid'")
    search_id: Optional[str] = Field(None, description="Existing search_id from categorized search")


class HelseDirectorateSearchRequest(BaseModel):
    """Request model for Helsedirektoratet API search endpoint."""
    query: str = Field(..., min_length=1, description="Search query text", alias="QueryText")
    filter: Optional[str] = Field(None, description="OData filter expression", alias="Filter")
    search_mode: Optional[str] = Field(None, description="Search mode: 'Any' or 'All'", alias="SearchMode")
    query_type: Optional[str] = Field(None, description="Query type: 'Simple' or 'Full'", alias="QueryType")
    get_full_infobits: bool = Field(True, description="Return full infobit content (default: True for complete data)", alias="getFullInfobits")


class TagRequest(BaseModel):
    """Request model for tagging endpoint."""
    content_id: Optional[str] = None
    text: Optional[str] = None


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    question: str = Field(..., min_length=1)
    role: Optional[str] = None
