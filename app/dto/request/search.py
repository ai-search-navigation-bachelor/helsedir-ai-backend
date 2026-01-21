"""
Search request DTOs.
"""

from pydantic import BaseModel, Field
from typing import Optional


class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., min_length=1, description="Search query")
    role: Optional[str] = Field(None, description="User role for filtering")
    k: int = Field(10, ge=1, le=100, description="Number of results to return")
    method: str = Field("hybrid", description="Search method: 'keyword', 'semantic', or 'hybrid'")


class HelseDirectorateSearchRequest(BaseModel):
    """Request model for Helsedirektoratet API search endpoint."""
    query: str = Field(..., min_length=1, description="Search query text", alias="QueryText")
    filter: Optional[str] = Field(None, description="OData filter expression", alias="Filter")
    search_mode: Optional[str] = Field(None, description="Search mode: 'Any' or 'All'", alias="SearchMode")
    query_type: Optional[str] = Field(None, description="Query type: 'Simple' or 'Full'", alias="QueryType")
    get_full_infobits: bool = Field(False, description="Return full infobit content", alias="getFullInfobits")


class TagRequest(BaseModel):
    """Request model for tagging endpoint."""
    content_id: Optional[str] = None
    text: Optional[str] = None


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    question: str = Field(..., min_length=1)
    role: Optional[str] = None
