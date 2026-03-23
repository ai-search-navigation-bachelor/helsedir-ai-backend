"""
Statistics response DTOs.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class StatisticsAttachmentResponse(BaseModel):
    """Frontend-friendly attachment metadata for NKI indicators."""

    title: str
    url: str


class StatisticsPointResponse(BaseModel):
    """Single normalized datapoint."""

    x: Optional[str] = None
    y: float
    location: Optional[str] = None
    parent_location: Optional[str] = None
    time_from: Optional[str] = None
    time_to: Optional[str] = None
    period_type: Optional[str] = None


class StatisticsSeriesResponse(BaseModel):
    """Series grouped by measure name."""

    name: str
    points: List[StatisticsPointResponse] = Field(default_factory=list)


class StatisticsDimensionsResponse(BaseModel):
    """Filter lists derived from the normalized dataset."""

    measures: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    parent_locations: List[str] = Field(default_factory=list)
    period_types: List[str] = Field(default_factory=list)


class ContentStatisticsResponse(BaseModel):
    """Statistics response for a single content page."""

    has_statistics: bool
    statistics_status: str
    content_id: str
    nki_indicator_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    attachments: List[StatisticsAttachmentResponse] = Field(default_factory=list)
    series: List[StatisticsSeriesResponse] = Field(default_factory=list)
    dimensions: StatisticsDimensionsResponse = Field(default_factory=StatisticsDimensionsResponse)
