"""
Theme pages route for listing theme pages.
"""

from fastapi import APIRouter, HTTPException, Depends
from app.dto.request.search import ThemePageRequest
from app.dto.response.search import ThemePageResponse
from app.controllers.search_controller import search_controller

router = APIRouter(prefix="/theme-pages", tags=["theme-pages"])


@router.get("", response_model=ThemePageResponse)
async def get_theme_pages(request: ThemePageRequest = Depends()):
    """
    Get all theme pages (temasider), optionally filtered by category.

    Query params:
        - category: Filter by category slug (optional)
          Valid values:
            - forebygging-diagnose-og-behandling
            - digitalisering-og-e-helse
            - lov-og-forskrift
            - helseberedskap
            - autorisasjon-og-spesialistutdanning
            - tilskudd-og-finansiering
            - statistikk-registre-og-rapporter

    Returns:
        ThemePageResponse with all theme pages
    """
    try:
        return await search_controller.get_theme_pages(
            category=request.category,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get theme pages: {str(e)}")
