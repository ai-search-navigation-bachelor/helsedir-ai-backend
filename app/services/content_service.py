"""
Content service for loading and managing health content.

Loads content from database cache or Helsedirektoratet API.
"""

from typing import List, Optional
from app.models.schemas import ContentItem
from app.services.database_service import database_service


class ContentService:
    """Service for loading and managing content data."""

    def __init__(self):
        self.content: List[ContentItem] = []
        self.content_by_id: dict = {}
        self.load_content()

    def load_content(self):
        """Load content from database cache."""
        db_content = database_service.get_all_content()

        if not db_content:
            print("No content in database cache. Use load_from_api() to fetch content.")
            return

        self.content = []
        for item in db_content:
            content_item = ContentItem(
                id=str(item.get("id", "")),
                title=item.get("tittel") or "",
                body=item.get("tekst") or "",
                url=item.get("url") or "",
                content_type=item.get("info_type") or "unknown",
                target_groups=[],
                tags=[],
            )
            self.content.append(content_item)

        self.content_by_id = {item.id: item for item in self.content}
        print(f"Loaded {len(self.content)} content items from database cache")

    def load_from_api(self, query_text: Optional[str] = None, max_items: int = 100):
        """
        Load content from Helsedirektoratet API and cache in database.

        Args:
            query_text: Optional search query to filter results
            max_items: Maximum number of items to load

        Example:
            >>> content_service.load_from_api(query_text="helse", max_items=50)
        """
        from app.services.helsedir_api_service import (
            helsedir_api_service,
            HelseDirectorateAPIError,
        )

        try:
            print(f"Loading content from Helsedirektoratet API...")

            results = helsedir_api_service.search_infobits(
                query_text=query_text,
                get_full_infobits=True,
            )

            api_items = results[:max_items] if isinstance(results, list) else []

            # Cache in database
            cached_count = database_service.cache_content_batch(api_items)
            print(f"Cached {cached_count} content items in database")

            # Parse into ContentItem format
            self.content = []
            for item in api_items:
                content_item = ContentItem(
                    id=str(item.get("id", item.get("infoId", ""))),
                    title=item.get("tittel", ""),
                    body=item.get("tekst", ""),
                    url=item.get("url", ""),
                    content_type=item.get("infoType", "unknown"),
                    target_groups=item.get("maalgruppe", []),
                    tags=[],
                )
                self.content.append(content_item)

            self.content_by_id = {item.id: item for item in self.content}
            print(f"Loaded {len(self.content)} content items from API")

        except HelseDirectorateAPIError as e:
            print(f"Error loading from API: {e}")
            print("Falling back to database cache...")
            self.load_content()

    def get_all_content(self) -> List[ContentItem]:
        """Get all content items."""
        return self.content

    def get_content_by_id(self, content_id: str) -> Optional[ContentItem]:
        """Get a specific content item by ID."""
        return self.content_by_id.get(content_id)

    def reload_content(self):
        """Reload content from database cache."""
        self.load_content()


# Global instance
content_service = ContentService()
