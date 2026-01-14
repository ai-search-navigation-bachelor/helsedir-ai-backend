import json
from typing import List, Optional
from pathlib import Path
from app.models.schemas import ContentItem
from app.config import settings


class ContentService:
    """Service for loading and managing content data."""

    def __init__(self):
        self.content: List[ContentItem] = []
        self.content_by_id: dict = {}
        self.load_content()

    def load_content(self):
        """Load content from JSON file."""
        content_path = Path(settings.content_file)

        if not content_path.exists():
            print(f"Warning: Content file not found at {content_path}")
            return

        with open(content_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.content = [ContentItem(**item) for item in data]
        self.content_by_id = {item.id: item for item in self.content}
        print(f"Loaded {len(self.content)} content items from {content_path}")

    def load_from_api(self, query_text: Optional[str] = None, max_items: int = 100):
        """
        Load content from Helsedirektoratet API instead of local file.

        Args:
            query_text: Optional search query to filter results
            max_items: Maximum number of items to load

        Example:
            >>> content_service.load_from_api(query_text="helse", max_items=50)
        """
        from app.services.helsedir_api_service import helsedir_api_service, HelseDirectorateAPIError

        try:
            print(f"Loading content from Helsedirektoratet API...")

            results = helsedir_api_service.search_infobits(
                query_text=query_text,
                get_full_infobits=True,
            )

            # Parse API results into ContentItem format
            # API returns a list directly
            api_items = results[:max_items] if isinstance(results, list) else []
            self.content = []

            for item in api_items:
                # Map Norwegian API fields to ContentItem schema
                content_item = ContentItem(
                    id=str(item.get('id', item.get('infoId', ''))),
                    title=item.get('tittel', ''),
                    body=item.get('tekst', ''),
                    url=item.get('url', ''),
                    content_type=item.get('infoType', 'unknown'),
                    published_at=item.get('publiseringsdato', ''),
                    target_groups=item.get('maalgruppe', []),
                    tags=[],  # API doesn't seem to have tags in this format
                )
                self.content.append(content_item)

            self.content_by_id = {item.id: item for item in self.content}
            print(f"Loaded {len(self.content)} content items from API")

        except HelseDirectorateAPIError as e:
            print(f"Error loading from API: {e}")
            print("Falling back to local content file...")
            self.load_content()

    def get_all_content(self) -> List[ContentItem]:
        """Get all content items."""
        return self.content

    def get_content_by_id(self, content_id: str) -> Optional[ContentItem]:
        """Get a specific content item by ID."""
        return self.content_by_id.get(content_id)

    def reload_content(self):
        """Reload content from file (useful for development)."""
        self.load_content()


# Global instance
content_service = ContentService()
