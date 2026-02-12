"""
Content service for loading and managing health content.

Loads content from database cache or Helsedirektoratet API.
"""

import json
import logging
from typing import List, Optional
from pydantic import ValidationError
from app.entities.content import ContentItem, ContentLink, AnbefalingFields
from app.services.data.database_service import database_service

logger = logging.getLogger(__name__)


class ContentService:
    """Service for loading and managing content data."""

    def __init__(self):
        self.content: List[ContentItem] = []
        self.content_by_id: dict = {}
        self.load_content()

    def _parse_json_field(self, value, default=None):
        """Parse a JSON field that may be a string or already parsed."""
        if value is None:
            return default if default is not None else []
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return default if default is not None else []
        return value

    def _parse_links(self, links_data) -> List[ContentLink]:
        """Parse links from database JSON into ContentLink objects."""
        links = self._parse_json_field(links_data, [])
        if not isinstance(links, list):
            return []

        result = []
        for link in links:
            if isinstance(link, dict):
                try:
                    result.append(ContentLink(
                        rel=link.get("rel", ""),
                        type=link.get("type", ""),
                        tittel=link.get("tittel"),
                        href=link.get("href", ""),
                        strukturId=link.get("strukturId"),
                    ))
                except ValidationError as e:
                    logger.warning(f"Malformed link data skipped: {link}, error: {e}")
                    continue
        return result

    def load_content(self):
        """Load content from database cache."""
        db_content = database_service.get_all_content()

        if not db_content:
            print("No content in database cache. Use load_from_api() to fetch content.")
            return

        self.content = []
        for item in db_content:
            # Parse JSON fields
            maalgruppe = self._parse_json_field(item.get("maalgruppe"), [])
            links = self._parse_links(item.get("links"))

            # Parse anbefaling-specific fields if this is an anbefaling
            anbefaling_fields = None
            if item.get("info_type") == "anbefaling":
                # Check if any anbefaling field has data (from LEFT JOIN)
                if any([
                    item.get("praktisk"),
                    item.get("rasjonale"),
                    item.get("fordeler_ulemper"),
                    item.get("verdier_preferanser"),
                    item.get("kvalitet_dokumentasjon"),
                    item.get("ressurshensyn"),
                    item.get("styrke"),
                ]):
                    anbefaling_fields = AnbefalingFields(
                        praktisk=item.get("praktisk"),
                        rasjonale=item.get("rasjonale"),
                        fordeler_ulemper=item.get("fordeler_ulemper"),
                        verdier_preferanser=item.get("verdier_preferanser"),
                        kvalitet_dokumentasjon=item.get("kvalitet_dokumentasjon"),
                        ressurshensyn=item.get("ressurshensyn"),
                        styrke=item.get("styrke"),
                    )

            content_item = ContentItem(
                id=str(item.get("id", "")),
                title=item.get("tittel") or "",
                body=item.get("tekst") or "",
                content_type=item.get("info_type") or "unknown",
                target_groups=maalgruppe if isinstance(maalgruppe, list) else [],
                links=links,
                anbefaling_fields=anbefaling_fields,
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
        from app.services.external.helsedir_api_service import helsedir_api_service
        from app.exceptions.helsedir import HelseDirectorateAPIError

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
                maalgruppe = self._parse_json_field(item.get("maalgruppe"), [])
                links = self._parse_links(item.get("links"))

                content_item = ContentItem(
                    id=str(item.get("id", item.get("infoId", ""))),
                    title=item.get("tittel", ""),
                    body=item.get("tekst", ""),
                    content_type=item.get("infoType", "unknown"),
                    target_groups=maalgruppe if isinstance(maalgruppe, list) else [],
                    links=links,
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
