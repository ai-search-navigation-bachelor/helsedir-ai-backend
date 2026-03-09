"""
Content service for loading and managing health content.

Loads content from database cache or Helsedirektoratet API.
"""

import json
import logging
from typing import List, Optional, Set
from pydantic import ValidationError
from app.entities.content import ContentItem, ContentLink, AnbefalingFields
from app.services.data.database_service import database_service
from app.services.data.document_metadata import compute_document_metadata
from app.services.repositories.content_repository import content_repository
from app.constants import ALLOWED_INFO_TYPES_SET

logger = logging.getLogger(__name__)


class ContentService:
    """Service for loading and managing content data."""

    def __init__(self):
        self.content: List[ContentItem] = []
        self.content_by_id: dict = {}
        self.content_by_path: dict = {}
        self.searchable_types: Set[str] = set()
        self._content_version: int = 0
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
                        id=link.get("id"),  # Map 'id' field (for internal links)
                        href=link.get("href"),  # Don't default to "" - let validator catch missing values
                        strukturId=link.get("strukturId"),
                    ))
                except ValidationError as e:
                    logger.warning(f"Malformed link data skipped: {link}, error: {e}")
                    continue
        return result

    def load_content(self):
        """Load content from database cache."""
        # Load searchable info types from DB; fall back to hardcoded constants
        # if the content_type_config table doesn't exist yet.
        db_searchable = content_repository.get_searchable_info_types()
        self.searchable_types = db_searchable if db_searchable else ALLOWED_INFO_TYPES_SET
        logger.debug("Searchable info types (%d): %s", len(self.searchable_types), sorted(self.searchable_types))

        db_content = database_service.get_all_content()

        if not db_content:
            print("No content in database cache. Use load_from_api() to fetch content.")
            return

        self.content = []
        for item in db_content:
            # Parse JSON fields
            role_tags = self._parse_json_field(item.get("role_tags"), [])
            links = self._parse_links(item.get("links"))
            document_meta = compute_document_metadata(item)
            stored_has_text = item.get("has_text_content")

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
                path=item.get("path"),
                forst_publisert=item.get("forst_publisert"),
                sist_faglig_oppdatert=item.get("sist_faglig_oppdatert"),
                role_tags=role_tags if isinstance(role_tags, list) else [],
                links=links,
                has_text_content=(
                    bool(stored_has_text)
                    if stored_has_text is not None
                    else document_meta["has_text_content"]
                ),
                document_url=item.get("document_url") or document_meta["document_url"],
                anbefaling_fields=anbefaling_fields,
            )
            self.content.append(content_item)

        self._rebuild_lookup_dicts()
        self._content_version += 1
        print(f"Loaded {len(self.content)} content items from database cache")

        self._enrich_with_theme_page_links()

    def _rebuild_lookup_dicts(self):
        """Build content_by_id and content_by_path lookup dicts from self.content."""
        self.content_by_id = {item.id: item for item in self.content}
        self.content_by_path = {}
        for item in self.content:
            if not item.path:
                continue
            if item.path in self.content_by_path:
                existing = self.content_by_path[item.path]
                logger.warning(
                    "Duplicate content path '%s': keeping id='%s', skipping id='%s'",
                    item.path, existing.id, item.id,
                )
                continue
            self.content_by_path[item.path] = item

    def _enrich_with_theme_page_links(self):
        """Add theme page links to content items that belong to a theme page."""
        content_to_themes = content_repository.get_all_content_to_theme_pages()
        if not content_to_themes:
            return

        enriched_count = 0
        total_links = 0
        for content_id, theme_pages in content_to_themes.items():
            content_item = self.content_by_id.get(content_id)
            if not content_item:
                continue

            for tp in theme_pages:
                content_item.links.append(ContentLink(
                    rel="temaside",
                    type="temaside",
                    id=tp["id"],
                    tittel=tp["tittel"],
                    path=tp["path"],
                ))
                total_links += 1
            enriched_count += 1

        logger.info(f"Enriched {enriched_count} content items with {total_links} theme page links")

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
                role_tags = self._parse_json_field(item.get("role_tags"), [])
                links = self._parse_links(item.get("links"))
                document_meta = compute_document_metadata(item)

                content_item = ContentItem(
                    id=str(item.get("id", item.get("infoId", ""))),
                    title=item.get("tittel", ""),
                    body=item.get("tekst", ""),
                    content_type=item.get("infoType", "unknown"),
                    path=item.get("path"),
                    role_tags=role_tags if isinstance(role_tags, list) else [],
                    links=links,
                    has_text_content=document_meta["has_text_content"],
                    document_url=document_meta["document_url"],
                )
                self.content.append(content_item)

            self._rebuild_lookup_dicts()
            self._content_version += 1
            self._enrich_with_theme_page_links()
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

    def get_content_by_path(self, path: str) -> Optional[ContentItem]:
        """Get a specific content item by path."""
        return self.content_by_path.get(path)

    def reload_content(self):
        """Reload content from database cache."""
        self.load_content()


# Global instance
content_service = ContentService()
