"""
Content service for loading and managing health content.

Loads content from database cache or Helsedirektoratet API.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set
from pydantic import ValidationError
from app.entities.content import (
    ContentItem,
    ContentLink,
    AnbefalingFields,
    EhelsestandardAttachment,
    EhelsestandardFields,
)
from app.services.data.database_service import database_service
from app.services.data.document_metadata import compute_document_metadata
from app.services.repositories.content_repository import content_repository
from app.constants import ALLOWED_INFO_TYPES_SET, normalize_content_type

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

    def _parse_ehelsestandard_fields(self, item: Dict[str, Any]) -> Optional[EhelsestandardFields]:
        """Parse stored e-helsestandard fields from a content row."""
        fields_data = self._parse_json_field(item.get("ehelsestandard_fields_json"), None)
        if isinstance(fields_data, dict):
            attachments_data = fields_data.get("attachments")
        elif item.get("attachments_json") is None:
            return None
        else:
            fields_data = {}
            attachments_data = self._parse_json_field(item.get("attachments_json"), [])

        attachments: List[EhelsestandardAttachment] = []
        if isinstance(attachments_data, list):
            for attachment in attachments_data:
                if not isinstance(attachment, dict):
                    continue
                title = attachment.get("title")
                url = attachment.get("url")
                if not isinstance(title, str) or not title.strip():
                    continue
                if not isinstance(url, str) or not url.strip():
                    continue
                attachments.append(
                    EhelsestandardAttachment(
                        title=title.strip(),
                        url=url.strip(),
                        file_type=attachment.get("file_type")
                        if isinstance(attachment.get("file_type"), str)
                        else attachment.get("fileType")
                        if isinstance(attachment.get("fileType"), str)
                        else None,
                    )
                )
        return EhelsestandardFields(
            standard_id=fields_data.get("standard_id") if isinstance(fields_data.get("standard_id"), str) else None,
            standard_type=(
                fields_data.get("standard_type") if isinstance(fields_data.get("standard_type"), str) else None
            ),
            purpose_html=fields_data.get("purpose_html") if isinstance(fields_data.get("purpose_html"), str) else None,
            applies_to_html=(
                fields_data.get("applies_to_html")
                if isinstance(fields_data.get("applies_to_html"), str)
                else None
            ),
            attachments=attachments,
        )

    @staticmethod
    def _normalize_internal_path(value: Optional[str]) -> Optional[str]:
        """Normalize an internal path or href for content lookups."""
        if not value or not isinstance(value, str):
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if normalized.startswith("http://") or normalized.startswith("https://"):
            return None
        if not normalized.startswith("/"):
            normalized = f"/{normalized.lstrip('/')}"
        return normalized

    def _compute_dead_end_theme_page_flags(self) -> Dict[str, bool]:
        """Compute which theme pages are true dead ends and should get no_content."""
        theme_pages = [
            item for item in self.content
            if normalize_content_type(item.content_type) == "temaside"
        ]
        if not theme_pages:
            return {}

        theme_page_ids = [item.id for item in theme_pages]
        linked_children = content_repository.get_theme_pages_content_batch(theme_page_ids)

        direct_theme_children: Dict[str, Set[str]] = {theme_id: set() for theme_id in theme_page_ids}
        direct_non_theme_child: Dict[str, bool] = {theme_id: False for theme_id in theme_page_ids}

        for theme_page in theme_pages:
            children = linked_children.get(theme_page.id, [])
            for child in children:
                child_type = normalize_content_type(child.get("info_type"))
                child_id = child.get("id")
                if child_type == "temaside" and child_id:
                    direct_theme_children[theme_page.id].add(str(child_id))
                else:
                    direct_non_theme_child[theme_page.id] = True

            for link in theme_page.links:
                if link.rel != "barn":
                    continue

                resolved_child = None
                if link.id:
                    resolved_child = self.get_content_by_id(link.id)
                if resolved_child is None:
                    child_path = self._normalize_internal_path(link.path) or self._normalize_internal_path(link.href)
                    if child_path:
                        resolved_child = self.get_content_by_path(child_path)

                if resolved_child is not None:
                    child_type = normalize_content_type(resolved_child.content_type)
                    if child_type == "temaside":
                        direct_theme_children[theme_page.id].add(resolved_child.id)
                    else:
                        direct_non_theme_child[theme_page.id] = True
                    continue

                if normalize_content_type(link.type) != "temaside":
                    direct_non_theme_child[theme_page.id] = True

        theme_pages_by_id = {item.id: item for item in theme_pages}
        navigable_cache: Dict[str, bool] = {}
        in_progress: Set[str] = set()

        def is_navigable(theme_page_id: str) -> bool:
            if theme_page_id in navigable_cache:
                return navigable_cache[theme_page_id]

            theme_page = theme_pages_by_id.get(theme_page_id)
            if not theme_page:
                return False
            if theme_page.has_text_content or direct_non_theme_child.get(theme_page_id, False):
                navigable_cache[theme_page_id] = True
                return True
            if theme_page_id in in_progress:
                logger.warning("Detected cycle while computing dead-end theme page flags for %s", theme_page_id)
                return False

            in_progress.add(theme_page_id)
            navigable = any(
                is_navigable(child_id)
                for child_id in direct_theme_children.get(theme_page_id, set())
            )
            in_progress.remove(theme_page_id)
            navigable_cache[theme_page_id] = navigable
            return navigable

        return {
            theme_page.id: not is_navigable(theme_page.id)
            for theme_page in theme_pages
        }

    def refresh_dead_end_theme_page_flags(self, persist: bool = True) -> Dict[str, bool]:
        """Recompute dead-end theme page flags for current in-memory content."""
        dead_end_flags = self._compute_dead_end_theme_page_flags()
        for theme_page_id, is_dead_end in dead_end_flags.items():
            content_item = self.content_by_id.get(theme_page_id)
            if content_item:
                content_item.is_dead_end_theme_page = is_dead_end

        if persist and dead_end_flags:
            updated_count = content_repository.update_dead_end_theme_page_flags(dead_end_flags)
            if updated_count:
                logger.info("Updated dead-end theme page flags for %d theme pages", updated_count)

        return dead_end_flags

    def load_content(self):
        """Load content from database cache."""
        # Load searchable info types from DB; fall back to hardcoded constants
        # if the content_type_config table doesn't exist yet.
        db_searchable = content_repository.get_searchable_info_types()
        self.searchable_types = db_searchable if db_searchable else ALLOWED_INFO_TYPES_SET
        logger.debug("Searchable info types (%d): %s", len(self.searchable_types), sorted(self.searchable_types))
        content_repository.ensure_dead_end_theme_page_column()

        db_content = database_service.get_all_content()

        if not db_content:
            print("No content in database cache. Use load_from_api() to fetch content.")
            return

        self.content = []
        for item in db_content:
            raw_info_type = item.get("info_type") or ""
            canonical_info_type = normalize_content_type(raw_info_type)

            # Parse JSON fields
            role_tags = self._parse_json_field(item.get("role_tags"), [])
            links = self._parse_links(item.get("links"))
            document_meta = compute_document_metadata(item)
            stored_has_text = item.get("has_text_content")

            # Parse anbefaling-specific fields if this is an anbefaling
            anbefaling_fields = None
            if canonical_info_type == "anbefaling":
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

            ehelsestandard_fields = None
            if canonical_info_type == "ehelsestandard":
                ehelsestandard_fields = self._parse_ehelsestandard_fields(item)

            content_item = ContentItem(
                id=str(item.get("id", "")),
                title=item.get("tittel") or "",
                short_title=item.get("kort_tittel"),
                body=item.get("tekst") or "",
                content_type=canonical_info_type or "unknown",
                path=item.get("path"),
                nki_indicator_id=item.get("nki_indicator_id"),
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
                is_dead_end_theme_page=bool(item.get("is_dead_end_theme_page")),
                anbefaling_fields=anbefaling_fields,
                ehelsestandard_fields=ehelsestandard_fields,
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

            # Reload from DB so local-only fields like nki_indicator_id survive refreshes.
            self.load_content()
            self.refresh_dead_end_theme_page_flags(persist=True)
            print(f"Reloaded {len(self.content)} content items from database cache after API refresh")

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
