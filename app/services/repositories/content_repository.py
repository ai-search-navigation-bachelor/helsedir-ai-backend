"""
Repository for content operations.
"""

import json
import logging
from typing import List, Optional, Dict, Any, Set
import httpx
import mysql.connector

from app.services.repositories.base import db_pool
from app.services.data.document_metadata import compute_document_metadata_with_fallback

logger = logging.getLogger(__name__)


class ContentRepository:
    """Repository for content CRUD operations."""

    def __init__(self):
        self._warned_missing_attachments_json = False
        self._warned_missing_kort_tittel = False
        self._warned_missing_nki_indicator_id = False
        self._warned_missing_dead_end_theme_page = False

    def _serialize_json_field(self, value) -> Optional[str]:
        """Serialize a field to JSON string if needed."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _is_unknown_column_error(error: mysql.connector.Error, column_name: str) -> bool:
        """Return True when MySQL reports a missing column for the given name."""
        return getattr(error, "errno", None) == 1054 and column_name in str(error)

    def has_dead_end_theme_page_column(self) -> bool:
        """Return True when content.is_dead_end_theme_page exists."""
        conn = db_pool.get_connection()
        if not conn:
            return False

        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute("SHOW COLUMNS FROM content LIKE 'is_dead_end_theme_page'")
            return cursor.fetchone() is not None
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def ensure_dead_end_theme_page_column(self) -> bool:
        """Add the dead-end temaside column if it does not exist."""
        if self.has_dead_end_theme_page_column():
            return True

        conn = db_pool.get_connection()
        if not conn:
            return False

        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                ALTER TABLE content
                ADD COLUMN is_dead_end_theme_page TINYINT(1) NOT NULL DEFAULT 0
                AFTER document_url
                """
            )
            conn.commit()
            logger.info("Added content.is_dead_end_theme_page column")
            return True
        except mysql.connector.Error as e:
            if getattr(e, "errno", None) == 1060:
                logger.info(
                    "content.is_dead_end_theme_page column already exists; treating concurrent ALTER TABLE as success"
                )
                return True
            conn.rollback()
            logger.error("Failed to ensure content.is_dead_end_theme_page column: %s", e)
            return False
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def update_dead_end_theme_page_flags(self, flags_by_theme_page_id: Dict[str, bool]) -> int:
        """Persist dead-end flags for theme pages."""
        if not flags_by_theme_page_id:
            return 0
        if not self.has_dead_end_theme_page_column():
            if not self._warned_missing_dead_end_theme_page:
                logger.warning(
                    "Skipping dead-end theme page flag update because content.is_dead_end_theme_page is missing"
                )
                self._warned_missing_dead_end_theme_page = True
            return 0

        conn = db_pool.get_connection()
        if not conn:
            return 0

        cursor = None
        try:
            cursor = conn.cursor()
            cursor.executemany(
                """
                UPDATE content
                SET is_dead_end_theme_page = %s
                WHERE id = %s AND info_type = 'temaside'
                """,
                [
                    (1 if is_dead_end else 0, theme_page_id)
                    for theme_page_id, is_dead_end in flags_by_theme_page_id.items()
                ],
            )
            conn.commit()
            return cursor.rowcount
        except mysql.connector.Error as e:
            conn.rollback()
            logger.error("Failed to update dead-end theme page flags: %s", e)
            return 0
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def _execute_content_upsert(
        self,
        cursor,
        *,
        content_id,
        title,
        short_title,
        text,
        info_type,
        koder_json,
        role_tags_json,
        links_json,
        forst_publisert,
        sist_faglig_oppdatert,
        path,
        has_text_content,
        document_url,
        nki_indicator_id,
        attachments_json,
        ehelsestandard_fields_json,
    ) -> None:
        """Insert/update content, with compatibility fallback for older schemas."""
        def execute_upsert(
            *,
            include_short_title: bool,
            include_nki_indicator_id: bool,
            include_attachments_json: bool,
            include_ehelsestandard_fields_json: bool,
        ) -> None:
            insert_columns = [
                "id",
                "tittel",
            ]
            values = [
                content_id,
                title,
            ]
            update_assignments = [
                "tittel = VALUES(tittel)",
            ]

            if include_short_title:
                insert_columns.append("kort_tittel")
                values.append(short_title)
                update_assignments.append(
                    "kort_tittel = COALESCE(NULLIF(VALUES(kort_tittel), ''), kort_tittel)"
                )

            insert_columns.extend(
                [
                    "tekst",
                    "info_type",
                    "koder",
                    "role_tags",
                    "links",
                    "forst_publisert",
                    "sist_faglig_oppdatert",
                    "path",
                    "has_text_content",
                    "document_url",
                ]
            )
            values.extend(
                [
                    text,
                    info_type,
                    koder_json,
                    role_tags_json,
                    links_json,
                    forst_publisert,
                    sist_faglig_oppdatert,
                    path,
                    has_text_content,
                    document_url,
                ]
            )
            update_assignments.extend(
                [
                    "tekst = VALUES(tekst)",
                    "info_type = VALUES(info_type)",
                    "koder = COALESCE(VALUES(koder), koder)",
                    "role_tags = COALESCE(VALUES(role_tags), role_tags)",
                    "links = COALESCE(VALUES(links), links)",
                    "forst_publisert = COALESCE(VALUES(forst_publisert), forst_publisert)",
                    "sist_faglig_oppdatert = COALESCE(VALUES(sist_faglig_oppdatert), sist_faglig_oppdatert)",
                    "has_text_content = COALESCE(VALUES(has_text_content), has_text_content)",
                    "document_url = COALESCE(VALUES(document_url), document_url)",
                ]
            )

            if include_nki_indicator_id:
                insert_columns.append("nki_indicator_id")
                values.append(nki_indicator_id)
                update_assignments.append(
                    "nki_indicator_id = COALESCE(VALUES(nki_indicator_id), nki_indicator_id)"
                )

            if include_attachments_json:
                insert_columns.append("attachments_json")
                values.append(attachments_json)
                update_assignments.append(
                    "attachments_json = COALESCE(VALUES(attachments_json), attachments_json)"
                )

            if include_ehelsestandard_fields_json:
                insert_columns.append("ehelsestandard_fields_json")
                values.append(ehelsestandard_fields_json)
                update_assignments.append(
                    "ehelsestandard_fields_json = COALESCE(VALUES(ehelsestandard_fields_json), ehelsestandard_fields_json)"
                )

            update_assignments.append("path = VALUES(path)")
            placeholders = ", ".join(["%s"] * len(insert_columns))
            update_sql = ",\n                    ".join(update_assignments)

            cursor.execute(
                f"""
                INSERT INTO content (
                    {", ".join(insert_columns)}
                )
                VALUES ({placeholders})
                ON DUPLICATE KEY UPDATE
                    {update_sql}
                """,
                tuple(values),
            )

        try:
            execute_upsert(
                include_short_title=True,
                include_nki_indicator_id=True,
                include_attachments_json=True,
                include_ehelsestandard_fields_json=True,
            )
        except mysql.connector.Error as e:
            missing_kort_tittel = self._is_unknown_column_error(e, "kort_tittel")
            missing_nki_indicator_id = self._is_unknown_column_error(e, "nki_indicator_id")
            missing_attachments_json = self._is_unknown_column_error(e, "attachments_json")
            missing_ehelsestandard_fields_json = self._is_unknown_column_error(
                e, "ehelsestandard_fields_json"
            )
            if not (
                missing_kort_tittel
                or missing_nki_indicator_id
                or missing_attachments_json
                or missing_ehelsestandard_fields_json
            ):
                raise

            if missing_kort_tittel and not self._warned_missing_kort_tittel:
                logger.warning(
                    "content schema is missing kort_tittel; retrying without it until migration is applied: %s",
                    e,
                )
                self._warned_missing_kort_tittel = True

            if missing_nki_indicator_id and not self._warned_missing_nki_indicator_id:
                logger.warning(
                    "content schema is missing nki_indicator_id; retrying without it until migration is applied: %s",
                    e,
                )
                self._warned_missing_nki_indicator_id = True

            if (missing_attachments_json or missing_ehelsestandard_fields_json) and not self._warned_missing_attachments_json:
                logger.warning(
                    "content schema is missing ehelsestandard metadata columns; retrying without them until migration is applied: %s",
                    e,
                )
                self._warned_missing_attachments_json = True

            execute_upsert(
                include_short_title=not missing_kort_tittel,
                include_nki_indicator_id=not missing_nki_indicator_id,
                include_attachments_json=not missing_attachments_json,
                include_ehelsestandard_fields_json=not missing_ehelsestandard_fields_json,
            )

    def _cache_anbefaling_details(self, content_id: str, content: Dict[str, Any], cursor) -> bool:
        """
        Cache anbefaling-specific fields in anbefaling_details table.

        Args:
            content_id: The content ID
            content: Full content dict from API (should have 'data' key)
            cursor: Active database cursor

        Returns:
            True if saved successfully
        """
        try:
            data = content.get("data", {})
            nokkel_info = data.get("nokkelInfo", {})

            cursor.execute(
                """
                INSERT INTO anbefaling_details
                (content_id, praktisk, rasjonale, fordeler_ulemper, verdier_preferanser,
                 kvalitet_dokumentasjon, ressurshensyn, styrke)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    praktisk = COALESCE(VALUES(praktisk), praktisk),
                    rasjonale = COALESCE(VALUES(rasjonale), rasjonale),
                    fordeler_ulemper = COALESCE(VALUES(fordeler_ulemper), fordeler_ulemper),
                    verdier_preferanser = COALESCE(VALUES(verdier_preferanser), verdier_preferanser),
                    kvalitet_dokumentasjon = COALESCE(VALUES(kvalitet_dokumentasjon), kvalitet_dokumentasjon),
                    ressurshensyn = COALESCE(VALUES(ressurshensyn), ressurshensyn),
                    styrke = COALESCE(VALUES(styrke), styrke)
                """,
                (
                    content_id,
                    data.get("praktisk"),
                    data.get("rasjonale"),
                    nokkel_info.get("fordelerogulemper"),
                    nokkel_info.get("verdierogpreferanser"),
                    nokkel_info.get("kvalitetdokumentasjon"),
                    nokkel_info.get("ressurshensyn"),
                    data.get("styrke"),
                ),
            )
            return True
        except mysql.connector.Error as e:
            print(f"Error caching anbefaling details for {content_id}: {e}")
            return False

    def cache_content(self, content: Dict[str, Any]) -> bool:
        """
        Cache a content item from Helsedir API or theme page.

        Args:
            content: Content dict with id, tittel, tekst, koder, role_tags, etc.
                     For theme pages: also includes path, info_type='temaside'
                     For anbefaling: also includes data.praktisk, data.rasjonale, etc.

        Returns:
            True if cached successfully
        """
        conn = db_pool.get_connection()
        if not conn:
            return False

        try:
            cursor = conn.cursor()

            koder_json = self._serialize_json_field(content.get("koder"))
            role_tags_json = self._serialize_json_field(content.get("role_tags"))
            links_json = self._serialize_json_field(content.get("links"))
            info_type = content.get("info_type") or content.get("infoType") or content.get("dokumentType")
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                document_meta = compute_document_metadata_with_fallback(content, client=client)
            has_text_content = int(
                content["has_text_content"] if "has_text_content" in content else document_meta["has_text_content"]
            )
            document_url = (
                content["document_url"] if "document_url" in content else document_meta["document_url"]
            )
            attachments_json = self._serialize_json_field(content.get("attachments_json"))
            ehelsestandard_fields_json = self._serialize_json_field(content.get("ehelsestandard_fields_json"))
            self._execute_content_upsert(
                cursor,
                content_id=content.get("id"),
                title=content.get("tittel"),
                short_title=content.get("kortTittel") or content.get("kort_tittel"),
                text=content.get("tekst"),
                info_type=info_type,
                koder_json=koder_json,
                role_tags_json=role_tags_json,
                links_json=links_json,
                forst_publisert=content.get("forstPublisert") or content.get("forst_publisert") or None,
                sist_faglig_oppdatert=content.get("sistFagligOppdatert") or content.get("sist_faglig_oppdatert") or None,
                path=content.get("path"),
                has_text_content=has_text_content,
                document_url=document_url,
                nki_indicator_id=content.get("nki_indicator_id"),
                attachments_json=attachments_json,
                ehelsestandard_fields_json=ehelsestandard_fields_json,
            )

            # If anbefaling, also save anbefaling-specific details
            if info_type == "anbefaling":
                details_ok = self._cache_anbefaling_details(content.get("id"), content, cursor)
                if not details_ok:
                    conn.rollback()
                    return False

            conn.commit()
            return True
        except mysql.connector.Error as e:
            print(f"Error caching content: {e}")
            return False
        finally:
            cursor.close()
            conn.close()

    def cache_content_batch(self, contents: List[Dict[str, Any]]) -> int:
        """
        Cache multiple content items.

        Args:
            contents: List of content dicts

        Returns:
            Number of items cached
        """
        conn = db_pool.get_connection()
        if not conn:
            return 0

        try:
            cursor = conn.cursor()
            cached = 0
            resolved_contents = []

            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                for content in contents:
                    document_meta = compute_document_metadata_with_fallback(content, client=client)
                    resolved_contents.append((content, document_meta))

            for content, document_meta in resolved_contents:
                try:
                    koder_json = self._serialize_json_field(content.get("koder"))
                    role_tags_json = self._serialize_json_field(content.get("role_tags"))
                    links_json = self._serialize_json_field(content.get("links"))
                    info_type = content.get("info_type") or content.get("infoType") or content.get("dokumentType")
                    has_text_content = int(
                        content["has_text_content"] if "has_text_content" in content else document_meta["has_text_content"]
                    )
                    document_url = (
                        content["document_url"] if "document_url" in content else document_meta["document_url"]
                    )
                    attachments_json = self._serialize_json_field(content.get("attachments_json"))
                    ehelsestandard_fields_json = self._serialize_json_field(
                        content.get("ehelsestandard_fields_json")
                    )
                    self._execute_content_upsert(
                        cursor,
                        content_id=content.get("id"),
                        title=content.get("tittel"),
                        short_title=content.get("kortTittel") or content.get("kort_tittel"),
                        text=content.get("tekst"),
                        info_type=info_type,
                        koder_json=koder_json,
                        role_tags_json=role_tags_json,
                        links_json=links_json,
                        forst_publisert=content.get("forstPublisert") or content.get("forst_publisert"),
                        sist_faglig_oppdatert=content.get("sistFagligOppdatert") or content.get("sist_faglig_oppdatert"),
                        path=content.get("path"),
                        has_text_content=has_text_content,
                        document_url=document_url,
                        nki_indicator_id=content.get("nki_indicator_id"),
                        attachments_json=attachments_json,
                        ehelsestandard_fields_json=ehelsestandard_fields_json,
                    )

                    # If anbefaling, also save anbefaling-specific details
                    if info_type == "anbefaling":
                        self._cache_anbefaling_details(content.get("id"), content, cursor)

                    cached += 1
                except mysql.connector.Error as e:
                    print(f"Error caching content {content.get('id')}: {e}")
                    continue

            conn.commit()
            return cached
        except mysql.connector.Error as e:
            print(f"Error caching content batch: {e}")
            return 0
        finally:
            cursor.close()
            conn.close()

    def get_content(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached content item by ID.

        For anbefaling content, also includes anbefaling_details fields.
        """
        conn = db_pool.get_connection()
        if not conn:
            return None

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT c.*,
                       a.praktisk, a.rasjonale, a.fordeler_ulemper,
                       a.verdier_preferanser, a.kvalitet_dokumentasjon,
                       a.ressurshensyn, a.styrke
                FROM content c
                LEFT JOIN anbefaling_details a ON c.id = a.content_id
                WHERE c.id = %s
                """,
                (content_id,)
            )
            return cursor.fetchone()
        except mysql.connector.Error as e:
            print(f"Error getting content: {e}")
            return None
        finally:
            cursor.close()
            conn.close()

    def get_all_content(self) -> List[Dict[str, Any]]:
        """
        Get all cached content.

        For anbefaling content, also includes anbefaling_details fields.
        """
        conn = db_pool.get_connection()
        if not conn:
            return []

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT c.*,
                       a.praktisk, a.rasjonale, a.fordeler_ulemper,
                       a.verdier_preferanser, a.kvalitet_dokumentasjon,
                       a.ressurshensyn, a.styrke
                FROM content c
                LEFT JOIN anbefaling_details a ON c.id = a.content_id
                """
            )
            return cursor.fetchall()
        except mysql.connector.Error as e:
            print(f"Error getting all content: {e}")
            return []
        finally:
            cursor.close()
            conn.close()

    def get_content_count(self) -> int:
        """Get total number of cached content items."""
        conn = db_pool.get_connection()
        if not conn:
            return 0

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM content")
            result = cursor.fetchone()
            return result[0] if result else 0
        except mysql.connector.Error as e:
            print(f"Error getting content count: {e}")
            return 0
        finally:
            cursor.close()
            conn.close()

    def get_all_content_ids(self) -> List[str]:
        """Get all content IDs from database (for deduplication/skip checks)."""
        conn = db_pool.get_connection()
        if not conn:
            return []

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM content")
            results = cursor.fetchall()
            return [row[0] for row in results]
        except mysql.connector.Error as e:
            print(f"Error getting content IDs: {e}")
            return []
        finally:
            cursor.close()
            conn.close()

    def check_content_exists_batch(self, content_ids: List[str]) -> Set[str]:
        """
        Check which content IDs exist in the database (batch operation).

        Args:
            content_ids: List of content IDs to check

        Returns:
            Set of content IDs that exist in database
        """
        if not content_ids:
            return set()

        conn = db_pool.get_connection()
        if not conn:
            return set()

        cursor = None
        try:
            cursor = conn.cursor()

            # Use IN clause for batch lookup
            placeholders = ','.join(['%s'] * len(content_ids))
            query = f"SELECT id FROM content WHERE id IN ({placeholders})"
            cursor.execute(query, tuple(content_ids))
            results = cursor.fetchall()

            return {row[0] for row in results}

        except mysql.connector.Error as e:
            print(f"Error checking content existence: {e}")
            return set()
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def get_content_stats_by_type(self) -> List[Dict[str, Any]]:
        """Get content count grouped by info_type."""
        conn = db_pool.get_connection()
        if not conn:
            return []

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT info_type, COUNT(*) as count
                FROM content
                GROUP BY info_type
                ORDER BY count DESC
                """
            )
            return cursor.fetchall()
        except mysql.connector.Error as e:
            print(f"Error getting content stats by type: {e}")
            return []
        finally:
            cursor.close()
            conn.close()

    def get_all_content_to_theme_pages(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all content_id -> theme page mappings in a single batch query.

        Returns:
            Dict mapping content_id -> list of {id, tittel, path} for each theme page
        """
        conn = db_pool.get_connection()
        if not conn:
            return {}

        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT tpc.content_id, tp.id AS theme_page_id, tp.tittel, tp.path
                FROM theme_page_content tpc
                INNER JOIN content tp ON tpc.theme_page_id = tp.id
                WHERE tp.info_type = 'temaside'
                ORDER BY tpc.content_id, tp.tittel
                """
            )
            rows = cursor.fetchall()

            result: Dict[str, List[Dict[str, Any]]] = {}
            for row in rows:
                content_id = row["content_id"]
                if content_id not in result:
                    result[content_id] = []
                result[content_id].append({
                    "id": row["theme_page_id"],
                    "tittel": row["tittel"],
                    "path": row["path"],
                })
        except mysql.connector.Error as e:
            logger.error("Error getting content to theme pages: %s", e)
            return {}
        else:
            return result
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def get_searchable_info_types(self) -> Set[str]:
        """
        Return the set of info_type values marked searchable=1 in content_type_config.
        Falls back to an empty set if the table doesn't exist yet.
        """
        conn = db_pool.get_connection()
        if not conn:
            return set()

        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT info_type FROM content_type_config WHERE searchable = 1"
            )
            return {row[0] for row in cursor.fetchall()}
        except mysql.connector.Error:
            # Table may not exist on older installs — return empty set so caller
            # can decide on a fallback.
            logger.debug("Could not read allowed info types from DB", exc_info=True)
            return set()
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def get_theme_page_content(self, theme_page_id: str) -> List[Dict[str, Any]]:
        """
        Get all content linked to a theme page via the junction table.

        Args:
            theme_page_id: ID of the theme page

        Returns:
            List of content items with their metadata
        """
        conn = db_pool.get_connection()
        if not conn:
            return []

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT c.* FROM content c
                INNER JOIN theme_page_content tpc ON c.id = tpc.content_id
                WHERE tpc.theme_page_id = %s
                ORDER BY tpc.display_order, c.tittel
                """,
                (theme_page_id,)
            )
            return cursor.fetchall()
        except mysql.connector.Error as e:
            print(f"Error getting theme page content: {e}")
            return []
        finally:
            cursor.close()
            conn.close()

    def get_all_theme_pages_content(self) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Pre-load ALL theme page children in a single query (for startup caching).

        Returns:
            Dict mapping theme_page_id -> list of linked content items (lightweight columns only).
            Returns None when the lookup could not be completed.
        """
        conn = db_pool.get_connection()
        if not conn:
            return None

        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT tpc.theme_page_id,
                       c.id, c.tittel, c.kort_tittel, c.info_type,
                       c.path, c.has_text_content, c.document_url
                FROM content c
                INNER JOIN theme_page_content tpc ON c.id = tpc.content_id
                ORDER BY tpc.theme_page_id, tpc.display_order, c.tittel
            """)
            rows = cursor.fetchall()

            result: Dict[str, List[Dict[str, Any]]] = {}
            for row in rows:
                theme_id = row.pop('theme_page_id')
                result.setdefault(theme_id, []).append(row)

            return result
        except mysql.connector.Error as e:
            logger.error("Error pre-loading all theme page children: %s", e)
            return None
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def get_theme_pages_content_batch(self, theme_page_ids: List[str]) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Get all content linked to multiple theme pages in a single query (batch operation).

        Args:
            theme_page_ids: List of theme page IDs

        Returns:
            Dict mapping theme_page_id -> list of linked content items.
            Returns None when the lookup could not be completed.
        """
        if not theme_page_ids:
            return {}

        conn = db_pool.get_connection()
        if not conn:
            return None

        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)

            # Use IN clause to fetch all at once — select only columns used by
            # _populate_theme_page_children to avoid transferring large tekst blobs.
            placeholders = ','.join(['%s'] * len(theme_page_ids))
            query = f"""
                SELECT tpc.theme_page_id,
                       c.id, c.tittel, c.kort_tittel, c.info_type,
                       c.path, c.has_text_content, c.document_url
                FROM content c
                INNER JOIN theme_page_content tpc ON c.id = tpc.content_id
                WHERE tpc.theme_page_id IN ({placeholders})
                ORDER BY tpc.theme_page_id, tpc.display_order, c.tittel
            """
            cursor.execute(query, tuple(theme_page_ids))
            rows = cursor.fetchall()

            # Group by theme_page_id
            result = {theme_id: [] for theme_id in theme_page_ids}
            for row in rows:
                theme_id = row.pop('theme_page_id')
                if theme_id in result:
                    result[theme_id].append(row)

            return result

        except mysql.connector.Error as e:
            print(f"Error getting theme pages content batch: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def get_non_empty_theme_page_ids(self, theme_page_ids: List[str]) -> Optional[Set[str]]:
        """
        Return theme-page IDs that have at least one linked child row.

        Returns:
            Set of non-empty theme-page IDs when the query succeeds.
            None when the lookup could not be completed.
        """
        if not theme_page_ids:
            return set()

        conn = db_pool.get_connection()
        if not conn:
            return None

        cursor = None
        try:
            cursor = conn.cursor()
            placeholders = ",".join(["%s"] * len(theme_page_ids))
            query = f"""
                SELECT DISTINCT theme_page_id
                FROM theme_page_content
                WHERE theme_page_id IN ({placeholders})
            """
            cursor.execute(query, tuple(theme_page_ids))
            return {row[0] for row in cursor.fetchall()}
        except mysql.connector.Error as e:
            logger.error("Error getting non-empty theme page ids: %s", e)
            return None
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def get_theme_pages(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all theme pages, optionally filtered by category.

        Args:
            category: Optional category slug to filter by (e.g., 'forebygging-diagnose-og-behandling')

        Returns:
            List of theme page items ordered by title
        """
        conn = db_pool.get_connection()
        if not conn:
            return []

        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)

            if category:
                # Filter by category using path prefix
                cursor.execute(
                    """
                    SELECT * FROM content
                    WHERE info_type = 'temaside'
                    AND path LIKE %s
                    ORDER BY tittel
                    """,
                    (f"/{category}%",)
                )
            else:
                # Get all theme pages
                cursor.execute(
                    """
                    SELECT * FROM content
                    WHERE info_type = 'temaside'
                    ORDER BY tittel
                    """
                )

            return cursor.fetchall()
        except mysql.connector.Error as e:
            print(f"Error getting theme pages: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()


# Global instance
content_repository = ContentRepository()
