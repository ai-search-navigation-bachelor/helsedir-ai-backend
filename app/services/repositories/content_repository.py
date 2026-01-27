"""
Repository for content operations.
"""

import json
from typing import List, Optional, Dict, Any
import mysql.connector

from app.services.repositories.base import db_pool


class ContentRepository:
    """Repository for content CRUD operations."""

    def _serialize_json_field(self, value) -> Optional[str]:
        """Serialize a field to JSON string if needed."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    def cache_content(self, content: Dict[str, Any]) -> bool:
        """
        Cache a content item from Helsedir API.

        Args:
            content: Content dict with id, tittel, tekst, koder, maalgruppe, etc.

        Returns:
            True if cached successfully
        """
        conn = db_pool.get_connection()
        if not conn:
            return False

        try:
            cursor = conn.cursor()

            koder_json = self._serialize_json_field(content.get("koder"))
            maalgruppe_json = self._serialize_json_field(content.get("maalgruppe"))
            links_json = self._serialize_json_field(content.get("links"))
            info_type = content.get("infoType") or content.get("dokumentType")

            cursor.execute(
                """
                INSERT INTO content (id, tittel, tekst, info_type, koder, maalgruppe, links)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    tittel = VALUES(tittel),
                    tekst = VALUES(tekst),
                    info_type = VALUES(info_type),
                    koder = COALESCE(VALUES(koder), koder),
                    maalgruppe = COALESCE(VALUES(maalgruppe), maalgruppe),
                    links = COALESCE(VALUES(links), links)
                """,
                (
                    content.get("id"),
                    content.get("tittel"),
                    content.get("tekst"),
                    info_type,
                    koder_json,
                    maalgruppe_json,
                    links_json,
                ),
            )
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

            for content in contents:
                try:
                    koder_json = self._serialize_json_field(content.get("koder"))
                    maalgruppe_json = self._serialize_json_field(content.get("maalgruppe"))
                    links_json = self._serialize_json_field(content.get("links"))
                    info_type = content.get("infoType") or content.get("dokumentType")

                    cursor.execute(
                        """
                        INSERT INTO content (id, tittel, tekst, info_type, koder, maalgruppe, links)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            tittel = VALUES(tittel),
                            tekst = VALUES(tekst),
                            info_type = VALUES(info_type),
                            koder = COALESCE(VALUES(koder), koder),
                            maalgruppe = COALESCE(VALUES(maalgruppe), maalgruppe),
                            links = COALESCE(VALUES(links), links)
                        """,
                        (
                            content.get("id"),
                            content.get("tittel"),
                            content.get("tekst"),
                            info_type,
                            koder_json,
                            maalgruppe_json,
                            links_json,
                        ),
                    )
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
        """Get a cached content item by ID."""
        conn = db_pool.get_connection()
        if not conn:
            return None

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM content WHERE id = %s", (content_id,))
            return cursor.fetchone()
        except mysql.connector.Error as e:
            print(f"Error getting content: {e}")
            return None
        finally:
            cursor.close()
            conn.close()

    def get_all_content(self) -> List[Dict[str, Any]]:
        """Get all cached content."""
        conn = db_pool.get_connection()
        if not conn:
            return []

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM content")
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


# Global instance
content_repository = ContentRepository()
