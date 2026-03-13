"""
Repository for search logging operations.
"""

import logging
from typing import List, Optional, Dict, Any
import mysql.connector

from app.services.repositories.base import db_pool

logger = logging.getLogger(__name__)


class SearchRepository:
    """Repository for search and click logging."""

    def __init__(self):
        self._warned_missing_search_log_columns = False
        self._warned_legacy_results_schema = False
        self._warned_missing_content_stats = False

    @staticmethod
    def _is_unknown_column_error(error: mysql.connector.Error, column_name: str) -> bool:
        """Return True when MySQL reports a missing column for the given name."""
        return getattr(error, "errno", None) == 1054 and column_name in str(error)

    @staticmethod
    def _is_missing_table_error(error: mysql.connector.Error, table_name: str) -> bool:
        """Return True when MySQL reports a missing table for the given name."""
        return getattr(error, "errno", None) == 1146 and table_name in str(error)

    def log_search(
        self,
        search_id: str,
        query: str,
        role: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Log a new search event to search_logs.

        Args:
            search_id: Unique ID for this search (UUID)
            query: The search query
            role: Optional user role
            session_id: Optional session ID
            user_id: Optional user ID

        Returns:
            True if logged successfully
        """
        conn = db_pool.get_connection()
        if not conn:
            return False

        cursor = None
        try:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO search_logs (search_id, query, role, session_id, user_id)
                    VALUES (%s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE query = VALUES(query)
                    """,
                    (search_id, query, role, session_id, user_id),
                )
            except mysql.connector.Error as e:
                if not (
                    self._is_unknown_column_error(e, "session_id")
                    or self._is_unknown_column_error(e, "user_id")
                ):
                    raise

                if not self._warned_missing_search_log_columns:
                    logger.warning(
                        "search_logs schema is missing session/user columns; "
                        "falling back to legacy insert until migration is applied: %s",
                        e,
                    )
                    self._warned_missing_search_log_columns = True
                cursor.execute(
                    """
                    INSERT INTO search_logs (search_id, query, role)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE query = VALUES(query)
                    """,
                    (search_id, query, role),
                )
            conn.commit()
            return True
        except mysql.connector.Error as e:
            print(f"Error logging search: {e}")
            return False
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def get_search_by_id(self, search_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a search record by search_id.

        Returns:
            Dict with query and role, or None if not found
        """
        conn = db_pool.get_connection()
        if not conn:
            return None

        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT query, role FROM search_logs WHERE search_id = %s",
                (search_id,)
            )
            return cursor.fetchone()
        except mysql.connector.Error as e:
            print(f"Error getting search: {e}")
            return None
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def log_search_results(
        self,
        search_id: str,
        results: List[Dict[str, Any]],
    ) -> bool:
        """
        Log search results shown with ML features.

        Args:
            search_id: The search ID
            results: List of results with features

        Returns:
            True if logged successfully
        """
        conn = db_pool.get_connection()
        if not conn:
            return False

        cursor = None
        try:
            cursor = conn.cursor()

            for result in results:
                # Skip results missing content_id to avoid DB errors
                content_id = result.get("content_id")
                if not content_id:
                    continue

                try:
                    cursor.execute(
                        """
                        INSERT INTO search_results_shown (
                            search_id, content_id, position, score,
                            semantic_score, bm25_score, rrf_score,
                            type_match, role_match, maalgruppe_match
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            search_id,
                            content_id,
                            result.get("position"),
                            result.get("score"),
                            result.get("semantic_score"),
                            result.get("bm25_score"),
                            result.get("rrf_score"),
                            result.get("type_match"),
                            result.get("role_match"),
                            result.get("maalgruppe_match", 0),
                        ),
                    )
                except mysql.connector.Error as e:
                    if not (
                        self._is_unknown_column_error(e, "type_match")
                        or self._is_unknown_column_error(e, "role_match")
                        or self._is_unknown_column_error(e, "maalgruppe_match")
                        or self._is_unknown_column_error(e, "bm25_score")
                        or self._is_unknown_column_error(e, "rrf_score")
                    ):
                        raise

                    if not self._warned_legacy_results_schema:
                        logger.warning(
                            "search_results_shown schema is outdated; "
                            "falling back to legacy insert until migration is applied: %s",
                            e,
                        )
                        self._warned_legacy_results_schema = True
                    cursor.execute(
                        """
                        INSERT INTO search_results_shown (
                            search_id, content_id, position, score, semantic_score
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            search_id,
                            content_id,
                            result.get("position"),
                            result.get("score"),
                            result.get("semantic_score"),
                        ),
                    )

            # Also update content_stats impressions (inline to avoid nested connection)
            seen_ids = set()
            for r in results:
                content_id = r.get("content_id")
                if not content_id or content_id in seen_ids:
                    continue
                seen_ids.add(content_id)
                try:
                    cursor.execute(
                        """
                        INSERT INTO content_stats (content_id, impressions, clicks)
                        VALUES (%s, 1, 0)
                        ON DUPLICATE KEY UPDATE impressions = impressions + 1
                        """,
                        (content_id,),
                    )
                except mysql.connector.Error as e:
                    if self._is_missing_table_error(e, "content_stats"):
                        if not self._warned_missing_content_stats:
                            logger.warning(
                                "content_stats table is missing; skipping impression updates until migration is applied: %s",
                                e,
                            )
                            self._warned_missing_content_stats = True
                        break
                    logger.warning("Failed to update content_stats for %s: %s", content_id, e)

            conn.commit()
            return True
        except mysql.connector.Error as e:
            print(f"Error logging search results: {e}")
            return False
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def log_click(self, search_id: str, content_id: str) -> bool:
        """
        Log a click event.

        Args:
            search_id: The search_id this click belongs to
            content_id: The clicked content ID

        Returns:
            True if logged successfully
        """
        conn = db_pool.get_connection()
        if not conn:
            return False

        cursor = None
        try:
            cursor = conn.cursor()

            # Look up position from search_results_shown
            cursor.execute(
                """
                SELECT position FROM search_results_shown
                WHERE search_id = %s AND content_id = %s
                LIMIT 1
                """,
                (search_id, content_id),
            )
            result = cursor.fetchone()
            position = result[0] if result else None

            # Insert click log
            cursor.execute(
                """
                INSERT INTO click_logs (search_id, content_id, position)
                VALUES (%s, %s, %s)
                """,
                (search_id, content_id, position),
            )

            # Also update content_stats clicks (inline to avoid nested connection)
            cursor.execute(
                """
                INSERT INTO content_stats (content_id, impressions, clicks)
                VALUES (%s, 0, 1)
                ON DUPLICATE KEY UPDATE clicks = clicks + 1
                """,
                (content_id,),
            )

            conn.commit()
            return True
        except mysql.connector.Error as e:
            if self._is_missing_table_error(e, "content_stats"):
                if not self._warned_missing_content_stats:
                    logger.warning(
                        "content_stats table is missing; skipping click updates until migration is applied: %s",
                        e,
                    )
                    self._warned_missing_content_stats = True
                return True
            print(f"Error logging click: {e}")
            return False
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def get_search_count(self) -> int:
        """Get total number of logged searches."""
        conn = db_pool.get_connection()
        if not conn:
            return 0

        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM search_logs")
            result = cursor.fetchone()
            return result[0] if result else 0
        except mysql.connector.Error as e:
            print(f"Error getting search count: {e}")
            return 0
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def get_click_count(self) -> int:
        """Get total number of logged clicks."""
        conn = db_pool.get_connection()
        if not conn:
            return 0

        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM click_logs")
            result = cursor.fetchone()
            return result[0] if result else 0
        except mysql.connector.Error as e:
            print(f"Error getting click count: {e}")
            return 0
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def get_max_position_for_search(self, search_id: str) -> int:
        """
        Get the maximum position already logged for a search_id.

        Returns:
            Max position, or 0 if no results logged yet
        """
        conn = db_pool.get_connection()
        if not conn:
            return 0

        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COALESCE(MAX(position), 0) FROM search_results_shown WHERE search_id = %s",
                (search_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else 0
        except mysql.connector.Error as e:
            print(f"Error getting max position: {e}")
            return 0
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def get_logged_content_ids_for_search(self, search_id: str) -> set:
        """
        Get content_ids already logged for a search_id.

        Returns:
            Set of content_ids already logged
        """
        conn = db_pool.get_connection()
        if not conn:
            return set()

        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT content_id FROM search_results_shown WHERE search_id = %s",
                (search_id,)
            )
            return {row[0] for row in cursor.fetchall()}
        except mysql.connector.Error as e:
            print(f"Error getting logged content_ids: {e}")
            return set()
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()


# Global instance
search_repository = SearchRepository()
