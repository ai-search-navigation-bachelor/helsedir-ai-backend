"""
Repository for search logging operations.
"""

from typing import List, Optional, Dict, Any
import mysql.connector

from app.services.repositories.base import db_pool
from app.services.repositories.stats_repository import stats_repository


class SearchRepository:
    """Repository for search and click logging."""

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
            cursor.execute(
                """
                INSERT INTO search_logs (search_id, query, role, session_id, user_id)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE query = VALUES(query)
                """,
                (search_id, query, role, session_id, user_id),
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

                cursor.execute(
                    """
                    INSERT INTO search_results_shown (
                        search_id, content_id, position, score,
                        semantic_similarity, keyword_score_total,
                        exact_title_proportion, full_coverage_proportion, title_keyword_proportion,
                        type_match, role_match, code_match_count, lis_match, maalgruppe_match
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        search_id,
                        content_id,
                        result.get("position"),
                        result.get("score"),
                        result.get("semantic_similarity"),
                        result.get("keyword_score_total"),
                        result.get("exact_title_proportion"),
                        result.get("full_coverage_proportion"),
                        result.get("title_keyword_proportion"),
                        result.get("type_match"),
                        result.get("role_match"),
                        result.get("code_match_count", 0),
                        result.get("lis_match", 0),
                        result.get("maalgruppe_match", 0),
                    ),
                )

            # Also update content_stats impressions
            content_ids = [r.get("content_id") for r in results if r.get("content_id")]
            stats_repository.record_impressions_batch(content_ids)

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

            # Also update content_stats clicks
            stats_repository.record_click(content_id)

            conn.commit()
            return True
        except mysql.connector.Error as e:
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


# Global instance
search_repository = SearchRepository()
