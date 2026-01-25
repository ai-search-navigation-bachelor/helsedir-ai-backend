"""
Database service for MySQL operations.

Handles caching of Helsedir content and tracking content statistics
for ML training (impressions, clicks, CTR).
"""

import json
from typing import List, Optional, Dict, Any
import mysql.connector
from mysql.connector import pooling

from app.config import settings


class DatabaseService:
    """Service for MySQL database operations."""

    def __init__(self):
        self.pool = None
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize connection pool."""
        try:
            self.pool = pooling.MySQLConnectionPool(
                pool_name="helsedir_pool",
                pool_size=5,
                host=settings.mysql_host,
                port=settings.mysql_port,
                user=settings.mysql_user,
                password=settings.mysql_password,
                database=settings.mysql_database,
                charset="utf8mb4",
                collation="utf8mb4_unicode_ci",
            )
            print(f"Database pool initialized: {settings.mysql_database}")
        except mysql.connector.Error as e:
            print(f"Error initializing database pool: {e}")
            self.pool = None

    def _get_connection(self):
        """Get a connection from the pool."""
        if self.pool is None:
            return None
        try:
            return self.pool.get_connection()
        except mysql.connector.Error as e:
            print(f"Error getting connection: {e}")
            return None

    def is_connected(self) -> bool:
        """Check if database is available."""
        conn = self._get_connection()
        if conn:
            conn.close()
            return True
        return False

    # ==================== Content Operations ====================

    def cache_content(self, content: Dict[str, Any]) -> bool:
        """
        Cache a content item from Helsedir API.

        Args:
            content: Content dict with id, tittel, tekst, koder, maalgruppe, etc.

        Returns:
            True if cached successfully
        """
        conn = self._get_connection()
        if not conn:
            return False

        try:
            cursor = conn.cursor()

            # Handle JSON fields - may already be JSON strings from API
            koder = content.get("koder")
            if koder is None:
                koder_json = None
            elif isinstance(koder, str):
                koder_json = koder  # Already a JSON string
            else:
                koder_json = json.dumps(koder, ensure_ascii=False)

            maalgruppe = content.get("maalgruppe")
            if maalgruppe is None:
                maalgrupp# The `get_ltr_training_rows()` method is a part of the database service
                # being tested in the script. This method is used to retrieve training rows
                # for Learning to Rank (LTR) models. In the context of a ranking model, LTR
                # is a machine learning technique that is used to improve the ranking of
                # search results based on relevance feedback.
                e_json = None
            elif isinstance(maalgruppe, str):
                maalgruppe_json = maalgruppe  # Already a JSON string
            else:
                maalgruppe_json = json.dumps(maalgruppe, ensure_ascii=False)

            # Get info_type - API uses 'infoType' or 'dokumentType' depending on mode
            info_type = content.get("infoType") or content.get("dokumentType")

            cursor.execute(
                """
                INSERT INTO content (id, tittel, tekst, info_type, koder, maalgruppe)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    tittel = VALUES(tittel),
                    tekst = VALUES(tekst),
                    info_type = VALUES(info_type),
                    koder = VALUES(koder),
                    maalgruppe = VALUES(maalgruppe)
                """,
                (
                    content.get("id"),
                    content.get("tittel"),
                    content.get("tekst"),
                    info_type,
                    koder_json,
                    maalgruppe_json,
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
        conn = self._get_connection()
        if not conn:
            return 0

        try:
            cursor = conn.cursor()
            cached = 0
            for content in contents:
                try:
                    # Handle JSON fields - may already be JSON strings from API
                    koder = content.get("koder")
                    if koder is None:
                        koder_json = None
                    elif isinstance(koder, str):
                        koder_json = koder  # Already a JSON string
                    else:
                        koder_json = json.dumps(koder, ensure_ascii=False)

                    maalgruppe = content.get("maalgruppe")
                    if maalgruppe is None:
                        maalgruppe_json = None
                    elif isinstance(maalgruppe, str):
                        maalgruppe_json = maalgruppe  # Already a JSON string
                    else:
                        maalgruppe_json = json.dumps(maalgruppe, ensure_ascii=False)

                    # Get info_type - API uses 'infoType' or 'dokumentType' depending on mode
                    info_type = content.get("infoType") or content.get("dokumentType")

                    cursor.execute(
                        """
                        INSERT INTO content (id, tittel, tekst, info_type, koder, maalgruppe)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            tittel = VALUES(tittel),
                            tekst = VALUES(tekst),
                            info_type = VALUES(info_type),
                            koder = VALUES(koder),
                            maalgruppe = VALUES(maalgruppe)
                        """,
                        (
                            content.get("id"),
                            content.get("tittel"),
                            content.get("tekst"),
                            info_type,
                            koder_json,
                            maalgruppe_json,
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
        """
        Get a cached content item by ID.

        Args:
            content_id: The content ID

        Returns:
            Content dict or None if not found
        """
        conn = self._get_connection()
        if not conn:
            return None

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM content WHERE id = %s",
                (content_id,),
            )
            result = cursor.fetchone()
            return result
        except mysql.connector.Error as e:
            print(f"Error getting content: {e}")
            return None
        finally:
            cursor.close()
            conn.close()

    def get_all_content(self) -> List[Dict[str, Any]]:
        """
        Get all cached content.

        Returns:
            List of content dicts
        """
        conn = self._get_connection()
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
        conn = self._get_connection()
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
        conn = self._get_connection()
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

    # ==================== Statistics Operations ====================

    def record_impression(self, content_id: str) -> bool:
        """
        Record an impression (content shown in search results).

        Args:
            content_id: The content ID

        Returns:
            True if recorded successfully
        """
        conn = self._get_connection()
        if not conn:
            return False

        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO content_stats (content_id, impressions, clicks)
                VALUES (%s, 1, 0)
                ON DUPLICATE KEY UPDATE impressions = impressions + 1
                """,
                (content_id,),
            )
            conn.commit()
            return True
        except mysql.connector.Error as e:
            print(f"Error recording impression: {e}")
            return False
        finally:
            cursor.close()
            conn.close()

    def record_impressions_batch(self, content_ids: List[str]) -> int:
        """
        Record impressions for multiple content items.

        Args:
            content_ids: List of content IDs

        Returns:
            Number of impressions recorded
        """
        conn = self._get_connection()
        if not conn:
            return 0

        try:
            cursor = conn.cursor()
            recorded = 0
            for content_id in content_ids:
                try:
                    cursor.execute(
                        """
                        INSERT INTO content_stats (content_id, impressions, clicks)
                        VALUES (%s, 1, 0)
                        ON DUPLICATE KEY UPDATE impressions = impressions + 1
                        """,
                        (content_id,),
                    )
                    recorded += 1
                except mysql.connector.Error:
                    continue
            conn.commit()
            return recorded
        except mysql.connector.Error as e:
            print(f"Error recording impressions batch: {e}")
            return 0
        finally:
            cursor.close()
            conn.close()

    def record_click(self, content_id: str) -> bool:
        """
        Record a click on a content item.

        Args:
            content_id: The content ID

        Returns:
            True if recorded successfully
        """
        conn = self._get_connection()
        if not conn:
            return False

        try:
            cursor = conn.cursor()
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
            print(f"Error recording click: {e}")
            return False
        finally:
            cursor.close()
            conn.close()

    def get_content_stats(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a content item.

        Args:
            content_id: The content ID

        Returns:
            Stats dict with impressions, clicks, ctr
        """
        conn = self._get_connection()
        if not conn:
            return None

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT content_id, impressions, clicks,
                       CASE WHEN impressions > 0 THEN clicks / impressions ELSE 0 END as ctr
                FROM content_stats
                WHERE content_id = %s
                """,
                (content_id,),
            )
            return cursor.fetchone()
        except mysql.connector.Error as e:
            print(f"Error getting content stats: {e}")
            return None
        finally:
            cursor.close()
            conn.close()

    def get_all_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all content items.

        Returns:
            List of stats dicts
        """
        conn = self._get_connection()
        if not conn:
            return []

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT content_id, impressions, clicks,
                       CASE WHEN impressions > 0 THEN clicks / impressions ELSE 0 END as ctr
                FROM content_stats
                ORDER BY ctr DESC
                """
            )
            return cursor.fetchall()
        except mysql.connector.Error as e:
            print(f"Error getting all stats: {e}")
            return []
        finally:
            cursor.close()
            conn.close()

    def get_ctr(self, content_id: str) -> float:
        """
        Get click-through rate for a content item.

        Args:
            content_id: The content ID

        Returns:
            CTR as float (0.0 if not found)
        """
        stats = self.get_content_stats(content_id)
        if stats:
            return stats.get("ctr", 0.0)
        return 0.0

    # ==================== Search Logging Operations ====================

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
        conn = self._get_connection()
        if not conn:
            return False

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
            cursor.close()
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
            results: List of results with features:
                - content_id, position
                - semantic_similarity, keyword_score_total
                - exact_title_proportion, full_coverage_proportion
                - title_keyword_proportion, body_keyword_proportion, exact_body_proportion
                - type_match, role_match, code_match_count, lis_match, maalgruppe_match

        Returns:
            True if logged successfully
        """
        conn = self._get_connection()
        if not conn:
            return False

        try:
            cursor = conn.cursor()

            for result in results:
                cursor.execute(
                    """
                    INSERT INTO search_results_shown (
                        search_id, content_id, position,
                        semantic_similarity, keyword_score_total,
                        exact_title_proportion, full_coverage_proportion,
                        title_keyword_proportion, body_keyword_proportion, exact_body_proportion,
                        type_match, role_match, code_match_count, lis_match, maalgruppe_match
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        search_id,
                        result.get("content_id"),
                        result.get("position"),
                        result.get("semantic_similarity"),
                        result.get("keyword_score_total"),
                        result.get("exact_title_proportion"),
                        result.get("full_coverage_proportion"),
                        result.get("title_keyword_proportion"),
                        result.get("body_keyword_proportion"),
                        result.get("exact_body_proportion"),
                        result.get("type_match"),
                        result.get("role_match"),
                        result.get("code_match_count", 0),
                        result.get("lis_match", 0),
                        result.get("maalgruppe_match", 0),
                    ),
                )

            # Also update content_stats impressions
            content_ids = [r.get("content_id") for r in results if r.get("content_id")]
            self.record_impressions_batch(content_ids)

            conn.commit()
            return True
        except mysql.connector.Error as e:
            print(f"Error logging search results: {e}")
            return False
        finally:
            cursor.close()
            conn.close()

    def log_click(
        self,
        search_id: str,
        content_id: str,
    ) -> bool:
        """
        Log a click event. Position is looked up from search_results_shown.

        Args:
            search_id: The search_id this click belongs to
            content_id: The clicked content ID

        Returns:
            True if logged successfully
        """
        conn = self._get_connection()
        if not conn:
            return False

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
            self.record_click(content_id)

            conn.commit()
            return True
        except mysql.connector.Error as e:
            print(f"Error logging click: {e}")
            return False
        finally:
            cursor.close()
            conn.close()

    def get_training_data(self) -> List[Dict[str, Any]]:
        """
        Get training data for learning-to-rank model.

        Returns searches that have at least one click, with all results
        and which ones were clicked.

        Returns:
            List of training examples with search info, results, and click labels
        """
        conn = self._get_connection()
        if not conn:
            return []

        try:
            cursor = conn.cursor(dictionary=True)

            # Get searches that have clicks
            cursor.execute(
                """
                SELECT DISTINCT sl.search_id, sl.query, sl.role
                FROM search_logs sl
                INNER JOIN click_logs cl ON sl.search_id = cl.search_id
                """
            )
            searches = cursor.fetchall()

            training_data = []
            for search in searches:
                search_id = search["search_id"]

                # Get results shown for this search
                cursor.execute(
                    """
                    SELECT content_id, position, score
                    FROM search_results_shown
                    WHERE search_id = %s
                    ORDER BY position
                    """,
                    (search_id,),
                )
                results = cursor.fetchall()

                # Get clicks for this search
                cursor.execute(
                    """
                    SELECT content_id
                    FROM click_logs
                    WHERE search_id = %s
                    """,
                    (search_id,),
                )
                clicks = {row["content_id"] for row in cursor.fetchall()}

                # Create training examples
                for result in results:
                    training_data.append(
                        {
                            "search_id": search_id,
                            "query": search["query"],
                            "role": search["role"],
                            "content_id": result["content_id"],
                            "position": result["position"],
                            "score": result["score"],
                            "clicked": 1 if result["content_id"] in clicks else 0,
                        }
                    )

            return training_data
        except mysql.connector.Error as e:
            print(f"Error getting training data: {e}")
            return []
        finally:
            cursor.close()
            conn.close()

    def get_search_count(self) -> int:
        """Get total number of logged searches."""
        conn = self._get_connection()
        if not conn:
            return 0

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM search_logs")
            result = cursor.fetchone()
            return result[0] if result else 0
        except mysql.connector.Error as e:
            print(f"Error getting search count: {e}")
            return 0
        finally:
            cursor.close()
            conn.close()

    def get_click_count(self) -> int:
        """Get total number of logged clicks."""
        conn = self._get_connection()
        if not conn:
            return 0

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM click_logs")
            result = cursor.fetchone()
            return result[0] if result else 0
        except mysql.connector.Error as e:
            print(f"Error getting click count: {e}")
            return 0
        finally:
            cursor.close()
            conn.close()

    def get_content_ctr(self) -> Dict[str, float]:
        """
        Get CTR (click-through rate) for all content items.

        Returns:
            Dictionary mapping content_id to CTR value (0.0-1.0)
        """
        conn = self._get_connection()
        if not conn:
            return {}

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT content_id, ctr
                FROM content_stats
                WHERE impressions > 0
                """
            )
            results = cursor.fetchall()
            return {row["content_id"]: float(row["ctr"] or 0.0) for row in results}
        except mysql.connector.Error as e:
            print(f"Error getting CTR data: {e}")
            return {}
        finally:
            cursor.close()
            conn.close()

    # ==================== Learning-to-Rank Operations ====================

    def get_ltr_training_rows(self, days_back: int = 180) -> List[Dict[str, Any]]:
        """
        Get training data for learning-to-rank model.

        Returns all search results shown with their features and click labels.
        Joins search_results_shown with click_logs to determine which results were clicked.

        Args:
            days_back: Number of days of history to include

        Returns:
            List of training rows with features and labels
        """
        conn = self._get_connection()
        if not conn:
            return []

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT
                    srs.search_id,
                    srs.content_id,
                    srs.position,
                    srs.semantic_similarity,
                    srs.keyword_score_total,
                    srs.exact_title_proportion,
                    srs.full_coverage_proportion,
                    srs.title_keyword_proportion,
                    srs.body_keyword_proportion,
                    srs.exact_body_proportion,
                    srs.type_match,
                    srs.role_match,
                    srs.code_match_count,
                    srs.lis_match,
                    srs.maalgruppe_match,
                    CASE WHEN cl.content_id IS NOT NULL THEN 1 ELSE 0 END as clicked
                FROM search_results_shown srs
                INNER JOIN search_logs sl ON srs.search_id = sl.search_id
                LEFT JOIN click_logs cl ON srs.search_id = cl.search_id
                    AND srs.content_id = cl.content_id
                WHERE sl.timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
                ORDER BY srs.search_id, srs.position
                """,
                (days_back,),
            )
            return cursor.fetchall()
        except mysql.connector.Error as e:
            print(f"Error getting LTR training rows: {e}")
            return []
        finally:
            cursor.close()
            conn.close()

    def get_content_stats_bulk(self) -> Dict[str, Dict[str, int]]:
        """
        Get content statistics for all content items in bulk.
        
        Returns:
            Dictionary mapping content_id to dict with 'clicks' and 'impressions'
        """
        conn = self._get_connection()
        if not conn:
            return {}

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT content_id, clicks, impressions
                FROM content_stats
                """
            )
            results = cursor.fetchall()
            return {
                row["content_id"]: {
                    "clicks": int(row["clicks"] or 0),
                    "impressions": int(row["impressions"] or 0),
                }
                for row in results
            }
        except mysql.connector.Error as e:
            print(f"Error getting content stats bulk: {e}")
            return {}
        finally:
            cursor.close()
            conn.close()

    def get_position_propensities(self) -> Dict[int, float]:
        """
        Get position propensities from database.
        
        Returns:
            Dictionary mapping position to propensity value
        """
        conn = self._get_connection()
        if not conn:
            return {}

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT position, propensity
                FROM position_propensity
                ORDER BY position
                """
            )
            results = cursor.fetchall()
            return {int(row["position"]): float(row["propensity"]) for row in results}
        except mysql.connector.Error as e:
            print(f"Error getting position propensities: {e}")
            return {}
        finally:
            cursor.close()
            conn.close()


# Global instance
database_service = DatabaseService()
