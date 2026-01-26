"""
Repository for content statistics operations.
"""

from typing import List, Optional, Dict, Any
import mysql.connector

from app.services.repositories.base import db_pool


class StatsRepository:
    """Repository for content statistics (impressions, clicks, CTR)."""

    def record_impression(self, content_id: str) -> bool:
        """Record an impression (content shown in search results)."""
        conn = db_pool.get_connection()
        if not conn:
            return False

        cursor = None
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
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def record_impressions_batch(self, content_ids: List[str]) -> int:
        """Record impressions for multiple content items."""
        conn = db_pool.get_connection()
        if not conn:
            return 0

        cursor = None
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
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def record_click(self, content_id: str) -> bool:
        """Record a click on a content item."""
        conn = db_pool.get_connection()
        if not conn:
            return False

        cursor = None
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
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def get_content_stats(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a content item."""
        conn = db_pool.get_connection()
        if not conn:
            return None

        cursor = None
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
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all content items."""
        conn = db_pool.get_connection()
        if not conn:
            return []

        cursor = None
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
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def get_ctr(self, content_id: str) -> float:
        """Get click-through rate for a content item."""
        stats = self.get_content_stats(content_id)
        if stats:
            return stats.get("ctr", 0.0)
        return 0.0

    def get_content_ctr(self) -> Dict[str, float]:
        """Get CTR for all content items."""
        conn = db_pool.get_connection()
        if not conn:
            return {}

        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT content_id, clicks, impressions
                FROM content_stats
                WHERE impressions > 0
                """
            )
            results = cursor.fetchall()
            return {
                row["content_id"]: float(row["clicks"]) / float(row["impressions"])
                for row in results
                if row["impressions"] > 0
            }
        except mysql.connector.Error as e:
            print(f"Error getting CTR data: {e}")
            return {}
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def get_content_ctr_windowed(self, days: int = 30) -> Dict[str, float]:
        """
        Get CTR for each content item within a time window.

        Uses smoothed CTR: (clicks + 1) / (impressions + 21)

        Args:
            days: Number of days to look back (default: 30)

        Returns:
            Dictionary mapping content_id to smoothed CTR value
        """
        conn = db_pool.get_connection()
        if not conn:
            return {}

        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT
                    srs.content_id,
                    COUNT(DISTINCT srs.id) as impressions,
                    COUNT(DISTINCT cl.id) as clicks
                FROM search_results_shown srs
                LEFT JOIN click_logs cl
                    ON srs.content_id = cl.content_id
                    AND srs.search_id = cl.search_id
                    AND cl.timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
                WHERE srs.timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
                GROUP BY srs.content_id
                """,
                (days, days),
            )
            results = cursor.fetchall()

            # Smoothed CTR: (clicks + alpha) / (impressions + alpha + beta)
            alpha = 1.0
            beta = 20.0
            ctr_dict = {}
            for row in results:
                clicks = int(row["clicks"] or 0)
                impressions = int(row["impressions"] or 0)
                ctr_dict[row["content_id"]] = (clicks + alpha) / (impressions + alpha + beta)

            return ctr_dict
        except mysql.connector.Error as e:
            print(f"Error getting windowed CTR data: {e}")
            return {}
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def get_content_stats_bulk(self) -> Dict[str, Dict[str, int]]:
        """Get content statistics for all content items in bulk."""
        conn = db_pool.get_connection()
        if not conn:
            return {}

        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT content_id, clicks, impressions FROM content_stats")
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
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()


# Global instance
stats_repository = StatsRepository()
