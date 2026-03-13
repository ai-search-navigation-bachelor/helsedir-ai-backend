"""
Repository for Learning-to-Rank training data operations.
"""

from typing import List, Dict, Any
import mysql.connector

from app.services.repositories.base import db_pool


class LtrRepository:
    """Repository for LTR training data."""

    def get_training_data(self) -> List[Dict[str, Any]]:
        """
        Get training data for learning-to-rank model.

        Returns searches that have at least one click, with all results
        and which ones were clicked.
        """
        conn = db_pool.get_connection()
        if not conn:
            return []

        cursor = None
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
                    training_data.append({
                        "search_id": search_id,
                        "query": search["query"],
                        "role": search["role"],
                        "content_id": result["content_id"],
                        "position": result["position"],
                        "score": result["score"],
                        "clicked": 1 if result["content_id"] in clicks else 0,
                    })

            return training_data
        except mysql.connector.Error as e:
            print(f"Error getting training data: {e}")
            return []
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def get_ltr_training_rows(self, days_back: int = 180) -> List[Dict[str, Any]]:
        """
        Get training data for learning-to-rank model.

        Returns all search results shown with their features and click labels.

        Args:
            days_back: Number of days of history to include

        Returns:
            List of training rows with features and labels
        """
        conn = db_pool.get_connection()
        if not conn:
            return []

        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT
                    srs.search_id,
                    srs.content_id,
                    srs.position,
                    srs.semantic_score,
                    srs.bm25_score,
                    srs.rrf_score,
                    srs.role_match,
                    sl.query,
                    sl.timestamp,
                    c.tittel AS title,
                    c.sist_faglig_oppdatert,
                    CASE WHEN EXISTS(
                        SELECT 1 FROM click_logs cl
                        WHERE cl.search_id = srs.search_id AND cl.content_id = srs.content_id
                    ) THEN 1 ELSE 0 END as clicked
                FROM search_results_shown srs
                INNER JOIN search_logs sl ON srs.search_id = sl.search_id
                LEFT JOIN content c ON srs.content_id = c.id
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
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def get_content_ctr_map(self, days_back: int = 30) -> Dict[str, float]:
        """
        Compute smoothed CTR per content_id over the last N days.

        Uses Bayesian smoothing: (clicks + 1) / (impressions + 21)
        This pulls the CTR toward a low prior for items with few impressions.

        Args:
            days_back: Number of days to look back

        Returns:
            Dict mapping content_id to smoothed CTR value
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
                    COUNT(DISTINCT CONCAT(srs.search_id, '-', srs.content_id)) AS impressions,
                    COUNT(DISTINCT cl.id) AS clicks
                FROM search_results_shown srs
                INNER JOIN search_logs sl ON srs.search_id = sl.search_id
                LEFT JOIN click_logs cl
                    ON srs.search_id = cl.search_id
                    AND srs.content_id = cl.content_id
                WHERE sl.timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
                GROUP BY srs.content_id
                """,
                (days_back,),
            )
            rows = cursor.fetchall()
            return {
                row["content_id"]: (int(row["clicks"]) + 1) / (int(row["impressions"]) + 21)
                for row in rows
            }
        except mysql.connector.Error as e:
            print(f"Error getting content CTR map: {e}")
            return {}
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def get_position_propensities(self) -> Dict[int, float]:
        """Get position propensities from database."""
        conn = db_pool.get_connection()
        if not conn:
            return {}

        cursor = None
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
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()


# Global instance
ltr_repository = LtrRepository()
