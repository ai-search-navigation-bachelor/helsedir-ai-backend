"""
Database service facade.

Provides backward-compatible interface while delegating to specialized repositories.
"""

from typing import List, Optional, Dict, Any

from app.services.repositories.base import db_pool
from app.services.repositories.content_repository import content_repository
from app.services.repositories.stats_repository import stats_repository
from app.services.repositories.search_repository import search_repository
from app.services.repositories.ltr_repository import ltr_repository


class DatabaseService:
    """
    Facade for database operations.

    Delegates to specialized repositories while maintaining backward compatibility.
    """

    def __init__(self):
        self._content = content_repository
        self._stats = stats_repository
        self._search = search_repository
        self._ltr = ltr_repository

    # ==================== Connection ====================

    def _get_connection(self):
        """Get a connection from the pool."""
        return db_pool.get_connection()

    def is_connected(self) -> bool:
        """Check if database is available."""
        return db_pool.is_connected()

    # ==================== Content Operations ====================

    def cache_content(self, content: Dict[str, Any]) -> bool:
        return self._content.cache_content(content)

    def cache_content_batch(self, contents: List[Dict[str, Any]]) -> int:
        return self._content.cache_content_batch(contents)

    def get_content(self, content_id: str) -> Optional[Dict[str, Any]]:
        return self._content.get_content(content_id)

    def get_all_content(self) -> List[Dict[str, Any]]:
        return self._content.get_all_content()

    def get_content_count(self) -> int:
        return self._content.get_content_count()

    def get_content_stats_by_type(self) -> List[Dict[str, Any]]:
        return self._content.get_content_stats_by_type()

    def get_all_content_ids(self) -> List[str]:
        return self._content.get_all_content_ids()

    def get_all_content_to_theme_pages(self) -> Dict[str, List[Dict[str, Any]]]:
        return self._content.get_all_content_to_theme_pages()

    # ==================== Statistics Operations ====================

    def record_impression(self, content_id: str) -> bool:
        return self._stats.record_impression(content_id)

    def record_impressions_batch(self, content_ids: List[str]) -> int:
        return self._stats.record_impressions_batch(content_ids)

    def record_click(self, content_id: str) -> bool:
        return self._stats.record_click(content_id)

    def get_content_stats(self, content_id: str) -> Optional[Dict[str, Any]]:
        return self._stats.get_content_stats(content_id)

    def get_all_stats(self) -> List[Dict[str, Any]]:
        return self._stats.get_all_stats()

    def get_ctr(self, content_id: str) -> float:
        return self._stats.get_ctr(content_id)

    def get_content_ctr(self) -> Dict[str, float]:
        return self._stats.get_content_ctr()

    def get_content_ctr_windowed(self, days: int = 30) -> Dict[str, float]:
        return self._stats.get_content_ctr_windowed(days)

    def get_content_stats_bulk(self) -> Dict[str, Dict[str, int]]:
        return self._stats.get_content_stats_bulk()

    # ==================== Search Logging Operations ====================

    def log_search(
        self,
        search_id: str,
        query: str,
        role: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        return self._search.log_search(search_id, query, role, session_id, user_id)

    def get_search_by_id(self, search_id: str) -> Optional[Dict[str, Any]]:
        return self._search.get_search_by_id(search_id)

    def log_search_results(self, search_id: str, results: List[Dict[str, Any]]) -> bool:
        return self._search.log_search_results(search_id, results)

    def log_click(self, search_id: str, content_id: str) -> bool:
        return self._search.log_click(search_id, content_id)

    def get_search_count(self) -> int:
        return self._search.get_search_count()

    def get_click_count(self) -> int:
        return self._search.get_click_count()

    def get_max_position_for_search(self, search_id: str) -> int:
        return self._search.get_max_position_for_search(search_id)

    def get_logged_content_ids_for_search(self, search_id: str) -> set:
        return self._search.get_logged_content_ids_for_search(search_id)

    # ==================== Learning-to-Rank Operations ====================

    def get_training_data(self) -> List[Dict[str, Any]]:
        return self._ltr.get_training_data()

    def get_ltr_training_rows(self, days_back: int = 180) -> List[Dict[str, Any]]:
        return self._ltr.get_ltr_training_rows(days_back)

    def get_position_propensities(self) -> Dict[int, float]:
        return self._ltr.get_position_propensities()


# Global instance
database_service = DatabaseService()
