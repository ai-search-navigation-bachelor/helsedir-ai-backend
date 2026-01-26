"""
Base database functionality with connection pooling.
"""

import mysql.connector
from mysql.connector import pooling
from typing import Optional

from app.config import settings


class DatabasePool:
    """Manages MySQL connection pool."""

    _instance: Optional["DatabasePool"] = None
    _pool: Optional[pooling.MySQLConnectionPool] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_pool()
        return cls._instance

    def _initialize_pool(self):
        """Initialize connection pool."""
        try:
            self._pool = pooling.MySQLConnectionPool(
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
            self._pool = None

    def get_connection(self):
        """Get a connection from the pool."""
        # Attempt lazy re-initialization if pool is None
        if self._pool is None:
            try:
                self._initialize_pool()
            except mysql.connector.Error as e:
                print(f"Error re-initializing pool: {e}")
                return None

        # Still None after re-init attempt
        if self._pool is None:
            return None

        try:
            return self._pool.get_connection()
        except mysql.connector.Error as e:
            print(f"Error getting connection: {e}")
            return None

    def is_connected(self) -> bool:
        """Check if database is available."""
        conn = self.get_connection()
        if conn:
            conn.close()
            return True
        return False


# Global pool instance
db_pool = DatabasePool()
