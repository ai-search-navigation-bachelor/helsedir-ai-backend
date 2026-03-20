"""
Repository for training dataset and preset management.

Tracks CSV files (stored in data/training_datasets/) and named training
presets (dataset + click simulation parameters) in the database.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import mysql.connector

from app.services.repositories.base import db_pool

# Where CSV files are stored on disk
DATASETS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "training_datasets"


class TrainingRepository:

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def get_all_datasets(self) -> List[Dict[str, Any]]:
        """List all registered training datasets."""
        conn = db_pool.get_connection()
        if not conn:
            return []
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT id, name, description, filename, row_count, created_at "
                "FROM training_datasets ORDER BY created_at DESC"
            )
            return cursor.fetchall()
        except mysql.connector.Error as e:
            print(f"Error listing datasets: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def get_dataset(self, dataset_id: int) -> Optional[Dict[str, Any]]:
        """Get a single dataset by ID."""
        conn = db_pool.get_connection()
        if not conn:
            return None
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT id, name, description, filename, row_count, created_at "
                "FROM training_datasets WHERE id = %s",
                (dataset_id,),
            )
            return cursor.fetchone()
        except mysql.connector.Error as e:
            print(f"Error getting dataset {dataset_id}: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def create_dataset(
        self,
        name: str,
        filename: str,
        row_count: int,
        description: str = "",
    ) -> int:
        """Register a new dataset. Returns the new ID."""
        conn = db_pool.get_connection()
        if not conn:
            raise RuntimeError("No database connection")
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO training_datasets (name, description, filename, row_count) "
                "VALUES (%s, %s, %s, %s)",
                (name, description, filename, row_count),
            )
            conn.commit()
            return cursor.lastrowid
        except mysql.connector.Error as e:
            conn.rollback()
            raise RuntimeError(f"Failed to create dataset: {e}") from e
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def delete_dataset(self, dataset_id: int) -> bool:
        """Delete a dataset record (does NOT delete the file on disk)."""
        conn = db_pool.get_connection()
        if not conn:
            return False
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM training_datasets WHERE id = %s", (dataset_id,))
            conn.commit()
            return cursor.rowcount > 0
        except mysql.connector.Error as e:
            conn.rollback()
            print(f"Error deleting dataset {dataset_id}: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
            conn.close()

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def get_all_presets(self) -> List[Dict[str, Any]]:
        """List all training presets with their dataset info."""
        conn = db_pool.get_connection()
        if not conn:
            return []
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT p.id, p.name, p.description,
                       p.dataset_id, d.name AS dataset_name, d.filename AS dataset_filename,
                       p.top_n, p.k,
                       p.target_click_min, p.target_click_max,
                       p.temaside_click_weight, p.retningslinje_click_weight,
                       p.other_click_weight,
                       p.created_at
                FROM training_presets p
                JOIN training_datasets d ON p.dataset_id = d.id
                ORDER BY p.created_at DESC
                """
            )
            return cursor.fetchall()
        except mysql.connector.Error as e:
            print(f"Error listing presets: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def get_preset(self, preset_id: int) -> Optional[Dict[str, Any]]:
        """Get a single preset by ID (includes dataset info)."""
        conn = db_pool.get_connection()
        if not conn:
            return None
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT p.id, p.name, p.description,
                       p.dataset_id, d.name AS dataset_name, d.filename AS dataset_filename,
                       p.top_n, p.k,
                       p.target_click_min, p.target_click_max,
                       p.temaside_click_weight, p.retningslinje_click_weight,
                       p.other_click_weight,
                       p.created_at
                FROM training_presets p
                JOIN training_datasets d ON p.dataset_id = d.id
                WHERE p.id = %s
                """,
                (preset_id,),
            )
            return cursor.fetchone()
        except mysql.connector.Error as e:
            print(f"Error getting preset {preset_id}: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def create_preset(
        self,
        name: str,
        dataset_id: int,
        description: str = "",
        top_n: int = 200,
        k: int = 15,
        target_click_min: float = 0.5,
        target_click_max: float = 0.9,
        temaside_click_weight: float = 0.35,
        retningslinje_click_weight: float = 0.30,
        other_click_weight: float = 0.15,
    ) -> int:
        """Create a new preset. Returns the new ID."""
        conn = db_pool.get_connection()
        if not conn:
            raise RuntimeError("No database connection")
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO training_presets
                    (name, description, dataset_id, top_n, k,
                     target_click_min, target_click_max,
                     temaside_click_weight, retningslinje_click_weight, other_click_weight)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    name, description, dataset_id, top_n, k,
                    target_click_min, target_click_max,
                    temaside_click_weight, retningslinje_click_weight, other_click_weight,
                ),
            )
            conn.commit()
            return cursor.lastrowid
        except mysql.connector.Error as e:
            conn.rollback()
            raise RuntimeError(f"Failed to create preset: {e}") from e
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def delete_preset(self, preset_id: int) -> bool:
        """Delete a preset."""
        conn = db_pool.get_connection()
        if not conn:
            return False
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM training_presets WHERE id = %s", (preset_id,))
            conn.commit()
            return cursor.rowcount > 0
        except mysql.connector.Error as e:
            conn.rollback()
            print(f"Error deleting preset {preset_id}: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
            conn.close()


# Global instance
training_repository = TrainingRepository()
