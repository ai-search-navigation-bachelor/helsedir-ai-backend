#!/usr/bin/env python3
"""
Migration: Replace maalgruppe column with role_tags in content table.

Usage:
    python scripts/data/migration/add_role_tags.py
    python scripts/data/migration/add_role_tags.py --dry-run
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.repositories.base import db_pool


def run_migration(dry_run: bool = False):
    conn = db_pool.get_connection()
    if not conn:
        print("ERROR: Could not connect to database")
        sys.exit(1)

    cursor = None
    try:
        cursor = conn.cursor()

        # Check if role_tags column already exists
        cursor.execute(
            "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'content' AND COLUMN_NAME = 'role_tags'"
        )
        has_role_tags = cursor.fetchone() is not None

        # Check if maalgruppe column exists
        cursor.execute(
            "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'content' AND COLUMN_NAME = 'maalgruppe'"
        )
        has_maalgruppe = cursor.fetchone() is not None

        if has_role_tags and not has_maalgruppe:
            print("Migration already applied (role_tags exists, maalgruppe removed).")
            return

        if dry_run:
            print("[DRY RUN] Would perform the following:")
            if not has_role_tags:
                print("  1. ADD COLUMN role_tags JSON DEFAULT NULL")
            if has_maalgruppe:
                print("  2. DROP COLUMN maalgruppe")
            return

        if not has_role_tags:
            print("Adding role_tags column...")
            cursor.execute("ALTER TABLE content ADD COLUMN role_tags JSON DEFAULT NULL")

        if has_maalgruppe:
            print("Dropping maalgruppe column...")
            cursor.execute("ALTER TABLE content DROP COLUMN maalgruppe")

        conn.commit()
        print("Migration completed successfully.")

    except Exception as e:
        print(f"ERROR: Migration failed: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        if cursor:
            cursor.close()
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add role_tags column, remove maalgruppe")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    args = parser.parse_args()
    run_migration(dry_run=args.dry_run)
