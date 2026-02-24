"""
Migration: Update search_results_shown to match the hybrid search pipeline.

Drops old keyword-based columns, renames semantic_similarity -> semantic_score,
and adds bm25_score + rrf_score columns.

Usage:
    python -m scripts.data.migration.migrate_search_results_shown
"""

import mysql.connector
from app.config import settings


def migrate():
    conn = mysql.connector.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
    )
    cursor = conn.cursor()

    try:
        # 1. Add new columns
        print("Adding bm25_score column...")
        cursor.execute(
            "ALTER TABLE search_results_shown ADD COLUMN bm25_score FLOAT AFTER semantic_similarity"
        )
    except mysql.connector.Error as e:
        if e.errno == 1060:  # Duplicate column
            print("  bm25_score already exists, skipping.")
        else:
            raise

    try:
        print("Adding rrf_score column...")
        cursor.execute(
            "ALTER TABLE search_results_shown ADD COLUMN rrf_score FLOAT AFTER bm25_score"
        )
    except mysql.connector.Error as e:
        if e.errno == 1060:
            print("  rrf_score already exists, skipping.")
        else:
            raise

    # 2. Rename semantic_similarity -> semantic_score
    try:
        print("Renaming semantic_similarity -> semantic_score...")
        cursor.execute(
            "ALTER TABLE search_results_shown CHANGE COLUMN semantic_similarity semantic_score FLOAT"
        )
    except mysql.connector.Error as e:
        if e.errno == 1054:  # Unknown column (already renamed)
            print("  semantic_similarity already renamed, skipping.")
        else:
            raise

    # 3. Drop old columns
    old_columns = [
        "keyword_score_total",
        "exact_title_proportion",
        "full_coverage_proportion",
        "title_keyword_proportion",
        "code_match_count",
        "lis_match",
    ]
    for col in old_columns:
        try:
            print(f"Dropping column {col}...")
            cursor.execute(f"ALTER TABLE search_results_shown DROP COLUMN {col}")
        except mysql.connector.Error as e:
            if e.errno == 1091:  # Can't DROP; check column exists
                print(f"  {col} already dropped, skipping.")
            else:
                raise

    conn.commit()
    print("Migration complete.")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    migrate()
