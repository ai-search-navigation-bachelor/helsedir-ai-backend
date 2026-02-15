#!/usr/bin/env python3
"""Quick test to verify database connection and content."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.config import settings
import mysql.connector

try:
    conn = mysql.connector.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
    )
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM content")
    print(f"Content rows: {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM content WHERE embedding IS NOT NULL")
    print(f"With embeddings: {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM search_logs")
    print(f"Search logs: {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM click_logs")
    print(f"Click logs: {cursor.fetchone()[0]}")

    print("\nConnection OK!")
    cursor.close()
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")
