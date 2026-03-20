"""
Generate filtered training CSV files from database content.

Instead of relying on a real page-view CSV, this script queries the content
database directly and builds a CSV compatible with generate_training_data.py.

Filters:
  --role       Only include content tagged for a specific role
                 (e.g. "Lege", "Sykepleier", "Fysioterapeut")
  --info-type  Only include a specific info_type
                 (e.g. "retningslinje", "veileder", "anbefaling")

Multiple filters can be combined (AND logic).

Views (popularity signal):
  All matched items get a synthetic view count. You can weight by recency
  (--weight-by-recency) so recently updated content gets higher priority
  as training targets.

Usage:
    python scripts/setup/generate_filtered_csv.py --output lege_fokus.csv --role Lege
    python scripts/setup/generate_filtered_csv.py --output retningslinjer.csv --info-type retningslinje
    python scripts/setup/generate_filtered_csv.py --output lege_retningslinjer.csv --role Lege --info-type retningslinje
    python scripts/setup/generate_filtered_csv.py --output alle_innhold.csv --weight-by-recency
    python scripts/setup/generate_filtered_csv.py --list-roles
    python scripts/setup/generate_filtered_csv.py --list-info-types
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import mysql.connector
from app.config import settings

DATASETS_DIR = project_root / "data" / "training_datasets"
BASE_URL = "https://www.helsedirektoratet.no"

# Default synthetic view count when not weighting by recency
DEFAULT_VIEWS = 1000
# Max views assigned to the most recently updated item (when weighting)
MAX_VIEWS = 5000
MIN_VIEWS = 200


def get_connection():
    return mysql.connector.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
        charset="utf8mb4",
        collation="utf8mb4_unicode_ci",
    )


def fetch_content(conn, role_filter=None, info_type_filter=None):
    """Fetch content rows with optional filters."""
    cursor = conn.cursor(dictionary=True)

    # Build query — we need id, tittel, path, info_type, role_tags, sist_faglig_oppdatert
    query = """
        SELECT id, tittel, kort_tittel, path, info_type, role_tags, sist_faglig_oppdatert
        FROM content
        WHERE path IS NOT NULL
          AND path != ''
          AND tittel IS NOT NULL
    """
    params = []

    if info_type_filter:
        query += " AND info_type = %s"
        params.append(info_type_filter)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    cursor.close()

    # Filter by role tag (JSON field — do in Python for simplicity)
    if role_filter:
        filtered = []
        for row in rows:
            tags = row.get("role_tags")
            if tags:
                if isinstance(tags, str):
                    try:
                        tags = json.loads(tags)
                    except (json.JSONDecodeError, TypeError):
                        tags = []
                if role_filter in (tags or []):
                    filtered.append(row)
        return filtered

    return rows


def compute_views(rows, weight_by_recency: bool) -> list:
    """Assign synthetic view counts to each row."""
    if not weight_by_recency:
        for row in rows:
            row["views"] = DEFAULT_VIEWS
        return rows

    # Parse dates and assign views proportional to recency
    now = datetime.now()
    dated = []
    undated = []
    for row in rows:
        d = row.get("sist_faglig_oppdatert")
        if d:
            if hasattr(d, "timestamp"):
                dated.append((row, d))
            else:
                undated.append(row)
        else:
            undated.append(row)

    if dated:
        oldest = min(d for _, d in dated)
        newest = max(d for _, d in dated)
        span = (newest - oldest).total_seconds() or 1

        for row, d in dated:
            age_ratio = (d - oldest).total_seconds() / span  # 0=oldest, 1=newest
            row["views"] = int(MIN_VIEWS + (MAX_VIEWS - MIN_VIEWS) * age_ratio)

    for row in undated:
        row["views"] = MIN_VIEWS

    return rows


def list_roles(conn):
    """Print all unique role tags found in the database."""
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT role_tags FROM content WHERE role_tags IS NOT NULL")
    all_tags = set()
    for row in cursor.fetchall():
        tags = row["role_tags"]
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except (json.JSONDecodeError, TypeError):
                continue
        if isinstance(tags, list):
            all_tags.update(tags)
    cursor.close()
    print("\nAvailable role tags:")
    for tag in sorted(all_tags):
        print(f"  {tag}")


def list_info_types(conn):
    """Print all unique info_types in the database."""
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT info_type, COUNT(*) as count FROM content "
        "WHERE info_type IS NOT NULL GROUP BY info_type ORDER BY count DESC"
    )
    print("\nAvailable info_types:")
    for row in cursor.fetchall():
        print(f"  {row['info_type']:<45} ({row['count']} items)")
    cursor.close()


def write_csv(rows, output_path: Path):
    """Write rows to CSV in the format expected by parse_page_view_csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "url", "device_type", "country", "views"])
        for row in rows:
            title = row.get("kort_tittel") or row.get("tittel") or ""
            path = row["path"]
            url = f"{BASE_URL}{path}"
            writer.writerow([title, url, "Desktop", "Norway", row["views"]])
    return len(rows)


def register_in_db(conn, name: str, filename: str, row_count: int, description: str):
    """Register the CSV in training_datasets table (upsert by name)."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO training_datasets (name, description, filename, row_count)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            description = VALUES(description),
            filename = VALUES(filename),
            row_count = VALUES(row_count)
        """,
        (name, description, filename, row_count),
    )
    conn.commit()
    cursor.execute("SELECT id FROM training_datasets WHERE name = %s", (name,))
    result = cursor.fetchone()
    cursor.close()
    return result[0] if result else None


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate filtered training CSV from database content"
    )
    parser.add_argument("--output", help="Output filename (saved to data/training_datasets/)")
    parser.add_argument("--role", help="Filter by role tag (e.g. 'Lege', 'Sykepleier')")
    parser.add_argument("--info-type", dest="info_type", help="Filter by info_type (e.g. 'retningslinje')")
    parser.add_argument("--weight-by-recency", action="store_true",
                        help="Give higher views to recently updated content")
    parser.add_argument("--name", help="Dataset name for DB registration (defaults to output filename)")
    parser.add_argument("--description", default="", help="Dataset description")
    parser.add_argument("--register", action="store_true",
                        help="Register the CSV in the training_datasets table")
    parser.add_argument("--list-roles", action="store_true", help="List all available role tags and exit")
    parser.add_argument("--list-info-types", action="store_true", help="List all info_types and exit")
    args = parser.parse_args()

    conn = get_connection()

    try:
        if args.list_roles:
            list_roles(conn)
            return

        if args.list_info_types:
            list_info_types(conn)
            return

        if not args.output:
            parser.error("--output is required unless using --list-roles or --list-info-types")

        print("\n" + "=" * 60)
        print("  Generate Filtered Training CSV")
        print("=" * 60)

        filters = []
        if args.role:
            filters.append(f"role={args.role}")
        if args.info_type:
            filters.append(f"info_type={args.info_type}")
        if not filters:
            filters.append("all searchable content")
        print(f"\nFilter: {', '.join(filters)}")

        print("Fetching content from database...")
        rows = fetch_content(conn, role_filter=args.role, info_type_filter=args.info_type)
        print(f"  {len(rows)} items matched")

        if not rows:
            print("\nNo content matched the filters. Use --list-roles or --list-info-types to see options.")
            return

        rows = compute_views(rows, weight_by_recency=args.weight_by_recency)

        if '/' in args.output or '\\' in args.output or '..' in args.output:
            print(f"Error: output must be a plain filename, got '{args.output}'")
            return

        output_path = DATASETS_DIR / args.output
        row_count = write_csv(rows, output_path)
        print(f"\nWrote {row_count} rows to: {output_path}")

        if args.register:
            dataset_name = args.name or Path(args.output).stem
            description = args.description or f"Generated: {', '.join(filters)}"
            dataset_id = register_in_db(conn, dataset_name, args.output, row_count, description)
            print(f"Registered in DB as '{dataset_name}' (id={dataset_id})")

        print("\nDone! Next steps:")
        if not args.register:
            print("  1. Upload via API: POST /dev/datasets/upload")
            print("     or re-run with --register to add directly to DB")
        print("  2. Create a preset: POST /dev/presets")
        print("  3. Generate training data: POST /dev/generate")
        print("  4. Train model: POST /dev/train")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
