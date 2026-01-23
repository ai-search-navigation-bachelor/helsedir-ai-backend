"""
Generate synthetic training data for ranking model.

This script creates realistic search logs, clicks, and statistics
to enable training the ranking model even without real user data.

Usage:
    python scripts/setup/generate_training_data.py
    python scripts/setup/generate_training_data.py --searches 100 --click-rate 0.3
"""

import sys
from pathlib import Path
import random
import uuid
from datetime import datetime, timedelta

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import mysql.connector
from app.config import settings


# Sample queries for different user types
SAMPLE_QUERIES = {
    "lege": [
        "diabetes behandling",
        "astma retningslinjer",
        "adhd diagnostikk",
        "covid-19 testing",
        "influensa vaksine",
        "hjerteinfarkt",
        "hjerneslag",
        "kreft screening",
        "blodtrykk normaler",
        "kolesterol grenser",
    ],
    "sykepleier": [
        "sårbehandling",
        "medikamenthåndtering",
        "pasientobservasjon",
        "infeksjonskontroll",
        "fall forebygging",
        "ernæring eldre",
        "palliativ omsorg",
        "smertevurdering",
        "diabetes oppfølging",
        "medisinhåndtering",
    ],
    "pasient": [
        "hodepine",
        "forkjølelse",
        "influensa symptomer",
        "covid-19 symptomer",
        "allergi",
        "magesmerter",
        "kvalme",
        "feber",
        "hoste",
        "utslett",
    ],
    None: [  # No role specified
        "helseinformasjon",
        "vaksiner",
        "kosthold",
        "trening",
        "søvn",
        "stress",
        "mental helse",
        "fysisk aktivitet",
        "ernæring",
        "forebygging",
    ],
}


def get_connection():
    """Create database connection."""
    try:
        return mysql.connector.connect(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password,
            database=settings.mysql_database,
            charset="utf8mb4",
            collation="utf8mb4_unicode_ci",
        )
    except mysql.connector.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)


def get_content_ids(conn):
    """Get all content IDs from database."""
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM content")
    ids = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return ids


def generate_search_logs(conn, num_searches=100, click_rate=0.3):
    """
    Generate synthetic search logs with realistic click patterns.

    Args:
        conn: Database connection
        num_searches: Number of searches to generate
        click_rate: Probability of clicking on a result (0-1)
    """
    content_ids = get_content_ids(conn)
    
    if not content_ids:
        print("Error: No content in database. Run import_content.py first.")
        return

    cursor = conn.cursor()
    
    print(f"\nGenerating {num_searches} searches...")
    
    searches_created = 0
    clicks_created = 0
    results_shown = 0
    
    start_date = datetime.now() - timedelta(days=30)  # 30 days of history
    
    for i in range(num_searches):
        # Pick random role and query
        role = random.choice([None, "lege", "sykepleier", "pasient"])
        query = random.choice(SAMPLE_QUERIES[role])
        
        # Generate unique search_id
        search_id = str(uuid.uuid4())
        
        # Random timestamp within last 30 days
        timestamp = start_date + timedelta(
            seconds=random.randint(0, 30 * 24 * 60 * 60)
        )
        
        # Simulate search results (5-10 results shown)
        num_results = random.randint(5, 10)
        shown_content = random.sample(content_ids, min(num_results, len(content_ids)))
        
        # Insert search log
        cursor.execute(
            """
            INSERT INTO search_logs (search_id, query, role, results_count, timestamp)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (search_id, query, role, num_results, timestamp),
        )
        searches_created += 1
        
        # Insert results shown
        for position, content_id in enumerate(shown_content, start=1):
            # Score decreases with position (simulating ranking)
            score = random.uniform(0.5, 1.0) / position
            
            cursor.execute(
                """
                INSERT INTO search_results_shown (search_id, content_id, position, score)
                VALUES (%s, %s, %s, %s)
                """,
                (search_id, content_id, position, score),
            )
            results_shown += 1
            
            # Simulate clicks (higher positions more likely to be clicked)
            # Position bias: position 1 has 50% click rate, decreases for lower positions
            position_bias = 1.0 / position
            click_probability = click_rate * position_bias
            
            if random.random() < click_probability:
                # Insert click log
                cursor.execute(
                    """
                    INSERT INTO click_logs (search_id, content_id, position, query, role, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (search_id, content_id, position, query, role, timestamp),
                )
                clicks_created += 1
        
        # Commit every 10 searches
        if (i + 1) % 10 == 0:
            conn.commit()
            print(f"  Progress: {i + 1}/{num_searches} searches generated...")
    
    # Final commit
    conn.commit()
    cursor.close()
    
    print(f"\n✓ Generated:")
    print(f"  {searches_created} searches")
    print(f"  {results_shown} results shown")
    print(f"  {clicks_created} clicks")


def update_content_stats(conn):
    """Calculate and update content statistics."""
    print("\nUpdating content statistics...")
    
    cursor = conn.cursor()
    
    # Clear existing stats
    cursor.execute("DELETE FROM content_stats")
    
    # Calculate impressions and clicks per content
    cursor.execute(
        """
        INSERT INTO content_stats (content_id, impressions, clicks)
        SELECT 
            s.content_id,
            COUNT(DISTINCT s.id) as impressions,
            COUNT(DISTINCT c.id) as clicks
        FROM search_results_shown s
        LEFT JOIN click_logs c ON s.search_id = c.search_id AND s.content_id = c.content_id
        GROUP BY s.content_id
        """
    )
    
    stats_count = cursor.rowcount
    conn.commit()
    cursor.close()
    
    print(f"✓ Updated statistics for {stats_count} content items")


def verify_data(conn):
    """Verify generated data."""
    cursor = conn.cursor(dictionary=True)
    
    print("\n" + "="*60)
    print("Training Data Verification:")
    print("="*60)
    
    # Count searches
    cursor.execute("SELECT COUNT(*) as count FROM search_logs")
    search_count = cursor.fetchone()["count"]
    print(f"✓ Searches: {search_count}")
    
    # Count clicks
    cursor.execute("SELECT COUNT(*) as count FROM click_logs")
    click_count = cursor.fetchone()["count"]
    print(f"✓ Clicks: {click_count}")
    
    # Count results shown
    cursor.execute("SELECT COUNT(*) as count FROM search_results_shown")
    shown_count = cursor.fetchone()["count"]
    print(f"✓ Results shown: {shown_count}")
    
    # Content stats
    cursor.execute("SELECT COUNT(*) as count FROM content_stats")
    stats_count = cursor.fetchone()["count"]
    print(f"✓ Content stats: {stats_count}")
    
    # Click-through rate
    if shown_count > 0:
        ctr = (click_count / shown_count) * 100
        print(f"✓ Overall CTR: {ctr:.2f}%")
    
    # Sample high CTR content
    cursor.execute(
        """
        SELECT content_id, impressions, clicks, ctr
        FROM content_stats
        WHERE impressions > 5
        ORDER BY ctr DESC
        LIMIT 5
        """
    )
    top_content = cursor.fetchall()
    
    if top_content:
        print(f"\nTop 5 content by CTR:")
        for row in top_content:
            print(f"  {row['content_id']}: {row['clicks']}/{row['impressions']} clicks (CTR: {row['ctr']*100:.1f}%)")
    
    print("="*60)
    
    # Check if we have enough data for training
    if search_count >= 50 and click_count >= 20:
        print("\n✓ Sufficient data for ranking model training!")
        print("  Run: python scripts/train/train_ranking_model.py")
    else:
        print(f"\n⚠ Need more data for training:")
        print(f"  Minimum: 50 searches, 20 clicks")
        print(f"  Current: {search_count} searches, {click_count} clicks")
    
    cursor.close()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for ranking model"
    )
    parser.add_argument(
        "--searches",
        type=int,
        default=100,
        help="Number of searches to generate (default: 100)",
    )
    parser.add_argument(
        "--click-rate",
        type=float,
        default=0.3,
        help="Base click rate 0-1 (default: 0.3 = 30%%)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing logs before generating new data",
    )
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Generate Training Data for Ranking Model")
    print("="*60 + "\n")
    
    print(f"Configuration:")
    print(f"  Searches: {args.searches}")
    print(f"  Click rate: {args.click_rate * 100:.0f}%")
    print(f"  Clear existing: {args.clear}")
    print()
    
    # Connect to database
    conn = get_connection()
    
    try:
        # Clear existing data if requested
        if args.clear:
            print("Clearing existing logs...")
            cursor = conn.cursor()
            cursor.execute("DELETE FROM click_logs")
            cursor.execute("DELETE FROM search_results_shown")
            cursor.execute("DELETE FROM search_logs")
            cursor.execute("DELETE FROM content_stats")
            conn.commit()
            cursor.close()
            print("✓ Existing logs cleared\n")
        
        # Generate data
        generate_search_logs(conn, args.searches, args.click_rate)
        update_content_stats(conn)
        verify_data(conn)
        
        print("\n✓ Training data generation completed!")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
