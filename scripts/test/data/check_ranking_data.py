"""
Check if there's enough data to train the ranking model.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.data.database_service import database_service

print("=" * 60)
print("Checking Ranking Model Training Data Availability")
print("=" * 60)

if not database_service.is_connected():
    print("\n✗ Database not connected")
    sys.exit(1)

print("\n[1] Search Logs")
print("-" * 60)
search_count = database_service.get_search_count()
print(f"Total searches logged: {search_count}")

print("\n[2] Click Logs")
print("-" * 60)
click_count = database_service.get_click_count()
print(f"Total clicks logged: {click_count}")

print("\n[3] LTR Training Data (Last 180 days)")
print("-" * 60)
ltr_rows = database_service.get_ltr_training_rows(days_back=180)
print(f"Total training rows: {len(ltr_rows)}")

if ltr_rows:
    # Count searches with clicks
    searches_with_clicks = {}
    for row in ltr_rows:
        search_id = row['search_id']
        if search_id not in searches_with_clicks:
            searches_with_clicks[search_id] = {'total': 0, 'clicked': 0}
        searches_with_clicks[search_id]['total'] += 1
        if row['clicked']:
            searches_with_clicks[search_id]['clicked'] += 1
    
    # Filter searches that have at least one click
    searches_with_any_click = {
        sid: data for sid, data in searches_with_clicks.items()
        if data['clicked'] > 0
    }
    
    # Filter searches with at least 5 results (min_group_size)
    searches_min_5_results = {
        sid: data for sid, data in searches_with_any_click.items()
        if data['total'] >= 5
    }
    
    print(f"\nSearches with results: {len(searches_with_clicks)}")
    print(f"Searches with at least 1 click: {len(searches_with_any_click)}")
    print(f"Searches with ≥5 results + clicks: {len(searches_min_5_results)}")
    
    if searches_min_5_results:
        print(f"\n✓ Enough data to train!")
        print(f"  Training groups: {len(searches_min_5_results)}")
        total_rows = sum(data['total'] for data in searches_min_5_results.values())
        print(f"  Training rows: {total_rows}")
        
        # Show sample
        print(f"\n  Sample search groups:")
        for i, (sid, data) in enumerate(list(searches_min_5_results.items())[:3]):
            print(f"    {i+1}. Search {sid}: {data['total']} results, {data['clicked']} clicks")
    else:
        print(f"\n✗ Not enough data to train")
        print(f"\n  Requirements:")
        print(f"    - At least 1 search with ≥5 results AND at least 1 click")
        print(f"\n  Current status:")
        print(f"    - Searches with any results: {len(searches_with_clicks)}")
        print(f"    - Searches with clicks: {len(searches_with_any_click)}")
        print(f"    - Searches meeting both criteria: {len(searches_min_5_results)}")
        
        if searches_with_any_click:
            # Show why searches are excluded
            excluded = {
                sid: data for sid, data in searches_with_any_click.items()
                if data['total'] < 5
            }
            if excluded:
                print(f"\n  Searches excluded (too few results):")
                for i, (sid, data) in enumerate(list(excluded.items())[:5]):
                    print(f"    - Search {sid}: {data['total']} results (need ≥5)")
else:
    print("No LTR training data found")
    print("\nTo collect training data:")
    print("  1. Users must perform searches")
    print("  2. Backend logs search_id and results shown (search_results_shown)")
    print("  3. Users click on results")
    print("  4. Backend logs clicks with search_id (click_logs)")

print("\n" + "=" * 60)

if ltr_rows and len(searches_min_5_results) > 0:
    print("✓ Ready to train! Run:")
    print("  python scripts/train/train_ranking_model.py")
else:
    print("⚠ Not ready to train yet")
    print("  Continue using the application to collect more data")

print("=" * 60)
