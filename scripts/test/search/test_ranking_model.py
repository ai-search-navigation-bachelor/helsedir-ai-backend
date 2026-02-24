"""
Test script for ranking model.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 50)
print("Testing Ranking Model Import")
print("=" * 50)

try:
    from app.ml.ranking_model import HealthContentReranker, RerankCandidate
    print("✓ Successfully imported HealthContentReranker and RerankCandidate")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("Testing Reranker Initialization")
print("=" * 50)

try:
    reranker = HealthContentReranker()
    print("✓ Successfully created HealthContentReranker instance")
    print(f"  Feature names: {reranker.feature_names}")
    print(f"  Model trained: {reranker.model is not None}")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("Testing RerankCandidate Creation")
print("=" * 50)

try:
    candidate = RerankCandidate(
        content_id="test-123",
        position=1,
        semantic_score=0.85,
        bm25_score=0.72,
        rrf_score=0.80,
        type_match=1.0,
        role_match=1.0,
        maalgruppe_match=1.0,
        smoothed_ctr=0.05,
    )
    print("✓ Successfully created RerankCandidate")
    print(f"  Content ID: {candidate.content_id}")
    print(f"  Semantic score: {candidate.semantic_score}")
    print(f"  Position: {candidate.position}")
except Exception as e:
    print(f"✗ Candidate creation failed: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("Testing Database Service Methods")
print("=" * 50)

try:
    from app.services.data.database_service import database_service
    
    # Test if database is connected
    if database_service.is_connected():
        print("✓ Database connected")
        
        # Test new methods
        print("\nTesting get_content_stats_bulk():")
        stats = database_service.get_content_stats_bulk()
        print(f"  Retrieved stats for {len(stats)} content items")
        if stats:
            sample_id = list(stats.keys())[0]
            print(f"  Sample: {sample_id} = {stats[sample_id]}")
        
        print("\nTesting get_position_propensities():")
        propensities = database_service.get_position_propensities()
        print(f"  Retrieved propensities for {len(propensities)} positions")
        if propensities:
            print(f"  Propensities: {propensities}")
        
        print("\nTesting get_ltr_training_rows():")
        rows = database_service.get_ltr_training_rows(days_back=30)
        print(f"  Retrieved {len(rows)} training rows")
        if rows:
            print(f"  Sample row keys: {list(rows[0].keys())}")
    else:
        print("⚠ Database not connected (this is OK for basic testing)")
        
except Exception as e:
    print(f"✗ Database service test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("All Tests Completed!")
print("=" * 50)
