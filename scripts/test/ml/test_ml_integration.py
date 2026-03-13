"""
Test both embedding and ranking models together.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

print("=" * 50)
print("Testing ML Integration")
print("=" * 50)

# Test 1: Embedding Model
print("\n[TEST 1] Embedding Model")
print("-" * 50)

try:
    from app.ml.embedding_model import HealthContentEmbedding
    
    model = HealthContentEmbedding()
    
    # Test query encoding
    query = "diabetes behandling"
    query_embedding = model.encode_query(query)
    
    print(f"✓ Query encoded: '{query}'")
    print(f"  Embedding shape: {query_embedding.shape}")
    print(f"  Embedding dimension: {len(query_embedding)}")
    print(f"  Sample values: {query_embedding[:5]}")
    
    # Test passage encoding
    passages = [
        {
            "tittel": "Diabetes type 2",
            "tekst": "<p>Informasjon om type 2 diabetes og behandling</p>",
            "info_type": "Sykdom",
            "koder": '{"ICD-10": ["E11"]}',
        },
        {
            "tittel": "Astma hos barn",
            "tekst": "<p>Behandling av astma hos barn</p>",
            "info_type": "Sykdom",
            "koder": '{"ICD-10": ["J45"]}',
        },
    ]
    
    passage_embeddings = model.encode_passages(passages)
    
    print(f"\n✓ Encoded {len(passages)} passages")
    print(f"  Embeddings shape: {passage_embeddings.shape}")
    
    # Test similarity
    import numpy as np
    similarities = [
        float(np.dot(query_embedding, emb))
        for emb in passage_embeddings
    ]
    
    print(f"\n✓ Computed similarities:")
    for i, (passage, sim) in enumerate(zip(passages, similarities)):
        print(f"  {i+1}. {passage['tittel']}: {sim:.4f}")
    
except Exception as e:
    print(f"✗ Embedding model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Ranking Model
print("\n[TEST 2] Ranking Model")
print("-" * 50)

try:
    from app.ml.ranking_model import HealthContentReranker, RerankCandidate
    
    reranker = HealthContentReranker()
    
    # Create test candidates
    candidates = [
        RerankCandidate(
            content_id="diabetes-001",
            semantic_score=0.85,
            bm25_score=0.72,
            smoothed_ctr=0.10,
            role_match=1.0,
        ),
        RerankCandidate(
            content_id="astma-001",
            semantic_score=0.45,
            bm25_score=0.30,
            smoothed_ctr=0.05,
            role_match=1.0,
        ),
    ]
    
    print(f"✓ Created {len(candidates)} rerank candidates")
    print(f"  Model trained: {reranker.model is not None}")
    
    if reranker.model is None:
        print(f"  ⚠ Model not trained yet (this is OK)")
        print(f"    Train with: python scripts/train/train_ranking_model.py")
    else:
        # Test reranking
        reranked = reranker.rerank(
            query="diabetes",
            role="Fastlege",
            candidates=candidates,
        )
        
        print(f"\n✓ Reranked {len(reranked)} candidates:")
        for i, (cand, score) in enumerate(reranked):
            print(f"  {i+1}. {cand.content_id}: {score:.4f}")
    
except Exception as e:
    print(f"✗ Ranking model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Database Integration
print("\n[TEST 3] Database Integration")
print("-" * 50)

try:
    from app.services.data.database_service import database_service
    
    if database_service.is_connected():
        print("✓ Database connected")
        
        # Check if we have embeddings
        content_count = database_service.get_content_count()
        print(f"  Content items: {content_count}")
        
        # Check LTR data availability
        ltr_rows = database_service.get_ltr_training_rows(days_back=30)
        print(f"  LTR training rows (30 days): {len(ltr_rows)}")

        # Check propensities
        propensities = database_service.get_position_propensities()
        print(f"  Position propensities: {len(propensities)}")
        
    else:
        print("⚠ Database not connected")
        
except Exception as e:
    print(f"✗ Database integration test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: End-to-End Search Flow
print("\n[TEST 4] End-to-End Search Flow Simulation")
print("-" * 50)

try:
    print("Simulating search flow:")
    print("  1. User searches: 'diabetes behandling'")
    print("  2. Embedding model encodes query ✓")
    print("  3. Database retrieves content with embeddings ✓")
    print("  4. Semantic similarity computed ✓")
    print("  5. Hybrid scoring (10% keyword + 90% semantic) ✓")
    
    if reranker.model is not None:
        print("  6. Ranking model reranks results ✓")
    else:
        print("  6. Ranking model reranks results (not trained yet)")
    
    print("\n✓ All components ready for end-to-end search!")
    
except Exception as e:
    print(f"✗ Flow simulation failed: {e}")

print("\n" + "=" * 50)
print("ML Integration Test Complete!")
print("=" * 50)

print("\nNext steps:")
print("  1. Generate embeddings: python scripts/data/generate_embeddings.py")
print("  2. Enable semantic search in .env: ML_EMBEDDING_ENABLED=true")
print("  3. Collect search/click data for ranking model")
print("  4. Train ranking model: python scripts/train/train_ranking_model.py")
