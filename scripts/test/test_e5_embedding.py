#!/usr/bin/env python3
"""
Test script to verify E5 embedding model integration.

Tests:
1. Model loading
2. Passage formatting with metadata
3. Query encoding with "query:" prefix
4. Document encoding with "passage:" prefix
5. Similarity computation

Usage:
    python scripts/test/test_e5_embedding.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np


def test_model_loading():
    """Test that E5 model loads correctly."""
    print("\n" + "="*60)
    print("TEST 1: Model Loading")
    print("="*60)
    
    from app.ml.embedding_model import HealthContentEmbedding
    
    model = HealthContentEmbedding(model_name="intfloat/multilingual-e5-base")
    print("✓ Model initialized")
    print(f"  Model name: {model.model_name}")
    
    # Trigger lazy loading
    test_text = ["test"]
    _ = model.encode(test_text, is_query=True)
    print("✓ Model loaded from HuggingFace")
    print(f"  Embedding dimension: {model.model.get_sentence_embedding_dimension()}")
    
    return model


def test_passage_formatting():
    """Test passage formatting with metadata."""
    print("\n" + "="*60)
    print("TEST 2: Passage Formatting")
    print("="*60)
    
    from app.ml.embedding_model import HealthContentEmbedding
    
    # Sample content item with all fields
    content_item = {
        "title": "Diabetes type 2",
        "content_type": "retningslinje",
        "icd10": "E11",
        "icpc2": "T90",
        "snomed_ct": "44054006",
        "lis_spesialitet": "Allmennmedisin",
        "lis_læringsmål": "Diagnostikk av diabetes",
        "body": "<p>Diabetes type 2 er en <strong>kronisk</strong> sykdom.</p>"
    }
    
    passage = HealthContentEmbedding.format_passage(content_item)
    print("✓ Passage formatted")
    print(f"\nFormatted passage:\n{passage}\n")
    
    # Verify HTML is stripped
    assert "<p>" not in passage
    assert "<strong>" not in passage
    assert "kronisk" in passage
    print("✓ HTML tags removed correctly")
    
    # Verify all fields are included
    assert "Tittel: Diabetes type 2" in passage
    assert "Type: retningslinje" in passage
    assert "ICD-10: E11" in passage
    assert "ICPC-2: T90" in passage
    assert "SNOMED-CT: 44054006" in passage
    assert "LIS-spesialitet: Allmennmedisin" in passage
    assert "LIS-læringsmål: Diagnostikk av diabetes" in passage
    print("✓ All metadata fields included")
    
    return passage


def test_query_encoding(model):
    """Test query encoding with 'query:' prefix."""
    print("\n" + "="*60)
    print("TEST 3: Query Encoding")
    print("="*60)
    
    query = "diabetes behandling"
    
    query_emb = model.encode_query(query)
    print(f"✓ Query encoded: '{query}'")
    print(f"  Embedding shape: {query_emb.shape}")
    print(f"  Embedding norm: {np.linalg.norm(query_emb):.4f}")
    
    # Verify L2 normalization (should be ~1.0)
    norm = np.linalg.norm(query_emb)
    assert 0.99 <= norm <= 1.01, f"Expected norm ~1.0, got {norm}"
    print("✓ L2 normalized (norm ≈ 1.0)")
    
    return query_emb


def test_document_encoding(model, passage):
    """Test document encoding with 'passage:' prefix."""
    print("\n" + "="*60)
    print("TEST 4: Document Encoding")
    print("="*60)
    
    doc_emb = model.encode([passage], is_query=False)[0]
    print(f"✓ Document encoded")
    print(f"  Embedding shape: {doc_emb.shape}")
    print(f"  Embedding norm: {np.linalg.norm(doc_emb):.4f}")
    
    # Verify L2 normalization
    norm = np.linalg.norm(doc_emb)
    assert 0.99 <= norm <= 1.01, f"Expected norm ~1.0, got {norm}"
    print("✓ L2 normalized (norm ≈ 1.0)")
    
    return doc_emb


def test_similarity(model):
    """Test similarity computation."""
    print("\n" + "="*60)
    print("TEST 5: Similarity Computation")
    print("="*60)
    
    # Create test documents
    doc1 = {"title": "Diabetes type 2", "body": "Diabetes er en kronisk sykdom med høyt blodsukker"}
    doc2 = {"title": "Hjerteinfarkt", "body": "Hjerteinfarkt er en alvorlig tilstand med brystsmerter"}
    doc3 = {"title": "Diabetes hos barn", "body": "Barn med diabetes type 1 trenger insulin"}
    
    # Format as passages
    passages = [
        model.format_passage(doc1),
        model.format_passage(doc2),
        model.format_passage(doc3),
    ]
    
    # Encode documents
    doc_embeddings = model.encode(passages, is_query=False)
    print(f"✓ Encoded {len(passages)} documents")
    
    # Encode query
    query = "diabetes behandling"
    query_emb = model.encode_query(query)
    
    # Compute similarities
    similarities = model.compute_similarity(query_emb, doc_embeddings)
    print(f"\n✓ Computed similarities for query: '{query}'")
    
    for i, (doc, sim) in enumerate(zip([doc1, doc2, doc3], similarities)):
        print(f"  {i+1}. {doc['title']}: {sim:.4f}")
    
    # Verify: diabetes docs should be more similar to diabetes query than heart attack
    assert similarities[0] > similarities[1], "Doc1 (diabetes) should be > Doc2 (hjerteinfarkt)"
    assert similarities[2] > similarities[1], "Doc3 (diabetes barn) should be > Doc2 (hjerteinfarkt)"
    print("\n✓ Similarity ranking correct (diabetes docs > heart attack)")
    
    # Verify similarities are in valid range [-1, 1] but typically [0, 1] for normalized
    assert all(-1 <= s <= 1 for s in similarities), "Similarities should be in [-1, 1]"
    print("✓ Similarities in valid range")


def test_batch_encoding(model):
    """Test batch encoding performance."""
    print("\n" + "="*60)
    print("TEST 6: Batch Encoding")
    print("="*60)
    
    # Create multiple documents
    docs = [
        {"title": f"Document {i}", "body": f"This is document number {i} about health"}
        for i in range(10)
    ]
    
    # Batch encode
    embeddings = model.encode_passages(docs, show_progress_bar=False)
    print(f"✓ Batch encoded {len(docs)} documents")
    print(f"  Output shape: {embeddings.shape}")
    print(f"  Expected: ({len(docs)}, 768)")
    
    assert embeddings.shape == (10, 768), f"Expected (10, 768), got {embeddings.shape}"
    print("✓ Batch encoding shape correct")


def main():
    print("\n" + "="*60)
    print("E5 EMBEDDING MODEL TEST SUITE")
    print("="*60)
    print("\nModel: intfloat/multilingual-e5-base")
    print("Source: https://huggingface.co/intfloat/multilingual-e5-base")
    
    try:
        # Run tests
        model = test_model_loading()
        passage = test_passage_formatting()
        query_emb = test_query_encoding(model)
        doc_emb = test_document_encoding(model, passage)
        test_similarity(model)
        test_batch_encoding(model)
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nE5 model integration is working correctly!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install sentence-transformers")
        print("  2. Generate embeddings: python scripts/data/generate_embeddings.py")
        print("  3. Enable in .env: ML_EMBEDDING_ENABLED=true")
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ TEST FAILED")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
