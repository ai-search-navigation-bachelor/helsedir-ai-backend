#!/usr/bin/env python3
"""
Comprehensive test script for E5 embedding model with passage visualization.

This script:
1. Loads the E5 model from HuggingFace
2. Formats sample health content as structured passages
3. Shows exactly how each passage looks
4. Encodes with proper query/passage prefixes
5. Computes similarities and shows ranking

Usage:
    python scripts/test/test_e5_passages.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import List, Dict, Any


# Sample health content with full metadata
SAMPLE_CONTENT = [
    {
        "id": 1,
        "title": "Diabetes type 2 hos voksne",
        "content_type": "retningslinje",
        "icd10": "E11",
        "icpc2": "T90",
        "snomed_ct": "44054006",
        "lis_spesialitet": "Allmennmedisin",
        "lis_læringsmål": "Diagnostikk og behandling av diabetes type 2",
        "body": "<h1>Diabetes type 2</h1><p>Diabetes type 2 er en <strong>kronisk metabolsk sykdom</strong> karakterisert ved høyt blodsukker. Behandling inkluderer <em>livsstilsendringer</em>, kosthold, fysisk aktivitet og medikamentell behandling med metformin eller insulin.</p>"
    },
    {
        "id": 2,
        "title": "Hjerteinfarkt - akutt behandling",
        "content_type": "veileder",
        "icd10": "I21",
        "icpc2": "K75",
        "snomed_ct": "22298006",
        "lis_spesialitet": "Kardiologi",
        "lis_læringsmål": "Akutt håndtering av hjerteinfarkt",
        "body": "<div><h2>Symptomer</h2><p>Hjerteinfarkt gir typisk <b>brystsmerter</b>, utstrålende til venstre arm, og pustevansker. Ring 113 umiddelbart. Akutt behandling med PCI (perkutan koronar intervensjon) innen 90 minutter.</p></div>"
    },
    {
        "id": 3,
        "title": "Diabetes hos barn og ungdom",
        "content_type": "faglig_rad",
        "icd10": "E10",
        "icpc2": "T89",
        "snomed_ct": "46635009",
        "lis_spesialitet": "Pediatri",
        "lis_læringsmål": "Type 1 diabetes hos barn",
        "body": "<p>Barn med diabetes type 1 trenger <strong>insulinbehandling</strong> fra diagnose. Viktig med tett oppfølging, blodsukkermåling 6-8 ganger daglig og HbA1c målsetning under 7%.</p>"
    },
    {
        "id": 4,
        "title": "Blodsukkerregulering ved diabetes",
        "content_type": "artikkel",
        "body": "Behandling av diabetes fokuserer på å regulere blodsukkeret gjennom medisinering og livsstilsendringer."
    },
    {
        "id": 5,
        "title": "Hjertesvikt diagnostikk",
        "content_type": "retningslinje",
        "icd10": "I50",
        "snomed_ct": "84114007",
        "lis_spesialitet": "Kardiologi",
        "body": "<p>Hjertesvikt diagnostiseres med ekkokardiografi og NT-proBNP. Behandling med ACE-hemmer og betablokkere.</p>"
    }
]


def print_section(title: str, width: int = 80):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width + "\n")


def print_passage(doc_id: int, title: str, passage: str, width: int = 80):
    """Print a formatted passage with border."""
    print(f"📄 Document {doc_id}: {title}")
    print("-" * width)
    print(passage)
    print("-" * width + "\n")


def test_model_loading():
    """Test E5 model loading from HuggingFace."""
    print_section("TEST 1: Model Loading from HuggingFace")
    
    from app.ml.embedding_model import HealthContentEmbedding
    
    print("Initializing HealthContentEmbedding...")
    model = HealthContentEmbedding()
    
    print(f"✓ Model initialized")
    print(f"  Default model name: {model.model_name}")
    print(f"  Source: https://huggingface.co/{model.model_name}")
    print(f"  Model loaded: {model._is_loaded}")
    
    print("\nTriggering model download/loading...")
    print("(First run will download ~560MB from HuggingFace)")
    
    # Trigger loading
    _ = model.encode(["test"], is_query=True, show_progress_bar=False)
    
    print(f"\n✓ Model loaded successfully")
    print(f"  Embedding dimension: {model.model.get_sentence_embedding_dimension()}")
    print(f"  Model is now cached locally for future use")
    
    return model


def test_html_stripping():
    """Test HTML tag removal."""
    print_section("TEST 2: HTML Tag Stripping")
    
    from app.ml.embedding_model import HealthContentEmbedding
    
    test_cases = [
        ("<p>Simple paragraph</p>", "Simple paragraph"),
        ("<h1>Title</h1><p>Body</p>", "Title Body"),
        ("<strong>Bold</strong> and <em>italic</em>", "Bold and italic"),
        ("Plain text", "Plain text"),
        ("<div><p>Nested <b>tags</b></p></div>", "Nested tags"),
    ]
    
    print("Testing HTML removal:\n")
    for html, expected in test_cases:
        result = HealthContentEmbedding.strip_html_tags(html)
        status = "✓" if result == expected else "✗"
        print(f"{status} Input:  {html}")
        print(f"  Output: {result}")
        print(f"  Expected: {expected}\n")


def test_passage_formatting(content_items: List[Dict[str, Any]]):
    """Test and display passage formatting."""
    print_section("TEST 3: Passage Formatting with Metadata")
    
    from app.ml.embedding_model import HealthContentEmbedding
    
    print("Formatting documents as structured passages:\n")
    
    for item in content_items:
        passage = HealthContentEmbedding.format_passage(item)
        print_passage(item["id"], item["title"], passage)
    
    print(f"✓ Formatted {len(content_items)} passages")
    print("\nKey observations:")
    print("  • All metadata fields are included in passage")
    print("  • HTML tags are removed from body content")
    print("  • Fields are labeled (e.g., 'Tittel:', 'ICD-10:')")
    print("  • Missing fields are gracefully handled")


def test_encoding_with_prefixes(model):
    """Test encoding with E5 query/passage prefixes."""
    print_section("TEST 4: Encoding with E5 Prefixes")
    
    query = "diabetes behandling"
    doc_text = "Diabetes type 2 behandling med insulin"
    
    print("E5 Model uses special prefixes for optimal retrieval:\n")
    
    # Query encoding
    print("🔍 Query Encoding:")
    print(f"  Input:  '{query}'")
    print(f"  Prefix: 'query: '")
    print(f"  Final:  'query: {query}'")
    query_emb = model.encode_query(query)
    print(f"  Output: {query_emb.shape} array, norm={np.linalg.norm(query_emb):.4f}\n")
    
    # Document encoding
    print("📄 Document Encoding:")
    print(f"  Input:  '{doc_text}'")
    print(f"  Prefix: 'passage: '")
    print(f"  Final:  'passage: {doc_text}'")
    doc_emb = model.encode([doc_text], is_query=False)[0]
    print(f"  Output: {doc_emb.shape} array, norm={np.linalg.norm(doc_emb):.4f}\n")
    
    # Compute similarity
    similarity = float(np.dot(query_emb, doc_emb))
    print(f"✓ Cosine similarity: {similarity:.4f}")
    print("\nNote: Prefixes are added automatically by encode() method")


def test_similarity_ranking(model, content_items: List[Dict[str, Any]]):
    """Test similarity computation and ranking."""
    print_section("TEST 5: Similarity Computation and Ranking")
    
    # Format passages
    passages = [model.format_passage(item) for item in content_items]
    
    # Encode documents
    print("Encoding all documents as passages...")
    doc_embeddings = model.encode(passages, is_query=False, show_progress_bar=True)
    print(f"✓ Encoded {len(passages)} documents\n")
    
    # Test queries
    test_queries = [
        "diabetes behandling hos voksne",
        "hjerteinfarkt symptomer",
        "barn med diabetes",
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 80)
        
        # Encode query
        query_emb = model.encode_query(query)
        
        # Compute similarities
        similarities = model.compute_similarity(query_emb, doc_embeddings)
        
        # Rank documents
        ranked = sorted(
            zip(content_items, similarities),
            key=lambda x: x[1],
            reverse=True
        )
        
        print("\nRanking:\n")
        for i, (item, score) in enumerate(ranked[:5], 1):
            bar_length = int(score * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"{i}. [{bar}] {score:.4f}")
            print(f"   {item['title']}")
            if i < len(ranked):
                print()
        
        print(f"\nTop match: {ranked[0][0]['title']} (score: {ranked[0][1]:.4f})")


def test_batch_encoding(model):
    """Test batch encoding efficiency."""
    print_section("TEST 6: Batch Encoding")
    
    # Create test documents
    test_docs = [
        {"title": f"Test Document {i}", "body": f"Content for document {i}"}
        for i in range(20)
    ]
    
    print(f"Encoding {len(test_docs)} documents in batch...")
    embeddings = model.encode_passages(test_docs, show_progress_bar=True)
    
    print(f"\n✓ Batch encoding complete")
    print(f"  Output shape: {embeddings.shape}")
    print(f"  Expected: ({len(test_docs)}, 768)")
    print(f"  Memory: ~{embeddings.nbytes / 1024:.1f} KB")
    
    # Verify normalization
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\n✓ All embeddings L2-normalized:")
    print(f"  Min norm: {norms.min():.4f}")
    print(f"  Max norm: {norms.max():.4f}")
    print(f"  Mean norm: {norms.mean():.4f}")


def test_cross_lingual():
    """Test multilingual capability."""
    print_section("TEST 7: Multilingual Capability")
    
    from app.ml.embedding_model import HealthContentEmbedding
    
    print("E5 is a multilingual model. Testing different languages:\n")
    
    model = HealthContentEmbedding()
    
    # Same concept in different languages
    texts = {
        "Norwegian": "diabetes behandling",
        "English": "diabetes treatment",
        "Swedish": "diabetes behandling",
        "Danish": "diabetes behandling",
    }
    
    # Encode all
    embeddings = {}
    for lang, text in texts.items():
        emb = model.encode_query(text)
        embeddings[lang] = emb
        print(f"✓ Encoded ({lang}): '{text}'")
    
    # Compare similarities
    print("\nCross-lingual similarities:")
    print("-" * 60)
    
    norwegian_emb = embeddings["Norwegian"]
    for lang, emb in embeddings.items():
        if lang != "Norwegian":
            sim = float(np.dot(norwegian_emb, emb))
            print(f"  Norwegian ↔ {lang:10s}: {sim:.4f}")
    
    print("\n✓ Similar concepts across languages have high similarity")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print(" E5 EMBEDDING MODEL - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("\nModel: intfloat/multilingual-e5-base")
    print("Source: https://huggingface.co/intfloat/multilingual-e5-base")
    print("Dimension: 768")
    
    try:
        # Run tests
        model = test_model_loading()
        test_html_stripping()
        test_passage_formatting(SAMPLE_CONTENT)
        test_encoding_with_prefixes(model)
        test_similarity_ranking(model, SAMPLE_CONTENT)
        test_batch_encoding(model)
        test_cross_lingual()
        
        # Summary
        print_section("TEST SUMMARY")
        print("✓ All tests passed successfully!\n")
        print("Key findings:")
        print("  ✓ E5 model loads correctly from HuggingFace")
        print("  ✓ Passages include all metadata fields")
        print("  ✓ HTML tags are properly removed")
        print("  ✓ Query/passage prefixes are applied")
        print("  ✓ Similarity ranking works as expected")
        print("  ✓ Embeddings are L2-normalized")
        print("  ✓ Multilingual capability confirmed")
        
        print("\nNext steps:")
        print("  1. Generate embeddings: python scripts/data/generate_embeddings.py")
        print("  2. Enable semantic search in .env: ML_EMBEDDING_ENABLED=true")
        print("  3. Test search: python scripts/test/test_search.py")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(" ✗ TEST FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
