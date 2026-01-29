#!/usr/bin/env python3
"""
Fine-tune embedding model using GPL (Generative Pseudo Labeling).

GPL uses an LLM to generate synthetic queries for each document, then trains
the embedding model contrastively to match queries to their source documents.

Usage:
    python scripts/train/finetune_gpl.py
    python scripts/train/finetune_gpl.py --queries-per-doc 5
    python scripts/train/finetune_gpl.py --skip-generation  # Use cached queries

Requirements:
    GROQ_API_KEY in .env (free from https://console.groq.com/keys)
"""

import argparse
import json
import sys
import time
import random
from pathlib import Path
from typing import List, Dict, Any

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def generate_queries_with_groq(
    api_key: str,
    title: str,
    body: str,
    info_type: str,
    num_queries: int = 3,
    _retries: int = 0,
) -> List[str]:
    """
    Use Groq API (free tier) to generate synthetic search queries.
    """
    import httpx

    body_truncated = body[:1500] if body else ""

    prompt = f"""Du er en ekspert på norsk helseinformasjon. Generer {num_queries} realistiske søkeord/fraser som en helsepersonell ville brukt for å finne dette dokumentet.

Dokumenttype: {info_type}
Tittel: {title}
Innhold: {body_truncated}

Regler:
- Skriv søk slik en lege eller sykepleier ville skrevet dem
- Bruk norske medisinske termer
- Varier mellom korte (1-2 ord) og lengre (3-5 ord) søk
- Ikke bruk anførselstegn eller nummerering

Returner kun søkefrasene, én per linje:"""

    try:
        response = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
                "max_tokens": 200,
            },
            timeout=30.0,
        )

        if response.status_code == 429:
            if _retries >= 3:
                print("  Rate limited 3 times, skipping")
                return []
            wait = 15 * (_retries + 1)
            print(f"  Rate limited, waiting {wait}s (retry {_retries + 1}/3)...")
            time.sleep(wait)
            return generate_queries_with_groq(api_key, title, body, info_type, num_queries, _retries + 1)

        if response.status_code != 200:
            print(f"  Groq error: {response.status_code} - {response.text[:200]}")
            return []

        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
        queries = [q.strip() for q in text.split("\n") if q.strip()]

        # Clean up
        cleaned = []
        for q in queries[:num_queries]:
            q = q.lstrip("0123456789.-) ")
            q = q.strip('"\'')
            if len(q) > 2 and len(q) < 100:
                cleaned.append(q)

        return cleaned

    except Exception as e:
        print(f"  Error with Groq: {e}")
        return []


def load_or_generate_queries(
    content_items: List[Dict[str, Any]],
    cache_path: Path,
    queries_per_doc: int,
    skip_generation: bool,
    api_key: str = "",
) -> Dict[str, List[str]]:
    """
    Load cached queries or generate new ones.

    Returns:
        Dict mapping content_id to list of queries
    """
    # Try to load from cache
    if cache_path.exists():
        print(f"Loading cached queries from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)

        if skip_generation:
            return cached

        # Check if we have queries for all items
        missing = [item for item in content_items if str(item["id"]) not in cached]
        if not missing:
            print(f"  All {len(cached)} documents have cached queries")
            return cached
        print(f"  Found {len(cached)} cached, {len(missing)} missing")
    else:
        cached = {}
        missing = content_items

    if skip_generation:
        print("Skipping generation, using only cached queries")
        return cached

    # Generate queries for missing documents
    print(f"\nGenerating queries for {len(missing)} documents...")
    print(f"  Queries per document: {queries_per_doc}")
    print(f"  Using: Groq API (llama-3.1-8b-instant)")

    from app.ml.embedding_model import HealthContentEmbedding

    failed = 0
    for i, item in enumerate(missing):
        content_id = str(item["id"])
        title = item.get("tittel") or item.get("title") or ""
        body = item.get("tekst") or item.get("body") or ""
        info_type = item.get("info_type") or ""

        # Clean body
        if body:
            body = HealthContentEmbedding.strip_html_tags(body)

        # Generate queries
        queries = generate_queries_with_groq(
            api_key=api_key,
            title=title,
            body=body,
            info_type=info_type,
            num_queries=queries_per_doc,
        )

        if queries:
            cached[content_id] = queries
            # Print first few to verify quality
            if len(cached) <= 3:
                print(f"\n  Doc {len(cached)}: {title[:60]}")
                for q in queries:
                    print(f"    -> {q}")
        else:
            failed += 1
            if failed <= 3:
                print(f"  Warning: No queries generated for '{title[:50]}'")

        # Progress
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(missing)}] Generated: {len(cached)}, Failed: {failed}")
            # Save intermediate results
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cached, f, ensure_ascii=False, indent=2)

        # Groq free tier: 30 req/min, use 3s delay to stay safe
        time.sleep(3)

    # Final save
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cached, f, ensure_ascii=False, indent=2)

    print(f"\n  Done! Generated: {len(cached)}, Failed: {failed}")
    print(f"  Saved to {cache_path}")
    return cached


def load_embeddings_from_db() -> Dict[str, "np.ndarray"]:
    """Load precomputed embeddings from database for hard negative mining."""
    import numpy as np
    from app.services.repositories.base import db_pool

    conn = db_pool.get_connection()
    if not conn:
        return {}

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, embedding FROM content WHERE embedding IS NOT NULL")
        rows = cursor.fetchall()
        embeddings = {}
        for content_id, embedding_bytes in rows:
            if embedding_bytes:
                embeddings[str(content_id)] = np.frombuffer(embedding_bytes, dtype=np.float32)
        print(f"  Loaded {len(embeddings)} embeddings for hard negative mining")
        return embeddings
    except Exception as e:
        print(f"  Warning: Could not load embeddings: {e}")
        return {}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def find_hard_negative(
    positive_id: str,
    candidate_ids: List[str],
    embeddings: Dict[str, "np.ndarray"],
) -> str:
    """
    Find the hardest negative: most similar to positive but not the positive itself.
    Returns the content_id of the hardest negative.
    """
    import numpy as np

    pos_emb = embeddings.get(positive_id)
    if pos_emb is None:
        return random.choice(candidate_ids)

    best_id = None
    best_sim = -2.0
    for cid in candidate_ids:
        emb = embeddings.get(cid)
        if emb is None:
            continue
        sim = float(np.dot(pos_emb, emb))
        if sim > best_sim:
            best_sim = sim
            best_id = cid

    return best_id or random.choice(candidate_ids)


def create_training_triplets(
    content_items: List[Dict[str, Any]],
    queries: Dict[str, List[str]],
    format_passage_fn,
) -> List[Dict[str, str]]:
    """
    Create (query, positive, negative) triplets for contrastive training.

    For each query:
    - Positive = the document the query was generated from
    - Negative = hardest negative (most similar embedding that isn't the positive)
    """
    # Load embeddings for hard negative mining
    embeddings = load_embeddings_from_db()
    use_hard_negatives = len(embeddings) > 0

    if use_hard_negatives:
        print("  Using hard negative mining (most similar non-positive document)")
    else:
        print("  Falling back to random negatives (no embeddings found)")

    triplets = []

    # Create id -> item mapping
    id_to_item = {str(item["id"]): item for item in content_items}

    # Create id -> passage mapping
    id_to_passage = {str(item["id"]): format_passage_fn(item) for item in content_items}

    all_ids = list(id_to_passage.keys())

    for content_id, query_list in queries.items():
        if content_id not in id_to_item:
            continue

        positive_passage = id_to_passage[content_id]
        candidate_ids = [cid for cid in all_ids if cid != content_id]

        # Find hard negative once per document (same negative for all queries of this doc)
        if use_hard_negatives:
            neg_id = find_hard_negative(content_id, candidate_ids, embeddings)
        else:
            neg_id = random.choice(candidate_ids)

        negative_passage = id_to_passage[neg_id]

        for query in query_list:
            triplets.append({
                "query": query,
                "positive": positive_passage,
                "negative": negative_passage,
            })

    return triplets


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune embedding model using GPL"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="intfloat/multilingual-e5-base",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/finetuned-e5-gpl",
        help="Directory to save fine-tuned model",
    )
    parser.add_argument(
        "--queries-per-doc",
        type=int,
        default=3,
        help="Number of queries to generate per document",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate (default: 5e-6, low to prevent catastrophic forgetting)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip query generation, use only cached queries",
    )
    args = parser.parse_args()

    # Check dependencies
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
        print("sentence-transformers available")
    except ImportError:
        print("Error: sentence-transformers required")
        sys.exit(1)

    from app.config import settings
    from app.services.data.database_service import database_service
    from app.ml.embedding_model import HealthContentEmbedding

    # Check database
    if not database_service.is_connected():
        print("Error: Cannot connect to database")
        sys.exit(1)

    # Check Groq API key
    if not args.skip_generation:
        if not settings.groq_api_key:
            print("Error: GROQ_API_KEY not configured in .env")
            print("Get a free key at: https://console.groq.com/keys")
            sys.exit(1)
        print(f"Using Groq API (llama-3.1-8b-instant)")

    # Load content
    print("\nLoading content from database...")
    content_items = database_service.get_all_content()
    print(f"Found {len(content_items)} content items")

    # Load or generate queries
    cache_path = project_root / "data" / "gpl_queries.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    queries = load_or_generate_queries(
        content_items,
        cache_path,
        args.queries_per_doc,
        args.skip_generation,
        api_key=settings.groq_api_key,
    )

    if not queries:
        print("Error: No queries generated or cached")
        sys.exit(1)

    total_queries = sum(len(q) for q in queries.values())
    print(f"\nTotal queries: {total_queries} for {len(queries)} documents")

    # Show examples
    print("\nExample queries:")
    for content_id, query_list in list(queries.items())[:3]:
        item = next((it for it in content_items if str(it["id"]) == content_id), None)
        if item:
            print(f"\n  Document: {item.get('tittel', '')[:50]}")
            for q in query_list[:3]:
                print(f"    - {q}")

    # Create training triplets
    print("\nCreating training triplets...")
    triplets = create_training_triplets(
        content_items,
        queries,
        HealthContentEmbedding.format_passage,
    )
    print(f"Created {len(triplets)} triplets")

    # Show example triplet
    if triplets:
        ex = triplets[0]
        print(f"\nExample triplet:")
        print(f"  Query: {ex['query']}")
        print(f"  Positive: {ex['positive'][:100]}...")
        print(f"  Negative: {ex['negative'][:100]}...")

    # Load base model
    print(f"\nLoading base model: {args.base_model}")
    model = SentenceTransformer(args.base_model)

    # Split into train / validation / test (70/15/15)
    random.shuffle(triplets)
    n = len(triplets)
    n_val = max(1, int(n * 0.15))
    n_test = max(1, int(n * 0.15))
    n_train = n - n_val - n_test

    train_triplets = triplets[:n_train]
    val_triplets = triplets[n_train:n_train + n_val]
    test_triplets = triplets[n_train + n_val:]

    print(f"  Train: {len(train_triplets)}, Validation: {len(val_triplets)}, Test: {len(test_triplets)}")

    # Create training data
    train_examples = [
        InputExample(texts=[t["query"], t["positive"], t["negative"]])
        for t in train_triplets
    ]

    train_dataloader = DataLoader(
        train_examples,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Create evaluators
    from sentence_transformers.evaluation import TripletEvaluator

    evaluator_val = TripletEvaluator(
        anchors=[t["query"] for t in val_triplets],
        positives=[t["positive"] for t in val_triplets],
        negatives=[t["negative"] for t in val_triplets],
        name="helsedir-val",
    )

    evaluator_test = TripletEvaluator(
        anchors=[t["query"] for t in test_triplets],
        positives=[t["positive"] for t in test_triplets],
        negatives=[t["negative"] for t in test_triplets],
        name="helsedir-test",
    )

    # Use TripletLoss for (query, positive, negative) training
    train_loss = losses.TripletLoss(model=model)

    # Calculate training steps
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.1)

    print(f"\nTraining configuration:")
    print(f"  Triplets: {len(triplets)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")

    # Pre-training sanity check
    print(f"\nPre-training sanity check:")
    sample = triplets[0]
    from sentence_transformers import util
    emb_q = model.encode(sample["query"])
    emb_p = model.encode(sample["positive"])
    emb_n = model.encode(sample["negative"])
    sim_pos = float(util.cos_sim(emb_q, emb_p)[0][0])
    sim_neg = float(util.cos_sim(emb_q, emb_n)[0][0])
    print(f"  Query: {sample['query']}")
    print(f"  Similarity to positive: {sim_pos:.4f}")
    print(f"  Similarity to negative: {sim_neg:.4f}")
    print(f"  Margin (pos - neg):     {sim_pos - sim_neg:.4f}")

    # Train
    print(f"\nStarting GPL training...")
    print("=" * 50)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        show_progress_bar=True,
        output_path=args.output_dir,
        save_best_model=True,
        evaluator=evaluator_val,
        evaluation_steps=steps_per_epoch,
    )

    print("=" * 50)

    # Evaluate on test set and post-training check
    try:
        print(f"\nTest set evaluation:")
        trained_model = SentenceTransformer(args.output_dir)
        test_result = evaluator_test(trained_model)
        if isinstance(test_result, dict):
            for key, value in test_result.items():
                print(f"  {key}: {value:.4f}")
        else:
            print(f"  Test accuracy: {test_result:.4f}")

        # Post-training check
        print(f"\nPost-training sanity check:")
        emb_q2 = trained_model.encode(sample["query"])
        emb_p2 = trained_model.encode(sample["positive"])
        emb_n2 = trained_model.encode(sample["negative"])
        sim_pos2 = float(util.cos_sim(emb_q2, emb_p2)[0][0])
        sim_neg2 = float(util.cos_sim(emb_q2, emb_n2)[0][0])
        print(f"  Query: {sample['query']}")
        print(f"  Similarity to positive: {sim_pos2:.4f} (was {sim_pos:.4f})")
        print(f"  Similarity to negative: {sim_neg2:.4f} (was {sim_neg:.4f})")
        print(f"  Margin (pos - neg):     {sim_pos2 - sim_neg2:.4f} (was {sim_pos - sim_neg:.4f})")

        improvement = (sim_pos2 - sim_neg2) - (sim_pos - sim_neg)
        if improvement > 0:
            print(f"  Margin improved by {improvement:.4f}")
        else:
            print(f"  Margin decreased by {abs(improvement):.4f} (may need more epochs)")
    except Exception as e:
        print(f"\nTest evaluation failed: {e}")
        print("Model was still saved successfully.")

    print(f"\nModel saved to: {args.output_dir}")

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print(f"""
1. Regenerate embeddings with the GPL-trained model:
   python scripts/data/generate_embeddings.py --model-name {args.output_dir}

2. Test search quality - relevant documents should now score higher

3. If needed, train for more epochs or generate more queries:
   python scripts/train/finetune_gpl.py --epochs 2 --queries-per-doc 5
""")


if __name__ == "__main__":
    main()
