"""
Enrich GPL queries with synonym variants.

For each document's query list, finds queries containing terms from the
synonym dictionary and creates variant queries using synonyms. Replaces
the most redundant queries (highest similarity to other queries) to stay
at 10 queries per document.

Usage:
    python scripts/data/maintenance/enrich_gpl_queries.py

    # Preview without writing
    python scripts/data/maintenance/enrich_gpl_queries.py --dry-run

    # Custom paths
    python scripts/data/maintenance/enrich_gpl_queries.py --input data/gpl_queries.json --output data/gpl_queries_enriched.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.search.synonyms import SYNONYM_LOOKUP

MAX_QUERIES = 10


def tokenize(text: str) -> List[str]:
    """Split text into lowercase tokens."""
    return re.findall(r"\w+", text.lower())


def find_synonym_matches(query: str) -> List[Tuple[str, FrozenSet[str]]]:
    """
    Find terms in a query that have synonyms.

    Returns list of (matched_term, synonym_group) tuples.
    Checks multi-word terms first, then single-word terms.
    """
    query_lower = query.lower()
    matches = []
    matched_spans: Set[Tuple[int, int]] = set()

    # Check multi-word synonyms first (longer matches take priority)
    multi_word_terms = sorted(
        [t for t in SYNONYM_LOOKUP if " " in t],
        key=len,
        reverse=True,
    )
    for term in multi_word_terms:
        start = query_lower.find(term)
        if start >= 0:
            end = start + len(term)
            # Check no overlap with already matched spans
            if not any(
                s < end and e > start for s, e in matched_spans
            ):
                matches.append((term, SYNONYM_LOOKUP[term]))
                matched_spans.add((start, end))

    # Then check single-word synonyms
    for token in tokenize(query):
        if token in SYNONYM_LOOKUP:
            # Check this token isn't part of an already matched multi-word term
            start = query_lower.find(token)
            end = start + len(token)
            if not any(
                s <= start and e >= end for s, e in matched_spans
            ):
                matches.append((token, SYNONYM_LOOKUP[token]))

    return matches


def create_synonym_variant(query: str, original_term: str, replacement: str) -> str:
    """
    Replace a term in a query with its synonym.

    Preserves the original casing style (title case, upper case, etc.).
    """
    query_lower = query.lower()
    idx = query_lower.find(original_term.lower())
    if idx < 0:
        return query

    original_in_query = query[idx : idx + len(original_term)]

    # Match casing
    if original_in_query.istitle() or (idx == 0 and original_in_query[0].isupper()):
        replacement = replacement.capitalize()
    elif original_in_query.isupper():
        replacement = replacement.upper()

    return query[:idx] + replacement + query[idx + len(original_term) :]


def compute_similarity(q1: str, q2: str) -> float:
    """Simple Jaccard similarity between two queries."""
    tokens1 = set(tokenize(q1))
    tokens2 = set(tokenize(q2))
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def find_most_redundant(queries: List[str], n: int = 1) -> List[int]:
    """
    Find indices of the N most redundant queries.

    A query is "redundant" if it has the highest average similarity
    to all other queries in the list.
    """
    if len(queries) <= n:
        return []

    avg_similarities = []
    for i, q in enumerate(queries):
        sims = [compute_similarity(q, queries[j]) for j in range(len(queries)) if j != i]
        avg_similarities.append((i, sum(sims) / len(sims) if sims else 0.0))

    avg_similarities.sort(key=lambda x: -x[1])
    return [idx for idx, _ in avg_similarities[:n]]


def generate_variants(queries: List[str]) -> List[str]:
    """
    Generate synonym variant queries for a document's query list.

    Returns only NEW variants (not already in the query list).
    """
    existing_lower = {q.lower() for q in queries}
    variants: List[str] = []
    seen_lower: Set[str] = set()

    for query in queries:
        matches = find_synonym_matches(query)
        for original_term, synonym_group in matches:
            for synonym in synonym_group:
                if synonym.lower() == original_term.lower():
                    continue

                variant = create_synonym_variant(query, original_term, synonym)
                variant_lower = variant.lower()

                if variant_lower not in existing_lower and variant_lower not in seen_lower:
                    variants.append(variant)
                    seen_lower.add(variant_lower)

    return variants


def enrich_queries(
    queries_by_doc: Dict[str, List[str]],
) -> Tuple[Dict[str, List[str]], Dict[str, object], Dict[str, Dict[str, List[str]]]]:
    """
    Enrich all documents' queries with synonym variants.

    Returns:
        enriched: The enriched queries dict.
        stats: Statistics about the enrichment.
        changelog: Per-document record of removed and added queries.
    """
    enriched: Dict[str, List[str]] = {}
    changelog: Dict[str, Dict[str, List[str]]] = {}
    total_replaced = 0
    total_variants_generated = 0
    docs_modified = 0

    for doc_id, queries in queries_by_doc.items():
        variants = generate_variants(queries)
        total_variants_generated += len(variants)

        if not variants:
            enriched[doc_id] = list(queries)
            continue

        current = list(queries)
        removed: List[str] = []
        added: List[str] = []

        # Determine how many we can add
        slots_available = MAX_QUERIES - len(current)

        if slots_available > 0:
            # There's room, just add variants
            to_add = variants[:slots_available]
            current.extend(to_add)
            added.extend(to_add)
            total_replaced += len(to_add)
        else:
            # Need to replace redundant queries
            n_to_replace = min(len(variants), max(1, len(current) // 3))
            redundant_indices = find_most_redundant(current, n=n_to_replace)

            # Replace redundant queries with variants
            for i, idx in enumerate(sorted(redundant_indices, reverse=True)):
                if i < len(variants):
                    removed.append(current[idx])
                    current[idx] = variants[i]
                    added.append(variants[i])
                    total_replaced += 1

        if current != queries:
            docs_modified += 1
            changelog[doc_id] = {"removed": removed, "added": added}

        enriched[doc_id] = current

    stats = {
        "total_documents": len(queries_by_doc),
        "documents_modified": docs_modified,
        "queries_replaced_or_added": total_replaced,
        "total_variants_generated": total_variants_generated,
    }
    return enriched, stats, changelog


def main():
    parser = argparse.ArgumentParser(description="Enrich GPL queries with synonym variants")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "data" / "gpl_queries.json"),
        help="Path to input GPL queries JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output file (default: overwrite input)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    print(f"Loading queries from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        queries_by_doc = json.load(f)

    print(f"Loaded {len(queries_by_doc)} documents, {sum(len(v) for v in queries_by_doc.values())} queries")
    print(f"Synonym dictionary: {len(SYNONYM_LOOKUP)} terms")
    print()

    enriched, stats, changelog = enrich_queries(queries_by_doc)

    print("=== Enrichment Results ===")
    for key, val in stats.items():
        print(f"  {key}: {val}")
    print()

    # Show some examples
    print("=== Examples ===")
    n_examples = 0
    for doc_id, changes in list(changelog.items())[:5]:
        print(f"\nDocument: {doc_id}")
        for q in changes["removed"]:
            print(f"  - {q}")
        for q in changes["added"]:
            print(f"  + {q}")
        n_examples += 1

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
    else:
        print(f"\nWriting enriched queries to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)

        changelog_path = output_path.with_name(
            output_path.stem + "_changelog" + output_path.suffix
        )
        print(f"Writing changelog to {changelog_path}")
        with open(changelog_path, "w", encoding="utf-8") as f:
            json.dump(changelog, f, ensure_ascii=False, indent=2)

        print(f"Done! {len(changelog)} documents changed.")


if __name__ == "__main__":
    main()
