#!/usr/bin/env python3
"""
Generate semantic tags for content based on title and text.

This script automatically generates tags by:
1. Extracting important keywords from title and text
2. Mapping to common medical terms
3. Storing in new 'tags' field

Usage:
    python scripts/generate_tags.py
    python scripts/generate_tags.py --dry-run --limit 10
"""

import argparse
import sys
import json
import re
from pathlib import Path
from collections import Counter
from typing import List, Set

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.data.database_service import database_service


# Norwegian stopwords (common words to ignore)
NORWEGIAN_STOPWORDS = {
    "og", "i", "til", "for", "på", "med", "av", "er", "en", "et", "om", "som",
    "har", "kan", "skal", "ved", "fra", "var", "den", "det", "de", "må", "deg",
    "vil", "være", "bli", "blir", "ble", "eller", "også", "når", "der", "hvor",
    "disse", "andre", "alle", "sin", "sine", "sitt", "noen", "hver", "etter",
    "over", "inn", "ut", "opp", "ned", "før", "mer", "slik", "bare", "selv",
    "mellom", "under", "gjør", "dette", "noe", "mange", "denne"
}

# Generic health/admin words that should not be tags (too common)
GENERIC_HEALTH_WORDS = {
    "pasienter", "pasient", "helse", "helsetjeneste", "tjeneste", "tjenester",
    "kommune", "kommuner", "arbeid", "tiltak", "behov", "bruk", "bruker",
    "brukere", "person", "personer", "side", "gjelder", "informasjon",
    "spørsmål", "svar", "kontakt", "dato", "årsak", "tilbud", "rett",
    "rettigheter", "plikter", "ansvar", "kvalitet", "utvikling", "opplæring",
    "kompetanse", "kunnskap", "erfaring", "forhold", "system", "organisering"
}

# Common medical term mappings (Norwegian)
MEDICAL_SYNONYMS = {
    "sukkersyke": "diabetes",
    "hjerteinfarkt": "hjerte",
    "hjerneslag": "slag",
    "høyt blodtrykk": "hypertensjon",
    "astma": "luftveier",
    "kols": "luftveier",
    "angst": "psykisk_helse",
    "depresjon": "psykisk_helse",
    "rus": "avhengighet",
}


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Remove special characters but keep Norwegian letters
    text = re.sub(r'[^a-zæøå\s-]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_keywords(
    text: str, 
    min_length: int = 4, 
    max_keywords: int = 20,
    min_frequency: int = 1,
    from_title: bool = False
) -> List[str]:
    """
    Extract important keywords from text.
    
    Args:
        text: Text to extract keywords from
        min_length: Minimum word length
        max_keywords: Maximum number of keywords to return
        min_frequency: Minimum times word must appear (only for body text)
        from_title: If True, looser frequency requirements (every word in title is important)
    """
    if not text:
        return []
    
    # Clean text
    text = clean_text(text)
    
    # Split into words
    words = text.split()
    
    # Filter stopwords, short words, and generic health words
    keywords = [
        word for word in words
        if len(word) >= min_length 
        and word not in NORWEGIAN_STOPWORDS
        and word not in GENERIC_HEALTH_WORDS
    ]
    
    # Count frequency
    word_counts = Counter(keywords)
    
    # Apply frequency filter (stricter for body text)
    if not from_title:
        # For body text: word must appear at least min_frequency times
        # OR be a long/specific medical term (≥8 characters)
        filtered_counts = {
            word: count for word, count in word_counts.items()
            if count >= min_frequency or len(word) >= 8
        }
    else:
        # For title: keep all words (title words are inherently important)
        filtered_counts = word_counts
    
    # Get most common, sorted by frequency then length (prefer longer words)
    top_keywords = sorted(
        filtered_counts.items(),
        key=lambda x: (x[1], len(x[0])),  # Sort by: frequency, then length
        reverse=True
    )
    
    return [word for word, count in top_keywords[:max_keywords]]


def extract_multi_word_terms(text: str) -> List[str]:
    """Extract common multi-word medical terms."""
    if not text:
        return []
    
    text = clean_text(text)
    
    # Common multi-word terms (add more as needed)
    multi_word_terms = [
        "type 2 diabetes",
        "type 1 diabetes", 
        "hjerte- og karsykdom",
        "kronisk obstruktiv lungesykdom",
        "psykisk helse",
        "fysisk aktivitet",
        "eldre pasienter",
        "barn og unge",
    ]
    
    found_terms = []
    for term in multi_word_terms:
        if term in text:
            found_terms.append(term.replace(" ", "_"))
    
    return found_terms


def generate_tags_for_content(tittel: str, tekst: str, info_type: str = None) -> List[str]:
    """
    Generate tags for a content item.
    
    Tags are sorted by:
    1. Frequency (most frequent first)
    2. Source: title words before text words
    3. Length (longer words first)
    4. Alphabetically
    
    Args:
        tittel: Content title
        tekst: Content text/body
        info_type: Content type (optional)
        
    Returns:
        List of generated tags, sorted by importance
    """
    # Track which tags came from title vs text
    title_tags = set()
    text_tags = set()
    tag_frequencies = {}
    
    # Extract from title (every word is important, no frequency filter)
    title_text = clean_text(tittel or "")
    title_keywords = extract_keywords(
        tittel or "", 
        max_keywords=10,
        from_title=True
    )
    title_tags.update(title_keywords)
    
    # Count frequencies in title
    title_words = title_text.split()
    for word in title_keywords:
        tag_frequencies[word] = title_words.count(word)
    
    # Extract from text (stricter: must appear 2+ times OR be long medical term)
    text_sample = (tekst or "")[:2000]
    text_sample_clean = clean_text(text_sample)
    text_keywords = extract_keywords(
        text_sample, 
        max_keywords=15,
        min_frequency=2,  # Must appear at least twice
        from_title=False
    )
    text_tags.update(text_keywords)
    
    # Count frequencies in text
    text_words = text_sample_clean.split()
    for word in text_keywords:
        tag_frequencies[word] = text_words.count(word)
    
    # Combine all tags
    all_tags = title_tags | text_tags
    
    # Extract multi-word terms
    combined_text = f"{tittel} {text_sample}"
    multi_terms = extract_multi_word_terms(combined_text)
    for term in multi_terms:
        all_tags.add(term)
        # Multi-word terms get high frequency (they're important)
        tag_frequencies[term] = 3
    
    # Add info_type as tag
    if info_type:
        info_tag = info_type.lower()
        all_tags.add(info_tag)
        # Info type is very important, give it high frequency
        tag_frequencies[info_tag] = 5
    
    # Map synonyms (preserve frequency from original)
    mapped_tags = {}
    for tag in all_tags:
        if tag in MEDICAL_SYNONYMS:
            new_tag = MEDICAL_SYNONYMS[tag]
            # Keep the higher frequency if synonym already exists
            if new_tag in mapped_tags:
                mapped_tags[new_tag] = max(
                    mapped_tags[new_tag],
                    tag_frequencies.get(tag, 1)
                )
            else:
                mapped_tags[new_tag] = tag_frequencies.get(tag, 1)
            # Track if it was from title
            if tag in title_tags:
                title_tags.add(new_tag)
        else:
            mapped_tags[tag] = tag_frequencies.get(tag, 1)
    
    # Sort by: frequency (desc), is_from_title (title first), length (desc), alphabetically
    sorted_tags = sorted(
        mapped_tags.keys(),
        key=lambda tag: (
            -mapped_tags[tag],           # 1. Higher frequency first (negative for descending)
            tag not in title_tags,        # 2. Title tags first (False < True, so title=False comes first)
            -len(tag),                    # 3. Longer words first (negative for descending)
            tag                           # 4. Alphabetically
        )
    )
    
    return sorted_tags[:20]  # Max 20 tags


def update_content_tags(content_id: str, tags: List[str]) -> bool:
    """Update tags for a content item in database."""
    conn = database_service._get_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        tags_json = json.dumps(tags, ensure_ascii=False)
        
        cursor.execute(
            "UPDATE content SET tags = %s WHERE id = %s",
            (tags_json, content_id)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error updating tags for {content_id}: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate semantic tags for content"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show generated tags without updating database"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items to process"
    )
    args = parser.parse_args()
    
    # Check database connection
    if not database_service.is_connected():
        print("Error: Cannot connect to database")
        sys.exit(1)
    
    # Get all content
    print("Loading content from database...")
    content_items = database_service.get_all_content()
    
    if args.limit:
        content_items = content_items[:args.limit]
    
    print(f"Processing {len(content_items)} content items...")
    
    updated_count = 0
    
    for i, item in enumerate(content_items, 1):
        content_id = item.get("id")
        tittel = item.get("tittel", "")
        tekst = item.get("tekst", "")
        info_type = item.get("info_type")
        
        # Generate tags
        tags = generate_tags_for_content(tittel, tekst, info_type)
        
        if args.dry_run:
            print(f"\n{i}. {tittel[:60]}")
            print(f"   Generated tags: {', '.join(tags[:10])}")
            if len(tags) > 10:
                print(f"   ... and {len(tags) - 10} more")
        else:
            # Update database
            if update_content_tags(content_id, tags):
                updated_count += 1
                if i % 10 == 0:
                    print(f"Processed {i}/{len(content_items)}...")
    
    if args.dry_run:
        print(f"\nDry run complete. Would update {len(content_items)} items.")
    else:
        print(f"\nSuccessfully updated tags for {updated_count}/{len(content_items)} items")


if __name__ == "__main__":
    main()
