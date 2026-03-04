#!/usr/bin/env python3
"""
Generate AI-based role tags for content items.

For each content item, builds context from title + body + child hierarchy,
sends to an LLM, and stores matching role slugs as JSON in the role_tags column.

Usage:
    python scripts/data/generation/generate_role_tags.py
    python scripts/data/generation/generate_role_tags.py --dry-run
    python scripts/data/generation/generate_role_tags.py --batch-size 20
    python scripts/data/generation/generate_role_tags.py --skip-existing
"""

import argparse
import json
import sys
import time
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.constants import ROLE_SLUGS, ROLE_INFO, ROLE_TAG_THRESHOLD
from app.config import settings
from app.services.repositories.base import db_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Text truncation limits per hierarchy level
BODY_LIMIT_MAIN = 1000
BODY_LIMIT_CHILD = 500
BODY_LIMIT_GRANDCHILD = 300
BODY_LIMIT_GREAT_GRANDCHILD = 200

ROLE_DESCRIPTIONS = """1. Lege — klinisk medisin, diagnostikk, behandling
2. Sykepleier — pleie, omsorg, prosedyrer
3. Annet helsepersonell — fysioterapi, ergoterapi, etc.
4. Leder i helsetjenesten — drift, organisering, kvalitet
5. Offentlig ansatt — kommune, fylke, departement
6. IT og e-helse — digitalisering, standarder, systemer
7. Forskning og utdanning — forskning, undervisning, akademia
8. Næringsliv og bransje — leverandører, organisasjoner
9. Media — journalistikk, kommunikasjon
10. Jus og regelverk — lover, forskrifter, juridisk"""

SYSTEM_PROMPT = "Du er en ekspert på å klassifisere helsefaglig innhold for ulike brukergrupper."

USER_PROMPT_TEMPLATE = """Vurder hvor relevant dette innholdet er for hver brukergruppe.
Gi en score fra 0.0 (ikke relevant) til 1.0 (svært relevant).

Innhold:
Tittel: {title}
Type: {content_type}
Tekst: {body}

{children_section}

Roller:
{role_descriptions}

Svar KUN med JSON-array av 10 desimaltall: [score1, score2, ...]"""


def get_all_content() -> List[Dict]:
    """Fetch all content with links from database."""
    conn = db_pool.get_connection()
    if not conn:
        return []
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, tittel, tekst, info_type, links, role_tags FROM content")
        return cursor.fetchall()
    finally:
        if cursor:
            cursor.close()
        conn.close()


def build_content_lookup(all_content: List[Dict]) -> Dict[str, Dict]:
    """Build id -> content dict for fast child lookups."""
    return {str(item["id"]): item for item in all_content}


def parse_links(links_data) -> List[Dict]:
    """Parse links JSON field."""
    if links_data is None:
        return []
    if isinstance(links_data, str):
        try:
            links_data = json.loads(links_data)
        except (json.JSONDecodeError, TypeError):
            return []
    return links_data if isinstance(links_data, list) else []


def truncate(text: Optional[str], limit: int) -> str:
    """Truncate text to limit characters."""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def build_children_section(item: Dict, lookup: Dict[str, Dict], depth: int = 0, max_depth: int = 3) -> str:
    """Build hierarchical children text for the prompt."""
    if depth >= max_depth:
        return ""

    links = parse_links(item.get("links"))
    child_links = [link for link in links if link.get("rel") == "barn"]

    if not child_links:
        return ""

    limits = [BODY_LIMIT_CHILD, BODY_LIMIT_GRANDCHILD, BODY_LIMIT_GREAT_GRANDCHILD]
    body_limit = limits[min(depth, len(limits) - 1)]
    indent = "  " * depth

    lines = []
    for link in child_links:
        child_id = link.get("id")
        if not child_id or child_id not in lookup:
            continue

        child = lookup[child_id]
        child_title = child.get("tittel", "")
        child_body = truncate(child.get("tekst"), body_limit)

        lines.append(f"{indent}- {child_title}: {child_body}")

        # Recurse for kapittel children
        child_type = child.get("info_type", "")
        if child_type == "kapittel":
            sub_section = build_children_section(child, lookup, depth + 1, max_depth)
            if sub_section:
                lines.append(sub_section)

    return "\n".join(lines)


def build_prompt(item: Dict, lookup: Dict[str, Dict]) -> str:
    """Build the full prompt for a content item."""
    title = item.get("tittel", "")
    content_type = item.get("info_type", "unknown")
    body = truncate(item.get("tekst"), BODY_LIMIT_MAIN)

    children_section = build_children_section(item, lookup)
    if children_section:
        children_section = f"Underinnhold:\n{children_section}"
    else:
        children_section = ""

    return USER_PROMPT_TEMPLATE.format(
        title=title,
        content_type=content_type,
        body=body,
        children_section=children_section,
        role_descriptions=ROLE_DESCRIPTIONS,
    )


def parse_scores(response_text: str) -> Optional[List[float]]:
    """Parse LLM response into list of 10 float scores."""
    # Try to extract JSON array from response
    match = re.search(r'\[[\s\S]*?\]', response_text)
    if not match:
        return None

    try:
        scores = json.loads(match.group())
        if not isinstance(scores, list) or len(scores) != 10:
            return None
        # Clamp to [0, 1]
        return [max(0.0, min(1.0, float(s))) for s in scores]
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def scores_to_tags(scores: List[float], threshold: float = ROLE_TAG_THRESHOLD) -> List[str]:
    """Convert scores to role tag slugs based on threshold."""
    tags = []
    for score, slug in zip(scores, ROLE_SLUGS):
        if score >= threshold:
            tags.append(slug)
    return tags


def save_role_tags(content_id: str, tags: List[str]) -> bool:
    """Save role tags to database."""
    conn = db_pool.get_connection()
    if not conn:
        return False
    cursor = None
    try:
        cursor = conn.cursor()
        tags_json = json.dumps(tags, ensure_ascii=False)
        cursor.execute(
            "UPDATE content SET role_tags = %s WHERE id = %s",
            (tags_json, content_id),
        )
        conn.commit()
        return True
    except Exception as e:
        logger.error("Error saving role tags for %s: %s", content_id, e)
        conn.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        conn.close()


def call_groq(prompt: str, api_key: str, max_retries: int = 3) -> Optional[str]:
    """Call Groq API with retry logic."""
    import httpx

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 100,
    }

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, headers=headers, json=payload)

            if response.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                logger.warning("Rate limited, waiting %ds...", wait)
                time.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except Exception as e:
            logger.warning("Groq API error (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return None


def main():
    parser = argparse.ArgumentParser(description="Generate AI role tags for content")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--batch-size", type=int, default=10, help="Items per batch")
    parser.add_argument("--skip-existing", action="store_true", help="Skip items that already have role tags")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls (seconds)")
    args = parser.parse_args()

    api_keys = settings.groq_api_keys
    if not api_keys:
        print("ERROR: No Groq API keys configured. Set GROQ_API_KEY in .env")
        sys.exit(1)

    print(f"Using {len(api_keys)} Groq API key(s)")
    print(f"Role tag threshold: {ROLE_TAG_THRESHOLD}")
    print(f"Roles: {', '.join(ROLE_SLUGS)}")
    print()

    all_content = get_all_content()
    if not all_content:
        print("No content found in database.")
        sys.exit(1)

    lookup = build_content_lookup(all_content)

    # Filter to items that need processing
    items_to_process = []
    for item in all_content:
        # Skip non-searchable hierarchy types
        if item.get("info_type") == "kapittel":
            continue

        if args.skip_existing:
            existing_tags = item.get("role_tags")
            if existing_tags:
                if isinstance(existing_tags, str):
                    try:
                        existing_tags = json.loads(existing_tags)
                    except (json.JSONDecodeError, TypeError):
                        existing_tags = None
                if existing_tags:
                    continue

        items_to_process.append(item)

    total = len(items_to_process)
    print(f"Processing {total} content items (out of {len(all_content)} total)")
    if args.dry_run:
        print("[DRY RUN MODE]")
    print()

    success_count = 0
    error_count = 0
    key_index = 0

    for i, item in enumerate(items_to_process):
        content_id = str(item["id"])
        title = item.get("tittel", "")[:80]
        progress = f"[{i + 1}/{total}]"

        prompt = build_prompt(item, lookup)

        # Rotate API keys
        api_key = api_keys[key_index % len(api_keys)]
        key_index += 1

        response_text = call_groq(prompt, api_key)
        if not response_text:
            logger.error("%s FAILED (no response): %s - %s", progress, content_id, title)
            error_count += 1
            continue

        scores = parse_scores(response_text)
        if not scores:
            logger.error("%s FAILED (bad response): %s - %s | Response: %s",
                         progress, content_id, title, response_text[:100])
            error_count += 1
            continue

        tags = scores_to_tags(scores)

        if args.dry_run:
            score_str = ", ".join(f"{s:.2f}" for s in scores)
            print(f"{progress} {title}")
            print(f"  Scores: [{score_str}]")
            print(f"  Tags: {tags}")
            print()
        else:
            if save_role_tags(content_id, tags):
                success_count += 1
                tag_str = ", ".join(tags) if tags else "(none)"
                logger.info("%s OK: %s → %s", progress, title, tag_str)
            else:
                error_count += 1
                logger.error("%s SAVE FAILED: %s", progress, content_id)

        # Rate limiting
        if args.delay > 0:
            time.sleep(args.delay)

    print()
    print(f"Done! Success: {success_count}, Errors: {error_count}, Total: {total}")


if __name__ == "__main__":
    main()
