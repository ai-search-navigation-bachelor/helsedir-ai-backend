#!/usr/bin/env python3
"""
Generate AI-based role tags for content items.

For each content item, builds context from title + body + child hierarchy,
sends to an LLM, and stores matching role slugs as JSON in the role_tags column.

Optimizations:
- Parallel async workers (one per Groq API key, up to 4x throughput)
- Batched DB writes (one connection per batch instead of per item)
- Reused httpx.AsyncClient per worker (no connection churn)
- HTML stripped from body text (saves tokens)
- Progress logging with ETA every 25 items

Usage:
    python scripts/data/generation/generate_role_tags.py
    python scripts/data/generation/generate_role_tags.py --dry-run
    python scripts/data/generation/generate_role_tags.py --skip-existing
"""

import argparse
import asyncio
import json
import re
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.constants import ROLE_SLUGS, ROLE_TAG_THRESHOLD
from app.config import settings
from app.services.repositories.base import db_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Text truncation limits per hierarchy level
BODY_LIMIT_MAIN = 1000
BODY_LIMIT_CHILD = 500
BODY_LIMIT_GRANDCHILD = 300
BODY_LIMIT_GREAT_GRANDCHILD = 200

# Regex to strip HTML tags (compiled once)
HTML_TAG_RE = re.compile(r"<[^>]+>")

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


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

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


def strip_html(text: Optional[str]) -> str:
    """Remove HTML tags from text."""
    if not text:
        return ""
    return HTML_TAG_RE.sub("", text)


def truncate(text: Optional[str], limit: int) -> str:
    """Strip HTML then truncate text to limit characters."""
    if not text:
        return ""
    text = strip_html(text).strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def parse_scores(response_text: str) -> Optional[List[float]]:
    """Parse LLM response into list of 10 float scores."""
    match = re.search(r'\[[\s\S]*?\]', response_text)
    if not match:
        return None

    try:
        scores = json.loads(match.group())
        if not isinstance(scores, list) or len(scores) != 10:
            return None
        return [max(0.0, min(1.0, float(s))) for s in scores]
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def scores_to_tags(scores: List[float], threshold: float = ROLE_TAG_THRESHOLD) -> List[str]:
    """Convert scores to role tag slugs based on threshold."""
    return [slug for score, slug in zip(scores, ROLE_SLUGS) if score >= threshold]


# ---------------------------------------------------------------------------
# Database (batched writes)
# ---------------------------------------------------------------------------

def save_role_tags_batch(updates: List[Tuple[str, List[str]]]) -> int:
    """
    Save role tags for multiple content items in a single transaction.

    Args:
        updates: List of (content_id, tags) tuples

    Returns:
        Number of successfully updated items
    """
    if not updates:
        return 0

    conn = db_pool.get_connection()
    if not conn:
        return 0
    cursor = None
    try:
        cursor = conn.cursor()
        for content_id, tags in updates:
            tags_json = json.dumps(tags, ensure_ascii=False)
            cursor.execute(
                "UPDATE content SET role_tags = %s WHERE id = %s",
                (tags_json, content_id),
            )
        conn.commit()
        return len(updates)
    except Exception as e:
        logger.error("Error saving role tags batch (%d items): %s", len(updates), e)
        conn.rollback()
        return 0
    finally:
        if cursor:
            cursor.close()
        conn.close()


# ---------------------------------------------------------------------------
# Async API worker
# ---------------------------------------------------------------------------

async def call_groq_async(
    client: httpx.AsyncClient,
    api_key: str,
    prompt: str,
    max_retries: int = 3,
) -> Optional[str]:
    """Call Groq API with retry logic (async, reuses client)."""
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
            response = await client.post(url, headers=headers, json=payload, timeout=30.0)

            if response.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                logger.warning("Worker rate limited, waiting %ds (attempt %d/%d)...", wait, attempt + 1, max_retries)
                await asyncio.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except Exception as e:
            logger.warning("Groq API error (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)

    return None


async def api_worker(
    worker_id: int,
    api_key: str,
    queue: asyncio.Queue,
    stats: Dict,
    pending_updates: List,
    updates_lock: asyncio.Lock,
    dry_run: bool,
    db_batch_size: int = 25,
) -> None:
    """
    Async worker that consumes items from the queue and calls the Groq API.

    Each worker uses its own API key and httpx client. Completed results are
    collected in pending_updates and flushed to the DB in batches.
    """
    async with httpx.AsyncClient() as client:
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            content_id = str(item["id"])
            title = (item.get("tittel") or "")[:80]

            prompt = item["_prompt"]

            response_text = await call_groq_async(client, api_key, prompt)

            if not response_text:
                async with updates_lock:
                    stats["errors"] += 1
                    stats["processed"] += 1
                logger.error("[Worker %d] FAILED (no response): %s - %s", worker_id, content_id, title)
                queue.task_done()
                await asyncio.sleep(2)
                continue

            scores = parse_scores(response_text)
            if not scores:
                async with updates_lock:
                    stats["errors"] += 1
                    stats["processed"] += 1
                logger.error("[Worker %d] FAILED (bad response): %s - %s | %s",
                             worker_id, content_id, title, response_text[:80])
                queue.task_done()
                await asyncio.sleep(2)
                continue

            tags = scores_to_tags(scores)
            tag_str = ", ".join(tags) if tags else "(none)"

            async with updates_lock:
                stats["success"] += 1
                stats["processed"] += 1

                if dry_run:
                    score_str = ", ".join(f"{s:.2f}" for s in scores)
                    logger.info("[%d/%d] %s -> %s  scores=[%s]",
                                stats["processed"], stats["total"], title, tag_str, score_str)
                else:
                    pending_updates.append((content_id, tags))
                    logger.info("[%d/%d] %s -> %s",
                                stats["processed"], stats["total"], title, tag_str)

                    # Flush to DB in batches
                    if len(pending_updates) >= db_batch_size:
                        batch = list(pending_updates)
                        pending_updates.clear()
                        saved = await asyncio.get_event_loop().run_in_executor(
                            None, save_role_tags_batch, batch
                        )
                        logger.info("DB batch saved: %d items", saved)

                # Log progress summary every 25 items
                if stats["processed"] % 25 == 0:
                    elapsed = time.monotonic() - stats["start_time"]
                    rate = stats["processed"] / elapsed if elapsed > 0 else 0
                    remaining = stats["total"] - stats["processed"]
                    eta_seconds = remaining / rate if rate > 0 else 0
                    eta_min = eta_seconds / 60
                    logger.info(
                        "--- Progress: %d/%d (%.0f%%) | %.1f items/s | ETA: %.1f min | Errors: %d ---",
                        stats["processed"], stats["total"],
                        100 * stats["processed"] / stats["total"],
                        rate, eta_min, stats["errors"],
                    )

            queue.task_done()
            await asyncio.sleep(2)  # ~30 req/min per key


async def run_workers(
    items: List[Dict],
    api_keys: List[str],
    dry_run: bool,
) -> Dict:
    """Fill the queue and launch one async worker per API key."""
    queue: asyncio.Queue = asyncio.Queue()
    for item in items:
        await queue.put(item)

    updates_lock = asyncio.Lock()
    pending_updates: List[Tuple[str, List[str]]] = []

    stats = {
        "processed": 0,
        "success": 0,
        "errors": 0,
        "total": len(items),
        "start_time": time.monotonic(),
    }

    workers = [
        api_worker(
            worker_id=i,
            api_key=key,
            queue=queue,
            stats=stats,
            pending_updates=pending_updates,
            updates_lock=updates_lock,
            dry_run=dry_run,
        )
        for i, key in enumerate(api_keys)
    ]

    logger.info("Starting %d parallel workers...", len(workers))
    await asyncio.gather(*workers)

    # Flush remaining pending updates
    if pending_updates and not dry_run:
        saved = save_role_tags_batch(pending_updates)
        logger.info("Final DB batch saved: %d items", saved)

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate AI role tags for content")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving to DB")
    parser.add_argument("--skip-existing", action="store_true", help="Skip items that already have role tags")
    args = parser.parse_args()

    api_keys = settings.groq_api_keys
    if not api_keys:
        print("ERROR: No Groq API keys configured. Set GROQ_API_KEY in .env")
        sys.exit(1)

    print("=" * 60)
    print("ROLE TAG GENERATION")
    print("=" * 60)
    print(f"  Groq API keys: {len(api_keys)}")
    print(f"  Threshold: {ROLE_TAG_THRESHOLD}")
    print(f"  Roles: {len(ROLE_SLUGS)}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Skip existing: {args.skip_existing}")
    print()

    logger.info("Loading content from database...")
    all_content = get_all_content()
    if not all_content:
        print("No content found in database.")
        sys.exit(1)

    lookup = build_content_lookup(all_content)
    logger.info("Loaded %d content items", len(all_content))

    # Filter to items that need processing
    items_to_process = []
    skipped_kapittel = 0
    skipped_existing = 0
    for item in all_content:
        if item.get("info_type") == "kapittel":
            skipped_kapittel += 1
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
                    skipped_existing += 1
                    continue

        items_to_process.append(item)

    logger.info("Skipped %d kapittel items, %d with existing tags", skipped_kapittel, skipped_existing)

    if not items_to_process:
        print("Nothing to process. All items already have tags (use without --skip-existing to regenerate).")
        return

    # Pre-build prompts (saves work in async workers)
    logger.info("Pre-building prompts for %d items...", len(items_to_process))
    for item in items_to_process:
        item["_prompt"] = build_prompt(item, lookup)

    total = len(items_to_process)
    num_workers = len(api_keys)
    estimated_seconds = (total / num_workers) * 2
    print(f"\n  Items to process: {total}")
    print(f"  Parallel workers: {num_workers}")
    print(f"  Effective rate: ~{num_workers * 30} req/min")
    print(f"  Estimated time: ~{estimated_seconds / 60:.1f} minutes")

    if args.dry_run:
        print("\n  [DRY RUN - no DB writes]")

    print()

    # Run async workers
    stats = asyncio.run(run_workers(
        items=items_to_process,
        api_keys=api_keys,
        dry_run=args.dry_run,
    ))

    # Summary
    elapsed = time.monotonic() - stats["start_time"]
    print()
    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Processed: {stats['processed']}/{stats['total']}")
    print(f"  Success: {stats['success']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Time: {elapsed / 60:.1f} minutes ({elapsed:.0f}s)")
    if stats["processed"] > 0:
        print(f"  Rate: {stats['processed'] / elapsed:.1f} items/s")


if __name__ == "__main__":
    main()
