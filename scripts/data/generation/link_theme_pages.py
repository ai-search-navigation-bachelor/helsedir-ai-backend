"""
Link theme pages to their content by scraping helsedirektoratet.no.

This script:
1. Reads theme_pages.json
2. For each theme page, fetches the page from helsedirektoratet.no
3. Extracts links from the page
4. Follows each link to get the pageID from meta tags
5. Matches pageID to content in database (ignoring prefix)
6. Creates junction table entries
"""

import json
import re
import time
from pathlib import Path
from typing import List, Set, Optional, Dict
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

import sys
# Add project root to path (scripts/data/generation -> scripts/data -> scripts -> project_root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.services.repositories.base import db_pool


BASE_URL = "https://www.helsedirektoratet.no"


def load_theme_pages() -> List[dict]:
    """Load theme pages from JSON."""
    # Path: scripts/data/generation/link_theme_pages.py -> project_root/data/theme_pages.json
    json_path = Path(__file__).parent.parent.parent.parent / "data" / "theme_pages.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_page_html(url: str, timeout: float = 30.0) -> Optional[str]:
    """Fetch HTML content from a URL."""
    try:
        with httpx.Client(follow_redirects=True) as client:
            response = client.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text
    except Exception as e:
        print(f"  ERROR fetching {url}: {e}")
        return None


def extract_page_id_from_html(html: str) -> Optional[str]:
    """Extract pageID from meta tag in HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    meta_tag = soup.find('meta', attrs={'name': 'pageID'})
    if meta_tag and meta_tag.get('content'):
        return meta_tag['content']
    return None


def extract_links_from_theme_page(html: str, theme_path: str) -> Set[str]:
    """
    Extract content links from a theme page HTML.

    Looks for links in navigation lists (like the screenshot shows).
    Filters out navigation links, external links, etc.
    """
    soup = BeautifulSoup(html, 'html.parser')
    links = set()

    # Find all links in navigation lists
    nav_lists = soup.find_all('nav', class_='b-nav-list')
    for nav in nav_lists:
        for link in nav.find_all('a', href=True):
            href = link['href']

            # Skip external links, anchors, etc.
            if href.startswith('http') and not href.startswith(BASE_URL):
                continue
            if href.startswith('#'):
                continue
            if href.startswith('mailto:'):
                continue

            # Convert to absolute URL
            if href.startswith('/'):
                full_url = BASE_URL + href
            else:
                full_url = urljoin(BASE_URL + theme_path, href)

            # Filter to only include content pages (not other theme pages)
            # Typically content pages have longer paths or specific patterns
            path = urlparse(full_url).path

            # Skip if it's just another theme page (root level)
            segments = [s for s in path.split('/') if s]
            if len(segments) <= 1:
                continue

            links.add(full_url)

    return links


def load_content_cache():
    """
    Load all content from database into memory for fast lookups.

    Returns:
        Tuple of (id_cache, path_cache):
        - id_cache: maps pageID suffix to content record
        - path_cache: maps helsedirektoratet.no path to content record
    """
    print("\nLoading content cache from database...")
    conn = db_pool.get_connection()
    if not conn:
        print("ERROR: Could not connect to database")
        return {}, {}

    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, tittel, path FROM content WHERE info_type != 'temaside'")
        all_content = cursor.fetchall()

        id_cache = {}
        path_cache = {}
        for content in all_content:
            content_id = content['id']
            parts = content_id.split('-')
            if len(parts) >= 2:
                id_cache[parts[-1]] = content

            if content.get('path'):
                normalized = content['path'].rstrip('/') or '/'
                path_cache[normalized] = content

        print(f"Loaded {len(id_cache)} items into ID cache, {len(path_cache)} into path cache")
        return id_cache, path_cache

    except Exception as e:
        print(f"ERROR loading content cache: {e}")
        return {}, {}
    finally:
        if cursor:
            cursor.close()
        conn.close()


def find_content_by_page_id(page_id: str, content_cache: Dict[str, dict]) -> Optional[dict]:
    """
    Find content by pageID using in-memory cache (optimized).
    """
    # Try direct lookup first
    if page_id in content_cache:
        return content_cache[page_id]

    # Fallback: search for partial matches
    for cached_id, content in content_cache.items():
        if page_id in cached_id or cached_id in page_id:
            return content

    return None


def create_junction_table():
    """Create the junction table if it doesn't exist."""
    conn = db_pool.get_connection()
    if not conn:
        print("ERROR: Could not connect to database")
        return False

    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS theme_page_content (
                id INT PRIMARY KEY AUTO_INCREMENT,
                theme_page_id VARCHAR(100) NOT NULL,
                content_id VARCHAR(100) NOT NULL,
                display_order INT DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (theme_page_id) REFERENCES content(id) ON DELETE CASCADE,
                FOREIGN KEY (content_id) REFERENCES content(id) ON DELETE CASCADE,
                INDEX idx_theme_page (theme_page_id),
                INDEX idx_content (content_id),
                UNIQUE KEY unique_theme_content (theme_page_id, content_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        conn.commit()
        print("✓ Junction table created/verified")
        return True
    except Exception as e:
        print(f"ERROR creating junction table: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def link_content_batch(theme_page_id: str, content_ids: List[str]) -> int:
    """
    Link multiple content items to a theme page in one batch (optimized).

    Returns:
        Number of items successfully linked
    """
    if not content_ids:
        return 0

    conn = db_pool.get_connection()
    if not conn:
        return 0

    cursor = None
    try:
        cursor = conn.cursor()

        # Batch insert with single transaction
        values = [(theme_page_id, content_id) for content_id in content_ids]
        cursor.executemany("""
            INSERT IGNORE INTO theme_page_content
            (theme_page_id, content_id)
            VALUES (%s, %s)
        """, values)

        conn.commit()
        return cursor.rowcount
    except Exception as e:
        print(f"    ERROR batch linking to {theme_page_id}: {e}")
        return 0
    finally:
        if cursor:
            cursor.close()
        conn.close()


def process_theme_page(theme_page: dict, id_cache: Dict[str, dict], path_cache: Dict[str, dict], verbose: bool = True) -> Dict:
    """
    Process a single theme page: fetch, extract links, match content.

    Args:
        theme_page: Theme page dict
        id_cache: Maps pageID suffix to content record
        path_cache: Maps helsedirektoratet.no path to content record
        verbose: Print detailed progress

    Returns:
        Dict with keys: linked, path_matches, pageid_matches, not_found, fetch_error
    """
    result = {"linked": 0, "path_matches": 0, "pageid_matches": 0, "not_found": 0, "fetch_error": False}

    theme_id = theme_page['id']
    theme_path = theme_page['path']
    theme_title = theme_page['tittel']

    if verbose:
        print(f"  Fetching...", end=" ", flush=True)

    html = fetch_page_html(BASE_URL + theme_path)
    if not html:
        print("ERROR: could not fetch page")
        result["fetch_error"] = True
        return result

    links = extract_links_from_theme_page(html, theme_path)
    if verbose:
        print(f"{len(links)} links found")

    if not links:
        return result

    matched_content_ids = []

    for link_url in sorted(links):
        link_path = urlparse(link_url).path.rstrip('/') or '/'

        # First: try direct path match (no HTTP request)
        content = path_cache.get(link_path)
        if content:
            matched_content_ids.append(content['id'])
            result["path_matches"] += 1
            if verbose:
                print(f"    ✓ [path]   {link_path}")
            continue

        # Fallback: fetch page and look for pageID meta tag
        link_html = fetch_page_html(link_url)
        if not link_html:
            result["not_found"] += 1
            if verbose:
                print(f"    ✗ [fetch]  {link_path}")
            continue

        page_id = extract_page_id_from_html(link_html)
        if not page_id:
            result["not_found"] += 1
            if verbose:
                print(f"    ✗ [no id]  {link_path}")
            continue

        content = find_content_by_page_id(page_id, id_cache)
        if not content:
            result["not_found"] += 1
            if verbose:
                print(f"    ✗ [no db]  {link_path}")
            continue

        matched_content_ids.append(content['id'])
        result["pageid_matches"] += 1
        if verbose:
            print(f"    ✓ [pageid] {link_path}")
        time.sleep(0.2)

    if matched_content_ids:
        result["linked"] = link_content_batch(theme_id, matched_content_ids)

    total = len(links)
    matched = result["path_matches"] + result["pageid_matches"]
    if verbose:
        print(f"  -> Linked {result['linked']}/{total} | path: {result['path_matches']}, pageID: {result['pageid_matches']}, no match: {result['not_found']}")

    return result


def has_children(theme_page: dict) -> bool:
    """Check if a theme page has any children."""
    links = theme_page.get('links', [])
    for link in links:
        if link.get('rel') == 'barn':
            return True
    return False


def main():
    print("="*60)
    print("THEME PAGE CONTENT LINKING (OPTIMIZED)")
    print("="*60)

    # Create junction table
    if not create_junction_table():
        print("\nERROR: Could not create junction table")
        return

    # Load content cache (optimization: all content in memory)
    id_cache, path_cache = load_content_cache()
    if not id_cache and not path_cache:
        print("\nERROR: Could not load content cache")
        return

    # Load theme pages
    print("\nLoading theme pages...")
    theme_pages = load_theme_pages()
    print(f"Loaded {len(theme_pages)} theme pages")

    # Filter to only leaf nodes (no children)
    leaf_theme_pages = [tp for tp in theme_pages if not has_children(tp)]
    print(f"Filtered to {len(leaf_theme_pages)} leaf theme pages (no children)")
    print(f"Skipping {len(theme_pages) - len(leaf_theme_pages)} parent theme pages")

    # Process each leaf theme page
    total_linked = 0
    total_path = 0
    total_pageid = 0
    total_not_found = 0
    processed = 0
    errors = 0

    for i, theme_page in enumerate(leaf_theme_pages, 1):
        print(f"\n[{i}/{len(leaf_theme_pages)}] {theme_page['tittel']}")
        print(f"  Path: {theme_page['path']}")

        try:
            result = process_theme_page(theme_page, id_cache, path_cache, verbose=True)
            total_linked += result["linked"]
            total_path += result["path_matches"]
            total_pageid += result["pageid_matches"]
            total_not_found += result["not_found"]
            if result["fetch_error"]:
                errors += 1
            else:
                processed += 1
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1
            continue

    print(f"\n\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Processed:        {processed}/{len(leaf_theme_pages)} theme pages")
    print(f"Errors:           {errors}")
    print(f"Total linked:     {total_linked}")
    print(f"  via path:       {total_path}")
    print(f"  via pageID:     {total_pageid}")
    print(f"  no match:       {total_not_found}")
    print(f"Parent pages skipped: {len(theme_pages) - len(leaf_theme_pages)}")


if __name__ == "__main__":
    main()
