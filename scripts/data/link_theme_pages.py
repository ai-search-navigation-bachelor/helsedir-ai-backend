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
from typing import List, Set, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

import sys
# Add project root to path (scripts/data -> scripts -> project_root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.repositories.base import db_pool


BASE_URL = "https://www.helsedirektoratet.no"


def load_theme_pages() -> List[dict]:
    """Load theme pages from JSON."""
    # Path: scripts/data/link_theme_pages.py -> project_root/data/theme_pages.json
    json_path = Path(__file__).parent.parent.parent / "data" / "theme_pages.json"
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


def find_content_by_page_id(page_id: str) -> Optional[dict]:
    """
    Find content in database by pageID.
    Searches for ID ending with the pageID (ignores 0006-XXXX- prefix).
    """
    conn = db_pool.get_connection()
    if not conn:
        return None

    try:
        cursor = conn.cursor(dictionary=True)

        # Search for ID ending with pageID
        query = "SELECT * FROM content WHERE id LIKE %s AND info_type != 'temaside'"
        cursor.execute(query, (f'%{page_id}',))

        result = cursor.fetchone()
        return result
    except Exception as e:
        print(f"    DB ERROR searching for {page_id}: {e}")
        return None
    finally:
        cursor.close()
        conn.close()


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


def link_content_to_theme(theme_page_id: str, content_id: str) -> bool:
    """Create a link between theme page and content."""
    conn = db_pool.get_connection()
    if not conn:
        return False

    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT IGNORE INTO theme_page_content
            (theme_page_id, content_id)
            VALUES (%s, %s)
        """, (theme_page_id, content_id))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"    ERROR linking {content_id} to {theme_page_id}: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def process_theme_page(theme_page: dict, verbose: bool = True) -> int:
    """
    Process a single theme page: fetch, extract links, match content.

    Returns:
        Number of content items linked
    """
    theme_id = theme_page['id']
    theme_path = theme_page['path']
    theme_title = theme_page['tittel']

    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {theme_title}")
        print(f"Path: {theme_path}")
        print(f"ID: {theme_id}")

    # Fetch theme page HTML
    theme_url = BASE_URL + theme_path
    if verbose:
        print(f"\nFetching: {theme_url}")

    html = fetch_page_html(theme_url)
    if not html:
        print("  ERROR: Could not fetch page")
        return 0

    # Extract links from theme page
    links = extract_links_from_theme_page(html, theme_path)
    if verbose:
        print(f"Found {len(links)} links on theme page")

    if not links:
        return 0

    # Process each link
    linked_count = 0
    for i, link_url in enumerate(sorted(links), 1):
        if verbose:
            print(f"\n  [{i}/{len(links)}] {link_url}")

        # Fetch linked page
        link_html = fetch_page_html(link_url)
        if not link_html:
            continue

        # Extract pageID
        page_id = extract_page_id_from_html(link_html)
        if not page_id:
            if verbose:
                print("    No pageID found")
            continue

        if verbose:
            print(f"    pageID: {page_id}")

        # Find matching content in database
        content = find_content_by_page_id(page_id)
        if not content:
            if verbose:
                print("    ✗ No matching content in database")
            continue

        content_id = content['id']
        content_title = content['tittel']
        if verbose:
            print(f"    ✓ Found: {content_id}")
            print(f"      Title: {content_title[:60]}...")

        # Link to theme page
        if link_content_to_theme(theme_id, content_id):
            linked_count += 1
            if verbose:
                print(f"    ✓ Linked to theme page")

        # Rate limiting
        time.sleep(0.2)

    if verbose:
        print(f"\n✓ Linked {linked_count} content items to theme page")

    return linked_count


def has_children(theme_page: dict) -> bool:
    """Check if a theme page has any children."""
    links = theme_page.get('links', [])
    for link in links:
        if link.get('rel') == 'barn':
            return True
    return False


def main():
    print("="*60)
    print("THEME PAGE CONTENT LINKING")
    print("="*60)

    # Create junction table
    if not create_junction_table():
        print("\nERROR: Could not create junction table")
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
    processed = 0
    for i, theme_page in enumerate(leaf_theme_pages, 1):
        print(f"\n\n{'#'*60}")
        print(f"THEME PAGE {i}/{len(leaf_theme_pages)}")
        print(f"{'#'*60}")

        try:
            linked = process_theme_page(theme_page, verbose=True)
            total_linked += linked
            processed += 1
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"\nERROR processing theme page: {e}")
            continue

    print(f"\n\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total content items linked: {total_linked}")
    print(f"Leaf theme pages processed: {processed}/{len(leaf_theme_pages)}")
    print(f"Parent theme pages skipped: {len(theme_pages) - len(leaf_theme_pages)}")


if __name__ == "__main__":
    main()
