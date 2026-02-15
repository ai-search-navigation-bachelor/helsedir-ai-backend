"""
Shared utilities for link processing.

This module provides shared functions for extracting and processing content links
across import and migration scripts.
"""

import re
from typing import Optional


def extract_content_id_from_href(href: str) -> Optional[str]:
    """
    Extract content ID from Helsedir API href URL.

    Extracts ID from URLs like:
    - https://api.helsedirektoratet.no/innhold/veiledere/0006-0008-2c686203-c2d2-4bd5-94eb-3e1366c57bfa
    - https://api.helsedirektoratet.no/innhold/kapitler/0006-0041-662959ab-1f9b-4438-8b98-9bbf9dcc1ac1

    Args:
        href: The URL to extract the ID from

    Returns:
        The extracted content ID, or None if not found
    """
    if not href:
        return None

    # Pattern: /innhold/{type}/{id}
    pattern = r'/innhold/[^/]+/([0-9]{4}-[0-9]{4}-[a-f0-9\-]+)'
    match = re.search(pattern, href, re.IGNORECASE)

    if match:
        return match.group(1)

    return None
