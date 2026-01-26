"""
Application constants.

Centralized location for configuration values used across the application.
"""

# =============================================================================
# Content Types (info_type)
# =============================================================================
# These are the content types that are returned from the /search endpoint.
# Content with other types is filtered out from search results.

ALLOWED_INFO_TYPES = [
    "anbefaling",
    "fil",
    "horing",
    "retningslinje",
    "rad",
    "regelverk-lov-eller-forskrift",
    "artikkel",
    "statistikk",
    "arrangement",
]

# Set for faster lookup
ALLOWED_INFO_TYPES_SET = set(ALLOWED_INFO_TYPES)


def is_allowed_info_type(info_type: str) -> bool:
    """Check if an info_type is allowed in search results."""
    if not info_type:
        return False
    return info_type.lower() in ALLOWED_INFO_TYPES_SET
