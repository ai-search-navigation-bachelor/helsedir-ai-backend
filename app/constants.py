"""
Application constants.

Centralized location for configuration values used across the application.
"""

# =============================================================================
# Content Types (info_type) with Display Names
# =============================================================================
# Single source of truth: info_type -> Norwegian display name
# Content with other types is filtered out from search results.

CATEGORY_INFO = {
    # Retningslinjer
    "retningslinje": "Retningslinje",
    "nasjonalt-forlop": "Nasjonalt forløp",
    "pakkeforlop-anbefaling": "Pakkeforløp-anbefaling",
    "normen-dokument": "Normen-dokument",
    "nasjonal-veileder": "Nasjonal veileder",
    "prioriteringsveileder": "Prioriteringsveileder",

    # Faglige råd
    "anbefaling": "Anbefaling",
    "rad": "Råd",
    "faglig-rad": "Faglig råd",
    "pico": "PICO",

    # Veiledere
    "takst-med-merknad": "Takst med merknad",
    "ehelsestandard": "E-helsestandard",
    "tilskudd": "Tilskudd",
    "veileder": "Veileder",
    "veiledning": "Veiledning",

    # Rundskriv
    "rundskriv": "Rundskriv",

    # Lovfortolkning
    "lov-eller-forskriftstekst-med-kommentar": "Lov/forskrift med kommentar",
    "regelverk-lov-eller-forskrift": "Regelverk",
    "veileder-lov-forskrift": "Veileder til lov og forskrift",
    "paragraf-med-kommentar": "Paragraf med kommentar",

    # Statistikk og rapporter
    "rapport": "Rapport",
    "statistikkelement": "Statistikkelement",
    "statistikk": "Statistikk",

    # Temasider (spesiell type)
    "temaside": "Temaside",
}

# Derived from CATEGORY_INFO
ALLOWED_INFO_TYPES = list(CATEGORY_INFO.keys())
ALLOWED_INFO_TYPES_SET = set(ALLOWED_INFO_TYPES)


def is_allowed_info_type(info_type: str) -> bool:
    """Check if an info_type is allowed in search results."""
    if not info_type:
        return False
    return info_type.lower() in ALLOWED_INFO_TYPES_SET


def get_category_display_name(info_type: str) -> str:
    """Get the Norwegian display name for a category."""
    if not info_type:
        return "Ukjent"
    return CATEGORY_INFO.get(info_type.lower(), info_type.title())


# =============================================================================
# Priority Categories (for categorized search)
# =============================================================================
# These categories show ALL results instead of just top N preview.
# They are displayed at the top of the search results page.

PRIORITY_CATEGORIES = [
    "temaside",  # Theme pages shown first with all results
    "retningslinje",
]

PRIORITY_CATEGORIES_SET = set(PRIORITY_CATEGORIES)


def is_priority_category(info_type: str) -> bool:
    """Check if a category should show all results (priority category)."""
    if not info_type:
        return False
    return info_type.lower() in PRIORITY_CATEGORIES_SET
