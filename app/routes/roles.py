"""
Roles endpoint.

Returns the list of available user roles for content personalization.
"""

from fastapi import APIRouter
from app.constants import ROLE_INFO

router = APIRouter(tags=["roles"])


@router.get("/roles")
async def get_roles():
    """
    Get all available user roles.

    Returns a list of role objects with slug and display name.
    """
    return {
        "roles": [
            {"slug": slug, "display_name": name}
            for slug, name in ROLE_INFO.items()
        ]
    }
