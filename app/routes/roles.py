"""
Roles endpoint.

Returns the list of available user roles for content personalization.
"""

from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from app.constants import ROLE_INFO

router = APIRouter(tags=["roles"])


class Role(BaseModel):
    slug: str
    display_name: str


class RolesResponse(BaseModel):
    roles: List[Role]


@router.get("/roles", response_model=RolesResponse)
async def get_roles() -> RolesResponse:
    """
    Get all available user roles.

    Returns a list of role objects with slug and display name.
    """
    return RolesResponse(
        roles=[
            Role(slug=slug, display_name=name)
            for slug, name in ROLE_INFO.items()
        ]
    )
