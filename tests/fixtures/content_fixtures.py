"""
Factory helpers for building ContentItem test data.
"""

from app.entities.content import ContentItem, ContentLink


def make_retningslinje(
    id: str = "r001",
    title: str = "Test Retningslinje",
    body: str = "Innhold.",
) -> ContentItem:
    return ContentItem(id=id, title=title, body=body, content_type="retningslinje")


def make_veileder(
    id: str = "v001",
    title: str = "Test Veileder",
    body: str = "Innhold.",
) -> ContentItem:
    return ContentItem(id=id, title=title, body=body, content_type="veileder")


def make_temaside(
    id: str = "t001",
    title: str = "Test Temaside",
    path: str = "/temasider/test",
) -> ContentItem:
    return ContentItem(id=id, title=title, body="", content_type="temaside", path=path)


def make_content_with_role(id: str, title: str, role: str) -> ContentItem:
    """Create a content item restricted to a specific role."""
    return ContentItem(
        id=id,
        title=title,
        body="",
        content_type="veileder",
        target_groups=[role],
    )


def make_internal_link(rel: str = "barn", type: str = "kapittel", id: str = "child-001") -> ContentLink:
    return ContentLink(rel=rel, type=type, id=id)


def make_external_link(
    rel: str = "publikasjon",
    type: str = "artikkel",
    href: str = "https://example.com/article/1234",
) -> ContentLink:
    return ContentLink(rel=rel, type=type, href=href)
