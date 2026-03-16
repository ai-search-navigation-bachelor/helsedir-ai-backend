"""
Helpers for normalizing e-helsestandard metadata shared across app layers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urljoin, urlparse

from app.config import settings


def resolve_attachment_url(file_uri: Optional[str]) -> Optional[str]:
    """Return an absolute public attachment URL for e-helsestandard files."""
    if not file_uri or not file_uri.strip():
        return None

    normalized = file_uri.strip()
    parsed = urlparse(normalized)
    if parsed.scheme and parsed.netloc:
        return normalized

    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return urljoin(settings.helsedir_public_base_url, normalized)


def normalize_attachment_entry(attachment: Mapping[str, Any]) -> Optional[Dict[str, Optional[str]]]:
    """Normalize a single attachment dict into a stable attachment shape."""
    attachment_url = resolve_attachment_url(
        attachment.get("fileUri")
        if isinstance(attachment.get("fileUri"), str)
        else attachment.get("file_uri")
        if isinstance(attachment.get("file_uri"), str)
        else attachment.get("url")
        if isinstance(attachment.get("url"), str)
        else attachment.get("href")
        if isinstance(attachment.get("href"), str)
        else None
    )
    if not attachment_url:
        return None

    title = (
        attachment.get("title")
        if isinstance(attachment.get("title"), str)
        else attachment.get("tittel")
        if isinstance(attachment.get("tittel"), str)
        else attachment.get("name")
        if isinstance(attachment.get("name"), str)
        else attachment_url.rstrip("/").rsplit("/", 1)[-1]
    )

    return {
        "title": title.strip(),
        "url": attachment_url,
        "file_type": attachment.get("fileType")
        if isinstance(attachment.get("fileType"), str)
        else attachment.get("file_type")
        if isinstance(attachment.get("file_type"), str)
        else attachment.get("type")
        if isinstance(attachment.get("type"), str)
        else None,
    }


def normalize_attachment_list(raw_attachments: Any) -> List[Dict[str, Optional[str]]]:
    """Normalize a raw attachments payload into a stable list."""
    if not isinstance(raw_attachments, list):
        return []

    normalized: List[Dict[str, Optional[str]]] = []
    for attachment in raw_attachments:
        if not isinstance(attachment, Mapping):
            continue
        normalized_attachment = normalize_attachment_entry(attachment)
        if normalized_attachment is None:
            continue
        normalized.append(normalized_attachment)
    return normalized
