"""
Helpers for normalizing text-content and document metadata.
"""

from __future__ import annotations

import html
import re
from typing import Any, Dict, Iterable, Mapping, Optional
from urllib.parse import urlparse


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HELSEDIR_PUBLIC_BASE_URL = "https://www.helsedirektoratet.no"


def _normalize_text(value: Optional[str]) -> str:
    """Strip HTML and whitespace so empty markup does not count as text."""
    if not value:
        return ""

    text = html.unescape(value)
    text = text.replace("\xa0", " ")
    text = _HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def has_visible_text(value: Optional[str]) -> bool:
    """Return True if the given HTML/text string contains visible content."""
    return bool(_normalize_text(value))


def _iter_document_candidates(payload: Mapping[str, Any]) -> Iterable[str]:
    data = payload.get("data")
    if isinstance(data, Mapping):
        file_url = data.get("fil")
        if isinstance(file_url, str) and file_url.strip():
            yield file_url.strip()

    attachments = payload.get("attachments")
    if isinstance(attachments, list):
        for attachment in attachments:
            if not isinstance(attachment, Mapping):
                continue
            for key in ("href", "url", "fil"):
                candidate = attachment.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    yield candidate.strip()

    for key in ("links", "lenker"):
        links = payload.get(key)
        if not isinstance(links, list):
            continue
        for link in links:
            if not isinstance(link, Mapping):
                continue
            for field_name in ("href", "url", "fil"):
                candidate = link.get(field_name)
                if isinstance(candidate, str) and candidate.strip().lower().endswith(".pdf"):
                    yield candidate.strip()


def extract_document_url(payload: Mapping[str, Any]) -> Optional[str]:
    """Return the first normalized document URL, if any."""
    for candidate in _iter_document_candidates(payload):
        return candidate
    return None


def resolve_public_document_url(
    path: Optional[str],
    stored_document_url: Optional[str],
) -> Optional[str]:
    """
    Return the public Helsedirektoratet URL frontend should open.

    Prefer the public content page derived from `path`. Fall back to the stored
    document URL (typically the raw PDF file) if no public path is available.
    """
    if path:
        normalized_path = path.strip()
        if normalized_path:
            parsed = urlparse(normalized_path)
            if parsed.scheme and parsed.netloc:
                return normalized_path
            if not normalized_path.startswith("/"):
                normalized_path = f"/{normalized_path}"
            return f"{_HELSEDIR_PUBLIC_BASE_URL}{normalized_path}"

    if stored_document_url and stored_document_url.strip():
        return stored_document_url.strip()

    return None


def compute_document_metadata(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Compute normalized text/document metadata from an API payload.

    Returns:
        {
            "has_text_content": bool,
            "document_url": Optional[str],
        }
    """
    text_value = payload.get("tekst")
    if text_value is None:
        text_value = payload.get("body")

    has_text_content = has_visible_text(text_value if isinstance(text_value, str) else None)
    document_url = extract_document_url(payload)

    return {
        "has_text_content": has_text_content,
        "document_url": document_url,
    }
