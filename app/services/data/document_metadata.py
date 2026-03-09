"""
Helpers for normalizing text-content and document metadata.
"""

from __future__ import annotations

import html
import json
import logging
import re
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import urljoin, urlparse

import httpx

from app.config import settings

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SHORTCODE_RE = re.compile(r"\[[a-z0-9_-]+[^\]]*/\]", re.IGNORECASE)
_ANCHOR_WITHOUT_HREF_RE = re.compile(
    r"<a(?![^>]*\bhref\s*=)[^>]*>.*?</a>",
    re.IGNORECASE | re.DOTALL,
)
_PDF_HREF_RE = re.compile(r"\.pdf(?:$|[?#])", re.IGNORECASE)
_PDF_PATH_SUFFIXES = ("/pdf-av-rapporten", "/pdf-versjon-av-rapporten")
_JSON_SCRIPT_RE = re.compile(
    r"<script[^>]*type=[\"']application/json[\"'][^>]*>(.*?)</script>",
    re.IGNORECASE | re.DOTALL,
)

logger = logging.getLogger(__name__)


class _PdfAnchorExtractor(HTMLParser):
    """Extract the first anchor href that points to a PDF."""

    def __init__(self):
        super().__init__()
        self.pdf_href: Optional[str] = None

    def handle_starttag(self, tag: str, attrs):
        if self.pdf_href or tag.lower() != "a":
            return

        href = None
        for key, value in attrs:
            if key.lower() == "href":
                href = value
                break

        if href and _PDF_HREF_RE.search(href.strip()):
            self.pdf_href = href.strip()


def is_pdf_report_chapter_path(path: Optional[str]) -> bool:
    """Return True when the path points to a PDF chapter page."""
    if not path:
        return False

    normalized_path = path.strip().rstrip("/")
    return any(normalized_path.endswith(suffix) for suffix in _PDF_PATH_SUFFIXES)


def _is_pdf_report_chapter_payload(payload: Mapping[str, Any]) -> bool:
    path = payload.get("path")
    if isinstance(path, str) and is_pdf_report_chapter_path(path):
        return True

    url = payload.get("url")
    if isinstance(url, str):
        parsed = urlparse(url.strip())
        if parsed.path and is_pdf_report_chapter_path(parsed.path):
            return True

    return False


def _strip_non_content_pdf_chapter_markup(text: str) -> str:
    """Strip button-like markup that should not count as body text."""
    text = _SHORTCODE_RE.sub(" ", text)
    text = _ANCHOR_WITHOUT_HREF_RE.sub(" ", text)
    return text


def _normalize_text(value: Optional[str], *, is_pdf_report_chapter: bool = False) -> str:
    """Strip HTML and whitespace so empty markup does not count as text."""
    if not value:
        return ""

    text = html.unescape(value)
    text = text.replace("\xa0", " ")
    if is_pdf_report_chapter:
        text = _strip_non_content_pdf_chapter_markup(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def has_visible_text(value: Optional[str], *, is_pdf_report_chapter: bool = False) -> bool:
    """Return True if the given HTML/text string contains visible content."""
    return bool(_normalize_text(value, is_pdf_report_chapter=is_pdf_report_chapter))


def _iter_document_candidates(payload: Mapping[str, Any]) -> Iterable[str]:
    # Intentional asymmetry: data["fil"] and attachment entries are treated as
    # document file URLs directly, while links/lenker may include other
    # references and are therefore restricted to explicit PDF URLs only.
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
            for key in ("fil", "href", "url"):
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
            for field_name in ("fil", "href", "url"):
                candidate = link.get(field_name)
                if isinstance(candidate, str) and candidate.strip().lower().endswith(".pdf"):
                    yield candidate.strip()


def extract_document_url(payload: Mapping[str, Any]) -> Optional[str]:
    """Return the first normalized document URL, if any."""
    for candidate in _iter_document_candidates(payload):
        return candidate
    return None


def _build_public_content_url(path: Optional[str], public_url: Optional[str] = None) -> Optional[str]:
    """Return the public HTML page URL for a content item."""
    if public_url and public_url.strip():
        return public_url.strip()

    if path:
        normalized_path = path.strip()
        if normalized_path:
            parsed = urlparse(normalized_path)
            if parsed.scheme and parsed.netloc:
                return normalized_path
            if not normalized_path.startswith("/"):
                normalized_path = f"/{normalized_path}"
            return f"{settings.helsedir_public_base_url}{normalized_path}"

    return None


def extract_pdf_url_from_public_html(
    html_content: str,
    *,
    base_url: Optional[str] = None,
) -> Optional[str]:
    """Return the first PDF link found in a public Helsedir HTML page."""
    parser = _PdfAnchorExtractor()
    parser.feed(html_content)
    parser.close()

    if not parser.pdf_href:
        return None

    if base_url:
        return urljoin(base_url, parser.pdf_href)
    return parser.pdf_href


def extract_related_links_from_public_html(
    html_content: str,
    *,
    base_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return normalized relatedLinks entries from a public Helsedir HTML page."""
    for match in _JSON_SCRIPT_RE.finditer(html_content):
        raw_json = html.unescape(match.group(1).strip())
        if not raw_json:
            continue
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            continue

        props = payload.get("props")
        related_links = props.get("relatedLinks") if isinstance(props, Mapping) else None
        if not isinstance(related_links, list):
            continue

        normalized: List[Dict[str, Any]] = []
        for item in related_links:
            if not isinstance(item, Mapping):
                continue
            title = item.get("tittel")
            url = item.get("url")
            if not isinstance(title, str) or not title.strip():
                continue
            if not isinstance(url, str) or not url.strip():
                continue

            normalized.append(
                {
                    "title": title.strip(),
                    "url": urljoin(base_url, url.strip()) if base_url else url.strip(),
                    "is_document": bool(item.get("type")),
                    "file_type": item.get("typeFile") if isinstance(item.get("typeFile"), str) else None,
                    "url_type": item.get("typeUrl") if isinstance(item.get("typeUrl"), str) else None,
                    "target": item.get("target") if isinstance(item.get("target"), str) else None,
                }
            )

        if normalized:
            return normalized

    return []


def resolve_public_related_links(
    path: Optional[str],
    *,
    public_url: Optional[str] = None,
    timeout: float = 10.0,
    client: Optional[httpx.Client] = None,
) -> List[Dict[str, Any]]:
    """Resolve related links from the public HTML page for textless report pages."""
    page_url = _build_public_content_url(path, public_url.strip() if isinstance(public_url, str) else public_url)
    if not page_url:
        return []

    owns_client = client is None
    if owns_client:
        client = httpx.Client(timeout=timeout, follow_redirects=True)

    try:
        response = client.get(page_url)
        response.raise_for_status()
        return extract_related_links_from_public_html(response.text, base_url=page_url)
    except httpx.HTTPError as exc:
        logger.warning("Failed to resolve related links from public page %s: %s", page_url, exc)
        return []
    finally:
        if owns_client and client is not None:
            client.close()


def resolve_pdf_report_chapter_document_url(
    path: Optional[str],
    *,
    public_url: Optional[str] = None,
    timeout: float = 10.0,
    client: Optional[httpx.Client] = None,
) -> Optional[str]:
    """
    Resolve a PDF URL by scraping the public HTML page of a PDF chapter.

    This function is intended for import/backfill flows, not request-time use.
    """
    normalized_public_url = public_url.strip() if isinstance(public_url, str) else None
    if not (
        is_pdf_report_chapter_path(path)
        or (normalized_public_url and is_pdf_report_chapter_path(urlparse(normalized_public_url).path))
    ):
        return None

    page_url = _build_public_content_url(path, normalized_public_url)
    if not page_url:
        return None

    owns_client = client is None
    if owns_client:
        client = httpx.Client(timeout=timeout, follow_redirects=True)

    try:
        response = client.get(page_url)
        response.raise_for_status()
        return extract_pdf_url_from_public_html(response.text, base_url=page_url)
    except httpx.HTTPError as exc:
        logger.warning("Failed to resolve PDF URL from public page %s: %s", page_url, exc)
        return None
    finally:
        if owns_client and client is not None:
            client.close()


def resolve_public_document_url(
    path: Optional[str],
    stored_document_url: Optional[str],
) -> Optional[str]:
    """
    Return the public Helsedirektoratet URL frontend should open.

    Prefer the public content page derived from `path`. Fall back to the stored
    document URL (typically the raw PDF file) if no public path is available.
    """
    if is_pdf_report_chapter_path(path) and stored_document_url and stored_document_url.strip():
        return stored_document_url.strip()

    if path:
        normalized_path = path.strip()
        if normalized_path:
            parsed = urlparse(normalized_path)
            if parsed.scheme and parsed.netloc:
                return normalized_path
            if not normalized_path.startswith("/"):
                normalized_path = f"/{normalized_path}"
            return f"{settings.helsedir_public_base_url}{normalized_path}"

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

    is_pdf_report_chapter = _is_pdf_report_chapter_payload(payload)
    has_text_content = has_visible_text(
        text_value if isinstance(text_value, str) else None,
        is_pdf_report_chapter=is_pdf_report_chapter,
    )
    document_url = extract_document_url(payload)

    return {
        "has_text_content": has_text_content,
        "document_url": document_url,
    }


def compute_document_metadata_with_fallback(
    payload: Mapping[str, Any],
    *,
    timeout: float = 10.0,
    client: Optional[httpx.Client] = None,
) -> Dict[str, Any]:
    """
    Compute metadata and, for PDF chapter payloads only, fall back to public
    HTML scraping when the primary metadata has no document URL.

    Intended for import/backfill flows, not request-time use. This will only
    attempt the network fallback when `_is_pdf_report_chapter_payload(payload)`
    is true, and `resolve_pdf_report_chapter_document_url` may perform blocking
    HTTP calls using the provided timeout/client.
    """
    meta = compute_document_metadata(payload)
    if meta["document_url"] or not _is_pdf_report_chapter_payload(payload):
        return meta

    fallback_document_url = resolve_pdf_report_chapter_document_url(
        payload.get("path") if isinstance(payload.get("path"), str) else None,
        public_url=payload.get("url") if isinstance(payload.get("url"), str) else None,
        timeout=timeout,
        client=client,
    )
    if fallback_document_url:
        meta["document_url"] = fallback_document_url

    return meta


def build_content_metadata(
    path: Optional[str],
    has_text_content: Any,
    stored_document_url: Optional[str],
) -> Dict[str, Any]:
    """Build canonical API metadata fields from stored content values."""
    has_text = bool(has_text_content)
    is_pdf_only = (not has_text) and bool(stored_document_url)
    document_url = (
        resolve_public_document_url(path, stored_document_url)
        if is_pdf_only
        else None
    )
    return {
        "has_text_content": has_text,
        "document_url": document_url,
        "is_pdf_only": is_pdf_only,
    }
