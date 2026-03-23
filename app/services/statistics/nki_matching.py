"""
Helpers for matching NKI indicators to content rows.
"""

import re
import unicodedata
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlparse


_URL_RE = re.compile(r"(https?://[^\s<>\"]+|/[^\s<>\"]+)")
_WHITESPACE_RE = re.compile(r"\s+")
_NORWEGIAN_TRANSLATION = str.maketrans(
    {
        "æ": "ae",
        "ø": "o",
        "å": "a",
        "Æ": "ae",
        "Ø": "o",
        "Å": "a",
    }
)


def normalize_match_text(value: Optional[str]) -> str:
    """Normalize titles for resilient matching."""
    if not value:
        return ""

    normalized = value.translate(_NORWEGIAN_TRANSLATION)
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.casefold().strip()
    return _WHITESPACE_RE.sub(" ", normalized)


def normalize_public_path(path_or_url: Optional[str]) -> Optional[str]:
    """Normalize a public Helsedirektoratet URL/path into a canonical path."""
    if not path_or_url:
        return None

    value = path_or_url.strip()
    if not value:
        return None

    if value.startswith("/"):
        normalized_path = value
    else:
        parsed = urlparse(value)
        if parsed.scheme and parsed.netloc and "helsedirektoratet.no" not in parsed.netloc:
            return None
        normalized_path = parsed.path

    normalized_path = unquote((normalized_path or "").strip())
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"
    normalized_path = normalized_path.rstrip("/") or "/"
    return normalized_path


def extract_indicator_paths(indicator: Dict[str, Any]) -> List[str]:
    """Extract public page paths from indicator payload fields."""
    candidates = []
    for key in ("url", "href", "tekst", "text", "description"):
        value = indicator.get(key)
        if not isinstance(value, str):
            continue
        if key in {"url", "href"}:
            match = normalize_public_path(value)
            if match:
                candidates.append(match)
            continue
        for url in _URL_RE.findall(value):
            match = normalize_public_path(url)
            if match:
                candidates.append(match)

    attachments = indicator.get("attachments")
    if isinstance(attachments, list):
        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            for key in ("url", "href", "fileUri"):
                match = normalize_public_path(attachment.get(key))
                if match:
                    candidates.append(match)

    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        deduped.append(candidate)
        seen.add(candidate)
    return deduped


def _build_path_index(content_rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}
    for row in content_rows:
        path = normalize_public_path(row.get("path"))
        if not path:
            continue
        index.setdefault(path, []).append(row)
    return index


def _build_title_index(
    content_rows: Iterable[Dict[str, Any]],
    *,
    normalized: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}
    seen_keys: Dict[str, set[str]] = {}
    for row in content_rows:
        row_id = str(row.get("id") or "")
        for field in ("tittel", "kort_tittel"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            key = normalize_match_text(value) if normalized else value.strip()
            row_ids_for_key = seen_keys.setdefault(key, set())
            if row_id in row_ids_for_key:
                continue
            index.setdefault(key, []).append(row)
            row_ids_for_key.add(row_id)
    return index


def build_match_indexes(
    content_rows: Iterable[Dict[str, Any]],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Build reusable indexes for indicator matching."""
    rows = list(content_rows)
    return {
        "path_index": _build_path_index(rows),
        "exact_title_index": _build_title_index(rows, normalized=False),
        "normalized_title_index": _build_title_index(rows, normalized=True),
    }


def find_indicator_match(
    indicator: Dict[str, Any],
    content_rows: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Find the safest content match for a single indicator."""
    indexes = build_match_indexes(content_rows)
    return _find_indicator_match_with_indexes(
        indicator,
        path_index=indexes["path_index"],
        exact_title_index=indexes["exact_title_index"],
        normalized_title_index=indexes["normalized_title_index"],
    )


def find_indicator_match_with_indexes(
    indicator: Dict[str, Any],
    indexes: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> Dict[str, Any]:
    """Find the safest content match using prebuilt indexes."""
    return _find_indicator_match_with_indexes(
        indicator,
        path_index=indexes["path_index"],
        exact_title_index=indexes["exact_title_index"],
        normalized_title_index=indexes["normalized_title_index"],
    )


def _find_indicator_match_with_indexes(
    indicator: Dict[str, Any],
    *,
    path_index: Dict[str, List[Dict[str, Any]]],
    exact_title_index: Dict[str, List[Dict[str, Any]]],
    normalized_title_index: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Match helper that reuses prebuilt lookup indexes."""
    for path in extract_indicator_paths(indicator):
        matches = path_index.get(path, [])
        if len(matches) == 1:
            return {"match": matches[0], "strategy": "url", "reason": None}
        if len(matches) > 1:
            return {"match": None, "strategy": None, "reason": f"ambiguous_url:{path}"}

    title = indicator.get("tittel") or indicator.get("title")
    if isinstance(title, str) and title.strip():
        exact_matches = exact_title_index.get(title.strip(), [])
        if len(exact_matches) == 1:
            return {"match": exact_matches[0], "strategy": "exact_title", "reason": None}
        if len(exact_matches) > 1:
            return {"match": None, "strategy": None, "reason": "ambiguous_exact_title"}

        normalized_title = normalize_match_text(title)
        normalized_matches = normalized_title_index.get(normalized_title, [])
        if len(normalized_matches) == 1:
            return {"match": normalized_matches[0], "strategy": "normalized_title", "reason": None}
        if len(normalized_matches) > 1:
            return {"match": None, "strategy": None, "reason": "ambiguous_normalized_title"}

    return {"match": None, "strategy": None, "reason": "no_match"}


def plan_indicator_backfill(
    indicators: Iterable[Dict[str, Any]],
    content_rows: Iterable[Dict[str, Any]],
    *,
    force: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build deterministic DB updates and skip reasons for indicator matching."""
    rows = list(content_rows)
    rows_by_id = {row["id"]: row for row in rows if row.get("id")}
    path_index = _build_path_index(rows)
    exact_title_index = _build_title_index(rows, normalized=False)
    normalized_title_index = _build_title_index(rows, normalized=True)
    preliminary_matches: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for indicator in indicators:
        indicator_id = indicator.get("id")
        result = _find_indicator_match_with_indexes(
            indicator,
            path_index=path_index,
            exact_title_index=exact_title_index,
            normalized_title_index=normalized_title_index,
        )
        if result["match"] is None:
            skipped.append(
                {
                    "indicator_id": indicator_id,
                    "title": indicator.get("tittel") or indicator.get("title"),
                    "reason": result["reason"],
                }
            )
            continue

        content_row = result["match"]
        preliminary_matches.append(
            {
                "content_id": content_row["id"],
                "indicator_id": indicator_id,
                "strategy": result["strategy"],
                "title": indicator.get("tittel") or indicator.get("title"),
            }
        )

    matches_by_content: Dict[str, List[Dict[str, Any]]] = {}
    for match in preliminary_matches:
        matches_by_content.setdefault(match["content_id"], []).append(match)

    updates: List[Dict[str, Any]] = []
    for content_id, matches in matches_by_content.items():
        if len(matches) > 1:
            skipped.append(
                {
                    "content_id": content_id,
                    "indicator_ids": [match["indicator_id"] for match in matches],
                    "reason": "content_conflict",
                }
            )
            continue

        match = matches[0]
        row = rows_by_id.get(content_id, {})
        existing_indicator_id = row.get("nki_indicator_id")
        if existing_indicator_id == match["indicator_id"]:
            continue
        if existing_indicator_id and not force:
            skipped.append(
                {
                    "content_id": content_id,
                    "indicator_id": match["indicator_id"],
                    "existing_indicator_id": existing_indicator_id,
                    "reason": "existing_mapping",
                }
            )
            continue

        updates.append(match)

    updates.sort(key=lambda item: (item["content_id"], item["indicator_id"]))
    skipped.sort(
        key=lambda item: (
            item.get("reason") or "",
            item.get("content_id") or "",
            item.get("indicator_id") or "",
            json.dumps(item, sort_keys=True, ensure_ascii=False),
        )
    )
    return updates, skipped
