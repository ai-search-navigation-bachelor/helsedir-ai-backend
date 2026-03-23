"""
Service for fetching and normalizing NKI statistics payloads.
"""

import asyncio
import copy
import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

from app.config import settings
from app.entities.content import ContentItem
from app.exceptions.helsedir import HelseDirectorateAPIError
from app.services.external.helsedir_api_service import helsedir_api_service

logger = logging.getLogger(__name__)


class NKIStatisticsService:
    """Fetch indicator metadata and dataset rows from the NKI API."""

    def __init__(self):
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._cache_lock = threading.Lock()
        self._cache_ttl_seconds = max(0, settings.nki_statistics_cache_ttl_seconds)

    def clear_cache(self) -> None:
        """Clear the in-memory indicator cache."""
        with self._cache_lock:
            self._cache.clear()

    async def get_statistics_for_content(self, content: ContentItem) -> Dict[str, Any]:
        """Return a frontend-friendly statistics payload for one content page."""
        if not content.nki_indicator_id:
            return self._empty_response(
                content_id=content.id,
                nki_indicator_id=None,
                status="not_configured",
            )

        cached = self._get_cached_indicator_payload(content.nki_indicator_id)
        if cached is None:
            try:
                indicator, dataset = await asyncio.gather(
                    helsedir_api_service.get_nki_quality_indicator_by_id_async(content.nki_indicator_id),
                    helsedir_api_service.get_nki_quality_indicator_data_async(content.nki_indicator_id),
                )
            except HelseDirectorateAPIError as exc:
                logger.warning(
                    "Failed to fetch NKI statistics for content_id=%s indicator_id=%s: %s",
                    content.id,
                    content.nki_indicator_id,
                    exc,
                )
                return self._empty_response(
                    content_id=content.id,
                    nki_indicator_id=content.nki_indicator_id,
                    status="unavailable",
                )

            cached = self._build_indicator_payload(
                indicator=indicator,
                dataset=dataset,
            )
            self._set_cached_indicator_payload(content.nki_indicator_id, cached)

        response = copy.deepcopy(cached)
        response["content_id"] = content.id
        response["nki_indicator_id"] = content.nki_indicator_id
        return response

    def _get_cached_indicator_payload(self, indicator_id: str) -> Optional[Dict[str, Any]]:
        with self._cache_lock:
            cached = self._cache.get(indicator_id)
            if not cached:
                return None

            expires_at, payload = cached
            if expires_at < time.monotonic():
                self._cache.pop(indicator_id, None)
                return None
            return payload

    def _set_cached_indicator_payload(self, indicator_id: str, payload: Dict[str, Any]) -> None:
        if self._cache_ttl_seconds <= 0:
            return
        with self._cache_lock:
            self._cache[indicator_id] = (time.monotonic() + self._cache_ttl_seconds, payload)

    def _empty_response(
        self,
        *,
        content_id: str,
        nki_indicator_id: Optional[str],
        status: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        attachments: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        return {
            "has_statistics": False,
            "statistics_status": status,
            "content_id": content_id,
            "nki_indicator_id": nki_indicator_id,
            "title": title,
            "description": description,
            "attachments": attachments or [],
            "series": [],
            "dimensions": {
                "measures": [],
                "locations": [],
                "parent_locations": [],
                "period_types": [],
            },
        }

    def _build_indicator_payload(
        self,
        *,
        indicator: Dict[str, Any],
        dataset: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        title = self._clean_string(indicator.get("tittel") or indicator.get("title"))
        description = self._clean_string(indicator.get("tekst") or indicator.get("description"))
        attachments = self._normalize_attachments(indicator.get("attachments"))

        rows = self._extract_rows(dataset)
        if not rows:
            return self._empty_response(
                content_id="",
                nki_indicator_id=None,
                status="no_data",
                title=title,
                description=description,
                attachments=attachments,
            )

        series_map: Dict[str, List[Dict[str, Any]]] = {}
        measures = set()
        locations = set()
        parent_locations = set()
        period_types = set()

        for row in rows:
            measure_name = self._clean_string(
                self._pick_value(row, "MeasureName", "measure_name", "Measure")
            ) or "Value"
            location_name = self._clean_string(
                self._pick_value(row, "LocationName", "location_name", "Location")
            )
            parent_name = self._clean_string(
                self._pick_value(row, "ParentName", "parent_name", "ParentLocationName")
            )
            period_type = self._clean_string(
                self._pick_value(row, "PeriodType", "period_type", "Period")
            )
            time_from = self._normalize_time_value(
                self._pick_value(row, "TimeFrom", "time_from", "FromDate")
            )
            time_to = self._normalize_time_value(
                self._pick_value(row, "TimeTo", "time_to", "ToDate")
            )
            value = self._parse_numeric_value(
                self._pick_value(row, "Value", "value", "ResultValue")
            )
            if value is None:
                continue

            measures.add(measure_name)
            if location_name:
                locations.add(location_name)
            if parent_name:
                parent_locations.add(parent_name)
            if period_type:
                period_types.add(period_type)

            series_map.setdefault(measure_name, []).append(
                {
                    "x": time_to or time_from,
                    "y": value,
                    "location": location_name,
                    "parent_location": parent_name,
                    "time_from": time_from,
                    "time_to": time_to,
                    "period_type": period_type,
                }
            )

        if not series_map:
            return self._empty_response(
                content_id="",
                nki_indicator_id=None,
                status="no_data",
                title=title,
                description=description,
                attachments=attachments,
            )

        series = []
        for measure_name in sorted(series_map):
            points = sorted(
                series_map[measure_name],
                key=lambda point: (
                    point.get("x") or "",
                    point.get("location") or "",
                    point.get("parent_location") or "",
                ),
            )
            series.append(
                {
                    "name": measure_name,
                    "points": points,
                }
            )

        return {
            "has_statistics": True,
            "statistics_status": "available",
            "content_id": "",
            "nki_indicator_id": None,
            "title": title,
            "description": description,
            "attachments": attachments,
            "series": series,
            "dimensions": {
                "measures": sorted(measures),
                "locations": sorted(locations),
                "parent_locations": sorted(parent_locations),
                "period_types": sorted(period_types),
            },
        }

    def _extract_rows(self, dataset: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(dataset, dict):
            return []

        nested_data = dataset.get("data")
        if isinstance(nested_data, dict) and (
            "AttachmentDataRows" in nested_data or "attachmentDataRows" in nested_data
        ):
            return self._extract_rows(nested_data)

        columns = dataset.get("ColumnNames") or dataset.get("columnNames") or []
        raw_rows = dataset.get("AttachmentDataRows") or dataset.get("attachmentDataRows") or []
        if not isinstance(raw_rows, list):
            return []

        if raw_rows and isinstance(raw_rows[0], dict):
            return [row for row in raw_rows if isinstance(row, dict)]

        if not columns or not isinstance(columns, list):
            return []

        normalized_rows: List[Dict[str, Any]] = []
        for row in raw_rows:
            if not isinstance(row, list):
                continue
            normalized_rows.append(
                {
                    str(columns[index]): row[index]
                    for index in range(min(len(columns), len(row)))
                }
            )
        return normalized_rows

    def _normalize_attachments(self, attachments: Any) -> List[Dict[str, str]]:
        if not isinstance(attachments, list):
            return []

        normalized = []
        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            title = self._clean_string(attachment.get("title"))
            url = self._clean_string(
                attachment.get("url") or attachment.get("href") or attachment.get("fileUri")
            )
            if not title or not url:
                continue
            if url.startswith("/"):
                url = urljoin(settings.helsedir_public_base_url, url)
            normalized.append({"title": title, "url": url})
        return normalized

    @staticmethod
    def _pick_value(row: Dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in row:
                return row.get(key)
        return None

    @staticmethod
    def _clean_string(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        return stripped or None

    @staticmethod
    def _parse_numeric_value(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            return None

        normalized = value.strip().replace("\u00a0", " ").replace(" ", "")
        if not normalized or normalized in {"-", "NA", "N/A", "null"}:
            return None

        if "," in normalized and "." not in normalized:
            normalized = normalized.replace(",", ".")
        elif "," in normalized and "." in normalized:
            if normalized.rfind(",") > normalized.rfind("."):
                normalized = normalized.replace(".", "").replace(",", ".")
            else:
                normalized = normalized.replace(",", "")

        try:
            return float(normalized)
        except ValueError:
            return None

    @staticmethod
    def _normalize_time_value(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None

        raw = value.strip()
        if not raw:
            return None

        normalized = raw.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return raw

        if parsed.hour == 0 and parsed.minute == 0 and parsed.second == 0 and parsed.microsecond == 0:
            return parsed.date().isoformat()
        return parsed.isoformat()


nki_statistics_service = NKIStatisticsService()
