"""
Hierarchy helpers for BM25 score propagation.
"""

import hashlib
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
from urllib.parse import urlparse

import numpy as np

from app.entities.content import ContentItem


@dataclass(frozen=True)
class BM25HierarchyConfig:
    enabled: bool
    max_depth: int
    decay: float
    source_top_k: int
    top_children: int
    tail_weight: float
    weight: float
    min_contribution: float


class BM25HierarchyIndex:
    """Builds parent graph and propagates BM25 scores up the hierarchy."""

    _HREF_ID_RE = re.compile(r"/innhold/[^/]+/([0-9]{4}-[0-9]{4}[-a-f0-9]+)", re.IGNORECASE)

    def __init__(self, config: BM25HierarchyConfig):
        self.config = config
        self._id_to_idx: Dict[str, int] = {}
        self._path_to_idx: Dict[str, int] = {}
        self.ancestors_by_idx: Dict[int, List[Tuple[int, int]]] = {}

    @staticmethod
    def normalize_path(path: Optional[str]) -> str:
        if not path:
            return ""
        p = str(path).strip()
        if not p:
            return ""
        if p.startswith("http://") or p.startswith("https://"):
            p = urlparse(p).path or ""
        if not p:
            return ""
        if not p.startswith("/"):
            p = "/" + p
        p = p.rstrip("/")
        return p or "/"

    @classmethod
    def fingerprint_item(cls, item: ContentItem) -> Tuple[str, str]:
        link_chunks = []
        for link in item.links or []:
            rel = (link.rel or "").lower()
            link_chunks.append(
                "|".join(
                    [
                        rel,
                        (link.type or "").lower(),
                        str(link.id or ""),
                        cls.normalize_path(getattr(link, "path", None)),
                        str(link.href or ""),
                    ]
                )
            )
        link_chunks.sort()

        tg = "|".join(sorted(item.target_groups or []))
        payload = "\x1f".join(
            [
                item.title or "",
                item.body or "",
                (item.content_type or "").lower(),
                cls.normalize_path(item.path),
                tg,
                "\x1e".join(link_chunks),
            ]
        )
        digest = hashlib.sha1(payload.encode("utf-8", "ignore")).hexdigest()
        return item.id, digest

    @classmethod
    def compute_signature(cls, items: List[ContentItem]) -> FrozenSet[Tuple[str, str]]:
        return frozenset(cls.fingerprint_item(item) for item in items)

    def _resolve_link_target_idx(self, link) -> Optional[int]:
        link_id = getattr(link, "id", None)
        if link_id and link_id in self._id_to_idx:
            return self._id_to_idx[link_id]

        link_path = self.normalize_path(getattr(link, "path", None))
        if link_path and link_path in self._path_to_idx:
            return self._path_to_idx[link_path]

        href = getattr(link, "href", None)
        if href:
            href_path = self.normalize_path(href)
            if href_path and href_path in self._path_to_idx:
                return self._path_to_idx[href_path]

            m = self._HREF_ID_RE.search(str(href))
            if m:
                content_id = m.group(1)
                if content_id in self._id_to_idx:
                    return self._id_to_idx[content_id]

        return None

    def build(self, items: List[ContentItem]) -> None:
        parent_by_child: Dict[int, Set[int]] = defaultdict(set)

        self._id_to_idx = {}
        self._path_to_idx = {}
        for idx, item in enumerate(items):
            self._id_to_idx[item.id] = idx
            p = self.normalize_path(item.path)
            if p and p not in self._path_to_idx:
                self._path_to_idx[p] = idx

        for idx, item in enumerate(items):
            for link in item.links or []:
                rel = (link.rel or "").lower()
                if rel not in {"barn", "forelder", "temaside"}:
                    continue

                target_idx = self._resolve_link_target_idx(link)
                if target_idx is None:
                    continue

                if rel == "barn":
                    child_idx, parent_idx = target_idx, idx
                else:
                    child_idx, parent_idx = idx, target_idx

                if child_idx != parent_idx:
                    parent_by_child[child_idx].add(parent_idx)

        self.ancestors_by_idx = {}
        if not self.config.enabled:
            return

        max_depth = self.config.max_depth
        for doc_idx in range(len(items)):
            seen = {doc_idx}
            q = deque([(doc_idx, 0)])
            ancestors: List[Tuple[int, int]] = []

            while q:
                current_idx, depth = q.popleft()
                if depth >= max_depth:
                    continue

                for parent_idx in parent_by_child.get(current_idx, set()):
                    if parent_idx in seen:
                        continue
                    seen.add(parent_idx)

                    next_depth = depth + 1
                    ancestors.append((parent_idx, next_depth))
                    q.append((parent_idx, next_depth))

            self.ancestors_by_idx[doc_idx] = ancestors

    def propagate(self, direct_scores: np.ndarray) -> np.ndarray:
        inherited_scores = np.zeros(len(direct_scores), dtype=np.float32)
        if not self.config.enabled or not self.ancestors_by_idx:
            return inherited_scores

        active_indices = np.flatnonzero(direct_scores > 0)
        if active_indices.size == 0:
            return inherited_scores

        top_k = min(self.config.source_top_k, int(active_indices.size))
        sorted_sources = sorted(
            active_indices.tolist(),
            key=lambda i: -float(direct_scores[i]),
        )[:top_k]

        propagated: Dict[int, List[float]] = defaultdict(list)
        for source_idx in sorted_sources:
            source_score = float(direct_scores[source_idx])
            for anc_idx, depth in self.ancestors_by_idx.get(source_idx, []):
                contribution = source_score * (self.config.decay ** depth)
                if contribution < self.config.min_contribution:
                    continue
                propagated[anc_idx].append(contribution)

        for anc_idx, values in propagated.items():
            values.sort(reverse=True)
            top_values = values[: self.config.top_children]
            if not top_values:
                continue
            best = top_values[0]
            tail_sum = sum(top_values[1:])
            aggregate = best + (self.config.tail_weight * tail_sum)
            inherited_scores[anc_idx] = float(self.config.weight * aggregate)

        return inherited_scores
