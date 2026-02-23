"""
Reciprocal Rank Fusion (RRF) utility.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class RRFResult:
    """Fused result for a single content item."""

    content_id: str
    score: float
    ranks: Dict[str, int]


def fuse_ranked_lists(
    ranked_lists: Dict[str, Sequence[str]],
    rrf_k: int = 60,
) -> List[RRFResult]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Score(doc) = sum_i 1 / (rrf_k + rank_i(doc))
    """
    if not ranked_lists:
        return []

    scores: Dict[str, float] = {}
    ranks_by_doc: Dict[str, Dict[str, int]] = {}

    safe_rrf_k = max(1, int(rrf_k))

    for source_name, doc_ids in ranked_lists.items():
        for rank, content_id in enumerate(doc_ids, start=1):
            if not content_id:
                continue

            scores[content_id] = scores.get(content_id, 0.0) + (1.0 / (safe_rrf_k + rank))
            doc_ranks = ranks_by_doc.setdefault(content_id, {})
            doc_ranks[source_name] = rank

    fused = [
        RRFResult(content_id=content_id, score=score, ranks=ranks_by_doc.get(content_id, {}))
        for content_id, score in scores.items()
    ]
    fused.sort(key=lambda x: (-x.score, x.content_id))
    return fused
