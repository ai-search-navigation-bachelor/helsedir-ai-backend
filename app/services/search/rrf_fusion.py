"""
Reciprocal Rank Fusion (RRF) utility.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass
class RRFResult:
    """Fused result for a single content item."""

    content_id: str
    score: float
    ranks: Dict[str, int]


def fuse_ranked_lists(
    ranked_lists: Dict[str, Sequence[str]],
    rrf_k: int = 60,
    weights: Optional[Dict[str, float]] = None,
) -> List[RRFResult]:
    """
    Fuse multiple ranked lists using Weighted Reciprocal Rank Fusion.

    Score(doc) = sum_i  w_i / (rrf_k + rank_i(doc))

    Args:
        ranked_lists: Mapping from source name to ordered list of content IDs.
        rrf_k: RRF smoothing constant.
        weights: Optional per-source weights. Sources without an explicit
                 weight default to 1.0.
    """
    if not ranked_lists:
        return []

    scores: Dict[str, float] = {}
    ranks_by_doc: Dict[str, Dict[str, int]] = {}

    safe_rrf_k = max(1, int(rrf_k))

    for source_name, doc_ids in ranked_lists.items():
        w = weights.get(source_name, 1.0) if weights else 1.0

        for rank, content_id in enumerate(doc_ids, start=1):
            if not content_id:
                continue

            scores[content_id] = scores.get(content_id, 0.0) + w / (safe_rrf_k + rank)
            doc_ranks = ranks_by_doc.setdefault(content_id, {})
            doc_ranks[source_name] = rank

    fused = [
        RRFResult(content_id=content_id, score=score, ranks=ranks_by_doc.get(content_id, {}))
        for content_id, score in scores.items()
    ]
    fused.sort(key=lambda x: (-x.score, x.content_id))
    return fused
