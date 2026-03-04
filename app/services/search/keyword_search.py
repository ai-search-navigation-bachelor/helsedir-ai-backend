"""
Keyword-based search functionality, modelled after Helsedirektoratet's current search.

Scoring combines:
- Which fields have hits (title > body)
- Content type authority (retningslinje > veileder > temaside > other)
- Norwegian stop word filtering
- Norwegian Snowball stemming for morphological matching
- Basic Norwegian medical synonyms
"""

import re
from typing import List, Dict, Any, Optional, Set, Tuple

from app.dto.response.search import SearchResult
from app.entities.content import ContentItem
from app.services.data.content_service import content_service
from app.config import settings


# ---------------------------------------------------------------------------
# Snowball stemmer (Norwegian)
# ---------------------------------------------------------------------------
try:
    from snowballstemmer import stemmer as _make_stemmer
    _nor_stemmer = _make_stemmer("norwegian")

    def _stem(word: str) -> str:
        return _nor_stemmer.stemWord(word)

except ImportError:
    def _stem(word: str) -> str:  # type: ignore[misc]
        return word


# ---------------------------------------------------------------------------
# Norwegian stop words
# ---------------------------------------------------------------------------
STOP_WORDS: Set[str] = {
    "og", "i", "er", "en", "et", "å", "av", "til", "for", "om",
    "på", "som", "med", "de", "vi", "at", "seg", "men", "fra",
    "den", "det", "ikke", "har", "blir", "ble", "var", "bli",
    "kan", "skal", "vil", "eller", "etter", "nå", "da", "her",
    "disse", "alle", "også", "så", "dette", "sin", "hans", "hennes",
    "sitt", "sine", "vår", "dere", "dem", "deg", "meg", "hun", "ham",
    "inn", "ut", "opp", "ned", "mot", "over", "under", "ved", "uten",
}

# ---------------------------------------------------------------------------
# Norwegian medical synonyms (bidirectional)
# ---------------------------------------------------------------------------
SYNONYMS: Dict[str, List[str]] = {
    "hjerteinfarkt": ["hjerteanfall"],
    "hjerteanfall": ["hjerteinfarkt"],
    "kreft": ["cancer"],
    "cancer": ["kreft"],
    "hypertensjon": ["høyt blodtrykk", "blodtrykk"],
    "sukkersyke": ["diabetes"],
    "diabetes": ["sukkersyke"],
    "adhd": ["konsentrasjonsvansker", "hyperaktivitet"],
    "konsentrasjonsvansker": ["adhd"],
    "depresjon": ["deprimert"],
    "deprimert": ["depresjon"],
}

# ---------------------------------------------------------------------------
# Content type authority boosts (applied as multiplier after field scoring)
# ---------------------------------------------------------------------------
TYPE_BOOST: Dict[str, float] = {
    "retningslinje": 1.3,
    "veileder": 1.2,
    "temaside": 1.15,
    "pakkeforlop": 1.1,
}

# Body field score weights (~50 % of title weights)
_BODY_EXACT_WEIGHT: float = 5.0
_BODY_KEYWORD_WEIGHT: float = 1.5


def _tokenize(text: str) -> Set[str]:
    """Tokenize text: lowercase, remove stop words, apply Snowball stemming."""
    words = re.findall(r"\w+", text.lower())
    return {_stem(w) for w in words if w not in STOP_WORDS}


def _expand_query(raw_query_lower: str, stemmed_keywords: Set[str]) -> Set[str]:
    """Expand query keywords with synonym stems."""
    raw_words = re.findall(r"\w+", raw_query_lower)
    expanded = set(stemmed_keywords)
    for word in raw_words:
        for synonym_phrase in SYNONYMS.get(word, []):
            expanded.update(_tokenize(synonym_phrase))
    return expanded


class KeywordSearch:
    """
    Keyword search modelled after Helsedirektoratet's current solution.

    Scoring priority:
    1. Title hits (exact phrase > keyword matches > full coverage)
    2. Body hits (exact phrase > keyword matches)
    3. Content type authority boost (retningslinje/veileder/temaside/pakkeforlop)

    Norwegian stop words are filtered out; Snowball stemming normalises word forms;
    a small synonym table expands the query for common medical terms.
    """

    def search(
        self,
        query: str,
        role: Optional[str] = None,
        k: int = 10,
    ) -> List[SearchResult]:
        query_lower = query.lower()
        stemmed_keywords = _tokenize(query_lower)
        query_keywords = _expand_query(query_lower, stemmed_keywords)

        results = []
        for item in content_service.get_all_content():
            if item.content_type not in content_service.searchable_types:
                continue
            if role and role not in item.target_groups:
                continue

            score, breakdown = self.calculate_score_with_breakdown(
                item, query_lower, query_keywords
            )
            if score > 0:
                results.append((item, score, breakdown))

        results.sort(key=lambda x: (-x[1], x[0].id))

        if not results:
            return []

        max_score = results[0][1]
        normalized = []
        for item, raw_score, breakdown in results[:k]:
            norm_score = raw_score / max_score if max_score > 0 else 0
            explanation = self._create_explanation(breakdown, role)
            explanation += f" | Score: {raw_score:.1f} → {norm_score:.2f}"
            normalized.append(
                SearchResult(
                    id=item.id,
                    title=item.title,
                    info_type=item.content_type,
                    path=item.path,
                    score=round(norm_score, 3),
                    explanation=explanation,
                )
            )
        return normalized

    def calculate_score(
        self,
        item: ContentItem,
        query_lower: str,
        query_keywords: Set[str],
    ) -> float:
        score, _ = self.calculate_score_with_breakdown(item, query_lower, query_keywords)
        return score

    def calculate_score_with_breakdown(
        self,
        item: ContentItem,
        query_lower: str,
        query_keywords: Set[str],
    ) -> Tuple[float, Dict[str, Any]]:
        """Score an item against the query using title + body + content type boost."""
        score = 0.0
        breakdown: Dict[str, Any] = {}

        # --- Title scoring ---
        title_lower = item.title.lower()
        title_stemmed = _tokenize(title_lower)

        if query_lower in title_lower:
            pts = settings.search_exact_phrase_title_weight
            score += pts
            breakdown["exact_title"] = pts

        title_matches = query_keywords & title_stemmed
        if title_matches:
            pts = len(title_matches) * settings.search_keyword_title_weight
            score += pts
            breakdown["title_keywords"] = {
                "count": len(title_matches),
                "matches": list(title_matches),
                "points": pts,
            }

        if title_stemmed and title_stemmed.issubset(query_keywords):
            pts = settings.search_full_title_coverage_weight
            score += pts
            breakdown["full_title_coverage"] = pts

        # --- Body scoring ---
        body = item.body or ""
        if body:
            body_lower = body.lower()
            body_stemmed = _tokenize(body_lower)

            if query_lower in body_lower:
                pts = _BODY_EXACT_WEIGHT
                score += pts
                breakdown["exact_body"] = pts

            body_matches = query_keywords & body_stemmed
            if body_matches:
                pts = len(body_matches) * _BODY_KEYWORD_WEIGHT
                score += pts
                breakdown["body_keywords"] = {
                    "count": len(body_matches),
                    "matches": list(body_matches)[:5],
                    "points": pts,
                }

        # --- Content type authority boost ---
        boost = TYPE_BOOST.get(item.content_type, 1.0)
        if boost != 1.0 and score > 0:
            score *= boost
            breakdown["type_boost"] = {"type": item.content_type, "multiplier": boost}

        return score, breakdown

    def _create_explanation(
        self,
        breakdown: Dict[str, Any],
        role: Optional[str],
    ) -> str:
        parts = []

        if "exact_title" in breakdown:
            parts.append(f"Exact title match (+{breakdown['exact_title']:.1f})")
        if "full_title_coverage" in breakdown:
            parts.append(f"Full title coverage (+{breakdown['full_title_coverage']:.1f})")
        if "title_keywords" in breakdown:
            kw = breakdown["title_keywords"]
            parts.append(f"Title words: {', '.join(kw['matches'][:3])} (+{kw['points']:.1f})")
        if "exact_body" in breakdown:
            parts.append(f"Exact body match (+{breakdown['exact_body']:.1f})")
        if "body_keywords" in breakdown:
            kw = breakdown["body_keywords"]
            parts.append(f"Body words: {kw['count']} (+{kw['points']:.1f})")
        if "type_boost" in breakdown:
            tb = breakdown["type_boost"]
            parts.append(f"Type boost {tb['type']} (×{tb['multiplier']})")
        if role:
            parts.append(f"Role: {role}")

        return " | ".join(parts) if parts else "Match"


# Global instance
keyword_search = KeywordSearch()
