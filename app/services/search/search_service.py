from typing import List, Optional, Dict, Any
from pathlib import Path

import numpy as np

from app.dto.response.search import SearchResult
from app.entities.content import ContentItem
from app.services.data.content_service import content_service
from app.services.data.database_service import database_service
from app.config import settings
import re


class SearchService:
    """Service for searching content with keyword and semantic search."""

    def __init__(self):
        self.content_service = content_service
        self.embedding_model = None
        self.content_embeddings = None  # Cache: {content_id: embedding}
        self._embeddings_loaded = False

    def search(self, query: str, role: Optional[str] = None, k: int = 10) -> List[SearchResult]:
        """
        Perform baseline keyword search.

        This is a simple implementation that scores based on:
        - Exact matches in title (high weight)
        - Keyword matches in title (medium weight)
        - Keyword matches in body (low weight)
        - Role filtering if specified
        """
        query_lower = query.lower()
        query_keywords = set(re.findall(r'\w+', query_lower))

        results = []
        for item in self.content_service.get_all_content():
            # Filter by role if specified
            if role and role not in item.target_groups:
                continue

            score, breakdown = self._calculate_score_with_breakdown(item, query_lower, query_keywords)

            if score > 0:
                snippet = self._create_snippet(item.body, query_keywords)
                explanation = self._create_explanation_with_breakdown(breakdown, role)

                results.append(
                    SearchResult(
                        id=item.id,
                        title=item.title,
                        url=item.url,
                        snippet=snippet,
                        score=score,
                        explanation=explanation,
                    )
                )

        # Sort by score (descending) and ensure consistent ordering
        results.sort(key=lambda x: (-x.score, x.id))

        return results[:k]

    def _calculate_score(self, item: ContentItem, query_lower: str, query_keywords: set) -> float:
        """Calculate relevance score for a content item using configurable weights."""
        score, _ = self._calculate_score_with_breakdown(item, query_lower, query_keywords)
        return score

    def _calculate_score_with_breakdown(self, item: ContentItem, query_lower: str, query_keywords: set) -> tuple[float, Dict[str, Any]]:
        """Calculate relevance score with detailed breakdown."""
        score = 0.0
        breakdown = {}

        title_lower = item.title.lower()
        body_lower = item.body.lower()

        # Exact phrase match in title (highest weight)
        if query_lower in title_lower:
            points = settings.search_exact_phrase_title_weight
            score += points
            breakdown['exact_title'] = points

        # Keyword matches in title
        title_keywords = set(re.findall(r'\w+', title_lower))
        title_matches = query_keywords & title_keywords
        if title_matches:
            points = len(title_matches) * settings.search_keyword_title_weight
            score += points
            breakdown['title_keywords'] = {'count': len(title_matches), 'matches': list(title_matches), 'points': points}

        # Full title coverage: all title words present in query
        if title_keywords and title_keywords.issubset(query_keywords):
            points = settings.search_full_title_coverage_weight
            score += points
            breakdown['full_title_coverage'] = points

        # Keyword matches in body
        body_keywords = set(re.findall(r'\w+', body_lower))
        body_matches = query_keywords & body_keywords
        if body_matches:
            points = len(body_matches) * settings.search_keyword_body_weight
            score += points
            breakdown['body_keywords'] = {'count': len(body_matches), 'points': points}

        # Exact phrase match in body
        if query_lower in body_lower:
            points = settings.search_exact_phrase_body_weight
            score += points
            breakdown['exact_body'] = points

        # Tag matches
        if item.tags:
            tag_text = " ".join(item.tags).lower()
            tag_keywords = set(re.findall(r'\w+', tag_text))
            tag_matches = query_keywords & tag_keywords
            if tag_matches:
                points = len(tag_matches) * settings.search_tag_match_weight
                score += points
                breakdown['tag_matches'] = {'count': len(tag_matches), 'matches': list(tag_matches), 'points': points}

        return score, breakdown

    def _create_snippet(self, body: str, query_keywords: set, max_length: int = 200) -> str:
        """Create a snippet highlighting relevant parts of the body."""
        # Find the first occurrence of any query keyword
        body_lower = body.lower()
        best_pos = len(body)

        for keyword in query_keywords:
            pos = body_lower.find(keyword)
            if pos != -1 and pos < best_pos:
                best_pos = pos

        # Extract snippet around the keyword
        if best_pos < len(body):
            start = max(0, best_pos - 50)
            end = min(len(body), best_pos + max_length - 50)

            snippet = body[start:end].strip()

            if start > 0:
                snippet = "..." + snippet
            if end < len(body):
                snippet = snippet + "..."

            return snippet
        else:
            # No keyword found, return beginning
            return body[:max_length].strip() + ("..." if len(body) > max_length else "")

    def _create_explanation(self, item: ContentItem, query_keywords: set, role: Optional[str]) -> str:
        """Create a short explanation of why this result matches."""
        score, breakdown = self._calculate_score_with_breakdown(item, item.title.lower() + " " + item.body.lower(), query_keywords)
        return self._create_explanation_with_breakdown(breakdown, role)

    def _create_explanation_with_breakdown(self, breakdown: Dict[str, Any], role: Optional[str]) -> str:
        """Create detailed explanation from score breakdown."""
        parts = []

        if 'exact_title' in breakdown:
            parts.append(f"Exact title match (+{breakdown['exact_title']:.1f})")
        
        if 'full_title_coverage' in breakdown:
            parts.append(f"Full title coverage (+{breakdown['full_title_coverage']:.1f})")
        
        if 'title_keywords' in breakdown:
            kw = breakdown['title_keywords']
            parts.append(f"Title words: {', '.join(kw['matches'][:3])} (+{kw['points']:.1f})")
        
        if 'body_keywords' in breakdown:
            bk = breakdown['body_keywords']
            parts.append(f"Body matches ({bk['count']}) (+{bk['points']:.1f})")
        
        if 'exact_body' in breakdown:
            parts.append(f"Exact in body (+{breakdown['exact_body']:.1f})")
        
        if 'tag_matches' in breakdown:
            tm = breakdown['tag_matches']
            parts.append(f"Tags: {', '.join(tm['matches'][:3])} (+{tm['points']:.1f})")
        
        if role:
            parts.append(f"Role: {role}")

        return " | ".join(parts) if parts else "Match"


    def _load_embedding_model(self) -> bool:
        """Load the E5 embedding model if available."""
        if self.embedding_model is not None:
            return True

        try:
            from app.ml.embedding_model import HealthContentEmbedding
            
            # E5 model doesn't need to be loaded from disk - it's pre-trained
            self.embedding_model = HealthContentEmbedding()
            # Model will be lazy-loaded from HuggingFace on first encode
            return True
        except Exception as e:
            print(f"Error initializing embedding model: {e}")
            return False

    def _load_content_embeddings(self) -> bool:
        """Load embeddings from database into memory cache."""
        if self._embeddings_loaded:
            return True

        from app.services.data.database_service import database_service

        conn = database_service._get_connection()
        if not conn:
            return False

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, embedding FROM content WHERE embedding IS NOT NULL")
            rows = cursor.fetchall()

            self.content_embeddings = {}
            for content_id, embedding_bytes in rows:
                if embedding_bytes:
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    self.content_embeddings[content_id] = embedding

            self._embeddings_loaded = True
            print(f"Loaded {len(self.content_embeddings)} embeddings into cache")
            return len(self.content_embeddings) > 0

        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False
        finally:
            cursor.close()
            conn.close()

    def search_semantic(
        self, query: str, role: Optional[str] = None, k: int = 10
    ) -> List[SearchResult]:
        """
        Perform semantic search using E5 embeddings.

        Returns empty list if embeddings not available.
        """
        # Load model and embeddings
        if not self._load_embedding_model():
            return []
        if not self._load_content_embeddings():
            return []

        # Encode query with "query:" prefix (E5 optimization)
        query_embedding = self.embedding_model.encode_query(query)

        # Calculate similarities
        similarities = []
        for item in self.content_service.get_all_content():
            # Filter by role if specified
            if role and role not in item.target_groups:
                continue

            if item.id in self.content_embeddings:
                doc_embedding = self.content_embeddings[item.id]
                # Cosine similarity (embeddings are L2-normalized)
                similarity = float(np.dot(query_embedding, doc_embedding))
                similarities.append((item, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: -x[1])

        # Create results
        results = []
        for item, score in similarities[:k]:
            snippet = self._create_snippet(item.body, set())
            results.append(
                SearchResult(
                    id=item.id,
                    title=item.title,
                    url=item.url,
                    snippet=snippet,
                    score=score,
                    explanation=f"Semantic similarity: {score:.3f}",
                )
            )

        return results

    def search_hybrid(
        self,
        query: str,
        role: Optional[str] = None,
        k: int = 10,
        keyword_weight: float = 0.2,
        semantic_weight: float = 0.8,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining keyword and semantic scores.

        Uses min-max normalization to scale both keyword and semantic scores
        to [0, 1] based on the actual min/max in the result set.

        Args:
            query: Search query
            role: Optional role filter
            k: Number of results
            keyword_weight: Weight for keyword score (0-1)
            semantic_weight: Weight for semantic score (0-1)

        Returns:
            Combined and re-ranked results
        """
        query_lower = query.lower()
        query_keywords = set(re.findall(r'\w+', query_lower))

        # Check if semantic search is available
        semantic_available = (
            self._load_embedding_model() and self._load_content_embeddings()
        )

        if semantic_available:
            query_embedding = self.embedding_model.encode([query])[0]
        else:
            # Fall back to keyword-only search
            return self.search(query, role, k)

        # Score all content (raw scores)
        scored_items = []

        for item in self.content_service.get_all_content():
            # Filter by role if specified
            if role and role not in item.target_groups:
                continue

            # Keyword score (raw)
            keyword_score = self._calculate_score(item, query_lower, query_keywords)

            # Semantic score (raw, -1 to 1)
            if item.id in self.content_embeddings:
                doc_embedding = self.content_embeddings[item.id]
                semantic_score = float(np.dot(query_embedding, doc_embedding))
            else:
                semantic_score = -1.0  # Lowest possible

            if keyword_score > 0 or semantic_score > -1.0:
                scored_items.append((item, keyword_score, semantic_score))

        if not scored_items:
            return []

        # Extract scores for normalization
        keyword_scores = [kw for _, kw, _ in scored_items]

        # Min-max normalization for keyword scores only
        kw_min, kw_max = min(keyword_scores), max(keyword_scores)

        # Normalize and combine
        normalized_items = []
        for item, kw_raw, sem_raw in scored_items:
            # Normalize keyword score (relative to result set)
            if kw_max > kw_min:
                kw_norm = (kw_raw - kw_min) / (kw_max - kw_min)
            else:
                kw_norm = 1.0 if kw_raw > 0 else 0.0

            # Semantic score: already in [0, 1], use as-is (absolute quality)
            sem_norm = sem_raw

            # Combined score
            combined_score = (
                keyword_weight * kw_norm +
                semantic_weight * sem_norm
            )

            normalized_items.append((item, combined_score, kw_raw, sem_raw, kw_norm, sem_norm))

        # Sort by combined score
        normalized_items.sort(key=lambda x: -x[1])

        # Apply ranking model if enabled
        if settings.ml_ranking_enabled:
            normalized_items = self._apply_ranking_model(
                normalized_items, query, query_keywords, role
            )

        # Create results
        results = []
        for item, combined, kw_raw, sem_raw, kw_norm, sem_norm in normalized_items[:k]:
            snippet = self._create_snippet(item.body, query_keywords)
            explanation = self._create_explanation(item, query_keywords, role)
            explanation += f" | Scores: kw={kw_raw:.1f}→{kw_norm:.2f}, sem={sem_raw:.2f}→{sem_norm:.2f}, final={combined:.2f}"

            results.append(
                SearchResult(
                    id=item.id,
                    title=item.title,
                    url=item.url,
                    snippet=snippet,
                    score=combined,
                    explanation=explanation,
                )
            )

        return results

    def _apply_ranking_model(
        self,
        items: List[tuple],
        query: str,
        query_keywords: set,
        role: Optional[str]
    ) -> List[tuple]:
        """
        Apply ranking model to re-rank results.

        Args:
            items: List of (item, combined, kw_raw, sem_raw, kw_norm, sem_norm)
            query: Search query
            query_keywords: Query keywords set
            role: User role

        Returns:
            Re-ranked list with new scores
        """
        try:
            from app.services.search.ml_service import ml_service

            if not ml_service.is_ranking_available():
                ml_service.load_ranking_model()
                if not ml_service.is_ranking_available():
                    return items  # Model not available, return as-is

            # Get CTR data
            ctr_data = database_service.get_content_ctr()

            # Extract features for each item
            features_list = []
            for item, combined, kw_raw, sem_raw, kw_norm, sem_norm in items:
                features = self._extract_ranking_features(
                    item, query, query_keywords, role,
                    kw_raw, sem_raw, ctr_data.get(item.id, 0.0)
                )
                features_list.append(features)

            # Get ranking scores from model
            ranking_scores = ml_service.get_ranking_scores(features_list)

            # Replace combined score with ranking score and re-sort
            re_ranked = []
            for i, (item, _, kw_raw, sem_raw, kw_norm, sem_norm) in enumerate(items):
                re_ranked.append((item, ranking_scores[i], kw_raw, sem_raw, kw_norm, sem_norm))

            re_ranked.sort(key=lambda x: -x[1])
            return re_ranked

        except Exception as e:
            print(f"Error applying ranking model: {e}")
            return items  # Return original if ranking fails

    def _extract_ranking_features(
        self,
        item: ContentItem,
        query: str,
        query_keywords: set,
        role: Optional[str],
        keyword_score: float,
        semantic_score: float,
        ctr: float
    ) -> Dict[str, float]:
        """
        Extract features for ranking model.

        Features match RANKING_FEATURES in ranking_model.py:
        1. title_keyword_score (normalized 0-1) - Ratio of query keywords in title
        2. body_keyword_score (normalized 0-1) - Ratio of query keywords in body
        3. tag_match_score (normalized 0-1) - Ratio of query keywords in tags
        4. exact_phrase_title (binary 0/1) - Full query phrase in title
        5. exact_phrase_body (binary 0/1) - Full query phrase in body
        6. semantic_similarity (0-1) - Cosine similarity from embedding
        7. content_type_encoded (0-1) - Authority level of content type
        8. historical_ctr (0-1) - Click-through rate
        9. role_match (0-1) - User role match with target groups
        
        All features are normalized to [0, 1] range for consistent learning.
        """
        query_lower = query.lower()
        title_lower = item.title.lower()
        body_lower = item.body.lower()

        # Title keyword score (normalized)
        title_keywords = set(re.findall(r'\w+', title_lower))
        title_matches = len(query_keywords & title_keywords)
        # Normalize: assume max 5 keyword matches in title
        title_kw_score = min(title_matches / 5.0, 1.0)

        # Body keyword score (normalized)
        body_keywords = set(re.findall(r'\w+', body_lower))
        body_matches = len(query_keywords & body_keywords)
        # Normalize: assume max 10 keyword matches in body
        body_kw_score = min(body_matches / 10.0, 1.0)

        # Tag match score (normalized)
        tag_score = 0.0
        if item.tags:
            tag_text = " ".join(item.tags).lower()
            tag_keywords = set(re.findall(r'\w+', tag_text))
            tag_matches = len(query_keywords & tag_keywords)
            # Normalize: assume max 3 tag matches
            tag_score = min(tag_matches / 3.0, 1.0)

        # Exact phrase matches (binary: 1.0 or 0.0)
        exact_title = 1.0 if query_lower in title_lower else 0.0
        exact_body = 1.0 if query_lower in body_lower else 0.0

        # Semantic similarity (already normalized 0-1)
        semantic = semantic_score

        # Content type encoding based on authority level
        content_type_map = {
            "retningslinje": 0.9,
            "veileder": 0.8,
            "fagprosedyre": 0.75,
            "faktaark": 0.6,
            "artikkel": 0.5,
        }
        content_type = content_type_map.get(
            item.info_type.lower() if item.info_type else None,
            0.5  # Default for unknown types
        )

        # Historical CTR (already normalized 0-1)
        historical_ctr = ctr

        # Role match - gradient based on specificity
        role_match = 0.0
        if role and item.target_groups:
            if role in item.target_groups:
                # Perfect match, but penalize if shared with many groups
                role_match = 1.0 / len(item.target_groups)
        elif not role and not item.target_groups:
            # General content for general search
            role_match = 0.5
        elif not item.target_groups:
            # Content for everyone
            role_match = 0.3

        return {
            "title_keyword_score": title_kw_score,
            "body_keyword_score": body_kw_score,
            "tag_match_score": tag_score,
            "exact_phrase_title": exact_title,
            "exact_phrase_body": exact_body,
            "semantic_similarity": semantic,
            "content_type_encoded": content_type,
            "historical_ctr": historical_ctr,
            "role_match": role_match,
        }


# Global instance
search_service = SearchService()
