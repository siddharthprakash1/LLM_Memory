"""
Re-ranker for Better Retrieval Results.

Uses cross-attention scoring and query-type specific boosting
to improve retrieval quality.
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Tuple


@dataclass
class RerankResult:
    """Result after re-ranking."""
    item: Any
    original_score: float
    reranked_score: float
    boost_reason: str = ""


class ReRanker:
    """
    Re-rank retrieved results for better accuracy.
    
    Uses:
    - Semantic similarity (via embeddings)
    - Lexical overlap (keyword matching)
    - Query-type specific boosting
    - Temporal/entity awareness
    """
    
    def __init__(self, embed_func: Callable[[str], np.ndarray] = None):
        """
        Initialize the re-ranker.
        
        Args:
            embed_func: Function to get embeddings (text -> np.ndarray)
        """
        self.embed_func = embed_func
    
    def rerank(
        self,
        query: str,
        candidates: list[Tuple[Any, float]],
        top_k: int = 10,
        include_reasoning: bool = False,
    ) -> list[Tuple[Any, float]] | list[RerankResult]:
        """
        Re-rank candidates based on the query.
        
        Args:
            query: The search query
            candidates: List of (item, initial_score) tuples
            top_k: Number of results to return
            include_reasoning: If True, return RerankResult with reasons
            
        Returns:
            Re-ranked list of (item, score) or RerankResult tuples
        """
        if not candidates:
            return []
        
        # Analyze query
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
        
        # Detect query type
        is_temporal = any(w in query_lower for w in [
            'when', 'date', 'time', 'year', 'day', 'month',
            'before', 'after', 'during', 'how long', 'ago',
        ])
        is_person = any(w in query_lower for w in ['who', 'whom', 'whose'])
        is_location = any(w in query_lower for w in ['where', 'location', 'place'])
        is_preference = any(w in query_lower for w in ['like', 'prefer', 'favorite', 'enjoy'])
        
        # Get query embedding if available
        query_emb = None
        if self.embed_func:
            try:
                query_emb = self.embed_func(query)
            except Exception:
                pass
        
        results = []
        
        for item, initial_score in candidates:
            # Get content from item
            content = self._get_content(item)
            content_lower = content.lower()
            content_words = set(re.findall(r'\b\w{3,}\b', content_lower))
            
            # 1. Semantic similarity (if embeddings available)
            semantic_score = 0.0
            if query_emb is not None and self.embed_func:
                try:
                    item_emb = self.embed_func(content)
                    semantic_score = self._cosine_similarity(query_emb, item_emb)
                except Exception:
                    pass
            
            # 2. Lexical overlap (keyword matching)
            overlap_score = len(query_words & content_words) / (len(query_words) + 1)
            
            # 3. Query-type specific boosting
            boost = 0.0
            boost_reason = ""
            
            if is_temporal:
                # Boost items with dates
                has_date = bool(re.search(
                    r'\b(\d{1,2}\s+\w+\s+\d{4}|\d{4}|\w+\s+\d{4})\b',
                    content
                ))
                if has_date:
                    boost += 0.2
                    boost_reason = "has_date"
                
                # Check for temporal keywords in content
                if any(m in content_lower for m in [
                    'january', 'february', 'march', 'april', 'may', 'june',
                    'july', 'august', 'september', 'october', 'november', 'december',
                ]):
                    boost += 0.15
                    boost_reason = "has_month"
            
            if is_person:
                # Boost items that mention specific people
                # Check if item has speaker attribute
                speaker = self._get_speaker(item)
                if speaker and speaker.lower() in query_lower:
                    boost += 0.25
                    boost_reason = "speaker_match"
            
            if is_location:
                # Boost items with location keywords
                if any(w in content_lower for w in ['in', 'at', 'to', 'from', 'near']):
                    boost += 0.1
                    boost_reason = "location_indicator"
            
            if is_preference:
                # Boost items with preference language
                if any(w in content_lower for w in ['like', 'love', 'enjoy', 'prefer', 'favorite']):
                    boost += 0.15
                    boost_reason = "preference_language"
            
            # 4. Combine scores
            if query_emb is not None:
                final_score = (
                    0.35 * semantic_score +
                    0.25 * overlap_score +
                    0.20 * initial_score +
                    0.20 * boost
                )
            else:
                # Without embeddings, rely more on keyword overlap
                final_score = (
                    0.50 * overlap_score +
                    0.30 * initial_score +
                    0.20 * boost
                )
            
            if include_reasoning:
                results.append(RerankResult(
                    item=item,
                    original_score=initial_score,
                    reranked_score=final_score,
                    boost_reason=boost_reason,
                ))
            else:
                results.append((item, final_score))
        
        # Sort by score
        if include_reasoning:
            results.sort(key=lambda x: x.reranked_score, reverse=True)
        else:
            results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def _get_content(self, item: Any) -> str:
        """Extract content string from an item."""
        if hasattr(item, 'content'):
            return str(item.content)
        if isinstance(item, dict):
            return str(item.get('content', item.get('text', str(item))))
        return str(item)
    
    def _get_speaker(self, item: Any) -> str | None:
        """Extract speaker from an item if available."""
        if hasattr(item, 'speaker'):
            return item.speaker
        if isinstance(item, dict):
            return item.get('speaker')
        return None
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def boost_by_recency(
        self,
        candidates: list[Tuple[Any, float]],
        time_field: str = 'timestamp',
        decay_hours: float = 168,  # 1 week
    ) -> list[Tuple[Any, float]]:
        """
        Apply recency boost to candidates.
        
        Args:
            candidates: List of (item, score) tuples
            time_field: Field name for timestamp
            decay_hours: Hours until full decay
            
        Returns:
            Boosted list of (item, score) tuples
        """
        from datetime import datetime, timedelta
        
        now = datetime.now()
        results = []
        
        for item, score in candidates:
            # Get timestamp
            ts = None
            if hasattr(item, time_field):
                ts = getattr(item, time_field)
            elif isinstance(item, dict):
                ts = item.get(time_field)
            
            # Calculate recency boost
            boost = 0.0
            if ts:
                try:
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    age_hours = (now - ts.replace(tzinfo=None)).total_seconds() / 3600
                    if age_hours < decay_hours:
                        boost = 0.2 * (1 - age_hours / decay_hours)
                except Exception:
                    pass
            
            results.append((item, score + boost))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
