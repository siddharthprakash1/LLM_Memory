"""
Multi-tier memory search.

Implements hierarchical search across memory tiers:
1. Short-term (immediate context)
2. Episodic (past experiences)
3. Semantic (general knowledge)
"""

from datetime import datetime
from typing import Any, Callable, Awaitable

from pydantic import BaseModel, Field

from llm_memory.models.base import BaseMemory, MemoryType
from llm_memory.models.short_term import ShortTermMemory
from llm_memory.models.episodic import EpisodicMemory
from llm_memory.models.semantic import SemanticMemory
from llm_memory.retrieval.intent import ClassifiedIntent, QueryIntent


def _utcnow() -> datetime:
    """Get current UTC time (naive, for compatibility with models)."""
    return datetime.utcnow()


class SearchResult(BaseModel):
    """A single search result."""

    memory_id: str
    memory_type: MemoryType
    content: str
    summary: str | None = None
    
    # Scores
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    final_score: float = Field(default=0.0, ge=0.0)
    
    # Metadata
    created_at: datetime | None = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = 0
    
    # Source tier info
    tier_rank: int = Field(default=0, description="Rank of tier in search order")


class SearchResults(BaseModel):
    """Collection of search results."""

    query: str
    intent: ClassifiedIntent
    results: list[SearchResult] = Field(default_factory=list)
    
    # Search metadata
    total_candidates: int = 0
    search_time_ms: float = 0.0
    tiers_searched: list[MemoryType] = Field(default_factory=list)
    
    searched_at: datetime = Field(default_factory=_utcnow)

    def top(self, n: int = 5) -> list[SearchResult]:
        """Get top N results."""
        return self.results[:n]

    def by_tier(self, tier: MemoryType) -> list[SearchResult]:
        """Get results from a specific tier."""
        return [r for r in self.results if r.memory_type == tier]


class SearchConfig(BaseModel):
    """Configuration for search behavior."""

    # Result limits
    max_results: int = Field(default=10, ge=1)
    max_per_tier: int = Field(default=5, ge=1)
    
    # Score thresholds
    min_similarity: float = Field(default=0.3, ge=0.0, le=1.0)
    min_relevance: float = Field(default=0.2, ge=0.0, le=1.0)
    
    # Tier behavior
    search_all_tiers: bool = Field(default=True)
    tier_fallback: bool = Field(
        default=True,
        description="Search lower priority tiers if high priority returns few results",
    )
    fallback_threshold: int = Field(
        default=3,
        description="Minimum results before triggering fallback",
    )
    
    # Intent-based adjustments
    use_intent_weights: bool = Field(default=True)
    boost_recent_stm: bool = Field(default=True)


class MemorySearcher:
    """
    Multi-tier memory searcher.
    
    Searches across STM, Episodic, and Semantic memories
    with intent-aware scoring and ranking.
    """

    def __init__(self, config: SearchConfig | None = None):
        self.config = config or SearchConfig()
        
        # Callbacks for memory access (set by memory system)
        self._get_stm: Callable[[], Awaitable[list[ShortTermMemory]]] | None = None
        self._get_episodic: Callable[[], Awaitable[list[EpisodicMemory]]] | None = None
        self._get_semantic: Callable[[], Awaitable[list[SemanticMemory]]] | None = None
        
        # Embedding function
        self._embed_fn: Callable[[str], Awaitable[list[float]]] | None = None

    def set_memory_callbacks(
        self,
        get_stm: Callable[[], Awaitable[list[ShortTermMemory]]],
        get_episodic: Callable[[], Awaitable[list[EpisodicMemory]]],
        get_semantic: Callable[[], Awaitable[list[SemanticMemory]]],
        embed_fn: Callable[[str], Awaitable[list[float]]] | None = None,
    ) -> None:
        """Set callbacks for memory access."""
        self._get_stm = get_stm
        self._get_episodic = get_episodic
        self._get_semantic = get_semantic
        self._embed_fn = embed_fn

    async def search(
        self,
        query: str,
        intent: ClassifiedIntent,
        embedding: list[float] | None = None,
    ) -> SearchResults:
        """
        Search memories based on query and intent.
        
        Args:
            query: Search query
            intent: Classified intent
            embedding: Pre-computed query embedding
            
        Returns:
            SearchResults with ranked results
        """
        start_time = _utcnow()
        all_results: list[SearchResult] = []
        total_candidates = 0
        tiers_searched = []

        # Get query embedding if needed
        if embedding is None and self._embed_fn:
            embedding = await self._embed_fn(query)

        # Search tiers in recommended order
        tiers_to_search = (
            intent.recommended_tiers
            if not self.config.search_all_tiers
            else [MemoryType.SHORT_TERM, MemoryType.EPISODIC, MemoryType.SEMANTIC]
        )

        for tier_rank, tier in enumerate(tiers_to_search):
            tier_results, candidates = await self._search_tier(
                tier, query, intent, embedding, tier_rank
            )
            all_results.extend(tier_results)
            total_candidates += candidates
            tiers_searched.append(tier)

            # Check if we have enough results
            if (
                not self.config.tier_fallback
                and len(all_results) >= self.config.fallback_threshold
            ):
                break

        # Sort by final score
        all_results.sort(key=lambda r: r.final_score, reverse=True)
        
        # Limit results
        all_results = all_results[:self.config.max_results]

        search_time = (_utcnow() - start_time).total_seconds() * 1000

        return SearchResults(
            query=query,
            intent=intent,
            results=all_results,
            total_candidates=total_candidates,
            search_time_ms=search_time,
            tiers_searched=tiers_searched,
        )

    async def _search_tier(
        self,
        tier: MemoryType,
        query: str,
        intent: ClassifiedIntent,
        embedding: list[float] | None,
        tier_rank: int,
    ) -> tuple[list[SearchResult], int]:
        """Search a single memory tier."""
        # Get memories from tier
        memories = await self._get_tier_memories(tier)
        if not memories:
            return [], 0

        results = []
        for memory in memories:
            # Calculate scores
            similarity = self._calculate_similarity(query, memory, embedding)
            
            if similarity < self.config.min_similarity:
                continue
            
            relevance = self._calculate_relevance(memory, intent)
            
            if relevance < self.config.min_relevance:
                continue
            
            final_score = self._calculate_final_score(
                similarity, relevance, memory, intent, tier_rank
            )
            
            result = SearchResult(
                memory_id=memory.id,
                memory_type=tier,
                content=memory.content,
                summary=memory.summary,
                similarity_score=similarity,
                relevance_score=relevance,
                final_score=final_score,
                created_at=memory.created_at,
                importance=memory.importance_score,
                access_count=memory.access_count,
                tier_rank=tier_rank,
            )
            results.append(result)

        # Sort and limit per tier
        results.sort(key=lambda r: r.final_score, reverse=True)
        results = results[:self.config.max_per_tier]

        return results, len(memories)

    async def _get_tier_memories(self, tier: MemoryType) -> list[BaseMemory]:
        """Get all memories from a tier."""
        if tier == MemoryType.SHORT_TERM and self._get_stm:
            return await self._get_stm()
        elif tier == MemoryType.EPISODIC and self._get_episodic:
            return await self._get_episodic()
        elif tier == MemoryType.SEMANTIC and self._get_semantic:
            return await self._get_semantic()
        return []

    def _calculate_similarity(
        self,
        query: str,
        memory: BaseMemory,
        embedding: list[float] | None,
    ) -> float:
        """Calculate similarity between query and memory."""
        # If we have embeddings, use cosine similarity
        if embedding and hasattr(memory, 'embedding') and memory.embedding:
            return self._cosine_similarity(embedding, memory.embedding)
        
        # Fallback to keyword similarity
        return self._keyword_similarity(query, memory.content)

    def _cosine_similarity(
        self,
        vec1: list[float],
        vec2: list[float],
    ) -> float:
        """Calculate cosine similarity between vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def _keyword_similarity(self, query: str, content: str) -> float:
        """Calculate keyword-based similarity."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words or not content_words:
            return 0.0
        
        intersection = len(query_words & content_words)
        union = len(query_words | content_words)
        
        return intersection / union if union > 0 else 0.0

    def _calculate_relevance(
        self,
        memory: BaseMemory,
        intent: ClassifiedIntent,
    ) -> float:
        """Calculate relevance score based on intent."""
        relevance = 0.5  # Base relevance
        
        # Check if memory type matches recommended tiers
        if memory.memory_type in intent.recommended_tiers:
            tier_index = intent.recommended_tiers.index(memory.memory_type)
            relevance += 0.3 * (1.0 - tier_index * 0.2)
        
        # Check key terms in content
        content_lower = memory.content.lower()
        term_matches = sum(
            1 for term in intent.key_terms
            if term in content_lower
        )
        if intent.key_terms:
            relevance += 0.2 * (term_matches / len(intent.key_terms))
        
        return min(1.0, relevance)

    def _calculate_final_score(
        self,
        similarity: float,
        relevance: float,
        memory: BaseMemory,
        intent: ClassifiedIntent,
        tier_rank: int,
    ) -> float:
        """Calculate final combined score."""
        if not self.config.use_intent_weights:
            # Simple average
            return (similarity + relevance) / 2
        
        # Weighted combination based on intent
        score = (
            similarity * intent.similarity_weight +
            relevance * intent.importance_weight
        )
        
        # Add recency bonus
        age_hours = ((_utcnow() - memory.created_at).total_seconds() / 3600
                     if memory.created_at else 0)
        recency_bonus = max(0, 1.0 - age_hours / 720)  # Decay over 30 days
        score += recency_bonus * intent.recency_weight * 0.3
        
        # Boost STM for context queries
        if (
            self.config.boost_recent_stm
            and memory.memory_type == MemoryType.SHORT_TERM
            and intent.primary_intent == QueryIntent.CONTEXT
        ):
            score *= 1.3
        
        # Tier rank penalty (prefer higher priority tiers)
        score *= (1.0 - tier_rank * 0.1)
        
        return score


class KeywordSearcher:
    """
    Simple keyword-based searcher for testing and fallback.
    
    Does not require embeddings.
    """

    def __init__(self, config: SearchConfig | None = None):
        self.config = config or SearchConfig()

    def search(
        self,
        query: str,
        memories: list[BaseMemory],
        intent: ClassifiedIntent | None = None,
    ) -> SearchResults:
        """
        Search memories by keyword matching.
        
        Args:
            query: Search query
            memories: Memories to search
            intent: Optional classified intent
            
        Returns:
            SearchResults
        """
        start_time = _utcnow()
        query_words = set(query.lower().split())
        
        results = []
        for memory in memories:
            content_words = set(memory.content.lower().split())
            
            # Calculate keyword overlap
            if not query_words:
                continue
            
            intersection = query_words & content_words
            if not intersection:
                continue
            
            similarity = len(intersection) / len(query_words)
            
            if similarity < self.config.min_similarity:
                continue
            
            relevance = memory.importance_score
            final_score = (similarity + relevance) / 2
            
            result = SearchResult(
                memory_id=memory.id,
                memory_type=memory.memory_type,
                content=memory.content,
                summary=memory.summary,
                similarity_score=similarity,
                relevance_score=relevance,
                final_score=final_score,
                created_at=memory.created_at,
                importance=memory.importance_score,
                access_count=memory.access_count,
            )
            results.append(result)

        # Sort by score
        results.sort(key=lambda r: r.final_score, reverse=True)
        results = results[:self.config.max_results]

        search_time = (_utcnow() - start_time).total_seconds() * 1000

        # Create default intent if not provided
        if intent is None:
            from llm_memory.retrieval.intent import IntentClassifier
            classifier = IntentClassifier()
            intent = classifier.classify(query)

        return SearchResults(
            query=query,
            intent=intent,
            results=results,
            total_candidates=len(memories),
            search_time_ms=search_time,
            tiers_searched=list(set(m.memory_type for m in memories)),
        )
