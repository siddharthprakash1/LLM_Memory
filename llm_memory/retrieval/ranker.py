"""
Result re-ranking with intent-aware weighting.

Provides post-retrieval ranking to optimize result quality:
- Intent-based boosting
- Diversity promotion
- Context relevance
- Temporal weighting
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from llm_memory.models.base import MemoryType
from llm_memory.retrieval.intent import ClassifiedIntent, QueryIntent
from llm_memory.retrieval.searcher import SearchResult, SearchResults


def _utcnow() -> datetime:
    """Get current UTC time (naive, for compatibility with models)."""
    return datetime.utcnow()


class RankingConfig(BaseModel):
    """Configuration for result ranking."""

    # Score weights
    similarity_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    relevance_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    recency_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    importance_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Diversity settings
    promote_diversity: bool = Field(default=True)
    diversity_penalty: float = Field(
        default=0.1,
        description="Penalty for results similar to already-selected results",
    )
    
    # Tier boosts
    stm_boost: float = Field(default=1.2, description="Boost for STM results")
    episodic_boost: float = Field(default=1.0, description="Boost for episodic results")
    semantic_boost: float = Field(default=0.9, description="Boost for semantic results")
    
    # Re-ranking options
    use_mmr: bool = Field(
        default=True,
        description="Use Maximal Marginal Relevance for diversity",
    )
    mmr_lambda: float = Field(
        default=0.7,
        description="MMR trade-off between relevance and diversity",
        ge=0.0,
        le=1.0,
    )


class RankedResult(BaseModel):
    """A re-ranked search result."""

    result: SearchResult
    original_rank: int
    new_rank: int
    ranking_score: float
    
    # Score breakdown
    similarity_contribution: float = 0.0
    relevance_contribution: float = 0.0
    recency_contribution: float = 0.0
    importance_contribution: float = 0.0
    diversity_penalty: float = 0.0
    tier_boost: float = 1.0


class RankedResults(BaseModel):
    """Collection of re-ranked results."""

    query: str
    intent: ClassifiedIntent
    ranked_results: list[RankedResult] = Field(default_factory=list)
    
    # Ranking metadata
    original_count: int = 0
    reranked_count: int = 0
    ranking_time_ms: float = 0.0

    def top(self, n: int = 5) -> list[SearchResult]:
        """Get top N results."""
        return [r.result for r in self.ranked_results[:n]]

    def get_reranked_positions(self) -> dict[str, tuple[int, int]]:
        """Get mapping of memory_id to (original_rank, new_rank)."""
        return {
            r.result.memory_id: (r.original_rank, r.new_rank)
            for r in self.ranked_results
        }


class ResultRanker:
    """
    Re-ranks search results with intent-aware weighting.
    
    Applies:
    - Intent-specific weight profiles
    - Diversity promotion (MMR)
    - Temporal and importance factors
    - Tier-based boosting
    """

    def __init__(self, config: RankingConfig | None = None):
        self.config = config or RankingConfig()

    def rank(
        self,
        results: SearchResults,
        context_embedding: list[float] | None = None,
    ) -> RankedResults:
        """
        Re-rank search results.
        
        Args:
            results: Original search results
            context_embedding: Optional context for diversity calculation
            
        Returns:
            RankedResults with optimized ordering
        """
        start_time = _utcnow()
        
        if not results.results:
            return RankedResults(
                query=results.query,
                intent=results.intent,
                original_count=0,
                reranked_count=0,
            )

        # Get intent-specific weights
        weights = self._get_intent_weights(results.intent)
        
        # Calculate ranking scores
        scored_results = []
        for idx, result in enumerate(results.results):
            ranking_score, breakdown = self._calculate_ranking_score(
                result, weights, results.intent
            )
            
            scored_results.append((idx, result, ranking_score, breakdown))

        # Apply diversity if enabled
        if self.config.use_mmr and len(scored_results) > 1:
            scored_results = self._apply_mmr(scored_results, results.intent)
        else:
            # Sort by score
            scored_results.sort(key=lambda x: x[2], reverse=True)

        # Create ranked results
        ranked = []
        for new_rank, (orig_rank, result, score, breakdown) in enumerate(scored_results):
            ranked_result = RankedResult(
                result=result,
                original_rank=orig_rank,
                new_rank=new_rank,
                ranking_score=score,
                similarity_contribution=breakdown.get("similarity", 0),
                relevance_contribution=breakdown.get("relevance", 0),
                recency_contribution=breakdown.get("recency", 0),
                importance_contribution=breakdown.get("importance", 0),
                diversity_penalty=breakdown.get("diversity_penalty", 0),
                tier_boost=breakdown.get("tier_boost", 1.0),
            )
            ranked.append(ranked_result)

        ranking_time = (_utcnow() - start_time).total_seconds() * 1000

        return RankedResults(
            query=results.query,
            intent=results.intent,
            ranked_results=ranked,
            original_count=len(results.results),
            reranked_count=len(ranked),
            ranking_time_ms=ranking_time,
        )

    def _get_intent_weights(self, intent: ClassifiedIntent) -> dict[str, float]:
        """Get weight profile based on intent."""
        # Use intent's own weights as base
        base_weights = {
            "similarity": intent.similarity_weight,
            "relevance": intent.importance_weight,
            "recency": intent.recency_weight,
            "importance": 0.3,  # Base importance weight
        }
        
        # Adjust based on specific intents
        if intent.primary_intent == QueryIntent.CONTEXT:
            base_weights["recency"] = 0.8
            base_weights["similarity"] = 0.5
        elif intent.primary_intent == QueryIntent.FACTUAL:
            base_weights["similarity"] = 0.7
            base_weights["recency"] = 0.2
        elif intent.primary_intent == QueryIntent.PREFERENCE:
            base_weights["recency"] = 0.7
            base_weights["importance"] = 0.5
        
        return base_weights

    def _calculate_ranking_score(
        self,
        result: SearchResult,
        weights: dict[str, float],
        intent: ClassifiedIntent,
    ) -> tuple[float, dict[str, float]]:
        """Calculate ranking score with breakdown."""
        breakdown = {}
        
        # Similarity contribution
        breakdown["similarity"] = result.similarity_score * weights["similarity"]
        
        # Relevance contribution
        breakdown["relevance"] = result.relevance_score * weights["relevance"]
        
        # Recency contribution
        recency_score = self._calculate_recency_score(result)
        breakdown["recency"] = recency_score * weights["recency"]
        
        # Importance contribution
        breakdown["importance"] = result.importance * weights["importance"]
        
        # Tier boost
        tier_boost = self._get_tier_boost(result.memory_type, intent)
        breakdown["tier_boost"] = tier_boost
        
        # Calculate total
        total = sum([
            breakdown["similarity"],
            breakdown["relevance"],
            breakdown["recency"],
            breakdown["importance"],
        ]) * tier_boost
        
        return total, breakdown

    def _calculate_recency_score(self, result: SearchResult) -> float:
        """Calculate recency score (0-1)."""
        if not result.created_at:
            return 0.5  # Default for unknown age
        
        age_hours = (_utcnow() - result.created_at).total_seconds() / 3600
        
        # Exponential decay over 30 days (720 hours)
        return max(0, 1.0 - (age_hours / 720))

    def _get_tier_boost(
        self,
        tier: MemoryType,
        intent: ClassifiedIntent,
    ) -> float:
        """Get boost factor for memory tier."""
        # Base boosts from config
        boosts = {
            MemoryType.SHORT_TERM: self.config.stm_boost,
            MemoryType.EPISODIC: self.config.episodic_boost,
            MemoryType.SEMANTIC: self.config.semantic_boost,
        }
        
        base_boost = boosts.get(tier, 1.0)
        
        # Additional boost if tier is in recommended tiers
        if tier in intent.recommended_tiers:
            tier_idx = intent.recommended_tiers.index(tier)
            base_boost *= (1.0 + 0.1 * (len(intent.recommended_tiers) - tier_idx))
        
        return base_boost

    def _apply_mmr(
        self,
        scored_results: list[tuple[int, SearchResult, float, dict]],
        intent: ClassifiedIntent,
    ) -> list[tuple[int, SearchResult, float, dict]]:
        """
        Apply Maximal Marginal Relevance for diversity.
        
        MMR = λ * Sim(d, q) - (1-λ) * max(Sim(d, d_j))
        where d_j are already selected documents
        """
        if not scored_results:
            return []

        lambda_param = self.config.mmr_lambda
        selected = []
        remaining = list(scored_results)
        
        # Select first result (highest score)
        remaining.sort(key=lambda x: x[2], reverse=True)
        selected.append(remaining.pop(0))
        
        # Select remaining using MMR
        while remaining:
            best_mmr = float('-inf')
            best_idx = 0
            
            for idx, (orig_rank, result, score, breakdown) in enumerate(remaining):
                # Calculate similarity to already selected
                max_sim_to_selected = 0.0
                for _, sel_result, _, _ in selected:
                    sim = self._content_similarity(result.content, sel_result.content)
                    max_sim_to_selected = max(max_sim_to_selected, sim)
                
                # MMR score
                mmr_score = lambda_param * score - (1 - lambda_param) * max_sim_to_selected
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx
            
            # Add diversity penalty to breakdown
            selected_item = remaining.pop(best_idx)
            orig_rank, result, score, breakdown = selected_item
            
            # Calculate actual penalty applied
            max_sim = 0.0
            for _, sel_result, _, _ in selected:
                sim = self._content_similarity(result.content, sel_result.content)
                max_sim = max(max_sim, sim)
            
            breakdown["diversity_penalty"] = max_sim * (1 - lambda_param)
            
            # Update score with MMR
            new_score = score - breakdown["diversity_penalty"]
            
            selected.append((orig_rank, result, new_score, breakdown))
        
        return selected

    def _content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


class ContextualRanker(ResultRanker):
    """
    Extended ranker with conversation context awareness.
    
    Boosts results that relate to recent conversation topics.
    """

    def __init__(
        self,
        config: RankingConfig | None = None,
        context_window: int = 5,
    ):
        super().__init__(config)
        self.context_window = context_window
        self._recent_topics: list[str] = []

    def add_context(self, topic: str) -> None:
        """Add a topic to recent context."""
        self._recent_topics.append(topic)
        if len(self._recent_topics) > self.context_window:
            self._recent_topics.pop(0)

    def clear_context(self) -> None:
        """Clear context history."""
        self._recent_topics.clear()

    def rank_with_context(
        self,
        results: SearchResults,
    ) -> RankedResults:
        """
        Rank results considering conversation context.
        
        Boosts results that relate to recent topics.
        """
        # First do normal ranking
        ranked = self.rank(results)
        
        if not self._recent_topics or not ranked.ranked_results:
            return ranked
        
        # Apply context boosting
        context_words = set()
        for topic in self._recent_topics:
            context_words.update(topic.lower().split())
        
        for ranked_result in ranked.ranked_results:
            content_words = set(ranked_result.result.content.lower().split())
            overlap = len(context_words & content_words)
            
            if overlap > 0:
                context_boost = min(0.2, overlap * 0.05)
                ranked_result.ranking_score *= (1 + context_boost)
        
        # Re-sort by new scores
        ranked.ranked_results.sort(key=lambda r: r.ranking_score, reverse=True)
        
        # Update ranks
        for new_rank, result in enumerate(ranked.ranked_results):
            result.new_rank = new_rank
        
        return ranked
