"""
Tests for intent-aware retrieval.
"""

import pytest
from datetime import datetime, timedelta

from llm_memory.models.base import MemoryType, ImportanceFactors
from llm_memory.models.semantic import SemanticMemory
from llm_memory.models.episodic import EpisodicMemory
from llm_memory.models.short_term import ShortTermMemory
from llm_memory.retrieval.intent import (
    QueryIntent,
    IntentSignal,
    ClassifiedIntent,
    IntentClassifier,
    LLMIntentClassifier,
)
from llm_memory.retrieval.searcher import (
    SearchResult,
    SearchResults,
    SearchConfig,
    MemorySearcher,
    KeywordSearcher,
)
from llm_memory.retrieval.ranker import (
    RankingConfig,
    RankedResult,
    RankedResults,
    ResultRanker,
    ContextualRanker,
)


def _utcnow() -> datetime:
    """Get current UTC time (naive)."""
    return datetime.utcnow()


def create_semantic(content: str, age_days: float = 0) -> SemanticMemory:
    """Create test semantic memory."""
    mem = SemanticMemory(content=content)
    mem.created_at = _utcnow() - timedelta(days=age_days)
    return mem


def create_episodic(content: str, age_days: float = 0) -> EpisodicMemory:
    """Create test episodic memory."""
    mem = EpisodicMemory(content=content)
    mem.created_at = _utcnow() - timedelta(days=age_days)
    return mem


def create_stm(content: str) -> ShortTermMemory:
    """Create test STM."""
    return ShortTermMemory(content=content)


def create_search_result(
    content: str,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    similarity: float = 0.8,
    relevance: float = 0.7,
    age_days: float = 0,
) -> SearchResult:
    """Create test search result."""
    return SearchResult(
        memory_id=f"mem_{_utcnow().timestamp()}",
        memory_type=memory_type,
        content=content,
        similarity_score=similarity,
        relevance_score=relevance,
        final_score=(similarity + relevance) / 2,
        created_at=_utcnow() - timedelta(days=age_days),
        importance=0.5,
    )


class TestQueryIntent:
    """Tests for QueryIntent enum."""

    def test_all_intents_defined(self):
        """All expected intents should be defined."""
        expected = [
            "factual", "episodic_recall", "preference", "procedural",
            "context", "problem_solving", "creative", "comparison", "general"
        ]
        actual = [i.value for i in QueryIntent]
        
        for exp in expected:
            assert exp in actual


class TestIntentSignal:
    """Tests for IntentSignal."""

    def test_default_values(self):
        """Should have sensible defaults."""
        signal = IntentSignal()
        
        assert signal.keywords == []
        assert signal.patterns == []
        assert signal.weight == 1.0

    def test_custom_values(self):
        """Should accept custom values."""
        signal = IntentSignal(
            keywords=["how", "what"],
            patterns=["how .* work"],
            weight=1.5,
        )
        
        assert len(signal.keywords) == 2
        assert signal.weight == 1.5


class TestClassifiedIntent:
    """Tests for ClassifiedIntent."""

    def test_default_values(self):
        """Should have sensible defaults."""
        intent = ClassifiedIntent(primary_intent=QueryIntent.GENERAL)
        
        assert intent.confidence == 0.5
        assert intent.secondary_intents == []
        assert intent.key_terms == []

    def test_classified_at_auto_set(self):
        """Should auto-set classification time."""
        intent = ClassifiedIntent(primary_intent=QueryIntent.FACTUAL)
        
        assert intent.classified_at is not None


class TestIntentClassifier:
    """Tests for IntentClassifier."""

    def test_classify_factual(self):
        """Should classify factual queries."""
        classifier = IntentClassifier()
        
        result = classifier.classify("What is Python?")
        
        assert result.primary_intent == QueryIntent.FACTUAL

    def test_classify_episodic(self):
        """Should classify episodic recall queries."""
        classifier = IntentClassifier()
        
        result = classifier.classify("Do you remember what we discussed last time?")
        
        assert result.primary_intent == QueryIntent.EPISODIC_RECALL

    def test_classify_preference(self):
        """Should classify preference queries."""
        classifier = IntentClassifier()
        
        result = classifier.classify("What do I prefer for code formatting?")
        
        assert result.primary_intent == QueryIntent.PREFERENCE

    def test_classify_procedural(self):
        """Should classify procedural queries."""
        classifier = IntentClassifier()
        
        result = classifier.classify("How to install numpy? Give me the steps and instructions")
        
        assert result.primary_intent == QueryIntent.PROCEDURAL

    def test_classify_problem_solving(self):
        """Should classify problem solving queries."""
        classifier = IntentClassifier()
        
        result = classifier.classify("I'm getting an error, can you fix this bug?")
        
        assert result.primary_intent == QueryIntent.PROBLEM_SOLVING

    def test_classify_creative(self):
        """Should classify creative queries."""
        classifier = IntentClassifier()
        
        result = classifier.classify("Can you suggest some ideas for my project?")
        
        assert result.primary_intent == QueryIntent.CREATIVE

    def test_classify_comparison(self):
        """Should classify comparison queries."""
        classifier = IntentClassifier()
        
        result = classifier.classify("Compare Python vs JavaScript")
        
        assert result.primary_intent == QueryIntent.COMPARISON

    def test_classify_with_context(self):
        """Should boost context intent when context provided."""
        classifier = IntentClassifier()
        
        result = classifier.classify("Continue with that", context="Previous discussion")
        
        # Context should be considered
        assert QueryIntent.CONTEXT in [result.primary_intent] + result.secondary_intents

    def test_classify_general_fallback(self):
        """Should fallback to general for unclear queries."""
        classifier = IntentClassifier()
        
        result = classifier.classify("xyz123")
        
        assert result.primary_intent == QueryIntent.GENERAL

    def test_recommended_tiers_set(self):
        """Should set recommended tiers based on intent."""
        classifier = IntentClassifier()
        
        result = classifier.classify("What is Python?")
        
        assert len(result.recommended_tiers) > 0
        assert MemoryType.SEMANTIC in result.recommended_tiers

    def test_key_terms_extracted(self):
        """Should extract key terms."""
        classifier = IntentClassifier()
        
        result = classifier.classify("How does Python asyncio work?")
        
        assert "python" in result.key_terms or "asyncio" in result.key_terms

    def test_weights_set(self):
        """Should set intent-specific weights."""
        classifier = IntentClassifier()
        
        factual = classifier.classify("What is Python?")
        episodic = classifier.classify("What happened last time?")
        
        # Factual should have higher similarity weight
        assert factual.similarity_weight >= 0.5
        # Episodic should have higher recency weight
        assert episodic.recency_weight >= 0.5


class TestLLMIntentClassifier:
    """Tests for LLM-based classifier."""

    def test_fallback_to_rule_based(self):
        """Should fallback when no LLM client."""
        classifier = LLMIntentClassifier()
        
        result = classifier.classify("What is Python?")
        
        assert result.primary_intent == QueryIntent.FACTUAL

    @pytest.mark.asyncio
    async def test_async_fallback(self):
        """Async classify should fallback without LLM."""
        classifier = LLMIntentClassifier()
        
        result = await classifier.classify_with_llm("What is Python?")
        
        assert result.primary_intent == QueryIntent.FACTUAL


class TestSearchResult:
    """Tests for SearchResult."""

    def test_default_scores(self):
        """Should have default scores."""
        result = SearchResult(
            memory_id="test",
            memory_type=MemoryType.SEMANTIC,
            content="Test content",
        )
        
        assert result.similarity_score == 0.0
        assert result.relevance_score == 0.0
        assert result.final_score == 0.0

    def test_custom_scores(self):
        """Should accept custom scores."""
        result = create_search_result("Test", similarity=0.9, relevance=0.8)
        
        assert result.similarity_score == 0.9
        assert result.relevance_score == 0.8


class TestSearchResults:
    """Tests for SearchResults collection."""

    def test_top_results(self):
        """Should return top N results."""
        classifier = IntentClassifier()
        intent = classifier.classify("test")
        
        results = SearchResults(
            query="test",
            intent=intent,
            results=[
                create_search_result("A", similarity=0.9),
                create_search_result("B", similarity=0.8),
                create_search_result("C", similarity=0.7),
            ],
        )
        
        top = results.top(2)
        
        assert len(top) == 2

    def test_by_tier(self):
        """Should filter results by tier."""
        classifier = IntentClassifier()
        intent = classifier.classify("test")
        
        results = SearchResults(
            query="test",
            intent=intent,
            results=[
                create_search_result("A", memory_type=MemoryType.SEMANTIC),
                create_search_result("B", memory_type=MemoryType.EPISODIC),
                create_search_result("C", memory_type=MemoryType.SEMANTIC),
            ],
        )
        
        semantic = results.by_tier(MemoryType.SEMANTIC)
        
        assert len(semantic) == 2


class TestSearchConfig:
    """Tests for SearchConfig."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = SearchConfig()
        
        assert config.max_results == 10
        assert config.min_similarity == 0.3
        assert config.search_all_tiers is True

    def test_custom_config(self):
        """Should accept custom values."""
        config = SearchConfig(
            max_results=5,
            min_similarity=0.5,
        )
        
        assert config.max_results == 5
        assert config.min_similarity == 0.5


class TestKeywordSearcher:
    """Tests for KeywordSearcher."""

    def test_search_finds_matching(self):
        """Should find memories with matching keywords."""
        searcher = KeywordSearcher()
        classifier = IntentClassifier()
        intent = classifier.classify("Python programming")
        
        memories = [
            create_semantic("Python is a programming language"),
            create_semantic("JavaScript is also popular"),
            create_semantic("Python has great libraries"),
        ]
        
        results = searcher.search("Python programming", memories, intent)
        
        assert len(results.results) >= 2

    def test_search_respects_min_similarity(self):
        """Should filter by minimum similarity."""
        config = SearchConfig(min_similarity=0.5)
        searcher = KeywordSearcher(config)
        
        memories = [
            create_semantic("Python programming"),
            create_semantic("Completely unrelated content xyz"),
        ]
        
        results = searcher.search("Python", memories)
        
        # Should only find Python memory
        assert len(results.results) == 1

    def test_search_respects_max_results(self):
        """Should limit results."""
        config = SearchConfig(max_results=2)
        searcher = KeywordSearcher(config)
        
        memories = [create_semantic(f"Python topic {i}") for i in range(10)]
        
        results = searcher.search("Python", memories)
        
        assert len(results.results) <= 2

    def test_search_records_metadata(self):
        """Should record search metadata."""
        searcher = KeywordSearcher()
        
        memories = [create_semantic("Python programming")]
        results = searcher.search("Python", memories)
        
        assert results.total_candidates == 1
        assert results.search_time_ms >= 0
        assert len(results.tiers_searched) > 0


class TestMemorySearcher:
    """Tests for MemorySearcher."""

    def test_cosine_similarity(self):
        """Should calculate cosine similarity correctly."""
        searcher = MemorySearcher()
        
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        
        sim = searcher._cosine_similarity(vec1, vec2)
        
        assert sim == 1.0

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors should have 0 similarity."""
        searcher = MemorySearcher()
        
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        
        sim = searcher._cosine_similarity(vec1, vec2)
        
        assert sim == 0.0

    def test_cosine_similarity_different_lengths(self):
        """Different length vectors should return 0."""
        searcher = MemorySearcher()
        
        vec1 = [1.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        
        sim = searcher._cosine_similarity(vec1, vec2)
        
        assert sim == 0.0

    def test_keyword_similarity(self):
        """Should calculate keyword similarity."""
        searcher = MemorySearcher()
        
        sim = searcher._keyword_similarity("Python programming", "Python is great")
        
        assert sim > 0

    def test_calculate_relevance(self):
        """Should calculate relevance based on intent."""
        searcher = MemorySearcher()
        classifier = IntentClassifier()
        intent = classifier.classify("Python programming")
        
        memory = create_semantic("Python programming is fun")
        
        relevance = searcher._calculate_relevance(memory, intent)
        
        assert relevance > 0.5


class TestRankingConfig:
    """Tests for RankingConfig."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = RankingConfig()
        
        assert config.similarity_weight > 0
        assert config.promote_diversity is True
        assert config.use_mmr is True

    def test_tier_boosts(self):
        """Should have tier boost defaults."""
        config = RankingConfig()
        
        assert config.stm_boost > config.semantic_boost


class TestRankedResult:
    """Tests for RankedResult."""

    def test_create_ranked_result(self):
        """Should create ranked result."""
        search_result = create_search_result("Test")
        
        ranked = RankedResult(
            result=search_result,
            original_rank=5,
            new_rank=1,
            ranking_score=0.95,
        )
        
        assert ranked.original_rank == 5
        assert ranked.new_rank == 1
        assert ranked.ranking_score == 0.95


class TestRankedResults:
    """Tests for RankedResults collection."""

    def test_top_results(self):
        """Should return top results."""
        classifier = IntentClassifier()
        intent = classifier.classify("test")
        
        ranked_list = [
            RankedResult(
                result=create_search_result(f"Result {i}"),
                original_rank=i,
                new_rank=i,
                ranking_score=1.0 - i * 0.1,
            )
            for i in range(5)
        ]
        
        results = RankedResults(
            query="test",
            intent=intent,
            ranked_results=ranked_list,
        )
        
        top = results.top(3)
        
        assert len(top) == 3

    def test_get_reranked_positions(self):
        """Should return position mapping."""
        classifier = IntentClassifier()
        intent = classifier.classify("test")
        
        ranked_list = [
            RankedResult(
                result=create_search_result("A"),
                original_rank=2,
                new_rank=0,
                ranking_score=0.9,
            ),
        ]
        ranked_list[0].result.memory_id = "mem_a"
        
        results = RankedResults(
            query="test",
            intent=intent,
            ranked_results=ranked_list,
        )
        
        positions = results.get_reranked_positions()
        
        assert positions["mem_a"] == (2, 0)


class TestResultRanker:
    """Tests for ResultRanker."""

    def test_rank_empty_results(self):
        """Should handle empty results."""
        ranker = ResultRanker()
        classifier = IntentClassifier()
        intent = classifier.classify("test")
        
        search_results = SearchResults(
            query="test",
            intent=intent,
            results=[],
        )
        
        ranked = ranker.rank(search_results)
        
        assert ranked.reranked_count == 0

    def test_rank_preserves_results(self):
        """Should preserve all results."""
        ranker = ResultRanker()
        classifier = IntentClassifier()
        intent = classifier.classify("test")
        
        search_results = SearchResults(
            query="test",
            intent=intent,
            results=[
                create_search_result("A"),
                create_search_result("B"),
            ],
        )
        
        ranked = ranker.rank(search_results)
        
        assert ranked.reranked_count == 2

    def test_rank_by_similarity(self):
        """Should rank by similarity score."""
        config = RankingConfig(use_mmr=False)
        ranker = ResultRanker(config)
        classifier = IntentClassifier()
        intent = classifier.classify("test")
        
        search_results = SearchResults(
            query="test",
            intent=intent,
            results=[
                create_search_result("Low", similarity=0.3),
                create_search_result("High", similarity=0.9),
                create_search_result("Medium", similarity=0.6),
            ],
        )
        
        ranked = ranker.rank(search_results)
        
        # High similarity should be first
        assert ranked.ranked_results[0].result.similarity_score == 0.9

    def test_recency_score(self):
        """Should calculate recency score."""
        ranker = ResultRanker()
        
        recent = create_search_result("Recent", age_days=1)
        old = create_search_result("Old", age_days=60)
        
        recent_score = ranker._calculate_recency_score(recent)
        old_score = ranker._calculate_recency_score(old)
        
        assert recent_score > old_score

    def test_tier_boost(self):
        """Should apply tier boost."""
        ranker = ResultRanker()
        classifier = IntentClassifier()
        intent = classifier.classify("What happened before?")  # Episodic
        
        stm_boost = ranker._get_tier_boost(MemoryType.SHORT_TERM, intent)
        semantic_boost = ranker._get_tier_boost(MemoryType.SEMANTIC, intent)
        
        # STM should have higher boost for episodic queries
        assert stm_boost >= semantic_boost

    def test_mmr_diversity(self):
        """Should promote diversity with MMR."""
        config = RankingConfig(use_mmr=True, mmr_lambda=0.7)
        ranker = ResultRanker(config)
        classifier = IntentClassifier()
        intent = classifier.classify("test")
        
        # Create similar results
        search_results = SearchResults(
            query="test",
            intent=intent,
            results=[
                create_search_result("Python programming is great", similarity=0.9),
                create_search_result("Python programming is awesome", similarity=0.85),
                create_search_result("JavaScript is different", similarity=0.8),
            ],
        )
        
        ranked = ranker.rank(search_results)
        
        # JavaScript should be boosted for diversity
        assert ranked.reranked_count == 3

    def test_content_similarity(self):
        """Should calculate content similarity."""
        ranker = ResultRanker()
        
        sim = ranker._content_similarity(
            "Python programming",
            "Python is great",
        )
        
        assert sim > 0


class TestContextualRanker:
    """Tests for ContextualRanker."""

    def test_add_context(self):
        """Should add context topics."""
        ranker = ContextualRanker()
        
        ranker.add_context("Python")
        ranker.add_context("async")
        
        assert len(ranker._recent_topics) == 2

    def test_context_window_limit(self):
        """Should respect context window."""
        ranker = ContextualRanker(context_window=3)
        
        for i in range(5):
            ranker.add_context(f"Topic {i}")
        
        assert len(ranker._recent_topics) == 3

    def test_clear_context(self):
        """Should clear context."""
        ranker = ContextualRanker()
        ranker.add_context("Python")
        
        ranker.clear_context()
        
        assert len(ranker._recent_topics) == 0

    def test_rank_with_context_boost(self):
        """Should boost results matching context."""
        ranker = ContextualRanker()
        ranker.add_context("Python")
        
        classifier = IntentClassifier()
        intent = classifier.classify("test")
        
        search_results = SearchResults(
            query="test",
            intent=intent,
            results=[
                create_search_result("Python programming", similarity=0.7),
                create_search_result("JavaScript coding", similarity=0.8),
            ],
        )
        
        ranked = ranker.rank_with_context(search_results)
        
        # Python should be boosted due to context
        assert ranked.ranked_results[0].result.content == "Python programming"

    def test_rank_without_context(self):
        """Should work without context."""
        ranker = ContextualRanker()
        classifier = IntentClassifier()
        intent = classifier.classify("test")
        
        search_results = SearchResults(
            query="test",
            intent=intent,
            results=[create_search_result("Test")],
        )
        
        ranked = ranker.rank_with_context(search_results)
        
        assert ranked.reranked_count == 1
