"""
Tests for consolidation pipeline.
"""

import pytest
from datetime import datetime, timedelta

from llm_memory.config import ConsolidationConfig, DecayConfig
from llm_memory.models.base import MemoryType, ImportanceFactors
from llm_memory.models.short_term import ShortTermMemory, WorkingContext, STMRole
from llm_memory.models.episodic import EpisodicMemory, Episode, EventType
from llm_memory.models.semantic import SemanticMemory, Fact, FactType
from llm_memory.consolidation.promoter import (
    PromotionCriteria,
    STMToEpisodicPromoter,
    EpisodicToSemanticPromoter,
    MemoryPromoter,
)
from llm_memory.consolidation.merger import (
    MemoryMerger,
    GarbageCollector,
    MemoryDeduplicator,
)
from llm_memory.consolidation.scheduler import (
    ManualConsolidator,
    ConsolidationScheduler,
)


def _utcnow() -> datetime:
    """Get current UTC time (naive, for compatibility with models)."""
    return datetime.utcnow()


def create_test_stm(
    content: str = "Test conversation",
    messages: list[tuple[str, STMRole]] | None = None,
    age_seconds: float = 120.0,
    importance: float = 0.5,
) -> ShortTermMemory:
    """Helper to create test STM."""
    stm = ShortTermMemory(content=content)
    
    if messages:
        for msg, role in messages:
            stm.add_message(msg, role=role)
    else:
        stm.add_message("Hello, I need help with Python", role=STMRole.USER)
        stm.add_message("Sure, I can help with Python", role=STMRole.ASSISTANT)
    
    # Set age
    stm.created_at = _utcnow() - timedelta(seconds=age_seconds)
    
    # Set importance
    stm.importance = ImportanceFactors(
        emotional_salience=importance,
        novelty=importance,
        causal_significance=importance,
    )
    
    return stm


def create_test_episodic(
    content: str = "Test episode",
    event_type: EventType = EventType.CONVERSATION,
    age_hours: float = 48.0,
) -> EpisodicMemory:
    """Helper to create test episodic memory."""
    episode = Episode(
        event_id=f"ep_{_utcnow().timestamp()}",
        event_type=event_type,
        description=content,
    )
    
    episodic = EpisodicMemory(content=content)
    episodic.add_episode(episode)
    episodic.created_at = _utcnow() - timedelta(hours=age_hours)
    
    return episodic


def create_test_semantic(
    content: str = "Test fact",
    fact_type: FactType = FactType.DEFINITION,
) -> SemanticMemory:
    """Helper to create test semantic memory."""
    fact = Fact(
        fact_id=f"fact_{_utcnow().timestamp()}",
        fact_type=fact_type,
        subject="Test",
        predicate="is",
        object="example",
        statement=content,
    )
    
    semantic = SemanticMemory(content=content)
    semantic.add_fact(fact)
    
    return semantic


class TestSTMToEpisodicPromoter:
    """Tests for STM → Episodic promotion."""

    def test_should_not_promote_empty_stm(self):
        """Empty STM should not be promoted."""
        promoter = STMToEpisodicPromoter()
        stm = ShortTermMemory(content="Empty")
        
        should, reason = promoter.should_promote(stm)
        
        assert should is False
        assert "empty" in reason.lower()

    def test_should_not_promote_too_recent(self):
        """Very recent STM should not be promoted."""
        criteria = PromotionCriteria(min_stm_age_seconds=60.0)
        promoter = STMToEpisodicPromoter(criteria=criteria)
        
        stm = create_test_stm(age_seconds=30.0)  # 30 seconds old
        
        should, reason = promoter.should_promote(stm)
        
        assert should is False
        assert "recent" in reason.lower()

    def test_should_promote_old_stm(self):
        """STM exceeding max age should be force-promoted."""
        criteria = PromotionCriteria(max_stm_age_seconds=300.0)
        promoter = STMToEpisodicPromoter(criteria=criteria)
        
        stm = create_test_stm(age_seconds=400.0)  # 400 seconds old
        
        should, reason = promoter.should_promote(stm)
        
        assert should is True
        assert "age" in reason.lower()

    def test_should_promote_high_importance(self):
        """High importance STM should be promoted."""
        criteria = PromotionCriteria(min_importance_for_episodic=0.4)
        promoter = STMToEpisodicPromoter(criteria=criteria)
        
        stm = create_test_stm(age_seconds=120.0, importance=0.8)
        
        should, reason = promoter.should_promote(stm)
        
        assert should is True
        assert "importance" in reason.lower()

    def test_promote_creates_episodic(self):
        """Promotion should create valid episodic memory."""
        promoter = STMToEpisodicPromoter()
        
        stm = create_test_stm(messages=[
            ("How do I sort a list?", STMRole.USER),
            ("Use sorted() or list.sort()", STMRole.ASSISTANT),
            ("Thanks, that worked!", STMRole.USER),
        ])
        
        result = promoter.promote(stm, force=True)
        
        assert result.success is True
        assert result.target_type == MemoryType.EPISODIC
        assert stm.is_consolidated is True

    def test_detect_event_type_error(self):
        """Should detect error event type."""
        promoter = STMToEpisodicPromoter()
        
        stm = create_test_stm(messages=[
            ("I'm getting a TypeError exception", STMRole.USER),
            ("Let me help debug that error", STMRole.ASSISTANT),
        ])
        
        event_type = promoter._detect_event_type(stm)
        
        assert event_type == EventType.ERROR

    def test_detect_event_type_task_completion(self):
        """Should detect task completion event type."""
        promoter = STMToEpisodicPromoter()
        
        stm = create_test_stm(messages=[
            ("Can you fix this?", STMRole.USER),
            ("Done! I've fixed the issue", STMRole.ASSISTANT),
        ])
        
        event_type = promoter._detect_event_type(stm)
        
        assert event_type == EventType.TASK_COMPLETION


class TestEpisodicToSemanticPromoter:
    """Tests for Episodic → Semantic promotion."""

    def test_find_candidates_empty(self):
        """No candidates from empty list."""
        promoter = EpisodicToSemanticPromoter()
        
        candidates = promoter.find_promotion_candidates([])
        
        assert candidates == []

    def test_find_candidates_too_few(self):
        """No candidates if fewer than minimum episodes."""
        criteria = PromotionCriteria(min_similar_episodes=3)
        promoter = EpisodicToSemanticPromoter(criteria=criteria)
        
        episodes = [create_test_episodic() for _ in range(2)]
        
        candidates = promoter.find_promotion_candidates(episodes)
        
        assert candidates == []

    def test_find_candidates_groups_by_type(self):
        """Should group similar episodes."""
        criteria = PromotionCriteria(
            min_similar_episodes=2,
            min_episode_age_hours=1.0,
        )
        promoter = EpisodicToSemanticPromoter(criteria=criteria)
        
        # Create similar error episodes
        episodes = [
            create_test_episodic("Python error in auth module", EventType.ERROR, age_hours=48),
            create_test_episodic("Python error in data module", EventType.ERROR, age_hours=48),
            create_test_episodic("Task completed successfully", EventType.TASK_COMPLETION, age_hours=48),
        ]
        
        candidates = promoter.find_promotion_candidates(episodes)
        
        # Should find the error group
        assert len(candidates) >= 1

    def test_promote_creates_semantic(self):
        """Promotion should create valid semantic memory."""
        criteria = PromotionCriteria(min_similar_episodes=2)
        promoter = EpisodicToSemanticPromoter(criteria=criteria)
        
        episodes = [
            create_test_episodic("User prefers Python", EventType.DECISION),
            create_test_episodic("User chose Python again", EventType.DECISION),
        ]
        
        result = promoter.promote(episodes)
        
        assert result.success is True
        assert result.target_type == MemoryType.SEMANTIC

    def test_promote_marks_consolidated(self):
        """Promotion should mark episodes as consolidated."""
        criteria = PromotionCriteria(min_similar_episodes=2)
        promoter = EpisodicToSemanticPromoter(criteria=criteria)
        
        episodes = [
            create_test_episodic("Pattern A"),
            create_test_episodic("Pattern A"),
        ]
        
        promoter.promote(episodes)
        
        for ep in episodes:
            assert ep.is_consolidated is True


class TestMemoryPromoter:
    """Tests for high-level MemoryPromoter."""

    def test_check_stm_promotion(self):
        """Should check STM promotion eligibility."""
        promoter = MemoryPromoter()
        stm = create_test_stm(age_seconds=5000, importance=0.8)
        
        should, reason = promoter.check_stm_promotion(stm)
        
        assert should is True

    def test_promote_stm_returns_episodic(self):
        """Should return created episodic memory."""
        promoter = MemoryPromoter()
        stm = create_test_stm()
        
        result, episodic = promoter.promote_stm(stm, force=True)
        
        assert result.success is True
        assert episodic is not None
        assert episodic.memory_type == MemoryType.EPISODIC


class TestMemoryMerger:
    """Tests for memory merging."""

    def test_merge_semantic_single(self):
        """Merging single memory returns itself."""
        merger = MemoryMerger()
        
        sem = create_test_semantic("Test fact")
        
        result_mem, result = merger.merge_semantic_memories([sem])
        
        assert result.success is True
        assert result_mem.id == sem.id

    def test_merge_semantic_multiple(self):
        """Merging multiple memories combines facts."""
        merger = MemoryMerger()
        
        memories = [
            create_test_semantic("Fact about Python"),
            create_test_semantic("Fact about TypeScript"),
        ]
        
        # Add more facts to second memory
        memories[1].add_fact(Fact(
            fact_id="extra",
            fact_type=FactType.DEFINITION,
            subject="TS",
            predicate="is",
            object="typed JS",
            statement="TypeScript is typed JavaScript",
        ))
        
        result_mem, result = merger.merge_semantic_memories(memories)
        
        assert result.success is True
        assert len(result.merged_ids) == 2
        assert len(result_mem.facts) >= 2

    def test_merge_strengthens_duplicate_facts(self):
        """Merging should strengthen duplicate facts."""
        merger = MemoryMerger()
        
        # Create memories with same fact
        memories = []
        for _ in range(3):
            mem = SemanticMemory(content="Python is great")
            mem.add_fact(Fact(
                fact_id=f"fact_{_}",
                fact_type=FactType.PREFERENCE,
                subject="User",
                predicate="likes",
                object="Python",
                statement="User likes Python",
                confidence=0.5,
            ))
            memories.append(mem)
        
        result_mem, result = merger.merge_semantic_memories(memories)
        
        # Check that fact confidence increased
        assert result_mem.facts[0].confidence > 0.5

    def test_should_merge_same_type(self):
        """Should only merge memories of same type."""
        merger = MemoryMerger()
        
        sem = create_test_semantic()
        ep = create_test_episodic()
        
        assert merger.should_merge(sem, ep) is False

    def test_should_merge_similar_content(self):
        """Should merge memories with similar content."""
        merger = MemoryMerger(similarity_threshold=0.5)
        
        sem1 = create_test_semantic("Python is a programming language for data science")
        sem2 = create_test_semantic("Python is a programming language for web development")
        
        assert merger.should_merge(sem1, sem2) is True


class TestGarbageCollector:
    """Tests for garbage collection."""

    def test_find_collectible_weak_memories(self):
        """Should find memories below threshold."""
        config = DecayConfig(min_strength_threshold=0.3)
        gc = GarbageCollector(config)
        
        memories = [
            create_test_semantic("Strong memory"),
            create_test_semantic("Weak memory"),
        ]
        memories[0].current_strength = 0.8
        memories[0].initial_strength = 0.8  # Set initial to prevent decay reset
        memories[1].current_strength = 0.1
        memories[1].initial_strength = 0.1
        
        collectible = gc.find_collectible(memories, threshold=0.3)
        
        assert len(collectible) == 1
        assert collectible[0].current_strength < 0.3

    def test_protect_user_marked(self):
        """Should not collect user-marked important memories."""
        config = DecayConfig(min_strength_threshold=0.3)
        gc = GarbageCollector(config)
        
        mem = create_test_semantic()
        mem.current_strength = 0.05  # Very weak
        mem.initial_strength = 0.05
        mem.importance.user_marked = 0.9  # But user marked important
        
        collectible = gc.find_collectible([mem], threshold=0.3)
        
        assert len(collectible) == 0

    def test_collect_marks_inactive(self):
        """Collection should mark memories as inactive."""
        config = DecayConfig(min_strength_threshold=0.3)
        gc = GarbageCollector(config)
        
        mem = create_test_semantic()
        mem.current_strength = 0.01
        mem.initial_strength = 0.01
        
        result = gc.collect([mem], threshold=0.3)
        
        assert result.collected_count == 1
        assert mem.is_active is False


class TestMemoryDeduplicator:
    """Tests for deduplication."""

    def test_find_exact_duplicates(self):
        """Should find exact duplicate content."""
        dedup = MemoryDeduplicator(similarity_threshold=0.9)
        
        memories = [
            create_test_semantic("Python is a programming language"),
            create_test_semantic("Python is a programming language"),  # Exact duplicate
            create_test_semantic("JavaScript is different"),
        ]
        
        duplicates = dedup.find_duplicates(memories)
        
        assert len(duplicates) == 1
        assert len(duplicates[0]) == 2

    def test_find_similar_duplicates(self):
        """Should find similar (near-duplicate) content."""
        dedup = MemoryDeduplicator(similarity_threshold=0.7)
        
        memories = [
            create_test_semantic("Python is great for data science"),
            create_test_semantic("Python is great for data analysis"),
            create_test_semantic("Rust is fast and safe"),
        ]
        
        duplicates = dedup.find_duplicates(memories)
        
        # Python memories should be grouped
        assert len(duplicates) >= 1

    def test_select_best_by_importance(self):
        """Should select memory with highest importance."""
        dedup = MemoryDeduplicator()
        
        memories = [
            create_test_semantic("Memory 1"),
            create_test_semantic("Memory 2"),
            create_test_semantic("Memory 3"),
        ]
        memories[1].importance.user_marked = 1.0  # Most important
        
        best = dedup.select_best(memories)
        
        assert best.importance.user_marked == 1.0


class TestManualConsolidator:
    """Tests for manual consolidation."""

    def test_process_stm(self):
        """Should process STM promotion."""
        consolidator = ManualConsolidator()
        stm = create_test_stm()
        
        result, episodic = consolidator.process_stm(stm, force=True)
        
        assert result.success is True
        assert episodic is not None

    def test_process_episodes(self):
        """Should process episode promotion."""
        config = ConsolidationConfig()
        criteria = PromotionCriteria(min_similar_episodes=2)
        consolidator = ManualConsolidator(config)
        consolidator.promoter = MemoryPromoter(config, criteria)
        
        episodes = [
            create_test_episodic("Pattern observation 1"),
            create_test_episodic("Pattern observation 2"),
        ]
        
        result, semantic = consolidator.process_episodes(episodes)
        
        assert result.success is True
        assert semantic is not None

    def test_collect_garbage(self):
        """Should collect weak memories."""
        decay_config = DecayConfig(min_strength_threshold=0.3)
        consolidator = ManualConsolidator(decay_config=decay_config)
        
        memories = [create_test_semantic() for _ in range(3)]
        memories[0].current_strength = 0.01
        memories[0].initial_strength = 0.01
        memories[1].current_strength = 0.02
        memories[1].initial_strength = 0.02
        memories[2].current_strength = 0.8
        memories[2].initial_strength = 0.8
        
        result = consolidator.collect_garbage(memories)
        
        assert result.collected_count == 2


class TestConsolidationScheduler:
    """Tests for consolidation scheduler."""

    def test_scheduler_not_running_initially(self):
        """Scheduler should not be running initially."""
        scheduler = ConsolidationScheduler()
        
        assert scheduler.is_running is False

    @pytest.mark.asyncio
    async def test_scheduler_start_stop(self):
        """Scheduler should start and stop."""
        scheduler = ConsolidationScheduler()
        
        await scheduler.start()
        assert scheduler.is_running is True
        
        await scheduler.stop()
        assert scheduler.is_running is False

    @pytest.mark.asyncio
    async def test_run_consolidation_without_callbacks(self):
        """Should run consolidation even without callbacks."""
        scheduler = ConsolidationScheduler()
        
        result = await scheduler.run_consolidation()
        
        assert result.success is True
        assert result.completed_at is not None
