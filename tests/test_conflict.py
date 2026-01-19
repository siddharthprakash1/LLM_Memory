"""
Tests for conflict detection and resolution.
"""

import pytest
from datetime import datetime, timedelta

from llm_memory.models.base import MemoryType, MemoryMetadata, MemorySource, ImportanceFactors
from llm_memory.models.semantic import SemanticMemory, Fact, FactType
from llm_memory.models.episodic import EpisodicMemory, Episode, EventType
from llm_memory.conflict.detector import (
    ConflictType,
    ConflictSeverity,
    DetectedConflict,
    ConflictDetector,
)
from llm_memory.conflict.strategies import (
    ResolutionStrategy,
    ResolutionAction,
    ResolutionResult,
    RecencyStrategy,
    ConfidenceStrategy,
    SourceReliabilityStrategy,
    FrequencyStrategy,
    ImportanceStrategy,
    MergeStrategy,
    KeepBothStrategy,
    UserGuidedStrategy,
    get_strategy,
    get_default_strategy_for_conflict,
)
from llm_memory.conflict.resolver import (
    ConflictResolutionConfig,
    ConflictResolver,
    BatchConflictResolver,
)


def _utcnow() -> datetime:
    """Get current UTC time (naive)."""
    return datetime.utcnow()


def create_semantic_memory(
    content: str,
    subject: str = "Test",
    predicate: str = "is",
    obj: str = "value",
    fact_type: FactType = FactType.DEFINITION,
    age_days: float = 0,
    source: MemorySource = MemorySource.USER_INPUT,
    confidence: float = 0.8,
) -> SemanticMemory:
    """Create a semantic memory for testing."""
    fact = Fact(
        fact_id=f"fact_{_utcnow().timestamp()}",
        fact_type=fact_type,
        subject=subject,
        predicate=predicate,
        object=obj,
        statement=content,
        confidence=confidence,
    )
    
    mem = SemanticMemory(
        content=content,
        metadata=MemoryMetadata(source=source),
    )
    mem.add_fact(fact)
    mem.created_at = _utcnow() - timedelta(days=age_days)
    
    return mem


def create_conflict(
    conflict_type: ConflictType = ConflictType.FACT_INCONSISTENCY,
    severity: ConflictSeverity = ConflictSeverity.MEDIUM,
    mem_a_id: str = "mem_a",
    mem_b_id: str = "mem_b",
) -> DetectedConflict:
    """Create a conflict for testing."""
    return DetectedConflict(
        conflict_type=conflict_type,
        severity=severity,
        memory_a_id=mem_a_id,
        memory_b_id=mem_b_id,
        memory_a_content="Content A",
        memory_b_content="Content B",
        conflicting_aspect="Test aspect",
        explanation="Test conflict",
    )


class TestConflictDetector:
    """Tests for conflict detection."""

    def test_no_conflicts_empty_list(self):
        """Empty list should have no conflicts."""
        detector = ConflictDetector()
        conflicts = detector.detect_conflicts([])
        assert conflicts == []

    def test_no_conflicts_single_memory(self):
        """Single memory can't conflict with itself."""
        detector = ConflictDetector()
        mem = create_semantic_memory("Python is great")
        
        conflicts = detector.detect_conflicts([mem])
        
        assert conflicts == []

    def test_detect_fact_inconsistency(self):
        """Should detect when same subject has different values."""
        detector = ConflictDetector()
        
        mem_a = create_semantic_memory(
            "User prefers Python",
            subject="User",
            predicate="prefers",
            obj="Python",
            fact_type=FactType.PREFERENCE,
        )
        mem_b = create_semantic_memory(
            "User prefers Rust",
            subject="User",
            predicate="prefers",
            obj="Rust",
            fact_type=FactType.PREFERENCE,
        )
        
        conflicts = detector.detect_conflicts([mem_a, mem_b])
        
        assert len(conflicts) >= 1
        assert conflicts[0].conflict_type in [
            ConflictType.FACT_INCONSISTENCY,
            ConflictType.PREFERENCE_CONFLICT,
        ]

    def test_detect_direct_contradiction(self):
        """Should detect negation patterns."""
        detector = ConflictDetector()
        
        mem_a = create_semantic_memory("The feature is enabled")
        mem_b = create_semantic_memory("The feature is not enabled")
        
        conflict = detector._check_content_contradiction(mem_a, mem_b)
        
        assert conflict is not None
        assert conflict.conflict_type == ConflictType.DIRECT_CONTRADICTION

    def test_no_conflict_different_subjects(self):
        """Different subjects should not conflict."""
        detector = ConflictDetector()
        
        mem_a = create_semantic_memory(
            "Python is interpreted",
            subject="Python",
            predicate="is",
            obj="interpreted",
        )
        mem_b = create_semantic_memory(
            "Rust is compiled",
            subject="Rust",
            predicate="is",
            obj="compiled",
        )
        
        conflicts = detector.detect_conflicts([mem_a, mem_b])
        
        # Should not find fact inconsistency (different subjects)
        fact_conflicts = [
            c for c in conflicts 
            if c.conflict_type == ConflictType.FACT_INCONSISTENCY
        ]
        assert len(fact_conflicts) == 0

    def test_detect_temporal_conflict(self):
        """Should detect outdated information."""
        detector = ConflictDetector(temporal_outdated_days=7)
        
        # Old memory
        old_mem = create_semantic_memory(
            "Python uses pip for packages",
            subject="Python",
            age_days=30,
        )
        # New memory with same subject
        new_mem = create_semantic_memory(
            "Python now uses uv for packages",
            subject="Python",
            age_days=0,
        )
        
        conflicts = detector._detect_temporal_conflicts([old_mem, new_mem])
        
        assert len(conflicts) >= 1
        assert conflicts[0].conflict_type == ConflictType.TEMPORAL_OUTDATED

    def test_is_negation(self):
        """Should detect negation patterns in text."""
        detector = ConflictDetector()
        
        assert detector._is_negation("The user likes Python", "The user dislikes Python")
        assert detector._is_negation("This is true", "This is false")
        assert detector._is_negation("Feature is enabled", "Feature is not enabled")
        assert not detector._is_negation("Python is great", "Rust is great")


class TestResolutionStrategies:
    """Tests for resolution strategies."""

    def test_recency_strategy_prefers_newer(self):
        """Recency strategy should prefer newer memory."""
        strategy = RecencyStrategy()
        
        old_mem = create_semantic_memory("Old info", age_days=30)
        new_mem = create_semantic_memory("New info", age_days=1)
        conflict = create_conflict(mem_a_id=old_mem.id, mem_b_id=new_mem.id)
        
        result = strategy.resolve(conflict, old_mem, new_mem)
        
        assert result.winner_id == new_mem.id
        assert result.loser_id == old_mem.id
        assert result.action == ResolutionAction.KEEP_B

    def test_confidence_strategy_prefers_higher(self):
        """Confidence strategy should prefer higher confidence."""
        strategy = ConfidenceStrategy()
        
        low_conf = create_semantic_memory("Low confidence", confidence=0.3)
        low_conf.overall_confidence = 0.3
        high_conf = create_semantic_memory("High confidence", confidence=0.9)
        high_conf.overall_confidence = 0.9
        conflict = create_conflict(mem_a_id=low_conf.id, mem_b_id=high_conf.id)
        
        result = strategy.resolve(conflict, low_conf, high_conf)
        
        assert result.winner_id == high_conf.id
        assert result.action == ResolutionAction.KEEP_B

    def test_source_reliability_strategy(self):
        """Source reliability should prefer user input over inference."""
        strategy = SourceReliabilityStrategy()
        
        user_mem = create_semantic_memory("User said", source=MemorySource.USER_INPUT)
        inferred_mem = create_semantic_memory("AI inferred", source=MemorySource.INFERENCE)
        conflict = create_conflict(mem_a_id=user_mem.id, mem_b_id=inferred_mem.id)
        
        result = strategy.resolve(conflict, user_mem, inferred_mem)
        
        assert result.winner_id == user_mem.id
        assert result.action == ResolutionAction.KEEP_A

    def test_frequency_strategy_prefers_more_accessed(self):
        """Frequency strategy should prefer more accessed memory."""
        strategy = FrequencyStrategy()
        
        rarely_accessed = create_semantic_memory("Rarely accessed")
        rarely_accessed.access_count = 2
        
        frequently_accessed = create_semantic_memory("Frequently accessed")
        frequently_accessed.access_count = 50
        
        conflict = create_conflict(
            mem_a_id=rarely_accessed.id,
            mem_b_id=frequently_accessed.id,
        )
        
        result = strategy.resolve(conflict, rarely_accessed, frequently_accessed)
        
        assert result.winner_id == frequently_accessed.id

    def test_importance_strategy_prefers_more_important(self):
        """Importance strategy should prefer higher importance."""
        strategy = ImportanceStrategy()
        
        low_imp = create_semantic_memory("Low importance")
        low_imp.importance = ImportanceFactors(
            emotional_salience=0.1,
            novelty=0.1,
        )
        
        high_imp = create_semantic_memory("High importance")
        high_imp.importance = ImportanceFactors(
            emotional_salience=0.9,
            novelty=0.9,
            user_marked=1.0,
        )
        
        conflict = create_conflict(mem_a_id=low_imp.id, mem_b_id=high_imp.id)
        
        result = strategy.resolve(conflict, low_imp, high_imp)
        
        assert result.winner_id == high_imp.id

    def test_merge_strategy_creates_merged_content(self):
        """Merge strategy should create combined content."""
        strategy = MergeStrategy()
        
        mem_a = create_semantic_memory("First perspective")
        mem_b = create_semantic_memory("Second perspective")
        conflict = create_conflict(mem_a_id=mem_a.id, mem_b_id=mem_b.id)
        
        result = strategy.resolve(conflict, mem_a, mem_b)
        
        assert result.action == ResolutionAction.MERGE
        assert result.merged_content is not None
        assert "First perspective" in result.merged_content
        assert "Second perspective" in result.merged_content

    def test_keep_both_strategy(self):
        """Keep both should not pick a winner."""
        strategy = KeepBothStrategy()
        
        mem_a = create_semantic_memory("Option A")
        mem_b = create_semantic_memory("Option B")
        conflict = create_conflict(mem_a_id=mem_a.id, mem_b_id=mem_b.id)
        
        result = strategy.resolve(conflict, mem_a, mem_b)
        
        assert result.action == ResolutionAction.KEEP_BOTH
        assert result.winner_id is None
        assert result.loser_id is None

    def test_user_guided_strategy_defers(self):
        """User guided should defer resolution."""
        strategy = UserGuidedStrategy()
        
        mem_a = create_semantic_memory("Option A")
        mem_b = create_semantic_memory("Option B")
        conflict = create_conflict(mem_a_id=mem_a.id, mem_b_id=mem_b.id)
        
        result = strategy.resolve(conflict, mem_a, mem_b)
        
        assert result.action == ResolutionAction.DEFER
        assert result.requires_user_confirmation is True

    def test_user_guided_apply_decision(self):
        """User guided should apply user decision."""
        strategy = UserGuidedStrategy()
        
        mem_a = create_semantic_memory("Option A")
        mem_b = create_semantic_memory("Option B")
        conflict = create_conflict(mem_a_id=mem_a.id, mem_b_id=mem_b.id)
        
        result = strategy.apply_user_decision(conflict, mem_a, mem_b, mem_b.id)
        
        assert result.action == ResolutionAction.KEEP_B
        assert result.winner_id == mem_b.id
        assert result.confidence == 1.0


class TestGetStrategy:
    """Tests for strategy factory."""

    def test_get_strategy_returns_correct_type(self):
        """Factory should return correct strategy type."""
        assert isinstance(get_strategy(ResolutionStrategy.RECENCY), RecencyStrategy)
        assert isinstance(get_strategy(ResolutionStrategy.CONFIDENCE), ConfidenceStrategy)
        assert isinstance(get_strategy(ResolutionStrategy.MERGE), MergeStrategy)

    def test_get_default_strategy_for_temporal(self):
        """Temporal conflicts should default to recency."""
        conflict = create_conflict(conflict_type=ConflictType.TEMPORAL_OUTDATED)
        
        strategy = get_default_strategy_for_conflict(conflict)
        
        assert strategy == ResolutionStrategy.RECENCY

    def test_get_default_strategy_for_fact(self):
        """Fact conflicts should default to confidence."""
        conflict = create_conflict(conflict_type=ConflictType.FACT_INCONSISTENCY)
        
        strategy = get_default_strategy_for_conflict(conflict)
        
        assert strategy == ResolutionStrategy.CONFIDENCE

    def test_critical_severity_defaults_to_user(self):
        """Critical conflicts should default to user-guided."""
        conflict = create_conflict(severity=ConflictSeverity.CRITICAL)
        
        strategy = get_default_strategy_for_conflict(conflict)
        
        assert strategy == ResolutionStrategy.USER_GUIDED


class TestConflictResolver:
    """Tests for the conflict resolver."""

    def test_detect_and_store_pending(self):
        """Resolver should store detected conflicts."""
        resolver = ConflictResolver()
        
        mem_a = create_semantic_memory(
            "User likes Python",
            subject="User",
            predicate="likes",
            obj="Python",
            fact_type=FactType.PREFERENCE,
        )
        mem_b = create_semantic_memory(
            "User likes Rust",
            subject="User",
            predicate="likes",
            obj="Rust",
            fact_type=FactType.PREFERENCE,
        )
        
        conflicts = resolver.detect_conflicts([mem_a, mem_b])
        
        assert resolver.pending_count >= 1

    def test_resolve_removes_from_pending(self):
        """Resolving should remove from pending."""
        resolver = ConflictResolver()
        
        mem_a = create_semantic_memory("Option A")
        mem_b = create_semantic_memory("Option B")
        conflict = create_conflict(mem_a_id=mem_a.id, mem_b_id=mem_b.id)
        
        resolver._pending_conflicts.append(conflict)
        initial_count = resolver.pending_count
        
        resolver.resolve_conflict(conflict, mem_a, mem_b, ResolutionStrategy.RECENCY)
        
        assert resolver.pending_count == initial_count - 1

    def test_resolve_adds_to_history(self):
        """Resolving should add to history."""
        resolver = ConflictResolver()
        
        mem_a = create_semantic_memory("Option A")
        mem_b = create_semantic_memory("Option B")
        conflict = create_conflict(mem_a_id=mem_a.id, mem_b_id=mem_b.id)
        
        resolver.resolve_conflict(conflict, mem_a, mem_b, ResolutionStrategy.RECENCY)
        
        assert len(resolver.history) == 1
        assert resolver.history[0].conflict.conflict_id == conflict.conflict_id

    def test_apply_resolution_keep_a(self):
        """Apply resolution should handle KEEP_A action."""
        resolver = ConflictResolver()
        
        mem_a = create_semantic_memory("Winner")
        mem_b = create_semantic_memory("Loser")
        
        result = ResolutionResult(
            conflict_id="test",
            strategy_used=ResolutionStrategy.RECENCY,
            action=ResolutionAction.KEEP_A,
            winner_id=mem_a.id,
            loser_id=mem_b.id,
            explanation="Test",
        )
        
        kept, discarded, merged = resolver.apply_resolution(result, mem_a, mem_b)
        
        assert kept == mem_a
        assert discarded == mem_b
        assert merged is None

    def test_apply_resolution_merge(self):
        """Apply resolution should create merged memory."""
        resolver = ConflictResolver()
        
        mem_a = create_semantic_memory("Content A")
        mem_b = create_semantic_memory("Content B")
        
        result = ResolutionResult(
            conflict_id="test",
            strategy_used=ResolutionStrategy.MERGE,
            action=ResolutionAction.MERGE,
            merged_content="Merged content",
            explanation="Test merge",
        )
        
        kept, discarded, merged = resolver.apply_resolution(result, mem_a, mem_b)
        
        assert kept is None
        assert discarded is None
        assert merged is not None
        assert "Merged" in merged.content

    def test_check_new_memory_for_conflicts(self):
        """Should check new memory against existing ones."""
        resolver = ConflictResolver()
        
        existing = create_semantic_memory(
            "User prefers dark mode",
            subject="User",
            predicate="prefers",
            obj="dark mode",
            fact_type=FactType.PREFERENCE,
        )
        new_mem = create_semantic_memory(
            "User prefers light mode",
            subject="User",
            predicate="prefers",
            obj="light mode",
            fact_type=FactType.PREFERENCE,
        )
        
        conflicts = resolver.check_new_memory(new_mem, [existing])
        
        assert len(conflicts) >= 1

    def test_get_statistics(self):
        """Should return correct statistics."""
        resolver = ConflictResolver()
        
        mem_a = create_semantic_memory("A")
        mem_b = create_semantic_memory("B")
        conflict = create_conflict(mem_a_id=mem_a.id, mem_b_id=mem_b.id)
        
        resolver.resolve_conflict(conflict, mem_a, mem_b, ResolutionStrategy.RECENCY)
        
        stats = resolver.get_statistics()
        
        assert stats["total_resolved"] == 1
        assert "by_strategy" in stats
        assert stats["by_strategy"]["recency"] == 1


class TestConflictResolutionConfig:
    """Tests for configuration."""

    def test_default_config(self):
        """Default config should have sensible defaults."""
        config = ConflictResolutionConfig()
        
        assert config.auto_detect is True
        assert config.auto_resolve_low_severity is True
        assert config.require_user_for_critical is True

    def test_auto_resolve_based_on_config(self):
        """Resolver should respect auto_resolve config."""
        config = ConflictResolutionConfig(
            auto_resolve_low_severity=True,
            auto_resolve_medium_severity=False,
        )
        resolver = ConflictResolver(config)
        
        low_conflict = create_conflict(severity=ConflictSeverity.LOW)
        medium_conflict = create_conflict(severity=ConflictSeverity.MEDIUM)
        
        assert resolver._should_auto_resolve(low_conflict) is True
        assert resolver._should_auto_resolve(medium_conflict) is False


class TestBatchConflictResolver:
    """Tests for batch resolution."""

    def test_batch_detection(self):
        """Should detect conflicts in batch."""
        batch_resolver = BatchConflictResolver()
        
        memories = [
            create_semantic_memory(
                "User prefers Python",
                subject="User",
                predicate="prefers",
                obj="Python",
                fact_type=FactType.PREFERENCE,
            ),
            create_semantic_memory(
                "User prefers Java",
                subject="User",
                predicate="prefers",
                obj="Java",
                fact_type=FactType.PREFERENCE,
            ),
        ]
        
        conflicts = batch_resolver.run_detection_batch(memories)
        
        # Should find the preference conflict
        assert len(conflicts) >= 1

    def test_batch_respects_size_limit(self):
        """Should respect batch size limit."""
        config = ConflictResolutionConfig(detection_batch_size=2)
        batch_resolver = BatchConflictResolver(config=config)
        
        memories = [create_semantic_memory(f"Memory {i}") for i in range(10)]
        
        # Should only process first 2
        conflicts = batch_resolver.run_detection_batch(memories)
        
        # Can't have conflicts with only 2 different memories
        # (no matching subjects)
        assert isinstance(conflicts, list)
