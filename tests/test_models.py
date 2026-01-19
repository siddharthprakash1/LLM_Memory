"""
Tests for memory data models.
"""

import pytest
from datetime import datetime, timedelta
import time

from llm_memory.models.base import (
    BaseMemory,
    MemoryType,
    MemorySource,
    ImportanceFactors,
    MemoryMetadata,
)
from llm_memory.models.short_term import (
    ShortTermMemory,
    WorkingContext,
    STMRole,
)
from llm_memory.models.episodic import (
    EpisodicMemory,
    Episode,
    EventType,
    TemporalContext,
)
from llm_memory.models.semantic import (
    SemanticMemory,
    Fact,
    FactType,
    Concept,
    ConceptType,
    Relationship,
    RelationType,
)


class TestImportanceFactors:
    """Tests for ImportanceFactors model."""

    def test_default_values(self):
        """Test default importance factor values."""
        factors = ImportanceFactors()
        assert factors.emotional_salience == 0.5
        assert factors.novelty == 0.5
        assert factors.relevance_frequency == 0.0
        assert factors.causal_significance == 0.5
        assert factors.user_marked == 0.0

    def test_composite_score_calculation(self):
        """Test composite score is calculated correctly."""
        factors = ImportanceFactors(
            emotional_salience=1.0,
            novelty=1.0,
            relevance_frequency=1.0,
            causal_significance=1.0,
            user_marked=1.0,
        )
        # All factors at 1.0 should give composite of 1.0
        assert factors.composite_score == 1.0

    def test_composite_score_mixed(self):
        """Test composite score with mixed values."""
        factors = ImportanceFactors(
            emotional_salience=0.8,
            novelty=0.6,
            relevance_frequency=0.4,
            causal_significance=0.7,
            user_marked=0.5,
        )
        # Should be between 0 and 1
        assert 0 <= factors.composite_score <= 1

    def test_bounds_validation(self):
        """Test that values are bounded correctly."""
        with pytest.raises(ValueError):
            ImportanceFactors(emotional_salience=1.5)

        with pytest.raises(ValueError):
            ImportanceFactors(novelty=-0.1)


class TestMemoryMetadata:
    """Tests for MemoryMetadata model."""

    def test_default_values(self):
        """Test default metadata values."""
        meta = MemoryMetadata()
        assert meta.user_id is None
        assert meta.scope == "global"
        assert meta.source == MemorySource.OBSERVATION
        assert meta.tags == []
        assert meta.confidence == 1.0

    def test_custom_values(self):
        """Test custom metadata values."""
        meta = MemoryMetadata(
            user_id="user_123",
            scope="project_abc",
            source=MemorySource.USER_INPUT,
            tags=["important", "preference"],
            confidence=0.9,
        )
        assert meta.user_id == "user_123"
        assert meta.scope == "project_abc"
        assert meta.source == MemorySource.USER_INPUT
        assert "important" in meta.tags
        assert meta.confidence == 0.9


class TestWorkingContext:
    """Tests for WorkingContext (STM item) model."""

    def test_creation(self):
        """Test creating a working context item."""
        ctx = WorkingContext(
            content="User asked about Python",
            role=STMRole.USER,
        )
        assert ctx.memory_type == MemoryType.SHORT_TERM
        assert ctx.content == "User asked about Python"
        assert ctx.role == STMRole.USER
        assert ctx.is_task_relevant is True

    def test_decay_rate(self):
        """Test STM has fast decay rate."""
        ctx = WorkingContext(content="Test")
        assert ctx.get_decay_rate() == 0.1

    def test_to_message_dict(self):
        """Test conversion to OpenAI message format."""
        ctx = WorkingContext(
            content="Hello!",
            role=STMRole.USER,
        )
        msg = ctx.to_message_dict()
        assert msg["role"] == "user"
        assert msg["content"] == "Hello!"


class TestShortTermMemory:
    """Tests for ShortTermMemory buffer."""

    def test_creation(self):
        """Test creating an STM buffer."""
        stm = ShortTermMemory(content="Session buffer")
        assert stm.memory_type == MemoryType.SHORT_TERM
        assert len(stm.items) == 0
        assert stm.max_items == 50

    def test_add_item(self):
        """Test adding items to STM."""
        stm = ShortTermMemory(content="Buffer")

        item = WorkingContext(content="First message", role=STMRole.USER)
        result = stm.add_item(item)

        assert result is True
        assert len(stm) == 1
        assert stm.items[0].sequence_number == 0

    def test_add_message(self):
        """Test convenience method for adding messages."""
        stm = ShortTermMemory(content="Buffer")

        item = stm.add_message("Hello!", role=STMRole.USER)

        assert len(stm) == 1
        assert item.content == "Hello!"
        assert item.role == STMRole.USER

    def test_get_recent(self):
        """Test getting recent items."""
        stm = ShortTermMemory(content="Buffer")

        for i in range(5):
            stm.add_message(f"Message {i}")

        recent = stm.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].content == "Message 4"

    def test_evict_oldest(self):
        """Test evicting oldest items."""
        stm = ShortTermMemory(content="Buffer")

        for i in range(5):
            stm.add_message(f"Message {i}")

        evicted = stm.evict_oldest(2)

        assert len(evicted) == 2
        assert evicted[0].content == "Message 0"
        assert len(stm) == 3
        assert stm.items[0].content == "Message 2"

    def test_to_messages(self):
        """Test conversion to messages list."""
        stm = ShortTermMemory(content="Buffer")
        stm.add_message("Hello", role=STMRole.USER)
        stm.add_message("Hi there!", role=STMRole.ASSISTANT)

        messages = stm.to_messages()

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_buffer_limit(self):
        """Test buffer respects max_items limit."""
        stm = ShortTermMemory(content="Buffer", max_items=3)

        for i in range(5):
            result = stm.add_item(WorkingContext(content=f"Item {i}"))
            if i < 3:
                assert result is True
            else:
                assert result is False

        assert len(stm) == 3


class TestEpisodicMemory:
    """Tests for EpisodicMemory model."""

    def test_creation(self):
        """Test creating episodic memory."""
        ep = EpisodicMemory(content="Debugging session")
        assert ep.memory_type == MemoryType.EPISODIC
        assert len(ep.episodes) == 0

    def test_add_episode(self):
        """Test adding episodes."""
        ep = EpisodicMemory(content="Task completion")

        episode = Episode(
            event_id="ep_001",
            event_type=EventType.TASK_COMPLETION,
            description="User completed authentication setup",
            outcome="success",
            was_successful=True,
        )

        ep.add_episode(episode)

        assert len(ep) == 1
        assert ep.primary_episode_id == "ep_001"

    def test_decay_rate(self):
        """Test episodic has medium decay rate."""
        ep = EpisodicMemory(content="Test")
        assert ep.get_decay_rate() == 0.01

    def test_temporal_context(self):
        """Test temporal context tracking."""
        temporal = TemporalContext(
            occurred_at=datetime.utcnow() - timedelta(days=2),
        )
        assert 1.9 <= temporal.age_days <= 2.1

    def test_from_stm_buffer(self):
        """Test creating episodic from STM."""
        ep = EpisodicMemory.from_stm_buffer(
            content="User asked about Python patterns",
            session_id="sess_123",
            task_id="task_456",
        )

        assert ep.memory_type == MemoryType.EPISODIC
        assert len(ep.episodes) == 1
        assert ep.episodes[0].event_type == EventType.CONVERSATION

    def test_get_episodes_by_type(self):
        """Test filtering episodes by type."""
        ep = EpisodicMemory(content="Mixed episodes")

        ep.add_episode(Episode(
            event_id="1",
            event_type=EventType.ERROR,
            description="Error occurred",
        ))
        ep.add_episode(Episode(
            event_id="2",
            event_type=EventType.TASK_COMPLETION,
            description="Task done",
        ))
        ep.add_episode(Episode(
            event_id="3",
            event_type=EventType.ERROR,
            description="Another error",
        ))

        errors = ep.get_episodes_by_type(EventType.ERROR)
        assert len(errors) == 2


class TestSemanticMemory:
    """Tests for SemanticMemory model."""

    def test_creation(self):
        """Test creating semantic memory."""
        sem = SemanticMemory(content="User preferences")
        assert sem.memory_type == MemoryType.SEMANTIC
        assert len(sem.facts) == 0

    def test_add_fact(self):
        """Test adding facts."""
        sem = SemanticMemory(content="Preferences")

        fact = Fact(
            fact_id="fact_001",
            fact_type=FactType.PREFERENCE,
            subject="User",
            predicate="prefers",
            object="Python",
            statement="User prefers Python for backend development",
        )

        sem.add_fact(fact)

        assert len(sem.facts) == 1
        assert sem.primary_fact_id == "fact_001"

    def test_decay_rate(self):
        """Test semantic has slow decay rate."""
        sem = SemanticMemory(content="Test")
        assert sem.get_decay_rate() == 0.001

    def test_get_preferences(self):
        """Test getting preference facts."""
        sem = SemanticMemory(content="Knowledge")

        sem.add_fact(Fact(
            fact_id="1",
            fact_type=FactType.PREFERENCE,
            subject="User",
            predicate="prefers",
            object="dark mode",
            statement="User prefers dark mode",
        ))
        sem.add_fact(Fact(
            fact_id="2",
            fact_type=FactType.DEFINITION,
            subject="Python",
            predicate="is",
            object="programming language",
            statement="Python is a programming language",
        ))

        prefs = sem.get_preferences()
        assert len(prefs) == 1
        assert prefs[0].fact_type == FactType.PREFERENCE

    def test_strengthen_fact(self):
        """Test strengthening a fact."""
        sem = SemanticMemory(content="Knowledge")

        fact = Fact(
            fact_id="1",
            fact_type=FactType.PREFERENCE,
            subject="User",
            predicate="likes",
            object="TypeScript",
            statement="User likes TypeScript",
            confidence=0.5,
        )
        sem.add_fact(fact)

        result = sem.strengthen_fact("1", boost=0.2)

        assert result is True
        assert sem.get_fact("1").confidence == 0.7
        assert sem.get_fact("1").evidence_count == 2

    def test_concepts_and_relationships(self):
        """Test adding concepts and relationships."""
        sem = SemanticMemory(content="Knowledge graph")

        concept1 = Concept(
            concept_id="c1",
            concept_type=ConceptType.TOOL,
            name="Python",
            description="Programming language",
        )
        concept2 = Concept(
            concept_id="c2",
            concept_type=ConceptType.TOOL,
            name="FastAPI",
            description="Web framework",
        )

        sem.add_concept(concept1)
        sem.add_concept(concept2)

        rel = Relationship(
            relationship_id="r1",
            relation_type=RelationType.USES,
            source_concept_id="c2",
            target_concept_id="c1",
        )
        sem.add_relationship(rel)

        assert len(sem.concepts) == 2
        assert len(sem.relationships) == 1

        rels = sem.get_relationships_for_concept("c1")
        assert len(rels) == 1


class TestMemoryDecay:
    """Tests for memory decay functionality."""

    def test_decay_calculation(self):
        """Test decay strength calculation."""
        ctx = WorkingContext(content="Test")

        # Set initial access time to past
        ctx.last_accessed_at = datetime.utcnow() - timedelta(hours=1)

        # Calculate decayed strength
        strength = ctx.calculate_decayed_strength(decay_rate=0.1)

        # Should be less than initial but not zero
        assert 0 < strength < 1.0

    def test_importance_affects_decay(self):
        """Test that high importance slows decay."""
        # Low importance memory
        low_imp = WorkingContext(content="Low importance")
        low_imp.importance = ImportanceFactors(
            emotional_salience=0.1,
            novelty=0.1,
            causal_significance=0.1,
        )
        low_imp.last_accessed_at = datetime.utcnow() - timedelta(hours=2)

        # High importance memory
        high_imp = WorkingContext(content="High importance")
        high_imp.importance = ImportanceFactors(
            emotional_salience=0.9,
            novelty=0.9,
            causal_significance=0.9,
        )
        high_imp.last_accessed_at = datetime.utcnow() - timedelta(hours=2)

        low_strength = low_imp.calculate_decayed_strength()
        high_strength = high_imp.calculate_decayed_strength()

        # High importance should decay slower
        assert high_strength > low_strength

    def test_rehearsal_boost(self):
        """Test that accessing memory boosts strength."""
        ctx = WorkingContext(content="Test")
        ctx.current_strength = 0.5
        initial_access_count = ctx.access_count

        ctx.apply_rehearsal_boost(boost=0.2)

        assert ctx.current_strength == 0.7
        assert ctx.access_count == initial_access_count + 1


class TestMemorySerialization:
    """Tests for memory serialization."""

    def test_to_dict(self):
        """Test converting memory to dict."""
        sem = SemanticMemory(content="Test knowledge")
        sem.add_fact(Fact(
            fact_id="f1",
            fact_type=FactType.DEFINITION,
            subject="X",
            predicate="is",
            object="Y",
            statement="X is Y",
        ))

        data = sem.to_dict()

        assert data["id"] == sem.id
        assert data["memory_type"] == "semantic"
        assert len(data["facts"]) == 1

    def test_from_dict(self):
        """Test creating memory from dict."""
        data = {
            "id": "test_id",
            "memory_type": "semantic",
            "content": "Test content",
            "facts": [{
                "fact_id": "f1",
                "fact_type": "definition",
                "subject": "A",
                "predicate": "is",
                "object": "B",
                "statement": "A is B",
            }],
        }

        # This would need proper model validation
        sem = SemanticMemory.model_validate(data)

        assert sem.id == "test_id"
        assert sem.content == "Test content"
