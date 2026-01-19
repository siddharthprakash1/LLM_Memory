"""
Tests for storage backends.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import asyncio

from llm_memory.config import StorageConfig
from llm_memory.models.base import MemoryType
from llm_memory.models.short_term import ShortTermMemory, WorkingContext, STMRole
from llm_memory.models.episodic import EpisodicMemory, Episode, EventType
from llm_memory.models.semantic import SemanticMemory, Fact, FactType
from llm_memory.storage.sqlite import SQLiteStorage
from llm_memory.storage.base import StorageError, MemoryNotFoundError


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_memory.db"


@pytest.fixture
def storage_config(temp_db_path):
    """Create a storage config with temp path."""
    return StorageConfig(sqlite_path=temp_db_path)


@pytest.fixture
async def sqlite_storage(storage_config):
    """Create and connect a SQLite storage instance."""
    storage = SQLiteStorage(storage_config)
    await storage.connect()
    yield storage
    await storage.disconnect()


class TestSQLiteStorageConnection:
    """Tests for SQLite storage connection."""

    @pytest.mark.asyncio
    async def test_connect_creates_database(self, storage_config):
        """Test that connecting creates the database file."""
        storage = SQLiteStorage(storage_config)
        
        assert not storage_config.sqlite_path.exists()
        
        await storage.connect()
        
        assert storage_config.sqlite_path.exists()
        assert await storage.is_connected()
        
        await storage.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect(self, storage_config):
        """Test disconnecting from storage."""
        storage = SQLiteStorage(storage_config)
        await storage.connect()
        
        assert await storage.is_connected()
        
        await storage.disconnect()
        
        assert not await storage.is_connected()

    @pytest.mark.asyncio
    async def test_context_manager(self, storage_config):
        """Test using storage as async context manager."""
        async with SQLiteStorage(storage_config) as storage:
            assert await storage.is_connected()
        
        # Connection should be closed after context


class TestSQLiteStorageCRUD:
    """Tests for CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_stm(self, sqlite_storage):
        """Test creating short-term memory."""
        stm = ShortTermMemory(content="Test STM buffer")
        stm.add_message("Hello!", role=STMRole.USER)
        
        memory_id = await sqlite_storage.create(stm)
        
        assert memory_id == stm.id

    @pytest.mark.asyncio
    async def test_read_stm(self, sqlite_storage):
        """Test reading short-term memory."""
        stm = ShortTermMemory(content="Test buffer")
        stm.add_message("Test message", role=STMRole.USER)
        
        await sqlite_storage.create(stm)
        
        loaded = await sqlite_storage.read(stm.id)
        
        assert loaded is not None
        assert loaded.id == stm.id
        assert loaded.memory_type == MemoryType.SHORT_TERM
        assert len(loaded.items) == 1

    @pytest.mark.asyncio
    async def test_read_nonexistent(self, sqlite_storage):
        """Test reading non-existent memory returns None."""
        result = await sqlite_storage.read("nonexistent_id")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_memory(self, sqlite_storage):
        """Test updating a memory."""
        stm = ShortTermMemory(content="Original")
        await sqlite_storage.create(stm)
        
        stm.add_message("New message")
        result = await sqlite_storage.update(stm)
        
        assert result is True
        
        loaded = await sqlite_storage.read(stm.id)
        assert len(loaded.items) == 1

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, sqlite_storage):
        """Test updating non-existent memory returns False."""
        stm = ShortTermMemory(content="Test")
        stm.id = "nonexistent_id"
        
        result = await sqlite_storage.update(stm)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_memory(self, sqlite_storage):
        """Test deleting a memory."""
        stm = ShortTermMemory(content="To delete")
        await sqlite_storage.create(stm)
        
        result = await sqlite_storage.delete(stm.id)
        
        assert result is True
        assert await sqlite_storage.read(stm.id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, sqlite_storage):
        """Test deleting non-existent memory returns False."""
        result = await sqlite_storage.delete("nonexistent_id")
        assert result is False


class TestSQLiteStorageEpisodicMemory:
    """Tests for episodic memory storage."""

    @pytest.mark.asyncio
    async def test_create_episodic(self, sqlite_storage):
        """Test creating episodic memory."""
        ep = EpisodicMemory(content="Debug session")
        ep.add_episode(Episode(
            event_id="ep_001",
            event_type=EventType.ERROR,
            description="TypeError in auth module",
            outcome="Fixed by adding type check",
            was_successful=True,
        ))
        
        memory_id = await sqlite_storage.create(ep)
        
        assert memory_id == ep.id

    @pytest.mark.asyncio
    async def test_read_episodic(self, sqlite_storage):
        """Test reading episodic memory with episodes."""
        ep = EpisodicMemory(content="Session")
        ep.add_episode(Episode(
            event_id="ep_001",
            event_type=EventType.TASK_COMPLETION,
            description="Task done",
        ))
        ep.add_episode(Episode(
            event_id="ep_002",
            event_type=EventType.LEARNING,
            description="Learned pattern",
        ))
        
        await sqlite_storage.create(ep)
        
        loaded = await sqlite_storage.read(ep.id)
        
        assert loaded is not None
        assert loaded.memory_type == MemoryType.EPISODIC
        assert len(loaded.episodes) == 2
        assert loaded.primary_episode_id == "ep_001"


class TestSQLiteStorageSemanticMemory:
    """Tests for semantic memory storage."""

    @pytest.mark.asyncio
    async def test_create_semantic(self, sqlite_storage):
        """Test creating semantic memory."""
        sem = SemanticMemory(content="User preferences")
        sem.add_fact(Fact(
            fact_id="f1",
            fact_type=FactType.PREFERENCE,
            subject="User",
            predicate="prefers",
            object="dark mode",
            statement="User prefers dark mode",
        ))
        
        memory_id = await sqlite_storage.create(sem)
        
        assert memory_id == sem.id

    @pytest.mark.asyncio
    async def test_read_semantic(self, sqlite_storage):
        """Test reading semantic memory with facts."""
        sem = SemanticMemory(content="Knowledge")
        sem.add_fact(Fact(
            fact_id="f1",
            fact_type=FactType.PREFERENCE,
            subject="User",
            predicate="prefers",
            object="Python",
            statement="User prefers Python",
            confidence=0.9,
        ))
        sem.add_fact(Fact(
            fact_id="f2",
            fact_type=FactType.DEFINITION,
            subject="FastAPI",
            predicate="is",
            object="web framework",
            statement="FastAPI is a web framework",
        ))
        
        await sqlite_storage.create(sem)
        
        loaded = await sqlite_storage.read(sem.id)
        
        assert loaded is not None
        assert loaded.memory_type == MemoryType.SEMANTIC
        assert len(loaded.facts) == 2
        
        # Check fact details preserved
        pref = next(f for f in loaded.facts if f.fact_id == "f1")
        assert pref.confidence == 0.9


class TestSQLiteStorageBatchOperations:
    """Tests for batch operations."""

    @pytest.mark.asyncio
    async def test_create_many(self, sqlite_storage):
        """Test creating multiple memories at once."""
        memories = [
            ShortTermMemory(content=f"Buffer {i}")
            for i in range(5)
        ]
        
        ids = await sqlite_storage.create_many(memories)
        
        assert len(ids) == 5
        for i, memory in enumerate(memories):
            assert ids[i] == memory.id

    @pytest.mark.asyncio
    async def test_read_many(self, sqlite_storage):
        """Test reading multiple memories at once."""
        memories = [
            ShortTermMemory(content=f"Buffer {i}")
            for i in range(3)
        ]
        await sqlite_storage.create_many(memories)
        
        ids = [m.id for m in memories]
        loaded = await sqlite_storage.read_many(ids)
        
        assert len(loaded) == 3

    @pytest.mark.asyncio
    async def test_delete_many(self, sqlite_storage):
        """Test deleting multiple memories at once."""
        memories = [
            ShortTermMemory(content=f"Buffer {i}")
            for i in range(5)
        ]
        await sqlite_storage.create_many(memories)
        
        ids_to_delete = [memories[0].id, memories[2].id, memories[4].id]
        deleted_count = await sqlite_storage.delete_many(ids_to_delete)
        
        assert deleted_count == 3
        assert await sqlite_storage.count() == 2


class TestSQLiteStorageQueries:
    """Tests for query operations."""

    @pytest.mark.asyncio
    async def test_query_by_type(self, sqlite_storage):
        """Test querying by memory type."""
        # Create mixed memories
        await sqlite_storage.create(ShortTermMemory(content="STM 1"))
        await sqlite_storage.create(ShortTermMemory(content="STM 2"))
        await sqlite_storage.create(EpisodicMemory(content="Episodic 1"))
        await sqlite_storage.create(SemanticMemory(content="Semantic 1"))
        
        stm_results = await sqlite_storage.query_by_type(MemoryType.SHORT_TERM)
        
        assert len(stm_results) == 2
        for mem in stm_results:
            assert mem.memory_type == MemoryType.SHORT_TERM

    @pytest.mark.asyncio
    async def test_query_by_user(self, sqlite_storage):
        """Test querying by user ID."""
        stm1 = ShortTermMemory(content="User A memory")
        stm1.metadata.user_id = "user_a"
        
        stm2 = ShortTermMemory(content="User B memory")
        stm2.metadata.user_id = "user_b"
        
        stm3 = ShortTermMemory(content="User A memory 2")
        stm3.metadata.user_id = "user_a"
        
        await sqlite_storage.create_many([stm1, stm2, stm3])
        
        results = await sqlite_storage.query_by_user("user_a")
        
        assert len(results) == 2
        for mem in results:
            assert mem.metadata.user_id == "user_a"

    @pytest.mark.asyncio
    async def test_query_by_scope(self, sqlite_storage):
        """Test querying by scope."""
        stm1 = ShortTermMemory(content="Global memory")
        stm1.metadata.scope = "global"
        
        stm2 = ShortTermMemory(content="Project memory")
        stm2.metadata.scope = "project_123"
        
        stm3 = ShortTermMemory(content="Project memory 2")
        stm3.metadata.scope = "project_123"
        
        await sqlite_storage.create_many([stm1, stm2, stm3])
        
        results = await sqlite_storage.query_by_scope("project_123")
        
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_by_strength(self, sqlite_storage):
        """Test querying by strength range."""
        memories = []
        for i, strength in enumerate([0.2, 0.5, 0.7, 0.9]):
            mem = ShortTermMemory(content=f"Memory {i}")
            mem.current_strength = strength
            memories.append(mem)
        
        await sqlite_storage.create_many(memories)
        
        results = await sqlite_storage.query_by_strength(
            min_strength=0.4,
            max_strength=0.8,
        )
        
        assert len(results) == 2
        for mem in results:
            assert 0.4 <= mem.current_strength <= 0.8

    @pytest.mark.asyncio
    async def test_query_weak_memories(self, sqlite_storage):
        """Test querying weak memories for GC."""
        memories = []
        for strength in [0.05, 0.08, 0.15, 0.3, 0.8]:
            mem = ShortTermMemory(content=f"Strength {strength}")
            mem.current_strength = strength
            memories.append(mem)
        
        await sqlite_storage.create_many(memories)
        
        weak = await sqlite_storage.query_weak_memories(threshold=0.1)
        
        assert len(weak) == 2
        for mem in weak:
            assert mem.current_strength < 0.1

    @pytest.mark.asyncio
    async def test_query_unconsolidated(self, sqlite_storage):
        """Test querying unconsolidated memories."""
        stm1 = ShortTermMemory(content="Not consolidated")
        stm1.is_consolidated = False
        
        stm2 = ShortTermMemory(content="Already consolidated")
        stm2.is_consolidated = True
        
        stm3 = ShortTermMemory(content="Also not consolidated")
        stm3.is_consolidated = False
        
        await sqlite_storage.create_many([stm1, stm2, stm3])
        
        results = await sqlite_storage.query_unconsolidated(MemoryType.SHORT_TERM)
        
        assert len(results) == 2
        for mem in results:
            assert mem.is_consolidated is False


class TestSQLiteStorageStatistics:
    """Tests for statistics operations."""

    @pytest.mark.asyncio
    async def test_count_total(self, sqlite_storage):
        """Test counting total memories."""
        memories = [ShortTermMemory(content=f"Mem {i}") for i in range(5)]
        await sqlite_storage.create_many(memories)
        
        count = await sqlite_storage.count()
        
        assert count == 5

    @pytest.mark.asyncio
    async def test_count_by_type(self, sqlite_storage):
        """Test counting by type."""
        await sqlite_storage.create(ShortTermMemory(content="STM"))
        await sqlite_storage.create(ShortTermMemory(content="STM 2"))
        await sqlite_storage.create(EpisodicMemory(content="Episodic"))
        
        stm_count = await sqlite_storage.count(MemoryType.SHORT_TERM)
        ep_count = await sqlite_storage.count(MemoryType.EPISODIC)
        
        assert stm_count == 2
        assert ep_count == 1

    @pytest.mark.asyncio
    async def test_get_stats(self, sqlite_storage):
        """Test getting storage statistics."""
        await sqlite_storage.create(ShortTermMemory(content="STM"))
        await sqlite_storage.create(EpisodicMemory(content="Episodic"))
        await sqlite_storage.create(SemanticMemory(content="Semantic"))
        
        stats = await sqlite_storage.get_stats()
        
        assert stats["total_memories"] == 3
        assert "by_type" in stats
        assert stats["by_type"]["short_term"] == 1
        assert stats["by_type"]["episodic"] == 1
        assert stats["by_type"]["semantic"] == 1


class TestSQLiteStorageMetadata:
    """Tests for metadata handling."""

    @pytest.mark.asyncio
    async def test_tags_stored(self, sqlite_storage):
        """Test that tags are stored correctly."""
        mem = ShortTermMemory(content="Tagged memory")
        mem.metadata.tags = ["important", "python", "debugging"]
        
        await sqlite_storage.create(mem)
        loaded = await sqlite_storage.read(mem.id)
        
        assert set(loaded.metadata.tags) == {"important", "python", "debugging"}

    @pytest.mark.asyncio
    async def test_importance_preserved(self, sqlite_storage):
        """Test that importance factors are preserved."""
        mem = ShortTermMemory(content="Important memory")
        mem.importance.emotional_salience = 0.9
        mem.importance.novelty = 0.8
        mem.importance.user_marked = 1.0
        
        await sqlite_storage.create(mem)
        loaded = await sqlite_storage.read(mem.id)
        
        assert loaded.importance.emotional_salience == 0.9
        assert loaded.importance.novelty == 0.8
        assert loaded.importance.user_marked == 1.0
