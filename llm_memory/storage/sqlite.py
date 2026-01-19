"""
SQLite storage backend for memory metadata and structured data.

Uses aiosqlite for async operations.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from llm_memory.config import StorageConfig
from llm_memory.models.base import BaseMemory, MemoryType
from llm_memory.models.short_term import ShortTermMemory, WorkingContext
from llm_memory.models.episodic import EpisodicMemory
from llm_memory.models.semantic import SemanticMemory
from llm_memory.storage.base import BaseStorage, MemoryNotFoundError, StorageError


# SQL Schema
SCHEMA = """
-- Main memories table
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    
    -- Embedding reference
    embedding_id TEXT,
    has_embedding INTEGER DEFAULT 0,
    
    -- Temporal
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    access_count INTEGER DEFAULT 0,
    
    -- Strength
    initial_strength REAL DEFAULT 1.0,
    current_strength REAL DEFAULT 1.0,
    
    -- Importance (stored as JSON)
    importance_json TEXT,
    
    -- Metadata (stored as JSON)
    metadata_json TEXT,
    
    -- Type-specific data (stored as JSON)
    type_data_json TEXT,
    
    -- Status
    is_active INTEGER DEFAULT 1,
    is_consolidated INTEGER DEFAULT 0
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_user_id ON memories(json_extract(metadata_json, '$.user_id'));
CREATE INDEX IF NOT EXISTS idx_scope ON memories(json_extract(metadata_json, '$.scope'));
CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_strength ON memories(current_strength);
CREATE INDEX IF NOT EXISTS idx_is_active ON memories(is_active);
CREATE INDEX IF NOT EXISTS idx_is_consolidated ON memories(is_consolidated);

-- Memory relationships table
CREATE TABLE IF NOT EXISTS memory_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_rel_source ON memory_relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON memory_relationships(target_id);

-- Memory tags table
CREATE TABLE IF NOT EXISTS memory_tags (
    memory_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (memory_id, tag),
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tag ON memory_tags(tag);

-- Statistics table
CREATE TABLE IF NOT EXISTS stats (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT
);
"""


def _memory_type_to_class(memory_type: MemoryType) -> type[BaseMemory]:
    """Map memory type to its class."""
    mapping = {
        MemoryType.SHORT_TERM: ShortTermMemory,
        MemoryType.EPISODIC: EpisodicMemory,
        MemoryType.SEMANTIC: SemanticMemory,
    }
    return mapping.get(memory_type, BaseMemory)


def _serialize_datetime(dt: datetime) -> str:
    """Serialize datetime to ISO format string."""
    return dt.isoformat()


def _deserialize_datetime(s: str) -> datetime:
    """Deserialize datetime from ISO format string."""
    return datetime.fromisoformat(s)


class SQLiteStorage(BaseStorage):
    """
    SQLite-based storage for memory metadata.
    
    Stores structured memory data with full query capabilities.
    Uses JSON columns for flexible nested data.
    """

    def __init__(self, config: StorageConfig | None = None):
        """
        Initialize SQLite storage.
        
        Args:
            config: Storage configuration
        """
        self.config = config or StorageConfig()
        self.db_path = self.config.sqlite_path
        self._connection: aiosqlite.Connection | None = None
        self._connected = False

    async def connect(self) -> None:
        """Initialize connection and create schema."""
        try:
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            self._connection = await aiosqlite.connect(str(self.db_path))
            self._connection.row_factory = aiosqlite.Row

            # Enable foreign keys
            await self._connection.execute("PRAGMA foreign_keys = ON")

            # Create schema
            await self._connection.executescript(SCHEMA)
            await self._connection.commit()

            self._connected = True
        except Exception as e:
            raise StorageError(f"Failed to connect to SQLite: {e}") from e

    async def disconnect(self) -> None:
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
        self._connected = False

    async def is_connected(self) -> bool:
        """Check if storage is connected."""
        return self._connected and self._connection is not None

    def _ensure_connected(self) -> None:
        """Raise error if not connected."""
        if not self._connected:
            raise StorageError("Not connected to database")

    def _memory_to_row(self, memory: BaseMemory) -> dict[str, Any]:
        """Convert a memory object to a database row."""
        # Get type-specific data
        type_data = {}
        if isinstance(memory, ShortTermMemory):
            type_data = {
                "items": [item.model_dump() for item in memory.items],
                "max_items": memory.max_items,
                "max_tokens": memory.max_tokens,
                "current_token_count": memory.current_token_count,
                "session_id": memory.session_id,
                "session_start": _serialize_datetime(memory.session_start),
                "current_task_id": memory.current_task_id,
                "current_task_description": memory.current_task_description,
            }
        elif isinstance(memory, EpisodicMemory):
            type_data = {
                "episodes": [ep.model_dump() for ep in memory.episodes],
                "primary_episode_id": memory.primary_episode_id,
                "temporal_context": memory.temporal_context.model_dump(),
                "narrative": memory.narrative,
                "lessons_learned": memory.lessons_learned,
                "consolidation_candidates": memory.consolidation_candidates,
                "times_pattern_matched": memory.times_pattern_matched,
            }
        elif isinstance(memory, SemanticMemory):
            type_data = {
                "facts": [f.model_dump() for f in memory.facts],
                "primary_fact_id": memory.primary_fact_id,
                "concepts": [c.model_dump() for c in memory.concepts],
                "relationships": [r.model_dump() for r in memory.relationships],
                "knowledge_domain": memory.knowledge_domain,
                "abstraction_level": memory.abstraction_level,
                "derived_from_episodic_ids": memory.derived_from_episodic_ids,
                "episode_count": memory.episode_count,
                "overall_confidence": memory.overall_confidence,
            }

        return {
            "id": memory.id,
            "memory_type": memory.memory_type.value,
            "content": memory.content,
            "summary": memory.summary,
            "embedding_id": memory.embedding_id,
            "has_embedding": 1 if memory.has_embedding else 0,
            "created_at": _serialize_datetime(memory.created_at),
            "updated_at": _serialize_datetime(memory.updated_at),
            "last_accessed_at": _serialize_datetime(memory.last_accessed_at),
            "access_count": memory.access_count,
            "initial_strength": memory.initial_strength,
            "current_strength": memory.current_strength,
            "importance_json": json.dumps(memory.importance.model_dump()),
            "metadata_json": json.dumps(memory.metadata.model_dump()),
            "type_data_json": json.dumps(type_data, default=str),
            "is_active": 1 if memory.is_active else 0,
            "is_consolidated": 1 if memory.is_consolidated else 0,
        }

    def _row_to_memory(self, row: aiosqlite.Row) -> BaseMemory:
        """Convert a database row to a memory object."""
        memory_type = MemoryType(row["memory_type"])
        memory_class = _memory_type_to_class(memory_type)

        # Parse JSON fields
        importance_data = json.loads(row["importance_json"]) if row["importance_json"] else {}
        metadata_data = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        type_data = json.loads(row["type_data_json"]) if row["type_data_json"] else {}

        # Base fields
        base_data = {
            "id": row["id"],
            "memory_type": memory_type,
            "content": row["content"],
            "summary": row["summary"],
            "embedding_id": row["embedding_id"],
            "has_embedding": bool(row["has_embedding"]),
            "created_at": _deserialize_datetime(row["created_at"]),
            "updated_at": _deserialize_datetime(row["updated_at"]),
            "last_accessed_at": _deserialize_datetime(row["last_accessed_at"]),
            "access_count": row["access_count"],
            "initial_strength": row["initial_strength"],
            "current_strength": row["current_strength"],
            "importance": importance_data,
            "metadata": metadata_data,
            "is_active": bool(row["is_active"]),
            "is_consolidated": bool(row["is_consolidated"]),
        }

        # Merge type-specific data
        base_data.update(type_data)

        return memory_class.model_validate(base_data)

    # CRUD Operations
    async def create(self, memory: BaseMemory) -> str:
        """Create a new memory in storage."""
        self._ensure_connected()

        row = self._memory_to_row(memory)

        columns = ", ".join(row.keys())
        placeholders = ", ".join(["?" for _ in row])

        query = f"INSERT INTO memories ({columns}) VALUES ({placeholders})"

        try:
            await self._connection.execute(query, list(row.values()))

            # Insert tags
            if memory.metadata.tags:
                await self._connection.executemany(
                    "INSERT INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                    [(memory.id, tag) for tag in memory.metadata.tags],
                )

            await self._connection.commit()
            return memory.id
        except Exception as e:
            await self._connection.rollback()
            raise StorageError(f"Failed to create memory: {e}") from e

    async def read(self, memory_id: str) -> BaseMemory | None:
        """Read a memory by ID."""
        self._ensure_connected()

        query = "SELECT * FROM memories WHERE id = ?"

        async with self._connection.execute(query, (memory_id,)) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        return self._row_to_memory(row)

    async def update(self, memory: BaseMemory) -> bool:
        """Update an existing memory."""
        self._ensure_connected()

        # Check if exists
        existing = await self.read(memory.id)
        if existing is None:
            return False

        row = self._memory_to_row(memory)
        del row["id"]  # Don't update ID

        set_clause = ", ".join([f"{k} = ?" for k in row.keys()])
        query = f"UPDATE memories SET {set_clause} WHERE id = ?"

        try:
            await self._connection.execute(query, list(row.values()) + [memory.id])

            # Update tags
            await self._connection.execute(
                "DELETE FROM memory_tags WHERE memory_id = ?", (memory.id,)
            )
            if memory.metadata.tags:
                await self._connection.executemany(
                    "INSERT INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                    [(memory.id, tag) for tag in memory.metadata.tags],
                )

            await self._connection.commit()
            return True
        except Exception as e:
            await self._connection.rollback()
            raise StorageError(f"Failed to update memory: {e}") from e

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        self._ensure_connected()

        query = "DELETE FROM memories WHERE id = ?"

        try:
            cursor = await self._connection.execute(query, (memory_id,))
            await self._connection.commit()
            return cursor.rowcount > 0
        except Exception as e:
            await self._connection.rollback()
            raise StorageError(f"Failed to delete memory: {e}") from e

    # Batch Operations
    async def create_many(self, memories: list[BaseMemory]) -> list[str]:
        """Create multiple memories in batch."""
        self._ensure_connected()

        ids = []
        try:
            for memory in memories:
                row = self._memory_to_row(memory)
                columns = ", ".join(row.keys())
                placeholders = ", ".join(["?" for _ in row])
                query = f"INSERT INTO memories ({columns}) VALUES ({placeholders})"
                await self._connection.execute(query, list(row.values()))

                if memory.metadata.tags:
                    await self._connection.executemany(
                        "INSERT INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                        [(memory.id, tag) for tag in memory.metadata.tags],
                    )

                ids.append(memory.id)

            await self._connection.commit()
            return ids
        except Exception as e:
            await self._connection.rollback()
            raise StorageError(f"Failed to create memories: {e}") from e

    async def read_many(self, memory_ids: list[str]) -> list[BaseMemory]:
        """Read multiple memories by IDs."""
        self._ensure_connected()

        if not memory_ids:
            return []

        placeholders = ", ".join(["?" for _ in memory_ids])
        query = f"SELECT * FROM memories WHERE id IN ({placeholders})"

        memories = []
        async with self._connection.execute(query, memory_ids) as cursor:
            async for row in cursor:
                memories.append(self._row_to_memory(row))

        return memories

    async def delete_many(self, memory_ids: list[str]) -> int:
        """Delete multiple memories by IDs."""
        self._ensure_connected()

        if not memory_ids:
            return 0

        placeholders = ", ".join(["?" for _ in memory_ids])
        query = f"DELETE FROM memories WHERE id IN ({placeholders})"

        try:
            cursor = await self._connection.execute(query, memory_ids)
            await self._connection.commit()
            return cursor.rowcount
        except Exception as e:
            await self._connection.rollback()
            raise StorageError(f"Failed to delete memories: {e}") from e

    # Query Operations
    async def query_by_type(
        self,
        memory_type: MemoryType,
        limit: int = 100,
        offset: int = 0,
    ) -> list[BaseMemory]:
        """Query memories by type."""
        self._ensure_connected()

        query = """
            SELECT * FROM memories 
            WHERE memory_type = ? AND is_active = 1
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """

        memories = []
        async with self._connection.execute(
            query, (memory_type.value, limit, offset)
        ) as cursor:
            async for row in cursor:
                memories.append(self._row_to_memory(row))

        return memories

    async def query_by_user(
        self,
        user_id: str,
        memory_type: MemoryType | None = None,
        limit: int = 100,
    ) -> list[BaseMemory]:
        """Query memories by user ID."""
        self._ensure_connected()

        query = """
            SELECT * FROM memories 
            WHERE json_extract(metadata_json, '$.user_id') = ? 
            AND is_active = 1
        """
        params = [user_id]

        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type.value)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        memories = []
        async with self._connection.execute(query, params) as cursor:
            async for row in cursor:
                memories.append(self._row_to_memory(row))

        return memories

    async def query_by_scope(
        self,
        scope: str,
        user_id: str | None = None,
        memory_type: MemoryType | None = None,
        limit: int = 100,
    ) -> list[BaseMemory]:
        """Query memories by scope."""
        self._ensure_connected()

        query = """
            SELECT * FROM memories 
            WHERE json_extract(metadata_json, '$.scope') = ? 
            AND is_active = 1
        """
        params = [scope]

        if user_id:
            query += " AND json_extract(metadata_json, '$.user_id') = ?"
            params.append(user_id)

        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type.value)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        memories = []
        async with self._connection.execute(query, params) as cursor:
            async for row in cursor:
                memories.append(self._row_to_memory(row))

        return memories

    async def query_by_time_range(
        self,
        start: datetime,
        end: datetime,
        memory_type: MemoryType | None = None,
        limit: int = 100,
    ) -> list[BaseMemory]:
        """Query memories by creation time range."""
        self._ensure_connected()

        query = """
            SELECT * FROM memories 
            WHERE created_at BETWEEN ? AND ? 
            AND is_active = 1
        """
        params = [_serialize_datetime(start), _serialize_datetime(end)]

        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type.value)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        memories = []
        async with self._connection.execute(query, params) as cursor:
            async for row in cursor:
                memories.append(self._row_to_memory(row))

        return memories

    async def query_by_strength(
        self,
        min_strength: float = 0.0,
        max_strength: float = 1.0,
        memory_type: MemoryType | None = None,
        limit: int = 100,
    ) -> list[BaseMemory]:
        """Query memories by strength range."""
        self._ensure_connected()

        query = """
            SELECT * FROM memories 
            WHERE current_strength BETWEEN ? AND ? 
            AND is_active = 1
        """
        params = [min_strength, max_strength]

        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type.value)

        query += " ORDER BY current_strength DESC LIMIT ?"
        params.append(limit)

        memories = []
        async with self._connection.execute(query, params) as cursor:
            async for row in cursor:
                memories.append(self._row_to_memory(row))

        return memories

    async def query_weak_memories(
        self,
        threshold: float = 0.1,
        memory_type: MemoryType | None = None,
    ) -> list[BaseMemory]:
        """Query memories below strength threshold."""
        self._ensure_connected()

        query = """
            SELECT * FROM memories 
            WHERE current_strength < ? 
            AND is_active = 1
        """
        params = [threshold]

        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type.value)

        query += " ORDER BY current_strength ASC"

        memories = []
        async with self._connection.execute(query, params) as cursor:
            async for row in cursor:
                memories.append(self._row_to_memory(row))

        return memories

    async def query_unconsolidated(
        self,
        memory_type: MemoryType,
        min_importance: float = 0.0,
        limit: int = 100,
    ) -> list[BaseMemory]:
        """Query memories that haven't been consolidated."""
        self._ensure_connected()

        query = """
            SELECT * FROM memories 
            WHERE memory_type = ? 
            AND is_consolidated = 0 
            AND is_active = 1
            ORDER BY created_at ASC
            LIMIT ?
        """

        memories = []
        async with self._connection.execute(
            query, (memory_type.value, limit)
        ) as cursor:
            async for row in cursor:
                memory = self._row_to_memory(row)
                if memory.importance_score >= min_importance:
                    memories.append(memory)

        return memories

    # Statistics
    async def count(self, memory_type: MemoryType | None = None) -> int:
        """Count memories in storage."""
        self._ensure_connected()

        if memory_type:
            query = "SELECT COUNT(*) FROM memories WHERE memory_type = ? AND is_active = 1"
            params = (memory_type.value,)
        else:
            query = "SELECT COUNT(*) FROM memories WHERE is_active = 1"
            params = ()

        async with self._connection.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        self._ensure_connected()

        stats = {
            "total_memories": await self.count(),
            "by_type": {},
            "by_strength": {},
        }

        # Count by type
        for mem_type in MemoryType:
            stats["by_type"][mem_type.value] = await self.count(mem_type)

        # Count by strength ranges
        ranges = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
        for min_s, max_s in ranges:
            query = """
                SELECT COUNT(*) FROM memories 
                WHERE current_strength >= ? AND current_strength < ? AND is_active = 1
            """
            async with self._connection.execute(query, (min_s, max_s)) as cursor:
                row = await cursor.fetchone()
                stats["by_strength"][f"{min_s}-{max_s}"] = row[0] if row else 0

        # DB file size
        if self.db_path.exists():
            stats["db_size_bytes"] = self.db_path.stat().st_size

        return stats
