"""
Memory API for programmatic access to the memory system.

Provides a clean interface for:
- Storing memories
- Retrieving memories
- Managing memory lifecycle
- Querying and searching
"""

from datetime import datetime
from typing import Any, Callable, Awaitable
from enum import Enum

from pydantic import BaseModel, Field

from llm_memory.config import MemoryConfig
from llm_memory.models.base import BaseMemory, MemoryType, MemoryMetadata, MemorySource
from llm_memory.models.short_term import ShortTermMemory, STMRole
from llm_memory.models.episodic import EpisodicMemory, Episode, EventType
from llm_memory.models.semantic import SemanticMemory, Fact, FactType
from llm_memory.retrieval.intent import IntentClassifier, ClassifiedIntent
from llm_memory.retrieval.searcher import SearchResults, KeywordSearcher, SearchConfig
from llm_memory.retrieval.ranker import ResultRanker, RankedResults, RankingConfig


def _utcnow() -> datetime:
    """Get current UTC time (naive)."""
    return datetime.utcnow()


class MemoryOperation(str, Enum):
    """Types of memory operations."""

    STORE = "store"
    RETRIEVE = "retrieve"
    UPDATE = "update"
    DELETE = "delete"
    SEARCH = "search"
    CONSOLIDATE = "consolidate"


class OperationResult(BaseModel):
    """Result of a memory operation."""

    success: bool
    operation: MemoryOperation
    memory_id: str | None = None
    memory_type: MemoryType | None = None
    message: str | None = None
    data: dict | None = None
    timestamp: datetime = Field(default_factory=_utcnow)


class StoreRequest(BaseModel):
    """Request to store a memory."""

    content: str
    memory_type: MemoryType = MemoryType.SEMANTIC
    summary: str | None = None
    user_id: str | None = None
    scope: str = "global"
    tags: list[str] = Field(default_factory=list)
    source: MemorySource = MemorySource.USER_INPUT
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Type-specific fields
    event_type: EventType | None = None  # For episodic
    fact_type: FactType | None = None  # For semantic
    role: STMRole | None = None  # For STM


class SearchRequest(BaseModel):
    """Request to search memories."""

    query: str
    memory_types: list[MemoryType] | None = None
    user_id: str | None = None
    scope: str | None = None
    limit: int = Field(default=10, ge=1, le=100)
    min_similarity: float = Field(default=0.3, ge=0.0, le=1.0)
    include_inactive: bool = False


class MemoryAPI:
    """
    High-level API for memory operations.
    
    Provides a simple interface for storing, retrieving,
    and searching memories across all tiers.
    """

    def __init__(self, config: MemoryConfig | None = None):
        self.config = config or MemoryConfig()
        
        # Components
        self._intent_classifier = IntentClassifier()
        self._searcher = KeywordSearcher(SearchConfig(
            max_results=self.config.retrieval.default_limit,
            min_similarity=0.3,
        ))
        self._ranker = ResultRanker()
        
        # In-memory storage (replaced by real storage in production)
        self._memories: dict[str, BaseMemory] = {}
        self._stm_sessions: dict[str, ShortTermMemory] = {}
        
        # Event hooks
        self._hooks: dict[str, list[Callable]] = {
            "pre_store": [],
            "post_store": [],
            "pre_retrieve": [],
            "post_retrieve": [],
            "pre_delete": [],
            "post_delete": [],
        }

    def register_hook(
        self,
        event: str,
        callback: Callable[[BaseMemory], Any],
    ) -> None:
        """Register a callback for memory events."""
        if event in self._hooks:
            self._hooks[event].append(callback)

    def _trigger_hooks(self, event: str, memory: BaseMemory) -> None:
        """Trigger all hooks for an event."""
        for callback in self._hooks.get(event, []):
            try:
                callback(memory)
            except Exception:
                pass  # Don't let hook errors break operations

    def store(self, request: StoreRequest) -> OperationResult:
        """
        Store a new memory.
        
        Args:
            request: Store request with content and metadata
            
        Returns:
            OperationResult with the new memory ID
        """
        # Create memory based on type
        memory = self._create_memory(request)
        
        # Trigger pre-store hooks
        self._trigger_hooks("pre_store", memory)
        
        # Store in memory
        self._memories[memory.id] = memory
        
        # Trigger post-store hooks
        self._trigger_hooks("post_store", memory)
        
        return OperationResult(
            success=True,
            operation=MemoryOperation.STORE,
            memory_id=memory.id,
            memory_type=memory.memory_type,
            message=f"Stored {memory.memory_type.value} memory",
        )

    def _create_memory(self, request: StoreRequest) -> BaseMemory:
        """Create memory object from request."""
        metadata = MemoryMetadata(
            user_id=request.user_id,
            scope=request.scope,
            source=request.source,
            tags=request.tags,
        )
        
        if request.memory_type == MemoryType.SHORT_TERM:
            memory = ShortTermMemory(
                content=request.content,
                summary=request.summary,
                metadata=metadata,
            )
            if request.role:
                memory.add_message(request.content, role=request.role)
        
        elif request.memory_type == MemoryType.EPISODIC:
            memory = EpisodicMemory(
                content=request.content,
                summary=request.summary,
                metadata=metadata,
            )
            if request.event_type:
                episode = Episode(
                    event_id=f"ep_{_utcnow().timestamp()}",
                    event_type=request.event_type,
                    description=request.content[:200],
                )
                memory.add_episode(episode)
        
        else:  # SEMANTIC
            memory = SemanticMemory(
                content=request.content,
                summary=request.summary,
                metadata=metadata,
            )
            if request.fact_type:
                fact = Fact(
                    fact_id=f"fact_{_utcnow().timestamp()}",
                    fact_type=request.fact_type,
                    subject="",
                    predicate="",
                    object="",
                    statement=request.content,
                )
                memory.add_fact(fact)
        
        # Set importance
        memory.importance.user_marked = request.importance
        
        return memory

    def get(self, memory_id: str) -> BaseMemory | None:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: The memory's unique identifier
            
        Returns:
            The memory if found, None otherwise
        """
        memory = self._memories.get(memory_id)
        
        if memory:
            self._trigger_hooks("pre_retrieve", memory)
            memory.mark_accessed()
            self._trigger_hooks("post_retrieve", memory)
        
        return memory

    def update(self, memory_id: str, **updates) -> OperationResult:
        """
        Update an existing memory.
        
        Args:
            memory_id: The memory to update
            **updates: Fields to update
            
        Returns:
            OperationResult
        """
        memory = self._memories.get(memory_id)
        
        if not memory:
            return OperationResult(
                success=False,
                operation=MemoryOperation.UPDATE,
                memory_id=memory_id,
                message="Memory not found",
            )
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(memory, key):
                setattr(memory, key, value)
        
        memory.updated_at = _utcnow()
        
        return OperationResult(
            success=True,
            operation=MemoryOperation.UPDATE,
            memory_id=memory_id,
            memory_type=memory.memory_type,
            message="Memory updated",
        )

    def delete(self, memory_id: str, soft: bool = True) -> OperationResult:
        """
        Delete a memory.
        
        Args:
            memory_id: The memory to delete
            soft: If True, mark as inactive; if False, remove completely
            
        Returns:
            OperationResult
        """
        memory = self._memories.get(memory_id)
        
        if not memory:
            return OperationResult(
                success=False,
                operation=MemoryOperation.DELETE,
                memory_id=memory_id,
                message="Memory not found",
            )
        
        self._trigger_hooks("pre_delete", memory)
        
        if soft:
            memory.is_active = False
        else:
            del self._memories[memory_id]
        
        self._trigger_hooks("post_delete", memory)
        
        return OperationResult(
            success=True,
            operation=MemoryOperation.DELETE,
            memory_id=memory_id,
            message="Memory deleted" if not soft else "Memory deactivated",
        )

    def search(self, request: SearchRequest) -> RankedResults:
        """
        Search memories.
        
        Args:
            request: Search request with query and filters
            
        Returns:
            RankedResults with matching memories
        """
        # Classify intent
        intent = self._intent_classifier.classify(request.query)
        
        # Filter memories
        memories = self._filter_memories(request)
        
        # Search
        search_results = self._searcher.search(
            request.query,
            memories,
            intent,
        )
        
        # Rank results
        ranked = self._ranker.rank(search_results)
        
        return ranked

    def _filter_memories(self, request: SearchRequest) -> list[BaseMemory]:
        """Filter memories based on request criteria."""
        memories = []
        
        for memory in self._memories.values():
            # Filter by active status
            if not request.include_inactive and not memory.is_active:
                continue
            
            # Filter by type
            if request.memory_types and memory.memory_type not in request.memory_types:
                continue
            
            # Filter by user
            if request.user_id and memory.metadata.user_id != request.user_id:
                continue
            
            # Filter by scope
            if request.scope and memory.metadata.scope != request.scope:
                continue
            
            memories.append(memory)
        
        return memories

    def add_to_stm(
        self,
        session_id: str,
        content: str,
        role: STMRole = STMRole.USER,
    ) -> OperationResult:
        """
        Add a message to short-term memory for a session.
        
        Args:
            session_id: The session identifier
            content: The message content
            role: The role (user, assistant, system)
            
        Returns:
            OperationResult
        """
        # Get or create STM for session
        if session_id not in self._stm_sessions:
            self._stm_sessions[session_id] = ShortTermMemory(
                content=f"Session {session_id}",
                session_id=session_id,
            )
        
        stm = self._stm_sessions[session_id]
        stm.add_message(content, role=role)
        
        # Also store in main memory store
        self._memories[stm.id] = stm
        
        return OperationResult(
            success=True,
            operation=MemoryOperation.STORE,
            memory_id=stm.id,
            memory_type=MemoryType.SHORT_TERM,
            message=f"Added {role.value} message to STM",
            data={"session_id": session_id, "message_count": len(stm.items)},
        )

    def get_stm(self, session_id: str) -> ShortTermMemory | None:
        """Get the STM for a session."""
        return self._stm_sessions.get(session_id)

    def clear_stm(self, session_id: str) -> OperationResult:
        """Clear the STM for a session."""
        if session_id in self._stm_sessions:
            stm = self._stm_sessions[session_id]
            stm.clear()
            return OperationResult(
                success=True,
                operation=MemoryOperation.DELETE,
                memory_id=stm.id,
                message="STM cleared",
            )
        
        return OperationResult(
            success=False,
            operation=MemoryOperation.DELETE,
            message="Session not found",
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get memory system statistics."""
        stats = {
            "total_memories": len(self._memories),
            "active_memories": sum(1 for m in self._memories.values() if m.is_active),
            "by_type": {},
            "active_sessions": len(self._stm_sessions),
        }
        
        for mem_type in MemoryType:
            count = sum(
                1 for m in self._memories.values()
                if m.memory_type == mem_type
            )
            stats["by_type"][mem_type.value] = count
        
        return stats

    def list_memories(
        self,
        memory_type: MemoryType | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[BaseMemory]:
        """
        List memories with pagination.
        
        Args:
            memory_type: Filter by type
            limit: Maximum memories to return
            offset: Starting offset
            
        Returns:
            List of memories
        """
        memories = list(self._memories.values())
        
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]
        
        # Sort by creation time (newest first)
        memories.sort(key=lambda m: m.created_at, reverse=True)
        
        return memories[offset:offset + limit]
