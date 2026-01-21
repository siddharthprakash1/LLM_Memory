"""
Memory System Orchestrator.

The main entry point for the LLM Memory system.
Ties together all components into a unified interface.
"""

import asyncio
from datetime import datetime
from typing import Any, Callable, Awaitable

from pydantic import BaseModel, Field

from llm_memory.config import MemoryConfig
from llm_memory.models.base import BaseMemory, MemoryType, MemorySource, MemoryMetadata
from llm_memory.models.short_term import ShortTermMemory, STMRole
from llm_memory.models.episodic import EpisodicMemory, Episode, EventType
from llm_memory.models.semantic import SemanticMemory, Fact, FactType

from llm_memory.encoding.embedder import create_embedder, BaseEmbedder
from llm_memory.encoding.summarizer import create_summarizer, BaseSummarizer
from llm_memory.decay.functions import MemoryDecayCalculator
from llm_memory.decay.importance import ImportanceScorer

from llm_memory.consolidation.promoter import MemoryPromoter
from llm_memory.consolidation.merger import MemoryMerger, GarbageCollector
from llm_memory.consolidation.scheduler import ConsolidationScheduler

from llm_memory.conflict.detector import ConflictDetector
from llm_memory.conflict.resolver import ConflictResolver

from llm_memory.retrieval.intent import IntentClassifier
from llm_memory.retrieval.searcher import KeywordSearcher, SearchConfig
from llm_memory.retrieval.ranker import ResultRanker, RankedResults

from llm_memory.storage.sqlite import SQLiteStorage

from llm_memory.api.hooks import (
    HookRegistry, HookEvent, HookContext,
    trigger_memory_created, trigger_memory_accessed,
)


def _utcnow() -> datetime:
    """Get current UTC time (naive)."""
    return datetime.utcnow()


class MemorySystemConfig(BaseModel):
    """Configuration for the memory system."""

    # Enable/disable components
    enable_embeddings: bool = True
    enable_summarization: bool = True
    enable_consolidation: bool = True
    enable_conflict_resolution: bool = True
    
    # Background tasks
    auto_consolidate: bool = False
    consolidation_interval_seconds: int = 300
    
    # Storage
    use_persistent_storage: bool = False
    storage_path: str = "./data/memory.db"


class MemorySystem:
    """
    The main Memory System orchestrator.
    
    Provides a unified interface for all memory operations:
    - Storing memories across tiers
    - Intelligent retrieval with intent classification
    - Automatic consolidation
    - Conflict resolution
    - Memory decay management
    
    Usage:
        system = MemorySystem()
        await system.initialize()
        
        # Store a memory
        memory = await system.remember("User prefers Python")
        
        # Recall memories
        results = await system.recall("What language does user prefer?")
        
        # Add to conversation
        await system.add_message("Hello!", role="user", session_id="session_1")
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        system_config: MemorySystemConfig | None = None,
    ):
        self.config = config or MemoryConfig()
        self.system_config = system_config or MemorySystemConfig()
        
        # Core components
        self._embedder: BaseEmbedder | None = None
        self._summarizer: BaseSummarizer | None = None
        self._decay_calculator = MemoryDecayCalculator()
        self._importance_scorer = ImportanceScorer()
        
        # Memory management
        self._promoter = MemoryPromoter()
        self._merger = MemoryMerger()
        self._garbage_collector = GarbageCollector()
        self._consolidation_scheduler: ConsolidationScheduler | None = None
        
        # Conflict handling
        self._conflict_detector = ConflictDetector()
        self._conflict_resolver = ConflictResolver()
        
        # Retrieval
        self._intent_classifier = IntentClassifier()
        self._searcher = KeywordSearcher(SearchConfig(
            max_results=self.config.retrieval.default_limit,
        ))
        self._ranker = ResultRanker()
        
        # Storage
        self._storage: SQLiteStorage | None = None
        
        # In-memory caches
        self._memories: dict[str, BaseMemory] = {}
        self._stm_sessions: dict[str, ShortTermMemory] = {}
        
        # Hooks
        self._hooks = HookRegistry()
        
        # State
        self._initialized = False
        self._running = False

    async def initialize(self) -> None:
        """Initialize the memory system."""
        if self._initialized:
            return
        
        # Initialize embedder
        if self.system_config.enable_embeddings:
            self._embedder = create_embedder(self.config.embedding)
        
        # Initialize summarizer
        if self.system_config.enable_summarization:
            self._summarizer = create_summarizer(self.config.llm)
        
        # Initialize storage
        if self.system_config.use_persistent_storage:
            self._storage = SQLiteStorage(self.system_config.storage_path)
            await self._storage.initialize()
        
        # Initialize consolidation scheduler
        if self.system_config.enable_consolidation:
            self._consolidation_scheduler = ConsolidationScheduler(
                self.config.consolidation,
                self.config.decay,
            )
        
        self._initialized = True

    async def start(self) -> None:
        """Start background processes."""
        if not self._initialized:
            await self.initialize()
        
        if self._running:
            return
        
        self._running = True
        
        # Start consolidation scheduler
        if (
            self.system_config.auto_consolidate
            and self._consolidation_scheduler
        ):
            await self._consolidation_scheduler.start()

    async def stop(self) -> None:
        """Stop background processes."""
        self._running = False
        
        if self._consolidation_scheduler:
            await self._consolidation_scheduler.stop()

    async def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        user_id: str | None = None,
        scope: str = "global",
        tags: list[str] | None = None,
        source: MemorySource = MemorySource.USER_INPUT,
        **kwargs,
    ) -> BaseMemory:
        """
        Store a new memory.
        
        This is the primary method for adding memories to the system.
        
        Args:
            content: The memory content
            memory_type: Type of memory (semantic, episodic, short_term)
            user_id: User identifier for scoping
            scope: Memory scope (global, project, session)
            tags: Optional tags for categorization
            source: Source of the memory
            **kwargs: Additional type-specific arguments
            
        Returns:
            The created memory object
        """
        # Create metadata
        metadata = MemoryMetadata(
            user_id=user_id,
            scope=scope,
            source=source,
            tags=tags or [],
        )
        
        # Create memory based on type
        if memory_type == MemoryType.SHORT_TERM:
            memory = ShortTermMemory(content=content, metadata=metadata)
        elif memory_type == MemoryType.EPISODIC:
            memory = EpisodicMemory(content=content, metadata=metadata)
            if "event_type" in kwargs:
                episode = Episode(
                    event_id=f"ep_{_utcnow().timestamp()}",
                    event_type=kwargs["event_type"],
                    description=content[:200],
                )
                memory.add_episode(episode)
        else:
            memory = SemanticMemory(content=content, metadata=metadata)
            if "fact_type" in kwargs:
                fact = Fact(
                    fact_id=f"fact_{_utcnow().timestamp()}",
                    fact_type=kwargs["fact_type"],
                    subject=kwargs.get("subject", ""),
                    predicate=kwargs.get("predicate", ""),
                    object=kwargs.get("object", ""),
                    statement=content,
                )
                memory.add_fact(fact)
        
        # Calculate importance
        if self.system_config.enable_summarization:
            importance = self._importance_scorer.calculate_importance(memory)
            memory.importance = importance
        
        # Generate embedding
        if self._embedder and self.system_config.enable_embeddings:
            try:
                embedding = await self._embedder.embed(content)
                memory.embedding = embedding
            except Exception:
                pass  # Continue without embedding
        
        # Generate summary
        if self._summarizer and len(content) > 200:
            try:
                result = await self._summarizer.summarize(content)
                memory.summary = result.summary
            except Exception:
                pass  # Continue without summary
        
        # Check for conflicts
        if self.system_config.enable_conflict_resolution:
            existing = list(self._memories.values())
            conflicts = self._conflict_resolver.check_new_memory(memory, existing)
            if conflicts:
                # Auto-resolve low severity conflicts
                for conflict in conflicts:
                    if conflict.auto_resolvable:
                        self._conflict_resolver.resolve_conflict(
                            conflict,
                            memory,
                            self._memories.get(conflict.memory_b_id),
                        )
        
        # Store in memory
        self._memories[memory.id] = memory
        
        # Persist if storage enabled
        if self._storage:
            await self._storage.store(memory)
        
        # Trigger hooks
        trigger_memory_created(memory)
        
        return memory

    async def recall(
        self,
        query: str,
        memory_types: list[MemoryType] | None = None,
        user_id: str | None = None,
        limit: int = 10,
    ) -> RankedResults:
        """
        Recall relevant memories.
        
        Uses intent classification to optimize retrieval.
        
        Args:
            query: The search query
            memory_types: Filter by memory types
            user_id: Filter by user
            limit: Maximum results
            
        Returns:
            RankedResults with relevant memories
        """
        # Classify intent
        intent = self._intent_classifier.classify(query)
        
        # Filter memories
        memories = []
        for mem in self._memories.values():
            if not mem.is_active:
                continue
            if memory_types and mem.memory_type not in memory_types:
                continue
            if user_id and mem.metadata.user_id != user_id:
                continue
            memories.append(mem)
        
        # Search
        search_results = self._searcher.search(query, memories, intent)
        
        # Rank
        ranked = self._ranker.rank(search_results)
        
        # Mark accessed
        for result in ranked.ranked_results[:limit]:
            mem = self._memories.get(result.result.memory_id)
            if mem:
                mem.mark_accessed()
                trigger_memory_accessed(mem)
        
        return ranked

    async def add_message(
        self,
        content: str,
        role: str = "user",
        session_id: str = "default",
    ) -> ShortTermMemory:
        """
        Add a message to short-term memory.
        
        Used for conversation tracking.
        
        Args:
            content: Message content
            role: Role (user, assistant, system)
            session_id: Session identifier
            
        Returns:
            The updated STM
        """
        # Get or create STM for session
        if session_id not in self._stm_sessions:
            self._stm_sessions[session_id] = ShortTermMemory(
                content=f"Session {session_id}",
                session_id=session_id,
            )
            self._memories[self._stm_sessions[session_id].id] = self._stm_sessions[session_id]
        
        stm = self._stm_sessions[session_id]
        
        # Map role string to enum
        role_map = {
            "user": STMRole.USER,
            "assistant": STMRole.ASSISTANT,
            "system": STMRole.SYSTEM,
        }
        stm_role = role_map.get(role.lower(), STMRole.USER)
        
        # Add message
        stm.add_message(content, role=stm_role)
        
        return stm

    async def get_context(
        self,
        session_id: str = "default",
        include_relevant: bool = True,
        query: str | None = None,
    ) -> dict[str, Any]:
        """
        Get context for a conversation.
        
        Returns STM history and optionally relevant long-term memories.
        
        Args:
            session_id: Session identifier
            include_relevant: Whether to include relevant memories
            query: Query for finding relevant memories
            
        Returns:
            Dictionary with history and context
        """
        result = {
            "history": [],
            "relevant_memories": [],
        }
        
        # Get STM history
        stm = self._stm_sessions.get(session_id)
        if stm:
            result["history"] = [
                {"role": item.role.value, "content": item.content}
                for item in stm.items
            ]
        
        # Get relevant memories
        if include_relevant and query:
            recall_results = await self.recall(
                query,
                memory_types=[MemoryType.SEMANTIC, MemoryType.EPISODIC],
                limit=5,
            )
            result["relevant_memories"] = [
                {
                    "content": r.result.content,
                    "type": r.result.memory_type.value,
                    "score": r.ranking_score,
                }
                for r in recall_results.ranked_results
            ]
        
        return result

    async def consolidate(self) -> dict[str, int]:
        """
        Manually trigger consolidation.
        
        Returns:
            Statistics about the consolidation
        """
        stats = {
            "stm_promoted": 0,
            "episodes_promoted": 0,
            "memories_merged": 0,
            "garbage_collected": 0,
        }
        
        # Promote eligible STMs
        for stm in list(self._stm_sessions.values()):
            if stm.is_consolidated:
                continue
            
            result, episodic = self._promoter.promote_stm(stm)
            if result.success and episodic:
                self._memories[episodic.id] = episodic
                stats["stm_promoted"] += 1
        
        # Run garbage collection
        all_memories = list(self._memories.values())
        gc_result = self._garbage_collector.collect(all_memories)
        stats["garbage_collected"] = gc_result.collected_count
        
        return stats

    def get_statistics(self) -> dict[str, Any]:
        """Get system statistics."""
        return {
            "total_memories": len(self._memories),
            "active_sessions": len(self._stm_sessions),
            "by_type": {
                mt.value: sum(1 for m in self._memories.values() if m.memory_type == mt)
                for mt in MemoryType
            },
            "pending_conflicts": self._conflict_resolver.pending_count,
            "initialized": self._initialized,
            "running": self._running,
        }

    def register_hook(self, event: HookEvent, callback: Callable) -> None:
        """Register an event hook."""
        self._hooks.register(event, callback)

    async def forget(self, memory_id: str, hard: bool = False) -> bool:
        """
        Forget (delete) a memory.
        
        Args:
            memory_id: Memory to delete
            hard: If True, remove completely; if False, mark inactive
            
        Returns:
            True if successful
        """
        memory = self._memories.get(memory_id)
        if memory is None:
            return False
        
        if hard:
            del self._memories[memory_id]
            if self._storage:
                await self._storage.delete(memory_id)
        else:
            memory.is_active = False
            if self._storage:
                await self._storage.update(memory)
        
        return True

    async def close(self) -> None:
        """Clean up resources."""
        await self.stop()
        
        if self._storage:
            await self._storage.close()
