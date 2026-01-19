"""
Consolidation scheduler for background memory maintenance.

Handles:
- Periodic consolidation runs
- Memory decay updates
- Garbage collection
- Pattern detection
"""

import asyncio
from datetime import datetime
from typing import Any, Callable, Awaitable
from enum import Enum

from pydantic import BaseModel, Field

from llm_memory.config import ConsolidationConfig, DecayConfig
from llm_memory.models.base import BaseMemory, MemoryType
from llm_memory.models.short_term import ShortTermMemory
from llm_memory.models.episodic import EpisodicMemory
from llm_memory.models.semantic import SemanticMemory
from llm_memory.consolidation.promoter import MemoryPromoter, PromotionResult
from llm_memory.consolidation.merger import (
    MemoryMerger,
    GarbageCollector,
    MemoryDeduplicator,
    GarbageCollectionResult,
)


def _utcnow() -> datetime:
    """Get current UTC time (naive, for compatibility with models)."""
    return datetime.utcnow()


class ConsolidationPhase(str, Enum):
    """Phases of consolidation."""

    DECAY_UPDATE = "decay_update"
    STM_PROMOTION = "stm_promotion"
    EPISODIC_PROMOTION = "episodic_promotion"
    GARBAGE_COLLECTION = "garbage_collection"
    DEDUPLICATION = "deduplication"


class ConsolidationRunResult(BaseModel):
    """Result of a consolidation run."""

    run_id: str
    started_at: datetime
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    
    # Phase results
    stm_promotions: int = 0
    episodic_promotions: int = 0
    memories_collected: int = 0
    duplicates_merged: int = 0
    decay_updates: int = 0
    
    # Errors
    errors: list[str] = Field(default_factory=list)
    
    # Status
    success: bool = True


class ConsolidationScheduler:
    """
    Background scheduler for memory consolidation.
    
    Runs periodic consolidation tasks:
    1. Update memory decay
    2. Promote eligible STM to episodic
    3. Promote episodic patterns to semantic
    4. Run garbage collection
    5. Deduplicate memories
    """

    def __init__(
        self,
        consolidation_config: ConsolidationConfig | None = None,
        decay_config: DecayConfig | None = None,
    ):
        self.consolidation_config = consolidation_config or ConsolidationConfig()
        self.decay_config = decay_config or DecayConfig()
        
        # Components
        self.promoter = MemoryPromoter(self.consolidation_config)
        self.merger = MemoryMerger()
        self.garbage_collector = GarbageCollector(self.decay_config)
        self.deduplicator = MemoryDeduplicator()
        
        # State
        self._running = False
        self._task: asyncio.Task | None = None
        self._run_count = 0
        self._last_run: ConsolidationRunResult | None = None
        
        # Callbacks for memory access (set by memory system)
        self._get_stm_memories: Callable[[], Awaitable[list[ShortTermMemory]]] | None = None
        self._get_episodic_memories: Callable[[], Awaitable[list[EpisodicMemory]]] | None = None
        self._get_semantic_memories: Callable[[], Awaitable[list[SemanticMemory]]] | None = None
        self._save_memory: Callable[[BaseMemory], Awaitable[None]] | None = None
        self._delete_memory: Callable[[str], Awaitable[None]] | None = None

    def set_memory_callbacks(
        self,
        get_stm: Callable[[], Awaitable[list[ShortTermMemory]]],
        get_episodic: Callable[[], Awaitable[list[EpisodicMemory]]],
        get_semantic: Callable[[], Awaitable[list[SemanticMemory]]],
        save_memory: Callable[[BaseMemory], Awaitable[None]],
        delete_memory: Callable[[str], Awaitable[None]],
    ) -> None:
        """Set callbacks for memory access."""
        self._get_stm_memories = get_stm
        self._get_episodic_memories = get_episodic
        self._get_semantic_memories = get_semantic
        self._save_memory = save_memory
        self._delete_memory = delete_memory

    async def start(self) -> None:
        """Start the background consolidation scheduler."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the background consolidation scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run_loop(self) -> None:
        """Main consolidation loop."""
        while self._running:
            try:
                await self.run_consolidation()
            except Exception as e:
                # Log error but continue
                print(f"Consolidation error: {e}")
            
            await asyncio.sleep(self.consolidation_config.consolidation_interval_seconds)

    async def run_consolidation(self) -> ConsolidationRunResult:
        """
        Run a single consolidation cycle.
        
        This can be called manually or by the background scheduler.
        """
        self._run_count += 1
        run_id = f"run_{self._run_count}_{_utcnow().timestamp()}"
        started_at = _utcnow()
        
        result = ConsolidationRunResult(
            run_id=run_id,
            started_at=started_at,
        )

        try:
            # Phase 1: Update decay for all memories
            result.decay_updates = await self._update_decay()

            # Phase 2: Promote eligible STM
            result.stm_promotions = await self._promote_stm()

            # Phase 3: Promote episodic patterns
            result.episodic_promotions = await self._promote_episodic()

            # Phase 4: Garbage collection
            result.memories_collected = await self._run_garbage_collection()

            # Phase 5: Deduplication (less frequent)
            if self._run_count % 5 == 0:  # Every 5th run
                result.duplicates_merged = await self._run_deduplication()

        except Exception as e:
            result.success = False
            result.errors.append(str(e))

        result.completed_at = _utcnow()
        result.duration_seconds = (result.completed_at - started_at).total_seconds()
        
        self._last_run = result
        return result

    async def _update_decay(self) -> int:
        """Update decay for all memories."""
        count = 0
        
        if self._get_stm_memories:
            stm_list = await self._get_stm_memories()
            for stm in stm_list:
                stm.update_strength(self.decay_config.stm_decay_rate)
                count += 1
        
        if self._get_episodic_memories:
            ep_list = await self._get_episodic_memories()
            for ep in ep_list:
                ep.update_strength(self.decay_config.episodic_decay_rate)
                count += 1
        
        if self._get_semantic_memories:
            sem_list = await self._get_semantic_memories()
            for sem in sem_list:
                sem.update_strength(self.decay_config.semantic_decay_rate)
                count += 1
        
        return count

    async def _promote_stm(self) -> int:
        """Promote eligible STM to episodic."""
        if not self._get_stm_memories or not self._save_memory:
            return 0
        
        promoted = 0
        stm_list = await self._get_stm_memories()
        
        for stm in stm_list:
            if stm.is_consolidated:
                continue
            
            result, episodic = self.promoter.promote_stm(stm)
            if result.success and episodic:
                await self._save_memory(episodic)
                promoted += 1
        
        return promoted

    async def _promote_episodic(self) -> int:
        """Promote episodic patterns to semantic."""
        if not self._get_episodic_memories or not self._save_memory:
            return 0
        
        promoted = 0
        ep_list = await self._get_episodic_memories()
        
        # Find promotion candidates
        groups = self.promoter.find_episodic_patterns(ep_list)
        
        for group in groups:
            result, semantic = self.promoter.promote_episodes(group)
            if result.success and semantic:
                await self._save_memory(semantic)
                promoted += 1
        
        return promoted

    async def _run_garbage_collection(self) -> int:
        """Run garbage collection on weak memories."""
        if not self._delete_memory:
            return 0
        
        collected = 0
        
        # Collect from each memory type
        for get_fn in [
            self._get_stm_memories,
            self._get_episodic_memories,
            self._get_semantic_memories,
        ]:
            if not get_fn:
                continue
            
            memories = await get_fn()
            result = self.garbage_collector.collect(memories)
            
            for mem_id in result.collected_ids:
                await self._delete_memory(mem_id)
                collected += 1
        
        return collected

    async def _run_deduplication(self) -> int:
        """Run deduplication on memories."""
        if not self._get_semantic_memories or not self._save_memory or not self._delete_memory:
            return 0
        
        merged = 0
        sem_list = await self._get_semantic_memories()
        
        # Find duplicate groups
        duplicate_groups = self.deduplicator.find_duplicates(sem_list)
        
        for group in duplicate_groups:
            # Merge the group
            result_mem, merge_result = self.merger.merge_semantic_memories(group)
            
            if merge_result.success and result_mem:
                # Save merged memory
                await self._save_memory(result_mem)
                
                # Delete original memories (except first which was used as base)
                for mem in group[1:]:
                    await self._delete_memory(mem.id)
                    merged += 1
        
        return merged

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    @property
    def last_run(self) -> ConsolidationRunResult | None:
        """Get result of last consolidation run."""
        return self._last_run


class ManualConsolidator:
    """
    Manual consolidation without background scheduler.
    
    Useful for:
    - Testing
    - On-demand consolidation
    - Single-run consolidation
    """

    def __init__(
        self,
        consolidation_config: ConsolidationConfig | None = None,
        decay_config: DecayConfig | None = None,
    ):
        self.config = consolidation_config or ConsolidationConfig()
        self.decay_config = decay_config or DecayConfig()
        
        self.promoter = MemoryPromoter(self.config)
        self.merger = MemoryMerger()
        self.garbage_collector = GarbageCollector(self.decay_config)
        self.deduplicator = MemoryDeduplicator()

    def process_stm(
        self,
        stm: ShortTermMemory,
        force: bool = False,
    ) -> tuple[PromotionResult, EpisodicMemory | None]:
        """
        Process a single STM for promotion.
        
        Returns the promotion result and new episodic memory if created.
        """
        return self.promoter.promote_stm(stm, force)

    def process_episodes(
        self,
        episodes: list[EpisodicMemory],
        pattern_description: str | None = None,
    ) -> tuple[PromotionResult, SemanticMemory | None]:
        """
        Process episodes for semantic promotion.
        
        Returns the promotion result and new semantic memory if created.
        """
        return self.promoter.promote_episodes(episodes, pattern_description)

    def find_promotion_candidates(
        self,
        episodes: list[EpisodicMemory],
    ) -> list[list[EpisodicMemory]]:
        """Find episode groups ready for promotion."""
        return self.promoter.find_episodic_patterns(episodes)

    def collect_garbage(
        self,
        memories: list[BaseMemory],
    ) -> GarbageCollectionResult:
        """Run garbage collection on memories."""
        return self.garbage_collector.collect(memories)

    def find_duplicates(
        self,
        memories: list[BaseMemory],
    ) -> list[list[BaseMemory]]:
        """Find duplicate memory groups."""
        return self.deduplicator.find_duplicates(memories)

    def merge_memories(
        self,
        memories: list[SemanticMemory],
    ) -> tuple[SemanticMemory | None, Any]:
        """Merge semantic memories."""
        return self.merger.merge_semantic_memories(memories)
