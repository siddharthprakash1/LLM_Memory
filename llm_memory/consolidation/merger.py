"""
Memory merging and garbage collection.

Handles:
- Merging similar memories
- Garbage collection of weak memories
- Memory deduplication
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from llm_memory.config import DecayConfig
from llm_memory.models.base import BaseMemory, MemoryType, MemoryMetadata, MemorySource
from llm_memory.models.episodic import EpisodicMemory, Episode
from llm_memory.models.semantic import SemanticMemory, Fact, FactType


def _utcnow() -> datetime:
    """Get current UTC time (naive, for compatibility with models)."""
    return datetime.utcnow()


class MergeResult(BaseModel):
    """Result of a merge operation."""

    success: bool
    merged_ids: list[str]
    result_id: str | None = None
    reason: str | None = None
    merged_at: datetime = Field(default_factory=_utcnow)


class GarbageCollectionResult(BaseModel):
    """Result of garbage collection."""

    collected_count: int
    collected_ids: list[str]
    total_checked: int
    collection_time: datetime = Field(default_factory=_utcnow)


class MemoryMerger:
    """
    Merges similar memories to reduce redundancy.
    
    Handles:
    - Merging duplicate facts
    - Combining similar episodes
    - Strengthening existing memories instead of creating new ones
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def merge_semantic_memories(
        self,
        memories: list[SemanticMemory],
    ) -> tuple[SemanticMemory, MergeResult]:
        """
        Merge multiple semantic memories into one.
        
        Combines facts, increases confidence, and preserves provenance.
        """
        if not memories:
            return None, MergeResult(
                success=False,
                merged_ids=[],
                reason="No memories to merge",
            )

        if len(memories) == 1:
            return memories[0], MergeResult(
                success=True,
                merged_ids=[memories[0].id],
                result_id=memories[0].id,
                reason="Single memory, no merge needed",
            )

        # Use first memory as base
        base = memories[0]
        
        # Collect all facts
        all_facts: dict[str, Fact] = {}
        for fact in base.facts:
            all_facts[self._fact_key(fact)] = fact

        # Merge facts from other memories
        source_ids = [base.id]
        episodic_ids = list(base.derived_from_episodic_ids)
        
        for mem in memories[1:]:
            source_ids.append(mem.id)
            episodic_ids.extend(mem.derived_from_episodic_ids)
            
            for fact in mem.facts:
                key = self._fact_key(fact)
                if key in all_facts:
                    # Strengthen existing fact
                    existing = all_facts[key]
                    existing.confidence = min(1.0, existing.confidence + 0.1)
                    existing.evidence_count += fact.evidence_count
                    existing.source_memory_ids.extend(fact.source_memory_ids)
                else:
                    # Add new fact
                    all_facts[key] = fact

        # Create merged memory
        merged = SemanticMemory(
            content=self._merge_content([m.content for m in memories]),
            summary=self._merge_summaries([m.summary for m in memories if m.summary]),
            facts=list(all_facts.values()),
            primary_fact_id=base.primary_fact_id,
            derived_from_episodic_ids=list(set(episodic_ids)),
            episode_count=sum(m.episode_count for m in memories),
            overall_confidence=min(1.0, base.overall_confidence + len(memories) * 0.05),
            metadata=MemoryMetadata(
                user_id=base.metadata.user_id,
                scope=base.metadata.scope,
                source=MemorySource.CONSOLIDATION,
                source_ids=source_ids,
            ),
            importance=base.importance,
        )

        return merged, MergeResult(
            success=True,
            merged_ids=[m.id for m in memories],
            result_id=merged.id,
            reason=f"Merged {len(memories)} memories into one",
        )

    def merge_episodic_memories(
        self,
        memories: list[EpisodicMemory],
    ) -> tuple[EpisodicMemory, MergeResult]:
        """
        Merge multiple episodic memories into one.
        
        Combines episodes and updates temporal span.
        """
        if not memories:
            return None, MergeResult(
                success=False,
                merged_ids=[],
                reason="No memories to merge",
            )

        if len(memories) == 1:
            return memories[0], MergeResult(
                success=True,
                merged_ids=[memories[0].id],
                result_id=memories[0].id,
                reason="Single memory, no merge needed",
            )

        # Use first memory as base
        base = memories[0]
        
        # Collect all episodes
        all_episodes = list(base.episodes)
        source_ids = [base.id]
        lessons = list(base.lessons_learned)
        
        for mem in memories[1:]:
            source_ids.append(mem.id)
            all_episodes.extend(mem.episodes)
            lessons.extend(mem.lessons_learned)

        # Create merged memory
        merged = EpisodicMemory(
            content=self._merge_content([m.content for m in memories]),
            summary=self._merge_summaries([m.summary for m in memories if m.summary]),
            episodes=all_episodes,
            primary_episode_id=base.primary_episode_id,
            lessons_learned=list(set(lessons)),
            metadata=MemoryMetadata(
                user_id=base.metadata.user_id,
                scope=base.metadata.scope,
                source=MemorySource.CONSOLIDATION,
                source_ids=source_ids,
            ),
            importance=base.importance,
        )

        # Update temporal span
        merged._update_temporal_span()

        return merged, MergeResult(
            success=True,
            merged_ids=[m.id for m in memories],
            result_id=merged.id,
            reason=f"Merged {len(memories)} episodic memories",
        )

    def should_merge(
        self,
        memory1: BaseMemory,
        memory2: BaseMemory,
        similarity_score: float | None = None,
    ) -> bool:
        """
        Determine if two memories should be merged.
        
        Args:
            memory1: First memory
            memory2: Second memory
            similarity_score: Pre-computed similarity (0-1)
            
        Returns:
            True if memories should be merged
        """
        # Must be same type
        if memory1.memory_type != memory2.memory_type:
            return False

        # Must be same user/scope
        if memory1.metadata.user_id != memory2.metadata.user_id:
            return False
        if memory1.metadata.scope != memory2.metadata.scope:
            return False

        # Check similarity
        if similarity_score is not None:
            return similarity_score >= self.similarity_threshold

        # Fallback to simple content comparison
        return self._simple_similarity(memory1, memory2) >= self.similarity_threshold

    def _fact_key(self, fact: Fact) -> str:
        """Generate key for fact deduplication."""
        return f"{fact.subject}:{fact.predicate}:{fact.object}".lower()

    def _merge_content(self, contents: list[str]) -> str:
        """Merge multiple content strings."""
        # Deduplicate and join
        unique = []
        seen = set()
        for content in contents:
            normalized = content.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(content.strip())
        
        return "\n\n".join(unique)

    def _merge_summaries(self, summaries: list[str]) -> str:
        """Merge multiple summaries."""
        if not summaries:
            return None
        if len(summaries) == 1:
            return summaries[0]
        
        # Use longest summary
        return max(summaries, key=len)

    def _simple_similarity(
        self,
        memory1: BaseMemory,
        memory2: BaseMemory,
    ) -> float:
        """Calculate simple keyword-based similarity."""
        words1 = set(memory1.content.lower().split())
        words2 = set(memory2.content.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


class GarbageCollector:
    """
    Garbage collection for weak memories.
    
    Removes or archives memories that have decayed below threshold.
    """

    def __init__(self, config: DecayConfig | None = None):
        self.config = config or DecayConfig()

    def find_collectible(
        self,
        memories: list[BaseMemory],
        threshold: float | None = None,
    ) -> list[BaseMemory]:
        """
        Find memories eligible for garbage collection.
        
        Args:
            memories: Memories to check
            threshold: Strength threshold (uses config default if None)
            
        Returns:
            List of memories below threshold
        """
        if threshold is None:
            threshold = self.config.min_strength_threshold

        collectible = []
        for memory in memories:
            # Update strength before checking
            memory.update_strength()
            
            if memory.current_strength < threshold:
                # Don't collect if user-marked as important
                if memory.importance.user_marked < 0.5:
                    collectible.append(memory)

        return collectible

    def collect(
        self,
        memories: list[BaseMemory],
        threshold: float | None = None,
    ) -> GarbageCollectionResult:
        """
        Mark memories for collection.
        
        Args:
            memories: Memories to check
            threshold: Strength threshold
            
        Returns:
            GarbageCollectionResult with collection stats
        """
        collectible = self.find_collectible(memories, threshold)
        
        # Mark as inactive
        for memory in collectible:
            memory.is_active = False

        return GarbageCollectionResult(
            collected_count=len(collectible),
            collected_ids=[m.id for m in collectible],
            total_checked=len(memories),
        )

    def should_collect(
        self,
        memory: BaseMemory,
        threshold: float | None = None,
    ) -> bool:
        """Check if single memory should be collected."""
        if threshold is None:
            threshold = self.config.min_strength_threshold

        memory.update_strength()
        
        if memory.current_strength >= threshold:
            return False
        
        # Protect user-marked memories
        if memory.importance.user_marked >= 0.5:
            return False
        
        return True


class MemoryDeduplicator:
    """
    Deduplication of memories.
    
    Finds and handles duplicate or near-duplicate memories.
    """

    def __init__(self, similarity_threshold: float = 0.9):
        self.similarity_threshold = similarity_threshold

    def find_duplicates(
        self,
        memories: list[BaseMemory],
        similarity_scores: dict[tuple[str, str], float] | None = None,
    ) -> list[list[BaseMemory]]:
        """
        Find groups of duplicate memories.
        
        Args:
            memories: Memories to check
            similarity_scores: Pre-computed similarity scores
            
        Returns:
            List of duplicate groups
        """
        if len(memories) < 2:
            return []

        # Group by type first
        by_type: dict[MemoryType, list[BaseMemory]] = {}
        for mem in memories:
            if mem.memory_type not in by_type:
                by_type[mem.memory_type] = []
            by_type[mem.memory_type].append(mem)

        duplicate_groups = []
        
        for mem_type, type_memories in by_type.items():
            # Find duplicates within type
            processed = set()
            
            for i, mem1 in enumerate(type_memories):
                if mem1.id in processed:
                    continue
                
                group = [mem1]
                processed.add(mem1.id)
                
                for mem2 in type_memories[i + 1:]:
                    if mem2.id in processed:
                        continue
                    
                    # Check similarity
                    if similarity_scores:
                        score = similarity_scores.get(
                            (mem1.id, mem2.id),
                            similarity_scores.get((mem2.id, mem1.id), 0)
                        )
                    else:
                        score = self._simple_similarity(mem1, mem2)
                    
                    if score >= self.similarity_threshold:
                        group.append(mem2)
                        processed.add(mem2.id)
                
                if len(group) > 1:
                    duplicate_groups.append(group)

        return duplicate_groups

    def _simple_similarity(
        self,
        memory1: BaseMemory,
        memory2: BaseMemory,
    ) -> float:
        """Calculate simple similarity between memories."""
        # Same content = definite duplicate
        if memory1.content.strip() == memory2.content.strip():
            return 1.0
        
        # Check keyword overlap
        words1 = set(memory1.content.lower().split())
        words2 = set(memory2.content.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0

    def select_best(self, duplicates: list[BaseMemory]) -> BaseMemory:
        """
        Select the best memory from a group of duplicates.
        
        Selection criteria:
        1. Highest importance
        2. Most recent
        3. Most accessed
        """
        if not duplicates:
            raise ValueError("No duplicates to select from")
        
        if len(duplicates) == 1:
            return duplicates[0]
        
        def score(mem: BaseMemory) -> tuple:
            return (
                mem.importance_score,
                mem.access_count,
                mem.created_at.timestamp(),
            )
        
        return max(duplicates, key=score)
