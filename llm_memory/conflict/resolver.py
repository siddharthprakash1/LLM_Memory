"""
Conflict resolution orchestrator.

Coordinates conflict detection and resolution:
- Runs detection on memory sets
- Selects appropriate resolution strategies
- Applies resolutions
- Tracks resolution history
"""

from datetime import datetime
from typing import Any, Callable, Awaitable

from pydantic import BaseModel, Field

from llm_memory.models.base import BaseMemory, MemoryType
from llm_memory.models.semantic import SemanticMemory
from llm_memory.conflict.detector import (
    ConflictDetector,
    LLMConflictDetector,
    DetectedConflict,
    ConflictType,
    ConflictSeverity,
)
from llm_memory.conflict.strategies import (
    ResolutionStrategy,
    ResolutionAction,
    ResolutionResult,
    BaseResolutionStrategy,
    get_strategy,
    get_default_strategy_for_conflict,
    RecencyStrategy,
    ConfidenceStrategy,
    MergeStrategy,
    UserGuidedStrategy,
)


def _utcnow() -> datetime:
    """Get current UTC time (naive, for compatibility with models)."""
    return datetime.utcnow()


class ConflictResolutionConfig(BaseModel):
    """Configuration for conflict resolution."""

    # Detection settings
    auto_detect: bool = Field(
        default=True,
        description="Automatically detect conflicts when new memories added",
    )
    detection_batch_size: int = Field(
        default=100,
        description="Max memories to check in single detection run",
    )
    
    # Resolution settings
    auto_resolve_low_severity: bool = Field(
        default=True,
        description="Automatically resolve low severity conflicts",
    )
    auto_resolve_medium_severity: bool = Field(
        default=False,
        description="Automatically resolve medium severity conflicts",
    )
    default_strategy: ResolutionStrategy = Field(
        default=ResolutionStrategy.RECENCY,
        description="Default resolution strategy",
    )
    
    # User interaction
    require_user_for_critical: bool = Field(
        default=True,
        description="Always require user for critical conflicts",
    )
    max_pending_conflicts: int = Field(
        default=50,
        description="Max unresolved conflicts before forcing resolution",
    )


class ConflictHistory(BaseModel):
    """Record of a resolved conflict."""

    conflict: DetectedConflict
    resolution: ResolutionResult
    resolved_at: datetime = Field(default_factory=_utcnow)
    applied: bool = False


class ConflictResolver:
    """
    Main conflict resolution orchestrator.
    
    Handles the full conflict lifecycle:
    1. Detection (via ConflictDetector)
    2. Strategy selection
    3. Resolution
    4. Application of resolution
    """

    def __init__(
        self,
        config: ConflictResolutionConfig | None = None,
        llm_client: Any = None,
    ):
        self.config = config or ConflictResolutionConfig()
        
        # Initialize detector
        if llm_client:
            self.detector = LLMConflictDetector(llm_client=llm_client)
        else:
            self.detector = ConflictDetector()
        
        # Pending conflicts awaiting resolution
        self._pending_conflicts: list[DetectedConflict] = []
        
        # Resolution history
        self._history: list[ConflictHistory] = []
        
        # User decision callback
        self._user_decision_callback: Callable[[DetectedConflict], Awaitable[str]] | None = None

    def set_user_decision_callback(
        self,
        callback: Callable[[DetectedConflict], Awaitable[str]],
    ) -> None:
        """Set callback for getting user decisions on conflicts."""
        self._user_decision_callback = callback

    def detect_conflicts(
        self,
        memories: list[BaseMemory],
        similarity_scores: dict[tuple[str, str], float] | None = None,
    ) -> list[DetectedConflict]:
        """
        Detect conflicts in a set of memories.
        
        Args:
            memories: Memories to check
            similarity_scores: Pre-computed similarity scores
            
        Returns:
            List of detected conflicts
        """
        conflicts = self.detector.detect_conflicts(memories, similarity_scores)
        
        # Add to pending
        for conflict in conflicts:
            if conflict not in self._pending_conflicts:
                self._pending_conflicts.append(conflict)
        
        return conflicts

    def check_new_memory(
        self,
        new_memory: BaseMemory,
        existing_memories: list[BaseMemory],
    ) -> list[DetectedConflict]:
        """
        Check if a new memory conflicts with existing ones.
        
        Called when adding new memories to the system.
        
        Args:
            new_memory: The new memory being added
            existing_memories: Existing memories to check against
            
        Returns:
            List of conflicts involving the new memory
        """
        conflicts = []
        
        for existing in existing_memories:
            conflict = self.detector.check_contradiction(new_memory, existing)
            if conflict:
                conflicts.append(conflict)
                self._pending_conflicts.append(conflict)
        
        return conflicts

    def resolve_conflict(
        self,
        conflict: DetectedConflict,
        memory_a: BaseMemory,
        memory_b: BaseMemory,
        strategy: ResolutionStrategy | None = None,
    ) -> ResolutionResult:
        """
        Resolve a single conflict.
        
        Args:
            conflict: The conflict to resolve
            memory_a: First memory in conflict
            memory_b: Second memory in conflict
            strategy: Strategy to use (auto-selects if None)
            
        Returns:
            ResolutionResult with the resolution decision
        """
        # Select strategy
        if strategy is None:
            strategy = self._select_strategy(conflict)
        
        # Get strategy implementation
        strategy_impl = get_strategy(strategy)
        
        # Resolve
        result = strategy_impl.resolve(conflict, memory_a, memory_b)
        
        # Record in history
        self._history.append(ConflictHistory(
            conflict=conflict,
            resolution=result,
        ))
        
        # Remove from pending
        if conflict in self._pending_conflicts:
            self._pending_conflicts.remove(conflict)
        
        return result

    async def resolve_conflict_async(
        self,
        conflict: DetectedConflict,
        memory_a: BaseMemory,
        memory_b: BaseMemory,
        strategy: ResolutionStrategy | None = None,
    ) -> ResolutionResult:
        """
        Resolve a conflict asynchronously (supports user interaction).
        
        Args:
            conflict: The conflict to resolve
            memory_a: First memory
            memory_b: Second memory
            strategy: Strategy to use
            
        Returns:
            ResolutionResult
        """
        # Select strategy
        if strategy is None:
            strategy = self._select_strategy(conflict)
        
        # If user-guided and we have callback
        if strategy == ResolutionStrategy.USER_GUIDED and self._user_decision_callback:
            user_choice = await self._user_decision_callback(conflict)
            
            user_strategy = UserGuidedStrategy()
            result = user_strategy.apply_user_decision(
                conflict, memory_a, memory_b, user_choice
            )
        else:
            result = self.resolve_conflict(conflict, memory_a, memory_b, strategy)
        
        return result

    def resolve_all_pending(
        self,
        memories_by_id: dict[str, BaseMemory],
    ) -> list[ResolutionResult]:
        """
        Resolve all pending conflicts.
        
        Args:
            memories_by_id: Dictionary mapping memory IDs to memories
            
        Returns:
            List of resolution results
        """
        results = []
        
        # Process conflicts by severity (low first for auto-resolve)
        pending = sorted(
            self._pending_conflicts.copy(),
            key=lambda c: {"low": 0, "medium": 1, "high": 2, "critical": 3}[c.severity.value]
        )
        
        for conflict in pending:
            # Check if we should auto-resolve
            if not self._should_auto_resolve(conflict):
                continue
            
            # Get memories
            mem_a = memories_by_id.get(conflict.memory_a_id)
            mem_b = memories_by_id.get(conflict.memory_b_id)
            
            if not mem_a or not mem_b:
                continue
            
            result = self.resolve_conflict(conflict, mem_a, mem_b)
            results.append(result)
        
        return results

    def apply_resolution(
        self,
        result: ResolutionResult,
        memory_a: BaseMemory,
        memory_b: BaseMemory,
    ) -> tuple[BaseMemory | None, BaseMemory | None, BaseMemory | None]:
        """
        Apply a resolution result to memories.
        
        Returns:
            Tuple of (kept_memory, discarded_memory, merged_memory)
            - kept_memory: Memory to keep
            - discarded_memory: Memory to mark as inactive/delete
            - merged_memory: New merged memory (if action was MERGE)
        """
        if result.action == ResolutionAction.KEEP_A:
            return memory_a, memory_b, None
        
        elif result.action == ResolutionAction.KEEP_B:
            return memory_b, memory_a, None
        
        elif result.action == ResolutionAction.MERGE:
            # Create merged memory
            merged = self._create_merged_memory(result, memory_a, memory_b)
            return None, None, merged
        
        elif result.action == ResolutionAction.KEEP_BOTH:
            # Mark both with conflict flag
            memory_a.metadata.tags.append("conflict:unresolved")
            memory_b.metadata.tags.append("conflict:unresolved")
            return memory_a, None, memory_b  # Keep both, no discard
        
        elif result.action == ResolutionAction.ARCHIVE_BOTH:
            return None, memory_a, memory_b  # Discard both
        
        else:  # DEFER
            return None, None, None

    def _select_strategy(self, conflict: DetectedConflict) -> ResolutionStrategy:
        """Select appropriate strategy for conflict."""
        # Critical always goes to user if configured
        if (
            conflict.severity == ConflictSeverity.CRITICAL
            and self.config.require_user_for_critical
        ):
            return ResolutionStrategy.USER_GUIDED
        
        # If auto-resolvable per conflict hint
        if conflict.auto_resolvable:
            return get_default_strategy_for_conflict(conflict)
        
        # Use default
        return self.config.default_strategy

    def _should_auto_resolve(self, conflict: DetectedConflict) -> bool:
        """Check if conflict should be auto-resolved."""
        if conflict.severity == ConflictSeverity.LOW:
            return self.config.auto_resolve_low_severity
        
        if conflict.severity == ConflictSeverity.MEDIUM:
            return self.config.auto_resolve_medium_severity
        
        if conflict.severity in [ConflictSeverity.HIGH, ConflictSeverity.CRITICAL]:
            return False
        
        return conflict.auto_resolvable

    def _create_merged_memory(
        self,
        result: ResolutionResult,
        memory_a: BaseMemory,
        memory_b: BaseMemory,
    ) -> SemanticMemory:
        """Create a merged memory from resolution result."""
        from llm_memory.models.base import MemoryMetadata, MemorySource
        
        merged = SemanticMemory(
            content=result.merged_content or f"{memory_a.content}\n\n{memory_b.content}",
            summary=f"Merged from conflict resolution: {result.explanation}",
            metadata=MemoryMetadata(
                user_id=memory_a.metadata.user_id,
                scope=memory_a.metadata.scope,
                source=MemorySource.CONSOLIDATION,
                source_ids=[memory_a.id, memory_b.id],
                tags=["conflict:resolved", "merged"],
            ),
        )
        
        # Combine importance
        merged.importance.relevance_frequency = max(
            memory_a.importance.relevance_frequency,
            memory_b.importance.relevance_frequency,
        )
        
        return merged

    @property
    def pending_conflicts(self) -> list[DetectedConflict]:
        """Get pending unresolved conflicts."""
        return self._pending_conflicts.copy()

    @property
    def pending_count(self) -> int:
        """Get number of pending conflicts."""
        return len(self._pending_conflicts)

    @property
    def history(self) -> list[ConflictHistory]:
        """Get resolution history."""
        return self._history.copy()

    def get_conflicts_for_memory(self, memory_id: str) -> list[DetectedConflict]:
        """Get all pending conflicts involving a specific memory."""
        return [
            c for c in self._pending_conflicts
            if c.memory_a_id == memory_id or c.memory_b_id == memory_id
        ]

    def clear_history(self) -> None:
        """Clear resolution history."""
        self._history.clear()

    def get_statistics(self) -> dict[str, Any]:
        """Get conflict resolution statistics."""
        total_resolved = len(self._history)
        
        # Count by strategy
        by_strategy: dict[str, int] = {}
        for h in self._history:
            strat = h.resolution.strategy_used.value
            by_strategy[strat] = by_strategy.get(strat, 0) + 1
        
        # Count by conflict type
        by_type: dict[str, int] = {}
        for h in self._history:
            ctype = h.conflict.conflict_type.value
            by_type[ctype] = by_type.get(ctype, 0) + 1
        
        return {
            "total_resolved": total_resolved,
            "pending_count": self.pending_count,
            "by_strategy": by_strategy,
            "by_conflict_type": by_type,
            "auto_resolved": sum(
                1 for h in self._history
                if not h.resolution.requires_user_confirmation
            ),
            "user_resolved": sum(
                1 for h in self._history
                if h.resolution.requires_user_confirmation
            ),
        }


class BatchConflictResolver:
    """
    Batch conflict resolution for memory system maintenance.
    
    Designed for periodic runs to clean up conflicts.
    """

    def __init__(
        self,
        resolver: ConflictResolver | None = None,
        config: ConflictResolutionConfig | None = None,
    ):
        self.resolver = resolver or ConflictResolver(config)
        self.config = config or ConflictResolutionConfig()

    def run_detection_batch(
        self,
        memories: list[BaseMemory],
        similarity_scores: dict[tuple[str, str], float] | None = None,
    ) -> list[DetectedConflict]:
        """Run conflict detection on a batch of memories."""
        # Limit batch size
        batch = memories[:self.config.detection_batch_size]
        return self.resolver.detect_conflicts(batch, similarity_scores)

    def run_resolution_batch(
        self,
        memories_by_id: dict[str, BaseMemory],
        max_resolutions: int = 50,
    ) -> list[ResolutionResult]:
        """Run resolution on pending conflicts."""
        results = []
        
        for conflict in self.resolver.pending_conflicts[:max_resolutions]:
            if not self.resolver._should_auto_resolve(conflict):
                continue
            
            mem_a = memories_by_id.get(conflict.memory_a_id)
            mem_b = memories_by_id.get(conflict.memory_b_id)
            
            if mem_a and mem_b:
                result = self.resolver.resolve_conflict(conflict, mem_a, mem_b)
                results.append(result)
        
        return results
