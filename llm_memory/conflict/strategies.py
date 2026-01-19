"""
Conflict resolution strategies.

Different strategies for resolving memory conflicts:
- Recency-based: Prefer newer information
- Confidence-based: Prefer higher confidence
- Source-based: Prefer more reliable sources
- Frequency-based: Prefer more frequently accessed
- User-guided: Ask user to resolve
- Merge: Combine conflicting information
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from llm_memory.models.base import BaseMemory, MemorySource
from llm_memory.models.semantic import SemanticMemory, Fact
from llm_memory.conflict.detector import DetectedConflict, ConflictType, ConflictSeverity


def _utcnow() -> datetime:
    """Get current UTC time (naive, for compatibility with models)."""
    return datetime.utcnow()


class ResolutionStrategy(str, Enum):
    """Available conflict resolution strategies."""

    RECENCY = "recency"  # Prefer newer memory
    CONFIDENCE = "confidence"  # Prefer higher confidence
    SOURCE_RELIABILITY = "source_reliability"  # Prefer more reliable source
    FREQUENCY = "frequency"  # Prefer more accessed memory
    IMPORTANCE = "importance"  # Prefer higher importance
    USER_GUIDED = "user_guided"  # Ask user
    MERGE = "merge"  # Combine both
    KEEP_BOTH = "keep_both"  # Mark as alternatives


class ResolutionAction(str, Enum):
    """Actions to take after resolution."""

    KEEP_A = "keep_a"  # Keep memory A, discard B
    KEEP_B = "keep_b"  # Keep memory B, discard A
    KEEP_BOTH = "keep_both"  # Keep both with conflict marker
    MERGE = "merge"  # Create merged memory
    ARCHIVE_BOTH = "archive_both"  # Archive both, create new
    DEFER = "defer"  # Defer to user


class ResolutionResult(BaseModel):
    """Result of conflict resolution."""

    conflict_id: str
    strategy_used: ResolutionStrategy
    action: ResolutionAction
    
    # Winner/loser
    winner_id: str | None = None
    loser_id: str | None = None
    
    # If merged
    merged_content: str | None = None
    
    # Explanation
    explanation: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    
    # Metadata
    resolved_at: datetime = Field(default_factory=_utcnow)
    requires_user_confirmation: bool = False


class BaseResolutionStrategy(ABC):
    """Base class for resolution strategies."""

    strategy_type: ResolutionStrategy

    @abstractmethod
    def resolve(
        self,
        conflict: DetectedConflict,
        memory_a: BaseMemory,
        memory_b: BaseMemory,
    ) -> ResolutionResult:
        """Resolve a conflict between two memories."""
        pass

    def can_resolve(self, conflict: DetectedConflict) -> bool:
        """Check if this strategy can resolve the given conflict."""
        return True


class RecencyStrategy(BaseResolutionStrategy):
    """
    Prefer more recent memory.
    
    Best for:
    - Temporal conflicts
    - Preference changes
    - Factual updates
    """

    strategy_type = ResolutionStrategy.RECENCY

    def resolve(
        self,
        conflict: DetectedConflict,
        memory_a: BaseMemory,
        memory_b: BaseMemory,
    ) -> ResolutionResult:
        """Resolve by preferring the newer memory."""
        # Compare creation times
        if memory_a.created_at > memory_b.created_at:
            winner, loser = memory_a, memory_b
            action = ResolutionAction.KEEP_A
        else:
            winner, loser = memory_b, memory_a
            action = ResolutionAction.KEEP_B

        age_diff = abs((memory_a.created_at - memory_b.created_at).days)
        
        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            strategy_used=self.strategy_type,
            action=action,
            winner_id=winner.id,
            loser_id=loser.id,
            explanation=f"Preferred newer memory (created {age_diff} days later)",
            confidence=min(0.95, 0.5 + age_diff * 0.01),  # More confident with larger gap
        )


class ConfidenceStrategy(BaseResolutionStrategy):
    """
    Prefer memory with higher confidence.
    
    Best for:
    - Fact conflicts
    - Semantic memory conflicts
    """

    strategy_type = ResolutionStrategy.CONFIDENCE

    def resolve(
        self,
        conflict: DetectedConflict,
        memory_a: BaseMemory,
        memory_b: BaseMemory,
    ) -> ResolutionResult:
        """Resolve by preferring higher confidence memory."""
        # Get confidence scores
        conf_a = self._get_confidence(memory_a)
        conf_b = self._get_confidence(memory_b)

        if conf_a >= conf_b:
            winner, loser = memory_a, memory_b
            action = ResolutionAction.KEEP_A
            confidence = conf_a
        else:
            winner, loser = memory_b, memory_a
            action = ResolutionAction.KEEP_B
            confidence = conf_b

        conf_diff = abs(conf_a - conf_b)
        
        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            strategy_used=self.strategy_type,
            action=action,
            winner_id=winner.id,
            loser_id=loser.id,
            explanation=f"Preferred memory with higher confidence ({confidence:.2f} vs {min(conf_a, conf_b):.2f})",
            confidence=min(0.9, 0.5 + conf_diff),
        )

    def _get_confidence(self, memory: BaseMemory) -> float:
        """Get confidence score for memory."""
        if isinstance(memory, SemanticMemory):
            return memory.overall_confidence
        return memory.current_strength


class SourceReliabilityStrategy(BaseResolutionStrategy):
    """
    Prefer memory from more reliable source.
    
    Source reliability ranking:
    1. USER_INPUT (highest)
    2. SYSTEM
    3. OBSERVATION
    4. ASSISTANT_OUTPUT
    5. INFERENCE
    6. CONSOLIDATION (lowest)
    """

    strategy_type = ResolutionStrategy.SOURCE_RELIABILITY

    SOURCE_RANKING = {
        MemorySource.USER_INPUT: 6,
        MemorySource.SYSTEM: 5,
        MemorySource.OBSERVATION: 4,
        MemorySource.ASSISTANT_OUTPUT: 3,
        MemorySource.INFERENCE: 2,
        MemorySource.CONSOLIDATION: 1,
    }

    def resolve(
        self,
        conflict: DetectedConflict,
        memory_a: BaseMemory,
        memory_b: BaseMemory,
    ) -> ResolutionResult:
        """Resolve by preferring more reliable source."""
        rank_a = self.SOURCE_RANKING.get(memory_a.metadata.source, 0)
        rank_b = self.SOURCE_RANKING.get(memory_b.metadata.source, 0)

        if rank_a >= rank_b:
            winner, loser = memory_a, memory_b
            action = ResolutionAction.KEEP_A
        else:
            winner, loser = memory_b, memory_a
            action = ResolutionAction.KEEP_B

        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            strategy_used=self.strategy_type,
            action=action,
            winner_id=winner.id,
            loser_id=loser.id,
            explanation=f"Preferred source: {winner.metadata.source.value} over {loser.metadata.source.value}",
            confidence=0.75 + (abs(rank_a - rank_b) * 0.05),
        )


class FrequencyStrategy(BaseResolutionStrategy):
    """
    Prefer more frequently accessed memory.
    
    Best for:
    - Memories that have been validated through use
    """

    strategy_type = ResolutionStrategy.FREQUENCY

    def resolve(
        self,
        conflict: DetectedConflict,
        memory_a: BaseMemory,
        memory_b: BaseMemory,
    ) -> ResolutionResult:
        """Resolve by preferring more accessed memory."""
        if memory_a.access_count >= memory_b.access_count:
            winner, loser = memory_a, memory_b
            action = ResolutionAction.KEEP_A
        else:
            winner, loser = memory_b, memory_a
            action = ResolutionAction.KEEP_B

        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            strategy_used=self.strategy_type,
            action=action,
            winner_id=winner.id,
            loser_id=loser.id,
            explanation=f"Preferred more accessed memory ({winner.access_count} vs {loser.access_count} accesses)",
            confidence=0.6 + min(0.3, abs(memory_a.access_count - memory_b.access_count) * 0.05),
        )


class ImportanceStrategy(BaseResolutionStrategy):
    """
    Prefer memory with higher importance score.
    
    Best for:
    - Conflicts where one memory is clearly more significant
    """

    strategy_type = ResolutionStrategy.IMPORTANCE

    def resolve(
        self,
        conflict: DetectedConflict,
        memory_a: BaseMemory,
        memory_b: BaseMemory,
    ) -> ResolutionResult:
        """Resolve by preferring higher importance memory."""
        imp_a = memory_a.importance_score
        imp_b = memory_b.importance_score

        if imp_a >= imp_b:
            winner, loser = memory_a, memory_b
            action = ResolutionAction.KEEP_A
        else:
            winner, loser = memory_b, memory_a
            action = ResolutionAction.KEEP_B

        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            strategy_used=self.strategy_type,
            action=action,
            winner_id=winner.id,
            loser_id=loser.id,
            explanation=f"Preferred more important memory ({max(imp_a, imp_b):.2f} vs {min(imp_a, imp_b):.2f})",
            confidence=min(0.95, 0.6 + abs(imp_a - imp_b)),
        )


class MergeStrategy(BaseResolutionStrategy):
    """
    Merge conflicting information.
    
    Creates a new memory that combines information from both,
    acknowledging the conflict.
    """

    strategy_type = ResolutionStrategy.MERGE

    def resolve(
        self,
        conflict: DetectedConflict,
        memory_a: BaseMemory,
        memory_b: BaseMemory,
    ) -> ResolutionResult:
        """Resolve by merging both memories."""
        merged_content = self._create_merged_content(conflict, memory_a, memory_b)

        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            strategy_used=self.strategy_type,
            action=ResolutionAction.MERGE,
            winner_id=None,
            loser_id=None,
            merged_content=merged_content,
            explanation="Merged conflicting information into single memory",
            confidence=0.7,
        )

    def _create_merged_content(
        self,
        conflict: DetectedConflict,
        memory_a: BaseMemory,
        memory_b: BaseMemory,
    ) -> str:
        """Create merged content from two memories."""
        # Create content that acknowledges both perspectives
        return f"""[Merged from conflict resolution]

Original information (from {memory_a.created_at.date()}):
{memory_a.content}

Updated/Alternative information (from {memory_b.created_at.date()}):
{memory_b.content}

Note: These memories had {conflict.conflict_type.value}. {conflict.explanation}"""


class KeepBothStrategy(BaseResolutionStrategy):
    """
    Keep both memories with conflict markers.
    
    Best for:
    - Uncertain conflicts
    - Different valid perspectives
    """

    strategy_type = ResolutionStrategy.KEEP_BOTH

    def resolve(
        self,
        conflict: DetectedConflict,
        memory_a: BaseMemory,
        memory_b: BaseMemory,
    ) -> ResolutionResult:
        """Resolve by keeping both memories."""
        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            strategy_used=self.strategy_type,
            action=ResolutionAction.KEEP_BOTH,
            winner_id=None,
            loser_id=None,
            explanation="Keeping both memories as valid alternatives",
            confidence=0.5,
        )


class UserGuidedStrategy(BaseResolutionStrategy):
    """
    Defer resolution to user.
    
    Best for:
    - Critical conflicts
    - Ambiguous situations
    - High-stakes information
    """

    strategy_type = ResolutionStrategy.USER_GUIDED

    def resolve(
        self,
        conflict: DetectedConflict,
        memory_a: BaseMemory,
        memory_b: BaseMemory,
    ) -> ResolutionResult:
        """Defer resolution to user."""
        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            strategy_used=self.strategy_type,
            action=ResolutionAction.DEFER,
            winner_id=None,
            loser_id=None,
            explanation="Conflict requires user input to resolve",
            confidence=0.0,
            requires_user_confirmation=True,
        )

    def apply_user_decision(
        self,
        conflict: DetectedConflict,
        memory_a: BaseMemory,
        memory_b: BaseMemory,
        keep_memory_id: str,
    ) -> ResolutionResult:
        """Apply user's decision to resolve conflict."""
        if keep_memory_id == memory_a.id:
            action = ResolutionAction.KEEP_A
            winner, loser = memory_a, memory_b
        else:
            action = ResolutionAction.KEEP_B
            winner, loser = memory_b, memory_a

        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            strategy_used=self.strategy_type,
            action=action,
            winner_id=winner.id,
            loser_id=loser.id,
            explanation="Resolved by user decision",
            confidence=1.0,
            requires_user_confirmation=False,
        )


# Strategy factory
def get_strategy(strategy_type: ResolutionStrategy) -> BaseResolutionStrategy:
    """Get resolution strategy by type."""
    strategies = {
        ResolutionStrategy.RECENCY: RecencyStrategy(),
        ResolutionStrategy.CONFIDENCE: ConfidenceStrategy(),
        ResolutionStrategy.SOURCE_RELIABILITY: SourceReliabilityStrategy(),
        ResolutionStrategy.FREQUENCY: FrequencyStrategy(),
        ResolutionStrategy.IMPORTANCE: ImportanceStrategy(),
        ResolutionStrategy.MERGE: MergeStrategy(),
        ResolutionStrategy.KEEP_BOTH: KeepBothStrategy(),
        ResolutionStrategy.USER_GUIDED: UserGuidedStrategy(),
    }
    return strategies.get(strategy_type, RecencyStrategy())


def get_default_strategy_for_conflict(conflict: DetectedConflict) -> ResolutionStrategy:
    """Get recommended strategy based on conflict type."""
    recommendations = {
        ConflictType.TEMPORAL_OUTDATED: ResolutionStrategy.RECENCY,
        ConflictType.PREFERENCE_CONFLICT: ResolutionStrategy.RECENCY,
        ConflictType.FACT_INCONSISTENCY: ResolutionStrategy.CONFIDENCE,
        ConflictType.DIRECT_CONTRADICTION: ResolutionStrategy.USER_GUIDED,
        ConflictType.SOURCE_DISAGREEMENT: ResolutionStrategy.SOURCE_RELIABILITY,
        ConflictType.PARTIAL_OVERLAP: ResolutionStrategy.MERGE,
    }
    
    # Critical severity always goes to user
    if conflict.severity == ConflictSeverity.CRITICAL:
        return ResolutionStrategy.USER_GUIDED
    
    return recommendations.get(conflict.conflict_type, ResolutionStrategy.RECENCY)
