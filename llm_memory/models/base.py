"""
Base memory models and types.

Defines the foundational classes that all memory types inherit from.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field
from ulid import ULID


class MemoryType(str, Enum):
    """Types of memory in the hierarchical system."""

    SHORT_TERM = "short_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class MemorySource(str, Enum):
    """Source of the memory."""

    USER_INPUT = "user_input"
    ASSISTANT_OUTPUT = "assistant_output"
    SYSTEM = "system"
    OBSERVATION = "observation"
    CONSOLIDATION = "consolidation"  # Created from consolidation pipeline
    INFERENCE = "inference"  # Inferred from other memories


class ImportanceFactors(BaseModel):
    """Breakdown of factors contributing to memory importance."""

    emotional_salience: float = Field(
        default=0.5,
        description="Emotional weight (0-1), detected via sentiment",
        ge=0.0,
        le=1.0,
    )
    novelty: float = Field(
        default=0.5,
        description="How different from existing memories (0-1)",
        ge=0.0,
        le=1.0,
    )
    relevance_frequency: float = Field(
        default=0.0,
        description="How often this memory is retrieved (normalized)",
        ge=0.0,
        le=1.0,
    )
    causal_significance: float = Field(
        default=0.5,
        description="Impact on downstream events (0-1)",
        ge=0.0,
        le=1.0,
    )
    user_marked: float = Field(
        default=0.0,
        description="Explicit user importance marking (0-1)",
        ge=0.0,
        le=1.0,
    )

    @computed_field
    @property
    def composite_score(self) -> float:
        """Calculate weighted composite importance score."""
        weights = {
            "emotional_salience": 0.2,
            "novelty": 0.25,
            "relevance_frequency": 0.2,
            "causal_significance": 0.25,
            "user_marked": 0.1,
        }
        score = (
            self.emotional_salience * weights["emotional_salience"]
            + self.novelty * weights["novelty"]
            + self.relevance_frequency * weights["relevance_frequency"]
            + self.causal_significance * weights["causal_significance"]
            + self.user_marked * weights["user_marked"]
        )
        return min(max(score, 0.0), 1.0)


class MemoryMetadata(BaseModel):
    """Metadata attached to every memory."""

    # Identifiers
    user_id: str | None = Field(default=None, description="User who owns this memory")
    scope: str = Field(default="global", description="Memory scope (global, project, session)")
    session_id: str | None = Field(default=None, description="Session this memory belongs to")
    project_id: str | None = Field(default=None, description="Project context")

    # Source tracking
    source: MemorySource = Field(default=MemorySource.OBSERVATION)
    source_ids: list[str] = Field(
        default_factory=list,
        description="IDs of source memories (for consolidated memories)",
    )

    # Tags and categorization
    tags: list[str] = Field(default_factory=list, description="User or system tags")
    categories: list[str] = Field(default_factory=list, description="Auto-categorized topics")

    # Relationships
    related_memory_ids: list[str] = Field(
        default_factory=list,
        description="IDs of related memories",
    )
    parent_memory_id: str | None = Field(
        default=None,
        description="ID of parent memory (for hierarchical relationships)",
    )

    # Conflict tracking
    conflicts_with: list[str] = Field(
        default_factory=list,
        description="IDs of memories this conflicts with",
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence in this memory's accuracy (0-1)",
        ge=0.0,
        le=1.0,
    )

    # Additional context
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional arbitrary metadata",
    )


class BaseMemory(BaseModel, ABC):
    """
    Abstract base class for all memory types.
    
    Provides common fields and methods for memory lifecycle management.
    """

    # Core identifiers
    id: str = Field(
        default_factory=lambda: str(ULID()),
        description="Unique memory identifier (ULID for time-ordering)",
    )
    memory_type: MemoryType = Field(description="Type of memory")

    # Content
    content: str = Field(description="The actual memory content")
    summary: str | None = Field(
        default=None,
        description="Summarized version of the content",
    )

    # Embeddings (stored separately in vector DB, but tracked here)
    embedding_id: str | None = Field(
        default=None,
        description="ID of the embedding in vector store",
    )
    has_embedding: bool = Field(default=False, description="Whether embedding exists")

    # Temporal information
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the memory was created",
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the memory was last updated",
    )
    last_accessed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the memory was last retrieved",
    )
    access_count: int = Field(default=0, description="Number of times accessed")

    # Strength and decay
    initial_strength: float = Field(
        default=1.0,
        description="Initial memory strength (0-1)",
        ge=0.0,
        le=1.0,
    )
    current_strength: float = Field(
        default=1.0,
        description="Current memory strength after decay (0-1)",
        ge=0.0,
        le=1.0,
    )

    # Importance
    importance: ImportanceFactors = Field(default_factory=ImportanceFactors)

    # Metadata
    metadata: MemoryMetadata = Field(default_factory=MemoryMetadata)

    # Status
    is_active: bool = Field(default=True, description="Whether memory is active")
    is_consolidated: bool = Field(
        default=False,
        description="Whether memory has been consolidated to next tier",
    )

    @computed_field
    @property
    def importance_score(self) -> float:
        """Get the composite importance score."""
        return self.importance.composite_score

    @computed_field
    @property
    def effective_strength(self) -> float:
        """Calculate effective strength (current strength * importance)."""
        return self.current_strength * (0.5 + 0.5 * self.importance_score)

    @computed_field
    @property
    def age_seconds(self) -> float:
        """Get the age of this memory in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()

    @computed_field
    @property
    def time_since_access_seconds(self) -> float:
        """Get time since last access in seconds."""
        return (datetime.utcnow() - self.last_accessed_at).total_seconds()

    def record_access(self) -> None:
        """Record that this memory was accessed."""
        self.last_accessed_at = datetime.utcnow()
        self.access_count += 1

    def apply_rehearsal_boost(self, boost: float = 0.2) -> None:
        """Apply rehearsal boost to memory strength."""
        self.current_strength = min(1.0, self.current_strength + boost)
        self.record_access()

    @abstractmethod
    def get_decay_rate(self) -> float:
        """Get the decay rate for this memory type."""
        pass

    def calculate_decayed_strength(self, decay_rate: float | None = None) -> float:
        """
        Calculate current strength using Ebbinghaus forgetting curve.
        
        Formula: S(t) = S₀ × e^(-λt/importance)
        
        Where:
        - S₀ = initial_strength
        - λ = decay_rate
        - t = time_since_access_seconds
        - importance = importance_score (affects decay rate)
        """
        import math

        rate = decay_rate if decay_rate is not None else self.get_decay_rate()
        t = self.time_since_access_seconds

        # Importance reduces effective decay rate
        effective_rate = rate / (0.5 + self.importance_score)

        decayed = self.initial_strength * math.exp(-effective_rate * t / 3600)  # Normalize to hours
        return max(0.0, min(1.0, decayed))

    def update_strength(self, decay_rate: float | None = None) -> None:
        """Update current_strength based on decay."""
        self.current_strength = self.calculate_decayed_strength(decay_rate)
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert memory to dictionary representation."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseMemory":
        """Create memory from dictionary."""
        return cls.model_validate(data)

    def __str__(self) -> str:
        """String representation of memory."""
        return f"{self.memory_type.value}[{self.id[:8]}]: {self.content[:50]}..."

    def __repr__(self) -> str:
        """Detailed representation of memory."""
        return (
            f"{self.__class__.__name__}("
            f"id={self.id!r}, "
            f"strength={self.current_strength:.2f}, "
            f"importance={self.importance_score:.2f})"
        )
