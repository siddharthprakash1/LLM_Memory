"""
Episodic memory models.

Episodic memory stores event-based memories with temporal context:
- Specific events and experiences
- Temporal tags (when things happened)
- Medium decay rate (days to weeks)
- Rich contextual information
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field

from llm_memory.models.base import BaseMemory, MemoryType


class EventType(str, Enum):
    """Types of events that can be stored as episodes."""

    CONVERSATION = "conversation"
    TASK_COMPLETION = "task_completion"
    ERROR = "error"
    DISCOVERY = "discovery"
    DECISION = "decision"
    FEEDBACK = "feedback"
    LEARNING = "learning"
    INTERACTION = "interaction"


class TemporalContext(BaseModel):
    """
    Temporal information about an episode.
    
    Captures when, how long, and temporal relationships.
    """

    # Absolute time
    occurred_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the event occurred",
    )
    ended_at: datetime | None = Field(
        default=None,
        description="When the event ended (if applicable)",
    )

    # Duration
    duration_seconds: float | None = Field(
        default=None,
        description="Duration of the event in seconds",
    )

    # Relative time markers
    day_of_week: str | None = Field(
        default=None,
        description="Day of week (Monday, Tuesday, etc.)",
    )
    time_of_day: str | None = Field(
        default=None,
        description="Time of day (morning, afternoon, evening, night)",
    )
    is_weekend: bool = Field(
        default=False,
        description="Whether event occurred on weekend",
    )

    # Sequence context
    preceded_by: str | None = Field(
        default=None,
        description="ID of event that preceded this one",
    )
    followed_by: str | None = Field(
        default=None,
        description="ID of event that followed this one",
    )

    # Recurrence
    is_recurring: bool = Field(
        default=False,
        description="Whether this is a recurring event pattern",
    )
    recurrence_pattern: str | None = Field(
        default=None,
        description="Pattern description (e.g., 'daily', 'weekly on Monday')",
    )

    @computed_field
    @property
    def age_days(self) -> float:
        """Get the age of this event in days."""
        return (datetime.utcnow() - self.occurred_at).total_seconds() / 86400

    def __post_init__(self) -> None:
        """Calculate derived fields after initialization."""
        if self.occurred_at:
            self.day_of_week = self.occurred_at.strftime("%A")
            hour = self.occurred_at.hour
            if 5 <= hour < 12:
                self.time_of_day = "morning"
            elif 12 <= hour < 17:
                self.time_of_day = "afternoon"
            elif 17 <= hour < 21:
                self.time_of_day = "evening"
            else:
                self.time_of_day = "night"
            self.is_weekend = self.occurred_at.weekday() >= 5

        if self.ended_at and self.occurred_at:
            self.duration_seconds = (self.ended_at - self.occurred_at).total_seconds()


class Participant(BaseModel):
    """A participant in an episode."""

    id: str = Field(description="Participant identifier")
    role: str = Field(description="Role in the episode (user, assistant, system)")
    name: str | None = Field(default=None, description="Display name")


class Episode(BaseModel):
    """
    A single episode/event in episodic memory.
    
    Represents a discrete event with full context.
    """

    # Event identification
    event_id: str = Field(description="Unique event identifier")
    event_type: EventType = Field(
        default=EventType.INTERACTION,
        description="Type of event",
    )

    # Content
    description: str = Field(description="What happened in this episode")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed structured information about the event",
    )

    # Temporal context
    temporal: TemporalContext = Field(default_factory=TemporalContext)

    # Participants
    participants: list[Participant] = Field(
        default_factory=list,
        description="Participants in this episode",
    )

    # Location/Context
    location: str | None = Field(
        default=None,
        description="Logical location (e.g., file path, project, feature)",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context about the episode",
    )

    # Outcome
    outcome: str | None = Field(
        default=None,
        description="Result or outcome of the episode",
    )
    was_successful: bool | None = Field(
        default=None,
        description="Whether the episode had a successful outcome",
    )

    # Emotional/Importance markers
    emotional_valence: float = Field(
        default=0.0,
        description="Emotional tone (-1 negative to +1 positive)",
        ge=-1.0,
        le=1.0,
    )
    surprise_level: float = Field(
        default=0.0,
        description="How surprising/unexpected this was (0-1)",
        ge=0.0,
        le=1.0,
    )


class EpisodicMemory(BaseMemory):
    """
    Episodic memory - stores event-based experiences.
    
    Contains one or more related episodes with rich
    temporal and contextual information.
    """

    memory_type: MemoryType = Field(default=MemoryType.EPISODIC, frozen=True)

    # Episodes
    episodes: list[Episode] = Field(
        default_factory=list,
        description="Episodes contained in this memory",
    )
    primary_episode_id: str | None = Field(
        default=None,
        description="ID of the main/primary episode",
    )

    # Temporal span
    temporal_context: TemporalContext = Field(
        default_factory=TemporalContext,
        description="Overall temporal context for this memory",
    )

    # Narrative
    narrative: str | None = Field(
        default=None,
        description="Narrative summary of the episodes",
    )

    # Learning extracted
    lessons_learned: list[str] = Field(
        default_factory=list,
        description="Lessons or insights extracted from these episodes",
    )

    # Consolidation tracking
    consolidation_candidates: list[str] = Field(
        default_factory=list,
        description="IDs of similar episodic memories for potential semantic consolidation",
    )
    times_pattern_matched: int = Field(
        default=0,
        description="How many times this episode matched a pattern",
    )

    def get_decay_rate(self) -> float:
        """Episodic memory has medium decay rate."""
        return 0.01  # Default, can be overridden by config

    def add_episode(self, episode: Episode) -> None:
        """Add an episode to this memory."""
        self.episodes.append(episode)
        if not self.primary_episode_id:
            self.primary_episode_id = episode.event_id
        self._update_temporal_span()
        self._update_summary()
        self.updated_at = datetime.utcnow()

    def get_episode(self, event_id: str) -> Episode | None:
        """Get an episode by its event ID."""
        for episode in self.episodes:
            if episode.event_id == event_id:
                return episode
        return None

    def get_primary_episode(self) -> Episode | None:
        """Get the primary episode."""
        if self.primary_episode_id:
            return self.get_episode(self.primary_episode_id)
        return self.episodes[0] if self.episodes else None

    def get_episodes_by_type(self, event_type: EventType) -> list[Episode]:
        """Get episodes of a specific type."""
        return [ep for ep in self.episodes if ep.event_type == event_type]

    def get_successful_episodes(self) -> list[Episode]:
        """Get episodes with successful outcomes."""
        return [ep for ep in self.episodes if ep.was_successful is True]

    def get_failed_episodes(self) -> list[Episode]:
        """Get episodes with failed outcomes."""
        return [ep for ep in self.episodes if ep.was_successful is False]

    def _update_temporal_span(self) -> None:
        """Update temporal context to span all episodes."""
        if not self.episodes:
            return

        earliest = min(ep.temporal.occurred_at for ep in self.episodes)
        latest_times = [
            ep.temporal.ended_at or ep.temporal.occurred_at for ep in self.episodes
        ]
        latest = max(latest_times)

        self.temporal_context.occurred_at = earliest
        self.temporal_context.ended_at = latest
        self.temporal_context.duration_seconds = (latest - earliest).total_seconds()

    def _update_summary(self) -> None:
        """Update summary based on episodes."""
        if not self.episodes:
            self.summary = "Empty episodic memory"
            return

        types = {}
        for ep in self.episodes:
            types[ep.event_type.value] = types.get(ep.event_type.value, 0) + 1

        type_summary = ", ".join(f"{k}: {v}" for k, v in types.items())
        self.summary = f"Episodic memory with {len(self.episodes)} episodes ({type_summary})"

    @classmethod
    def from_stm_buffer(
        cls,
        content: str,
        session_id: str | None = None,
        task_id: str | None = None,
        **kwargs,
    ) -> "EpisodicMemory":
        """
        Create an episodic memory from a short-term memory buffer.
        
        This is used during consolidation from STM to episodic.
        """
        episode = Episode(
            event_id=f"ep_{datetime.utcnow().timestamp()}",
            event_type=EventType.CONVERSATION,
            description=content,
            details={
                "session_id": session_id,
                "task_id": task_id,
            },
        )

        memory = cls(
            content=content,
            episodes=[episode],
            primary_episode_id=episode.event_id,
            **kwargs,
        )

        memory._update_summary()
        return memory

    def __len__(self) -> int:
        """Return number of episodes."""
        return len(self.episodes)
