"""
Short-term memory models.

Short-term memory (STM) represents the working context:
- Current conversation buffer
- Active task context
- Fast decay rate (minutes to hours)
- High capacity but transient
"""

from datetime import datetime
from enum import Enum

from pydantic import Field

from llm_memory.models.base import BaseMemory, MemoryType


class STMRole(str, Enum):
    """Role of the message in short-term memory."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    OBSERVATION = "observation"


class WorkingContext(BaseMemory):
    """
    A single item in short-term/working memory.
    
    Represents a single message, observation, or context item
    that the agent is actively working with.
    """

    memory_type: MemoryType = Field(default=MemoryType.SHORT_TERM, frozen=True)

    # STM-specific fields
    role: STMRole = Field(
        default=STMRole.OBSERVATION,
        description="Role of this memory item",
    )
    sequence_number: int = Field(
        default=0,
        description="Order in the conversation/task sequence",
    )

    # Task context
    task_id: str | None = Field(
        default=None,
        description="ID of the task this memory belongs to",
    )
    is_task_relevant: bool = Field(
        default=True,
        description="Whether this is relevant to the current task",
    )

    # Attention weight (for transformer-like attention simulation)
    attention_weight: float = Field(
        default=1.0,
        description="Attention weight for this memory (0-1)",
        ge=0.0,
        le=1.0,
    )

    # Turn information
    turn_number: int | None = Field(
        default=None,
        description="Conversation turn number",
    )

    def get_decay_rate(self) -> float:
        """STM has fast decay rate."""
        return 0.1  # Default, can be overridden by config

    def to_message_dict(self) -> dict:
        """Convert to OpenAI-style message format."""
        return {
            "role": self.role.value,
            "content": self.content,
        }


class ShortTermMemory(BaseMemory):
    """
    Container for short-term memory buffer.
    
    Holds multiple WorkingContext items and manages
    the overall working memory state for a session.
    """

    memory_type: MemoryType = Field(default=MemoryType.SHORT_TERM, frozen=True)

    # Buffer of working context items
    items: list[WorkingContext] = Field(
        default_factory=list,
        description="Working context items in this buffer",
    )

    # Buffer constraints
    max_items: int = Field(
        default=50,
        description="Maximum number of items in buffer",
    )
    max_tokens: int = Field(
        default=8000,
        description="Maximum total tokens in buffer",
    )
    current_token_count: int = Field(
        default=0,
        description="Current estimated token count",
    )

    # Session tracking
    session_id: str | None = Field(
        default=None,
        description="ID of the session this STM belongs to",
    )
    session_start: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this session started",
    )

    # Current task
    current_task_id: str | None = Field(
        default=None,
        description="ID of the current active task",
    )
    current_task_description: str | None = Field(
        default=None,
        description="Description of current task",
    )

    def get_decay_rate(self) -> float:
        """STM has fast decay rate."""
        return 0.1

    def add_item(self, item: WorkingContext) -> bool:
        """
        Add an item to the STM buffer.
        
        Returns True if added, False if buffer is full.
        """
        if len(self.items) >= self.max_items:
            return False

        # Set sequence number
        item.sequence_number = len(self.items)
        self.items.append(item)
        self.updated_at = datetime.utcnow()

        # Update summary
        self._update_summary()

        return True

    def add_message(
        self,
        content: str,
        role: STMRole = STMRole.USER,
        **kwargs,
    ) -> WorkingContext:
        """
        Convenience method to add a message to STM.
        
        Creates a WorkingContext and adds it to the buffer.
        """
        item = WorkingContext(
            content=content,
            role=role,
            task_id=self.current_task_id,
            **kwargs,
        )
        self.add_item(item)
        return item

    def get_recent(self, n: int = 10) -> list[WorkingContext]:
        """Get the n most recent items."""
        return self.items[-n:] if self.items else []

    def get_by_role(self, role: STMRole) -> list[WorkingContext]:
        """Get all items with a specific role."""
        return [item for item in self.items if item.role == role]

    def get_task_relevant(self) -> list[WorkingContext]:
        """Get items relevant to the current task."""
        return [item for item in self.items if item.is_task_relevant]

    def clear(self) -> list[WorkingContext]:
        """Clear the buffer and return the cleared items."""
        cleared = self.items.copy()
        self.items = []
        self.current_token_count = 0
        self.updated_at = datetime.utcnow()
        return cleared

    def evict_oldest(self, n: int = 1) -> list[WorkingContext]:
        """Evict the n oldest items from the buffer."""
        if n >= len(self.items):
            return self.clear()

        evicted = self.items[:n]
        self.items = self.items[n:]

        # Reindex sequence numbers
        for i, item in enumerate(self.items):
            item.sequence_number = i

        self.updated_at = datetime.utcnow()
        return evicted

    def evict_weakest(self, n: int = 1) -> list[WorkingContext]:
        """Evict the n items with lowest effective strength."""
        if n >= len(self.items):
            return self.clear()

        # Sort by effective strength
        sorted_items = sorted(self.items, key=lambda x: x.effective_strength)
        evicted = sorted_items[:n]
        remaining = sorted_items[n:]

        # Sort remaining back by sequence number
        self.items = sorted(remaining, key=lambda x: x.sequence_number)

        # Reindex
        for i, item in enumerate(self.items):
            item.sequence_number = i

        self.updated_at = datetime.utcnow()
        return evicted

    def to_messages(self) -> list[dict]:
        """Convert buffer to OpenAI-style messages list."""
        return [item.to_message_dict() for item in self.items]

    def _update_summary(self) -> None:
        """Update the summary field based on current items."""
        if not self.items:
            self.summary = "Empty working memory"
        else:
            roles = {}
            for item in self.items:
                roles[item.role.value] = roles.get(item.role.value, 0) + 1

            role_summary = ", ".join(f"{k}: {v}" for k, v in roles.items())
            self.summary = f"STM buffer with {len(self.items)} items ({role_summary})"

    @property
    def content(self) -> str:
        """Override content to return concatenated items."""
        return "\n".join(f"[{item.role.value}]: {item.content}" for item in self.items)

    @content.setter
    def content(self, value: str) -> None:
        """Content setter (required by Pydantic but not typically used)."""
        # For STM, content is derived from items
        pass

    def __len__(self) -> int:
        """Return number of items in buffer."""
        return len(self.items)
