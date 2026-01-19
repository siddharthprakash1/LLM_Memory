"""
Event hooks for memory lifecycle management.

Provides a pub/sub system for:
- Memory creation events
- Memory access events
- Consolidation events
- Conflict events
"""

from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable
from dataclasses import dataclass, field

from llm_memory.models.base import BaseMemory, MemoryType


def _utcnow() -> datetime:
    """Get current UTC time (naive)."""
    return datetime.utcnow()


class HookEvent(str, Enum):
    """Types of hook events."""

    # Memory lifecycle
    MEMORY_CREATED = "memory_created"
    MEMORY_ACCESSED = "memory_accessed"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_DELETED = "memory_deleted"
    
    # STM events
    STM_MESSAGE_ADDED = "stm_message_added"
    STM_CLEARED = "stm_cleared"
    
    # Consolidation events
    CONSOLIDATION_STARTED = "consolidation_started"
    CONSOLIDATION_COMPLETED = "consolidation_completed"
    STM_PROMOTED = "stm_promoted"
    EPISODIC_PROMOTED = "episodic_promoted"
    MEMORY_MERGED = "memory_merged"
    MEMORY_GARBAGE_COLLECTED = "memory_garbage_collected"
    
    # Conflict events
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"
    
    # Retrieval events
    SEARCH_STARTED = "search_started"
    SEARCH_COMPLETED = "search_completed"
    
    # Decay events
    DECAY_UPDATED = "decay_updated"
    STRENGTH_BELOW_THRESHOLD = "strength_below_threshold"


@dataclass
class HookContext:
    """Context passed to hook callbacks."""

    event: HookEvent
    timestamp: datetime = field(default_factory=_utcnow)
    memory: BaseMemory | None = None
    memory_id: str | None = None
    memory_type: MemoryType | None = None
    data: dict = field(default_factory=dict)
    source: str | None = None


# Type for hook callbacks
HookCallback = Callable[[HookContext], None]
AsyncHookCallback = Callable[[HookContext], Awaitable[None]]


class HookRegistry:
    """
    Registry for memory event hooks.
    
    Allows components to subscribe to memory events
    and react accordingly.
    """

    def __init__(self):
        self._sync_hooks: dict[HookEvent, list[HookCallback]] = {}
        self._async_hooks: dict[HookEvent, list[AsyncHookCallback]] = {}
        self._global_hooks: list[HookCallback] = []
        self._enabled = True

    def register(
        self,
        event: HookEvent,
        callback: HookCallback,
    ) -> None:
        """
        Register a synchronous hook for an event.
        
        Args:
            event: The event to hook
            callback: Function to call when event occurs
        """
        if event not in self._sync_hooks:
            self._sync_hooks[event] = []
        self._sync_hooks[event].append(callback)

    def register_async(
        self,
        event: HookEvent,
        callback: AsyncHookCallback,
    ) -> None:
        """
        Register an async hook for an event.
        
        Args:
            event: The event to hook
            callback: Async function to call when event occurs
        """
        if event not in self._async_hooks:
            self._async_hooks[event] = []
        self._async_hooks[event].append(callback)

    def register_global(self, callback: HookCallback) -> None:
        """
        Register a hook that fires for all events.
        
        Args:
            callback: Function to call for any event
        """
        self._global_hooks.append(callback)

    def unregister(
        self,
        event: HookEvent,
        callback: HookCallback,
    ) -> bool:
        """
        Unregister a hook.
        
        Args:
            event: The event to unhook
            callback: The callback to remove
            
        Returns:
            True if callback was found and removed
        """
        if event in self._sync_hooks and callback in self._sync_hooks[event]:
            self._sync_hooks[event].remove(callback)
            return True
        return False

    def trigger(self, context: HookContext) -> list[Exception]:
        """
        Trigger all hooks for an event.
        
        Args:
            context: The hook context
            
        Returns:
            List of any exceptions that occurred
        """
        if not self._enabled:
            return []
        
        errors = []
        
        # Trigger global hooks
        for callback in self._global_hooks:
            try:
                callback(context)
            except Exception as e:
                errors.append(e)
        
        # Trigger event-specific hooks
        if context.event in self._sync_hooks:
            for callback in self._sync_hooks[context.event]:
                try:
                    callback(context)
                except Exception as e:
                    errors.append(e)
        
        return errors

    async def trigger_async(self, context: HookContext) -> list[Exception]:
        """
        Trigger all async hooks for an event.
        
        Args:
            context: The hook context
            
        Returns:
            List of any exceptions that occurred
        """
        if not self._enabled:
            return []
        
        errors = []
        
        # Trigger sync hooks first
        errors.extend(self.trigger(context))
        
        # Trigger async hooks
        if context.event in self._async_hooks:
            for callback in self._async_hooks[context.event]:
                try:
                    await callback(context)
                except Exception as e:
                    errors.append(e)
        
        return errors

    def enable(self) -> None:
        """Enable hook triggering."""
        self._enabled = True

    def disable(self) -> None:
        """Disable hook triggering (for testing/debugging)."""
        self._enabled = False

    def clear(self, event: HookEvent | None = None) -> None:
        """
        Clear registered hooks.
        
        Args:
            event: Specific event to clear, or None for all
        """
        if event:
            self._sync_hooks.pop(event, None)
            self._async_hooks.pop(event, None)
        else:
            self._sync_hooks.clear()
            self._async_hooks.clear()
            self._global_hooks.clear()

    def get_hook_count(self, event: HookEvent | None = None) -> int:
        """Get count of registered hooks."""
        if event:
            sync = len(self._sync_hooks.get(event, []))
            async_ = len(self._async_hooks.get(event, []))
            return sync + async_
        
        total = len(self._global_hooks)
        for hooks in self._sync_hooks.values():
            total += len(hooks)
        for hooks in self._async_hooks.values():
            total += len(hooks)
        return total


# Global hook registry instance
_global_registry = HookRegistry()


def get_hook_registry() -> HookRegistry:
    """Get the global hook registry."""
    return _global_registry


def on_event(event: HookEvent):
    """
    Decorator to register a hook for an event.
    
    Usage:
        @on_event(HookEvent.MEMORY_CREATED)
        def my_hook(context: HookContext):
            print(f"Memory created: {context.memory_id}")
    """
    def decorator(func: HookCallback) -> HookCallback:
        _global_registry.register(event, func)
        return func
    return decorator


# Convenience functions
def trigger_memory_created(memory: BaseMemory, **extra_data) -> None:
    """Trigger memory created event."""
    context = HookContext(
        event=HookEvent.MEMORY_CREATED,
        memory=memory,
        memory_id=memory.id,
        memory_type=memory.memory_type,
        data=extra_data,
    )
    _global_registry.trigger(context)


def trigger_memory_accessed(memory: BaseMemory, **extra_data) -> None:
    """Trigger memory accessed event."""
    context = HookContext(
        event=HookEvent.MEMORY_ACCESSED,
        memory=memory,
        memory_id=memory.id,
        memory_type=memory.memory_type,
        data=extra_data,
    )
    _global_registry.trigger(context)


def trigger_memory_deleted(memory_id: str, memory_type: MemoryType, **extra_data) -> None:
    """Trigger memory deleted event."""
    context = HookContext(
        event=HookEvent.MEMORY_DELETED,
        memory_id=memory_id,
        memory_type=memory_type,
        data=extra_data,
    )
    _global_registry.trigger(context)


def trigger_conflict_detected(
    memory_a_id: str,
    memory_b_id: str,
    conflict_type: str,
    **extra_data,
) -> None:
    """Trigger conflict detected event."""
    context = HookContext(
        event=HookEvent.CONFLICT_DETECTED,
        data={
            "memory_a_id": memory_a_id,
            "memory_b_id": memory_b_id,
            "conflict_type": conflict_type,
            **extra_data,
        },
    )
    _global_registry.trigger(context)


def trigger_search_completed(
    query: str,
    result_count: int,
    search_time_ms: float,
    **extra_data,
) -> None:
    """Trigger search completed event."""
    context = HookContext(
        event=HookEvent.SEARCH_COMPLETED,
        data={
            "query": query,
            "result_count": result_count,
            "search_time_ms": search_time_ms,
            **extra_data,
        },
    )
    _global_registry.trigger(context)
