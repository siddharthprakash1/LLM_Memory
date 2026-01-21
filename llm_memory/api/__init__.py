"""
API module for LLM Memory system.

Provides:
- MemorySystem: Main orchestrator for all memory operations
- MemoryAPI: Programmatic API for memory management
- Hooks: Event system for memory lifecycle
- Integrations: LangChain and other framework integrations
"""

from llm_memory.api.memory_api import (
    MemoryAPI,
    MemoryOperation,
    OperationResult,
    StoreRequest,
    SearchRequest,
)
from llm_memory.api.memory_system import (
    MemorySystem,
    MemorySystemConfig,
)
from llm_memory.api.hooks import (
    HookEvent,
    HookContext,
    HookRegistry,
    HookCallback,
    get_hook_registry,
    on_event,
    trigger_memory_created,
    trigger_memory_accessed,
    trigger_memory_deleted,
    trigger_conflict_detected,
    trigger_search_completed,
)

__all__ = [
    # Memory API
    "MemoryAPI",
    "MemoryOperation",
    "OperationResult",
    "StoreRequest",
    "SearchRequest",
    # Memory System
    "MemorySystem",
    "MemorySystemConfig",
    # Hooks
    "HookEvent",
    "HookContext",
    "HookRegistry",
    "HookCallback",
    "get_hook_registry",
    "on_event",
    "trigger_memory_created",
    "trigger_memory_accessed",
    "trigger_memory_deleted",
    "trigger_conflict_detected",
    "trigger_search_completed",
]
