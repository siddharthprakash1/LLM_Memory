"""
LLM Memory - Hierarchical Long-Term Memory for AI Agents

A comprehensive memory system implementing:
- Three-tier memory hierarchy (STM → Episodic → Semantic)
- Memory decay with importance scoring
- Consolidation pipeline
- Conflict resolution
- Intent-aware retrieval

Quick Start:
    from llm_memory import MemorySystem
    
    # Initialize
    system = MemorySystem()
    await system.initialize()
    
    # Store a memory
    await system.remember("User prefers Python for data science")
    
    # Recall memories
    results = await system.recall("What language does user prefer?")
    
    # Add to conversation
    await system.add_message("Hello!", role="user", session_id="chat_1")
"""

from llm_memory.config import MemoryConfig
from llm_memory.models.base import MemoryType, MemorySource, BaseMemory
from llm_memory.models.short_term import ShortTermMemory, STMRole
from llm_memory.models.episodic import EpisodicMemory, EventType
from llm_memory.models.semantic import SemanticMemory, FactType

from llm_memory.api.memory_system import MemorySystem, MemorySystemConfig
from llm_memory.api.memory_api import MemoryAPI

__version__ = "0.1.0"

__all__ = [
    # Main entry points
    "MemorySystem",
    "MemorySystemConfig",
    "MemoryAPI",
    # Configuration
    "MemoryConfig",
    # Memory types
    "MemoryType",
    "MemorySource",
    "BaseMemory",
    "ShortTermMemory",
    "EpisodicMemory",
    "SemanticMemory",
    # Enums
    "STMRole",
    "EventType",
    "FactType",
]
