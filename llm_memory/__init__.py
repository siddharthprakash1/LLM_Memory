"""
LLM Memory - Hierarchical Long-Term Memory for Agents

A cognitive architecture implementing three-tier memory:
- Short-term Memory (STM): Working context, fast decay
- Episodic Memory: Event-based memories with temporal context
- Semantic Memory: Facts, patterns, generalizations

Features:
- Memory decay with importance scoring
- Consolidation pipeline (STM → Episodic → Semantic)
- Conflict detection and resolution
- Intent-aware retrieval
"""

__version__ = "0.1.0"
__author__ = "LLM Memory Project"

from llm_memory.config import MemoryConfig
from llm_memory.models.base import BaseMemory, MemoryMetadata, MemoryType
from llm_memory.models.short_term import ShortTermMemory, WorkingContext
from llm_memory.models.episodic import EpisodicMemory, Episode, TemporalContext
from llm_memory.models.semantic import SemanticMemory, Fact, Concept, Relationship

__all__ = [
    # Config
    "MemoryConfig",
    # Base
    "BaseMemory",
    "MemoryMetadata",
    "MemoryType",
    # Short-term
    "ShortTermMemory",
    "WorkingContext",
    # Episodic
    "EpisodicMemory",
    "Episode",
    "TemporalContext",
    # Semantic
    "SemanticMemory",
    "Fact",
    "Concept",
    "Relationship",
]
