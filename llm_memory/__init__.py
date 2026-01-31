"""
LLM Memory - Hierarchical Long-Term Memory for AI Agents

A comprehensive memory system implementing:
- Three-tier memory hierarchy (STM → Episodic → Semantic)
- Memory V3 with Knowledge Graph and Multi-hop Reasoning
- Multi-Agent LangGraph Framework
- Memory decay with importance scoring
- Consolidation pipeline
- Conflict resolution
- Intent-aware retrieval

Quick Start (Original API):
    from llm_memory import MemorySystem
    
    system = MemorySystem()
    await system.initialize()
    await system.remember("User prefers Python for data science")
    results = await system.recall("What language does user prefer?")

Quick Start (Memory V3 with Knowledge Graph):
    from llm_memory.memory_v3 import create_memory_v3
    
    memory = create_memory_v3(user_id="user1")
    memory.add("Alice loves hiking in the mountains", speaker="Alice")
    results = memory.search("What does Alice enjoy?")
    context = memory.get_context("Tell me about Alice")

Quick Start (Multi-Agent System):
    from llm_memory.agents import create_graph, run_graph
    
    graph = create_graph(user_id="user1")
    result = run_graph(graph, task="Research and analyze AI trends")
    print(result["final_response"])
"""

from llm_memory.config import MemoryConfig
from llm_memory.models.base import MemoryType, MemorySource, BaseMemory
from llm_memory.models.short_term import ShortTermMemory, STMRole
from llm_memory.models.episodic import EpisodicMemory, EventType
from llm_memory.models.semantic import SemanticMemory, FactType

from llm_memory.api.memory_system import MemorySystem, MemorySystemConfig
from llm_memory.api.memory_api import MemoryAPI

# Memory V3 with Knowledge Graph
from llm_memory.memory_v3 import (
    HierarchicalMemoryV3,
    MemoryItemV3,
    create_memory_v3,
    EntityExtractor,
    KnowledgeGraph,
    KGTriple,
    QueryDecomposer,
    ReRanker,
    CachedEmbedder,
)

__version__ = "0.2.0"

__all__ = [
    # Main entry points
    "MemorySystem",
    "MemorySystemConfig",
    "MemoryAPI",
    
    # Memory V3 (Knowledge Graph + Multi-hop)
    "HierarchicalMemoryV3",
    "MemoryItemV3",
    "create_memory_v3",
    "EntityExtractor",
    "KnowledgeGraph",
    "KGTriple",
    "QueryDecomposer",
    "ReRanker",
    "CachedEmbedder",
    
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
