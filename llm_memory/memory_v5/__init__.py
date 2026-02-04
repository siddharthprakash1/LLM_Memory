"""
Memory V5 - Next-Generation Cognitive Memory Architecture

This implements a SOTA memory system inspired by:
- Mem0/Mem0g: Graph-based memory with entity-relationship modeling
- Memory-R1: RL-trained memory manager (ADD/UPDATE/DELETE/NOOP)
- LightMem: Tiered memory (Sensory → STM → LTM) with topic clustering
- GraphFlow: Chain-of-Explorations retrieval with flow optimization
- RMM: Reflective Memory Management with prospective/retrospective reflection

Key Innovations over V4:
1. Graph Memory Store - Directed labeled graphs with typed entities
2. Tiered Architecture - Sensory/STM/LTM with automatic promotion
3. RL Memory Manager - Learned ADD/UPDATE/DELETE/NOOP policies
4. Advanced Retrieval - CoE + multi-hop graph traversal
5. Reflective Management - Dynamic summarization + retrieval refinement

Architecture Overview:
```
┌─────────────────────────────────────────────────────────────────┐
│                     MEMORY V5 ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   SENSORY    │───▶│    SHORT     │───▶│    LONG      │       │
│  │   MEMORY     │    │    TERM      │    │    TERM      │       │
│  │  (Filtering) │    │   (Working)  │    │  (Persistent)│       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌─────────────────────────────────────────────────────┐        │
│  │              GRAPH MEMORY STORE                      │        │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │        │
│  │  │ Entities │  │Relations│  │ Triplets│             │        │
│  │  │ (Nodes)  │  │ (Edges) │  │ (Facts) │             │        │
│  │  └─────────┘  └─────────┘  └─────────┘             │        │
│  └─────────────────────────────────────────────────────┘        │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────┐        │
│  │           RL MEMORY MANAGER                          │        │
│  │  ┌─────┐ ┌────────┐ ┌────────┐ ┌──────┐            │        │
│  │  │ ADD │ │ UPDATE │ │ DELETE │ │ NOOP │            │        │
│  │  └─────┘ └────────┘ └────────┘ └──────┘            │        │
│  └─────────────────────────────────────────────────────┘        │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────┐        │
│  │           ADVANCED RETRIEVAL                         │        │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │        │
│  │  │   CoE   │  │GraphFlow│  │ Re-Rank │             │        │
│  │  │ Search  │  │Traversal│  │ + Fuse  │             │        │
│  │  └─────────┘  └─────────┘  └─────────┘             │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

Usage:
    from llm_memory.memory_v5 import MemoryV5, create_memory_v5
    
    # Create memory instance
    memory = create_memory_v5(user_id="user1")
    
    # Add conversation turn (automatic tiered processing)
    memory.add_turn(speaker="User", text="I'm a vegetarian and allergic to dairy", date="2024-01-15")
    
    # Query with advanced retrieval
    context = memory.query("What dietary restrictions does the user have?")
    
    # Get graph-based context for multi-hop reasoning
    graph_context = memory.query_graph("What foods should I avoid?")
"""

from .graph_store import GraphMemoryStore, Entity, Relation, Triplet
from .tiered_memory import TieredMemory, SensoryMemory, ShortTermMemory, LongTermMemory
from .memory_manager import MemoryManager, MemoryOperation
from .retrieval_v5 import AdvancedRetriever, ChainOfExplorations
from .reflective import ReflectiveManager
from .memory_store_v5 import MemoryStoreV5, create_memory_v5

__all__ = [
    # Core Store
    "MemoryStoreV5",
    "create_memory_v5",
    
    # Graph Memory
    "GraphMemoryStore",
    "Entity", 
    "Relation",
    "Triplet",
    
    # Tiered Memory
    "TieredMemory",
    "SensoryMemory",
    "ShortTermMemory",
    "LongTermMemory",
    
    # Memory Manager
    "MemoryManager",
    "MemoryOperation",
    
    # Retrieval
    "AdvancedRetriever",
    "ChainOfExplorations",
    
    # Reflective
    "ReflectiveManager",
]
