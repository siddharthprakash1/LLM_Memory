"""
Storage backends for the LLM Memory system.

Provides:
- SQLite storage for metadata and structured data
- Vector storage (ChromaDB/Qdrant) for embeddings
- Abstract base classes for custom backends
"""

from llm_memory.storage.base import (
    BaseStorage,
    BaseVectorStorage,
    StorageError,
    MemoryNotFoundError,
)
from llm_memory.storage.sqlite import SQLiteStorage

# VectorStorage requires chromadb - import only if available
try:
    from llm_memory.storage.vector import VectorStorage
    _has_chromadb = True
except ImportError:
    VectorStorage = None  # type: ignore
    _has_chromadb = False

__all__ = [
    # Base
    "BaseStorage",
    "BaseVectorStorage",
    "StorageError",
    "MemoryNotFoundError",
    # Implementations
    "SQLiteStorage",
]

if _has_chromadb:
    __all__.append("VectorStorage")
