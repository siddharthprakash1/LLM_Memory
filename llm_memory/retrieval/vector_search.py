"""
Production-Grade Vector Search Engine

Implements:
- ChromaDB with HNSW indexing for fast ANN search
- Hybrid search (vector + keyword)
- Metadata filtering
- Similarity thresholds
- Batch operations

Based on: SPI, Milvus, and production RAG best practices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
import hashlib

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from llm_memory.models.base import BaseMemory, MemoryType


logger = logging.getLogger(__name__)


@dataclass
class VectorSearchConfig:
    """Configuration for vector search."""
    
    # ChromaDB settings
    collection_name: str = "llm_memory"
    persist_directory: str | None = None  # None = in-memory
    
    # HNSW parameters (tune for speed vs accuracy)
    hnsw_space: str = "cosine"  # cosine, l2, ip
    hnsw_construction_ef: int = 200  # Higher = better index quality, slower build
    hnsw_search_ef: int = 100  # Higher = better recall, slower search
    hnsw_m: int = 16  # Connections per node (16-64 typical)
    
    # Search parameters
    default_k: int = 10
    similarity_threshold: float = 0.5  # Minimum similarity score
    
    # Hybrid search weights
    vector_weight: float = 0.7
    keyword_weight: float = 0.3


@dataclass
class VectorSearchResult:
    """Result from vector search."""
    
    memory_id: str
    content: str
    similarity_score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_type: str = ""
    created_at: datetime | None = None


class VectorSearchEngine:
    """
    Production-grade vector search engine using ChromaDB.
    
    Features:
    - HNSW indexing for fast approximate nearest neighbor search
    - Hybrid search combining vector similarity + keyword matching
    - Metadata filtering (by type, user, time range)
    - Automatic embedding management
    
    Usage:
        engine = VectorSearchEngine(config)
        await engine.initialize()
        
        # Add memories
        await engine.add_memory(memory)
        
        # Search
        results = await engine.search(query_embedding, k=10)
    """
    
    def __init__(self, config: VectorSearchConfig | None = None):
        self.config = config or VectorSearchConfig()
        self._client: Any = None
        self._collection: Any = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the vector store."""
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available, using fallback")
            self._initialized = True
            return
        
        if self._initialized:
            return
        
        # Create ChromaDB client
        if self.config.persist_directory:
            self._client = chromadb.PersistentClient(
                path=self.config.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False),
            )
        
        # Create or get collection with HNSW settings
        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={
                "hnsw:space": self.config.hnsw_space,
                "hnsw:construction_ef": self.config.hnsw_construction_ef,
                "hnsw:M": self.config.hnsw_m,
            },
        )
        
        self._initialized = True
        logger.info(f"VectorSearchEngine initialized with collection: {self.config.collection_name}")
    
    async def add_memory(
        self,
        memory: BaseMemory,
        embedding: list[float] | None = None,
    ) -> bool:
        """
        Add a memory to the vector store.
        
        Args:
            memory: Memory to add
            embedding: Pre-computed embedding (uses memory.embedding if None)
            
        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()
        
        if not CHROMADB_AVAILABLE:
            return False
        
        # Get embedding
        emb = embedding or memory.embedding
        if not emb:
            logger.warning(f"No embedding for memory {memory.id}, skipping vector store")
            return False
        
        # Prepare metadata
        metadata = {
            "memory_type": memory.memory_type.value,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "importance": memory.importance_score,  # Use computed property
            "strength": memory.current_strength,  # Use current_strength
            "access_count": memory.access_count,
            "is_active": memory.is_active,
        }
        
        # Add user-specific metadata
        if memory.metadata:
            if memory.metadata.user_id:
                metadata["user_id"] = memory.metadata.user_id
            if memory.metadata.scope:
                metadata["scope"] = memory.metadata.scope
            if memory.metadata.tags:
                metadata["tags"] = ",".join(memory.metadata.tags)
        
        try:
            self._collection.upsert(
                ids=[memory.id],
                embeddings=[emb],
                documents=[memory.content],
                metadatas=[metadata],
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add memory to vector store: {e}")
            return False
    
    async def add_memories_batch(
        self,
        memories: list[BaseMemory],
        embeddings: list[list[float]] | None = None,
    ) -> int:
        """
        Add multiple memories in batch (more efficient).
        
        Returns:
            Number of memories successfully added
        """
        if not self._initialized:
            await self.initialize()
        
        if not CHROMADB_AVAILABLE:
            return 0
        
        ids = []
        embs = []
        docs = []
        metas = []
        
        for i, memory in enumerate(memories):
            emb = embeddings[i] if embeddings else memory.embedding
            if not emb:
                continue
            
            ids.append(memory.id)
            embs.append(emb)
            docs.append(memory.content)
            
            metadata = {
                "memory_type": memory.memory_type.value,
                "created_at": memory.created_at.isoformat(),
                "importance": memory.importance_score,
                "strength": memory.current_strength,
            }
            if memory.metadata and memory.metadata.user_id:
                metadata["user_id"] = memory.metadata.user_id
            metas.append(metadata)
        
        if not ids:
            return 0
        
        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embs,
                documents=docs,
                metadatas=metas,
            )
            return len(ids)
        except Exception as e:
            logger.error(f"Batch add failed: {e}")
            return 0
    
    async def search(
        self,
        query_embedding: list[float],
        k: int | None = None,
        filters: dict[str, Any] | None = None,
        similarity_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search for similar memories using vector similarity.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            filters: Metadata filters (e.g., {"memory_type": "semantic"})
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of search results sorted by similarity
        """
        if not self._initialized:
            await self.initialize()
        
        if not CHROMADB_AVAILABLE or not self._collection:
            return []
        
        k = k or self.config.default_k
        threshold = similarity_threshold or self.config.similarity_threshold
        
        # Build ChromaDB where clause from filters
        where = None
        if filters:
            where = self._build_where_clause(filters)
        
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k * 2,  # Get more, filter by threshold
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
        
        # Process results
        search_results = []
        
        if not results["ids"] or not results["ids"][0]:
            return []
        
        for i, memory_id in enumerate(results["ids"][0]):
            # Convert distance to similarity (for cosine, similarity = 1 - distance)
            distance = results["distances"][0][i] if results["distances"] else 0
            similarity = 1 - distance  # Cosine distance to similarity
            
            if similarity < threshold:
                continue
            
            content = results["documents"][0][i] if results["documents"] else ""
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            
            search_results.append(VectorSearchResult(
                memory_id=memory_id,
                content=content,
                similarity_score=similarity,
                metadata=metadata,
                memory_type=metadata.get("memory_type", ""),
                created_at=datetime.fromisoformat(metadata["created_at"]) if "created_at" in metadata else None,
            ))
        
        # Sort by similarity (highest first)
        search_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return search_results[:k]
    
    async def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """
        Hybrid search combining vector similarity and keyword matching.
        
        This improves recall by catching both semantic and lexical matches.
        """
        if not self._initialized:
            await self.initialize()
        
        # Get vector search results
        vector_results = await self.search(
            query_embedding,
            k=k or self.config.default_k * 2,
            filters=filters,
            similarity_threshold=0.3,  # Lower threshold for hybrid
        )
        
        # Score by keyword overlap
        query_words = set(query_text.lower().split())
        
        for result in vector_results:
            content_words = set(result.content.lower().split())
            overlap = len(query_words & content_words)
            keyword_score = overlap / max(len(query_words), 1)
            
            # Combine scores
            result.similarity_score = (
                self.config.vector_weight * result.similarity_score +
                self.config.keyword_weight * keyword_score
            )
        
        # Re-sort and filter
        vector_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        threshold = self.config.similarity_threshold
        filtered = [r for r in vector_results if r.similarity_score >= threshold]
        
        return filtered[:k or self.config.default_k]
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from the vector store."""
        if not CHROMADB_AVAILABLE or not self._collection:
            return False
        
        try:
            self._collection.delete(ids=[memory_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            return False
    
    async def update_memory(
        self,
        memory: BaseMemory,
        embedding: list[float] | None = None,
    ) -> bool:
        """Update a memory in the vector store."""
        return await self.add_memory(memory, embedding)  # Upsert handles updates
    
    async def get_stats(self) -> dict[str, Any]:
        """Get vector store statistics."""
        if not CHROMADB_AVAILABLE or not self._collection:
            return {"available": False}
        
        try:
            count = self._collection.count()
            return {
                "available": True,
                "collection_name": self.config.collection_name,
                "document_count": count,
                "hnsw_space": self.config.hnsw_space,
            }
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    async def clear(self) -> None:
        """Clear all data from the vector store."""
        if not CHROMADB_AVAILABLE or not self._client:
            return
        
        try:
            self._client.delete_collection(self.config.collection_name)
            self._collection = self._client.create_collection(
                name=self.config.collection_name,
                metadata={
                    "hnsw:space": self.config.hnsw_space,
                    "hnsw:construction_ef": self.config.hnsw_construction_ef,
                    "hnsw:M": self.config.hnsw_m,
                },
            )
        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")
    
    def _build_where_clause(self, filters: dict[str, Any]) -> dict:
        """Build ChromaDB where clause from filters."""
        conditions = []
        
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append({key: {"$in": value}})
            elif isinstance(value, dict):
                # Already a condition
                conditions.append({key: value})
            else:
                conditions.append({key: {"$eq": value}})
        
        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$and": conditions}
        
        return {}


# Singleton instance for easy access
_default_engine: VectorSearchEngine | None = None


def get_vector_engine(config: VectorSearchConfig | None = None) -> VectorSearchEngine:
    """Get or create the default vector search engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = VectorSearchEngine(config)
    return _default_engine
