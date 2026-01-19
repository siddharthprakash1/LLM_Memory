"""
Abstract base classes for storage backends.

Defines the interface that all storage backends must implement.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, TypeVar

from llm_memory.models.base import BaseMemory, MemoryType


# Custom exceptions
class StorageError(Exception):
    """Base exception for storage errors."""

    pass


class MemoryNotFoundError(StorageError):
    """Raised when a memory is not found."""

    pass


class ConnectionError(StorageError):
    """Raised when connection to storage fails."""

    pass


# Type variable for memory subclasses
T = TypeVar("T", bound=BaseMemory)


class BaseStorage(ABC):
    """
    Abstract base class for memory storage backends.
    
    Provides CRUD operations and query capabilities for memories.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Initialize connection to storage backend."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to storage backend."""
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if storage is connected."""
        pass

    # CRUD Operations
    @abstractmethod
    async def create(self, memory: BaseMemory) -> str:
        """
        Create a new memory in storage.
        
        Args:
            memory: The memory to store
            
        Returns:
            The ID of the created memory
        """
        pass

    @abstractmethod
    async def read(self, memory_id: str) -> BaseMemory | None:
        """
        Read a memory by ID.
        
        Args:
            memory_id: The ID of the memory to read
            
        Returns:
            The memory if found, None otherwise
        """
        pass

    @abstractmethod
    async def update(self, memory: BaseMemory) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory: The memory to update (must have existing ID)
            
        Returns:
            True if updated, False if not found
        """
        pass

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.
        
        Args:
            memory_id: The ID of the memory to delete
            
        Returns:
            True if deleted, False if not found
        """
        pass

    # Batch Operations
    @abstractmethod
    async def create_many(self, memories: list[BaseMemory]) -> list[str]:
        """
        Create multiple memories in batch.
        
        Args:
            memories: List of memories to create
            
        Returns:
            List of created memory IDs
        """
        pass

    @abstractmethod
    async def read_many(self, memory_ids: list[str]) -> list[BaseMemory]:
        """
        Read multiple memories by IDs.
        
        Args:
            memory_ids: List of memory IDs to read
            
        Returns:
            List of found memories (may be shorter than input if some not found)
        """
        pass

    @abstractmethod
    async def delete_many(self, memory_ids: list[str]) -> int:
        """
        Delete multiple memories by IDs.
        
        Args:
            memory_ids: List of memory IDs to delete
            
        Returns:
            Number of memories deleted
        """
        pass

    # Query Operations
    @abstractmethod
    async def query_by_type(
        self,
        memory_type: MemoryType,
        limit: int = 100,
        offset: int = 0,
    ) -> list[BaseMemory]:
        """
        Query memories by type.
        
        Args:
            memory_type: The type of memories to query
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of memories matching the type
        """
        pass

    @abstractmethod
    async def query_by_user(
        self,
        user_id: str,
        memory_type: MemoryType | None = None,
        limit: int = 100,
    ) -> list[BaseMemory]:
        """
        Query memories by user ID.
        
        Args:
            user_id: The user ID to query
            memory_type: Optional filter by memory type
            limit: Maximum number of results
            
        Returns:
            List of memories for the user
        """
        pass

    @abstractmethod
    async def query_by_scope(
        self,
        scope: str,
        user_id: str | None = None,
        memory_type: MemoryType | None = None,
        limit: int = 100,
    ) -> list[BaseMemory]:
        """
        Query memories by scope.
        
        Args:
            scope: The scope to query (global, project, session)
            user_id: Optional filter by user
            memory_type: Optional filter by type
            limit: Maximum number of results
            
        Returns:
            List of memories in the scope
        """
        pass

    @abstractmethod
    async def query_by_time_range(
        self,
        start: datetime,
        end: datetime,
        memory_type: MemoryType | None = None,
        limit: int = 100,
    ) -> list[BaseMemory]:
        """
        Query memories by creation time range.
        
        Args:
            start: Start of time range
            end: End of time range
            memory_type: Optional filter by type
            limit: Maximum number of results
            
        Returns:
            List of memories in the time range
        """
        pass

    @abstractmethod
    async def query_by_strength(
        self,
        min_strength: float = 0.0,
        max_strength: float = 1.0,
        memory_type: MemoryType | None = None,
        limit: int = 100,
    ) -> list[BaseMemory]:
        """
        Query memories by strength range.
        
        Args:
            min_strength: Minimum strength threshold
            max_strength: Maximum strength threshold
            memory_type: Optional filter by type
            limit: Maximum number of results
            
        Returns:
            List of memories within strength range
        """
        pass

    @abstractmethod
    async def query_weak_memories(
        self,
        threshold: float = 0.1,
        memory_type: MemoryType | None = None,
    ) -> list[BaseMemory]:
        """
        Query memories below strength threshold (candidates for garbage collection).
        
        Args:
            threshold: Strength threshold
            memory_type: Optional filter by type
            
        Returns:
            List of weak memories
        """
        pass

    @abstractmethod
    async def query_unconsolidated(
        self,
        memory_type: MemoryType,
        min_importance: float = 0.0,
        limit: int = 100,
    ) -> list[BaseMemory]:
        """
        Query memories that haven't been consolidated.
        
        Args:
            memory_type: Type of memories to query
            min_importance: Minimum importance score
            limit: Maximum number of results
            
        Returns:
            List of unconsolidated memories
        """
        pass

    # Statistics
    @abstractmethod
    async def count(self, memory_type: MemoryType | None = None) -> int:
        """
        Count memories in storage.
        
        Args:
            memory_type: Optional filter by type
            
        Returns:
            Number of memories
        """
        pass

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage stats
        """
        pass

    # Context manager support
    async def __aenter__(self) -> "BaseStorage":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()


class BaseVectorStorage(ABC):
    """
    Abstract base class for vector storage backends.
    
    Provides embedding storage and similarity search capabilities.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Initialize connection to vector storage."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to vector storage."""
        pass

    @abstractmethod
    async def create_collection(
        self,
        name: str,
        dimension: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Create a new collection for storing vectors.
        
        Args:
            name: Collection name
            dimension: Vector dimension
            metadata: Optional collection metadata
        """
        pass

    @abstractmethod
    async def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            name: Collection name
            
        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def collection_exists(self, name: str) -> bool:
        """
        Check if a collection exists.
        
        Args:
            name: Collection name
            
        Returns:
            True if exists
        """
        pass

    # Vector Operations
    @abstractmethod
    async def add_embedding(
        self,
        collection: str,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        document: str | None = None,
    ) -> str:
        """
        Add an embedding to a collection.
        
        Args:
            collection: Collection name
            id: Embedding ID
            embedding: The embedding vector
            metadata: Optional metadata
            document: Optional original document text
            
        Returns:
            The embedding ID
        """
        pass

    @abstractmethod
    async def add_embeddings(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> list[str]:
        """
        Add multiple embeddings to a collection.
        
        Args:
            collection: Collection name
            ids: List of embedding IDs
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dicts
            documents: Optional list of document texts
            
        Returns:
            List of embedding IDs
        """
        pass

    @abstractmethod
    async def get_embedding(
        self,
        collection: str,
        id: str,
    ) -> tuple[list[float], dict[str, Any]] | None:
        """
        Get an embedding by ID.
        
        Args:
            collection: Collection name
            id: Embedding ID
            
        Returns:
            Tuple of (embedding, metadata) or None if not found
        """
        pass

    @abstractmethod
    async def delete_embedding(self, collection: str, id: str) -> bool:
        """
        Delete an embedding by ID.
        
        Args:
            collection: Collection name
            id: Embedding ID
            
        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def update_embedding(
        self,
        collection: str,
        id: str,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        document: str | None = None,
    ) -> bool:
        """
        Update an existing embedding.
        
        Args:
            collection: Collection name
            id: Embedding ID
            embedding: Optional new embedding vector
            metadata: Optional new metadata
            document: Optional new document text
            
        Returns:
            True if updated, False if not found
        """
        pass

    # Search Operations
    @abstractmethod
    async def search(
        self,
        collection: str,
        query_embedding: list[float],
        limit: int = 10,
        filter: dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Search for similar embeddings.
        
        Args:
            collection: Collection name
            query_embedding: Query vector
            limit: Maximum number of results
            filter: Optional metadata filter
            include_embeddings: Whether to include embedding vectors in results
            
        Returns:
            List of results with id, score, metadata, and optionally embedding
        """
        pass

    @abstractmethod
    async def search_by_text(
        self,
        collection: str,
        query_text: str,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search by text (requires embedding generation).
        
        Args:
            collection: Collection name
            query_text: Query text
            limit: Maximum number of results
            filter: Optional metadata filter
            
        Returns:
            List of results
        """
        pass

    # Statistics
    @abstractmethod
    async def count(self, collection: str) -> int:
        """
        Count embeddings in a collection.
        
        Args:
            collection: Collection name
            
        Returns:
            Number of embeddings
        """
        pass

    # Context manager support
    async def __aenter__(self) -> "BaseVectorStorage":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()
