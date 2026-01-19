"""
Vector storage backend using ChromaDB.

Provides embedding storage and similarity search capabilities.
"""

from typing import Any

import chromadb
from chromadb.config import Settings

from llm_memory.config import StorageConfig, EmbeddingConfig
from llm_memory.storage.base import BaseVectorStorage, StorageError


class VectorStorage(BaseVectorStorage):
    """
    ChromaDB-based vector storage for embeddings.
    
    Supports:
    - Multiple collections for different memory types
    - Similarity search
    - Metadata filtering
    - Persistent storage
    """

    def __init__(
        self,
        storage_config: StorageConfig | None = None,
        embedding_config: EmbeddingConfig | None = None,
    ):
        """
        Initialize vector storage.
        
        Args:
            storage_config: Storage configuration
            embedding_config: Embedding configuration
        """
        self.storage_config = storage_config or StorageConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()
        self._client: chromadb.ClientAPI | None = None
        self._connected = False
        self._embedding_function = None

    async def connect(self) -> None:
        """Initialize ChromaDB client with persistence."""
        try:
            # Ensure directory exists
            persist_dir = self.storage_config.chroma_persist_directory
            persist_dir.mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB client
            self._client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # Set up embedding function if using OpenAI
            if self.embedding_config.provider == "openai":
                self._embedding_function = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
                    model_name=self.embedding_config.model,
                )

            self._connected = True
        except Exception as e:
            raise StorageError(f"Failed to connect to ChromaDB: {e}") from e

    async def disconnect(self) -> None:
        """Close ChromaDB connection."""
        # ChromaDB doesn't require explicit disconnection
        self._client = None
        self._connected = False

    def _ensure_connected(self) -> None:
        """Raise error if not connected."""
        if not self._connected or self._client is None:
            raise StorageError("Not connected to vector storage")

    async def create_collection(
        self,
        name: str,
        dimension: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a new collection for storing vectors."""
        self._ensure_connected()

        try:
            self._client.get_or_create_collection(
                name=name,
                metadata=metadata or {"dimension": dimension},
                embedding_function=self._embedding_function,
            )
        except Exception as e:
            raise StorageError(f"Failed to create collection {name}: {e}") from e

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        self._ensure_connected()

        try:
            self._client.delete_collection(name)
            return True
        except Exception:
            return False

    async def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        self._ensure_connected()

        try:
            collections = self._client.list_collections()
            return any(c.name == name for c in collections)
        except Exception:
            return False

    def _get_collection(self, name: str):
        """Get a collection by name."""
        self._ensure_connected()
        return self._client.get_collection(
            name=name,
            embedding_function=self._embedding_function,
        )

    # Vector Operations
    async def add_embedding(
        self,
        collection: str,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        document: str | None = None,
    ) -> str:
        """Add an embedding to a collection."""
        self._ensure_connected()

        try:
            coll = self._get_collection(collection)

            # Prepare data
            add_kwargs = {
                "ids": [id],
                "embeddings": [embedding],
            }

            if metadata:
                # ChromaDB doesn't support nested dicts, so flatten
                flat_metadata = self._flatten_metadata(metadata)
                add_kwargs["metadatas"] = [flat_metadata]

            if document:
                add_kwargs["documents"] = [document]

            coll.add(**add_kwargs)
            return id
        except Exception as e:
            raise StorageError(f"Failed to add embedding: {e}") from e

    async def add_embeddings(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> list[str]:
        """Add multiple embeddings to a collection."""
        self._ensure_connected()

        try:
            coll = self._get_collection(collection)

            add_kwargs = {
                "ids": ids,
                "embeddings": embeddings,
            }

            if metadatas:
                flat_metadatas = [self._flatten_metadata(m) for m in metadatas]
                add_kwargs["metadatas"] = flat_metadatas

            if documents:
                add_kwargs["documents"] = documents

            coll.add(**add_kwargs)
            return ids
        except Exception as e:
            raise StorageError(f"Failed to add embeddings: {e}") from e

    async def get_embedding(
        self,
        collection: str,
        id: str,
    ) -> tuple[list[float], dict[str, Any]] | None:
        """Get an embedding by ID."""
        self._ensure_connected()

        try:
            coll = self._get_collection(collection)
            result = coll.get(ids=[id], include=["embeddings", "metadatas"])

            if not result["ids"]:
                return None

            embedding = result["embeddings"][0] if result["embeddings"] else []
            metadata = result["metadatas"][0] if result["metadatas"] else {}

            return embedding, metadata
        except Exception:
            return None

    async def delete_embedding(self, collection: str, id: str) -> bool:
        """Delete an embedding by ID."""
        self._ensure_connected()

        try:
            coll = self._get_collection(collection)
            coll.delete(ids=[id])
            return True
        except Exception:
            return False

    async def update_embedding(
        self,
        collection: str,
        id: str,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        document: str | None = None,
    ) -> bool:
        """Update an existing embedding."""
        self._ensure_connected()

        try:
            coll = self._get_collection(collection)

            update_kwargs = {"ids": [id]}

            if embedding:
                update_kwargs["embeddings"] = [embedding]
            if metadata:
                update_kwargs["metadatas"] = [self._flatten_metadata(metadata)]
            if document:
                update_kwargs["documents"] = [document]

            coll.update(**update_kwargs)
            return True
        except Exception:
            return False

    # Search Operations
    async def search(
        self,
        collection: str,
        query_embedding: list[float],
        limit: int = 10,
        filter: dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        """Search for similar embeddings."""
        self._ensure_connected()

        try:
            coll = self._get_collection(collection)

            include = ["metadatas", "documents", "distances"]
            if include_embeddings:
                include.append("embeddings")

            # Build where clause for filtering
            where = None
            if filter:
                where = self._build_where_clause(filter)

            result = coll.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where,
                include=include,
            )

            # Convert to list of dicts
            results = []
            if result["ids"] and result["ids"][0]:
                for i, id in enumerate(result["ids"][0]):
                    item = {
                        "id": id,
                        "score": 1 - result["distances"][0][i] if result["distances"] else 0,
                        "distance": result["distances"][0][i] if result["distances"] else 0,
                    }

                    if result.get("metadatas") and result["metadatas"][0]:
                        item["metadata"] = result["metadatas"][0][i]

                    if result.get("documents") and result["documents"][0]:
                        item["document"] = result["documents"][0][i]

                    if include_embeddings and result.get("embeddings") and result["embeddings"][0]:
                        item["embedding"] = result["embeddings"][0][i]

                    results.append(item)

            return results
        except Exception as e:
            raise StorageError(f"Failed to search: {e}") from e

    async def search_by_text(
        self,
        collection: str,
        query_text: str,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search by text (uses embedding function)."""
        self._ensure_connected()

        try:
            coll = self._get_collection(collection)

            include = ["metadatas", "documents", "distances"]

            where = None
            if filter:
                where = self._build_where_clause(filter)

            result = coll.query(
                query_texts=[query_text],
                n_results=limit,
                where=where,
                include=include,
            )

            # Convert to list of dicts
            results = []
            if result["ids"] and result["ids"][0]:
                for i, id in enumerate(result["ids"][0]):
                    item = {
                        "id": id,
                        "score": 1 - result["distances"][0][i] if result["distances"] else 0,
                        "distance": result["distances"][0][i] if result["distances"] else 0,
                    }

                    if result.get("metadatas") and result["metadatas"][0]:
                        item["metadata"] = result["metadatas"][0][i]

                    if result.get("documents") and result["documents"][0]:
                        item["document"] = result["documents"][0][i]

                    results.append(item)

            return results
        except Exception as e:
            raise StorageError(f"Failed to search by text: {e}") from e

    # Statistics
    async def count(self, collection: str) -> int:
        """Count embeddings in a collection."""
        self._ensure_connected()

        try:
            coll = self._get_collection(collection)
            return coll.count()
        except Exception:
            return 0

    # Helper methods
    def _flatten_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Flatten nested metadata for ChromaDB compatibility."""
        flat = {}
        for key, value in metadata.items():
            if isinstance(value, dict):
                # Flatten nested dicts with dot notation
                for nested_key, nested_value in value.items():
                    if not isinstance(nested_value, (dict, list)):
                        flat[f"{key}.{nested_key}"] = nested_value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                if all(isinstance(v, (str, int, float)) for v in value):
                    flat[key] = ",".join(str(v) for v in value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                flat[key] = value

        return flat

    def _build_where_clause(self, filter: dict[str, Any]) -> dict[str, Any]:
        """Build ChromaDB where clause from filter dict."""
        # Simple conversion for basic filters
        # ChromaDB expects: {"field": {"$eq": value}} or {"$and": [...]}

        if not filter:
            return None

        conditions = []
        for key, value in filter.items():
            if isinstance(value, dict):
                # Already in ChromaDB format
                conditions.append({key: value})
            else:
                # Convert to ChromaDB format
                conditions.append({key: {"$eq": value}})

        if len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    async def ensure_collections(self) -> None:
        """Ensure all required collections exist."""
        self._ensure_connected()

        collections = [
            self.storage_config.stm_collection,
            self.storage_config.episodic_collection,
            self.storage_config.semantic_collection,
        ]

        for name in collections:
            await self.create_collection(
                name=name,
                dimension=self.embedding_config.dimensions,
            )
