"""
Embedding generation for memory content.

Supports multiple providers:
- Ollama (local, default)
- OpenAI
- Sentence Transformers (local)
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import BaseModel

from llm_memory.config import EmbeddingConfig


class EmbeddingResult(BaseModel):
    """Result of an embedding operation."""

    text: str
    embedding: list[float]
    model: str
    dimensions: int


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        pass

    async def embed_with_metadata(self, text: str) -> EmbeddingResult:
        """Generate embedding with metadata."""
        embedding = await self.embed(text)
        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.config.model,
            dimensions=len(embedding),
        )


class OllamaEmbedder(BaseEmbedder):
    """
    Ollama-based embedder for local embedding generation.
    
    Uses Ollama's embedding API with models like:
    - nomic-embed-text (768 dimensions)
    - mxbai-embed-large (1024 dimensions)
    - all-minilm (384 dimensions)
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.base_url = config.ollama_base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def embed(self, text: str) -> list[float]:
        """Generate embedding using Ollama."""
        client = await self._get_client()

        response = await client.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.config.model,
                "prompt": text,
            },
        )
        response.raise_for_status()

        data = response.json()
        return data["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        # Ollama doesn't have native batch support, so we parallelize
        tasks = [self.embed(text) for text in texts]
        
        # Process in batches to avoid overwhelming the server
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)

        return results

    async def __aenter__(self) -> "OllamaEmbedder":
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI-based embedder.
    
    Uses OpenAI's embedding API with models like:
    - text-embedding-3-small (1536 dimensions)
    - text-embedding-3-large (3072 dimensions)
    - text-embedding-ada-002 (1536 dimensions)
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI()
            except ImportError:
                raise ImportError("openai package required for OpenAI embeddings")
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Generate embedding using OpenAI."""
        client = self._get_client()

        response = await client.embeddings.create(
            model=self.config.model,
            input=text,
        )

        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        client = self._get_client()

        # OpenAI supports batch embeddings natively
        response = await client.embeddings.create(
            model=self.config.model,
            input=texts,
        )

        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Sentence Transformers embedder for fully local operation.
    
    Uses models like:
    - all-MiniLM-L6-v2 (384 dimensions)
    - all-mpnet-base-v2 (768 dimensions)
    - paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions)
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._model: Any = None

    def _get_model(self) -> Any:
        """Get or create the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.config.model)
            except ImportError:
                raise ImportError(
                    "sentence-transformers package required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    async def embed(self, text: str) -> list[float]:
        """Generate embedding using sentence transformers."""
        model = self._get_model()
        
        # Run in thread pool since sentence-transformers is sync
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, lambda: model.encode(text, convert_to_numpy=True)
        )

        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        model = self._get_model()

        # Run in thread pool
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: model.encode(texts, convert_to_numpy=True)
        )

        return embeddings.tolist()


def create_embedder(config: EmbeddingConfig | None = None) -> BaseEmbedder:
    """
    Factory function to create the appropriate embedder.
    
    Args:
        config: Embedding configuration. Uses defaults if None.
        
    Returns:
        Configured embedder instance.
    """
    if config is None:
        config = EmbeddingConfig()

    providers = {
        "ollama": OllamaEmbedder,
        "openai": OpenAIEmbedder,
        "sentence-transformers": SentenceTransformerEmbedder,
    }

    embedder_class = providers.get(config.provider)
    if embedder_class is None:
        raise ValueError(f"Unknown embedding provider: {config.provider}")

    return embedder_class(config)


class MemoryEmbedder:
    """
    High-level embedder for memory content.
    
    Handles memory-specific embedding logic like:
    - Combining content and summary
    - Caching embeddings
    - Batch processing memories
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self._embedder = create_embedder(self.config)
        self._cache: dict[str, list[float]] = {}

    async def embed_memory_content(
        self,
        content: str,
        summary: str | None = None,
        use_cache: bool = True,
    ) -> list[float]:
        """
        Generate embedding for memory content.
        
        If summary is provided, combines content and summary for richer embedding.
        """
        # Create cache key
        cache_key = f"{content[:100]}:{summary[:50] if summary else ''}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Combine content and summary for richer embedding
        if summary:
            text = f"{summary}\n\n{content}"
        else:
            text = content

        # Truncate if too long (most models have token limits)
        max_chars = 8000  # Conservative limit
        if len(text) > max_chars:
            text = text[:max_chars]

        embedding = await self._embedder.embed(text)

        if use_cache:
            self._cache[cache_key] = embedding

        return embedding

    async def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a search query.
        
        Queries are embedded directly without additional processing.
        """
        return await self._embedder.embed(query)

    async def embed_memories_batch(
        self,
        memories: list[tuple[str, str | None]],  # (content, summary) pairs
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple memories.
        
        Args:
            memories: List of (content, summary) tuples.
            
        Returns:
            List of embeddings.
        """
        texts = []
        for content, summary in memories:
            if summary:
                texts.append(f"{summary}\n\n{content}")
            else:
                texts.append(content)

        return await self._embedder.embed_batch(texts)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    async def close(self) -> None:
        """Close the embedder resources."""
        if hasattr(self._embedder, "close"):
            await self._embedder.close()

    async def __aenter__(self) -> "MemoryEmbedder":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
