"""
Cached Embedder for Efficient Embedding Generation.

Features:
- In-memory LRU cache for repeated queries
- Support for Ollama embeddings
- Fallback to simple hash-based embeddings
"""

import re
import hashlib
import numpy as np
from typing import Any
import httpx


class CachedEmbedder:
    """
    Embedding generator with caching for efficiency.
    
    Supports:
    - Ollama embeddings (local)
    - Simple fallback embeddings
    """
    
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dimensions: int = 768,
        max_cache_size: int = 10000,
    ):
        """
        Initialize the embedder.
        
        Args:
            model: Embedding model name (for Ollama)
            base_url: Ollama API base URL
            dimensions: Embedding dimensions
            max_cache_size: Maximum number of cached embeddings
        """
        self.model = model
        self.base_url = base_url
        self.dimensions = dimensions
        self.max_cache_size = max_cache_size
        
        # LRU-style cache (dict preserves insertion order in Python 3.7+)
        self._cache: dict[str, np.ndarray] = {}
    
    def embed(self, text: str) -> np.ndarray:
        """
        Get embedding for text, using cache if available.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Generate cache key
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Generate embedding
        try:
            embedding = self._ollama_embed(text)
        except Exception:
            # Fallback to simple embedding
            embedding = self._simple_embed(text)
        
        # Cache it (with LRU eviction)
        if len(self._cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[cache_key] = embedding
        return embedding
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings (shape: [len(texts), dimensions])
        """
        return np.array([self.embed(t) for t in texts])
    
    def _ollama_embed(self, text: str) -> np.ndarray:
        """Get embedding from Ollama API."""
        # Truncate very long texts
        text = text[:2000]
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            response.raise_for_status()
            
            data = response.json()
            embedding = data.get("embedding", [])
            
            if embedding:
                return np.array(embedding, dtype=np.float32)
            
            raise ValueError("No embedding returned")
    
    async def embed_async(self, text: str) -> np.ndarray:
        """Async version of embed."""
        # Generate cache key
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Generate embedding
        try:
            embedding = await self._ollama_embed_async(text)
        except Exception:
            embedding = self._simple_embed(text)
        
        # Cache it
        if len(self._cache) >= self.max_cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[cache_key] = embedding
        return embedding
    
    async def _ollama_embed_async(self, text: str) -> np.ndarray:
        """Async embedding from Ollama API."""
        text = text[:2000]
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            response.raise_for_status()
            
            data = response.json()
            embedding = data.get("embedding", [])
            
            if embedding:
                return np.array(embedding, dtype=np.float32)
            
            raise ValueError("No embedding returned")
    
    def _simple_embed(self, text: str) -> np.ndarray:
        """
        Fallback simple embedding based on word hashing.
        
        This is not as good as real embeddings but works offline.
        """
        vec = np.zeros(self.dimensions, dtype=np.float32)
        
        # Extract words (at least 3 characters)
        words = re.findall(r'\b\w{3,}\b', text.lower())
        
        for i, word in enumerate(words[:100]):
            # Hash word to get index
            idx = hash(word) % self.dimensions
            # Weighted by position (earlier words more important)
            vec[idx] += 1.0 / (i + 1)
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
    
    @property
    def cache_size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
