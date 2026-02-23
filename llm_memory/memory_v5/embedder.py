"""
Embedding & Reranking Infrastructure for Memory V5.

Provides two core capabilities:
1. MemoryEmbedder — Bi-encoder embeddings using sentence-transformers (all-MiniLM-L6-v2)
   - 384-dimensional dense vectors
   - Runs locally, no API key needed
   - Lazy model loading (loads on first encode() call)

2. RetrievalReranker — Cross-encoder reranking using ms-marco-MiniLM-L6-v2
   - Pairwise (query, passage) scoring for precise relevance
   - Used as post-retrieval reranking step
   - Runs locally, no API key needed

Both models are ~23MB each and download from HuggingFace on first use.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Sentinel for lazy imports
_SentenceTransformer = None
_CrossEncoder = None


def _lazy_import_sentence_transformer():
    """Lazy import to avoid loading torch at module import time."""
    global _SentenceTransformer
    if _SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SentenceTransformer = SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embedding-based retrieval. "
                "Install it with: pip install sentence-transformers>=2.2.0"
            )
    return _SentenceTransformer


def _lazy_import_cross_encoder():
    """Lazy import to avoid loading torch at module import time."""
    global _CrossEncoder
    if _CrossEncoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _CrossEncoder = CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install it with: pip install sentence-transformers>=2.2.0"
            )
    return _CrossEncoder


class MemoryEmbedder:
    """
    Bi-encoder embedding model for semantic search.

    Uses sentence-transformers/all-MiniLM-L6-v2:
    - 384-dimensional embeddings
    - ~23MB model size
    - Runs on CPU (fast enough for <10K facts)
    - No API key required

    Usage:
        embedder = MemoryEmbedder()
        vecs = embedder.encode(["Alice likes hiking", "Bob works at Google"])
        query_vec = embedder.encode(["What does Alice enjoy?"])
        scores = embedder.similarity(query_vec[0], vecs)
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None

    def _load_model(self):
        """Load the sentence transformer model (lazy, first call only)."""
        if self._model is None:
            SentenceTransformer = _lazy_import_sentence_transformer()
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded (dim={self.EMBEDDING_DIM})")
        return self._model

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode texts into dense embeddings.

        Args:
            texts: List of strings to embed.
            batch_size: Batch size for encoding (higher = faster but more RAM).
            normalize: Whether to L2-normalize embeddings (recommended for cosine sim).
            show_progress: Show progress bar during encoding.

        Returns:
            np.ndarray of shape (len(texts), EMBEDDING_DIM).
        """
        if not texts:
            return np.array([]).reshape(0, self.EMBEDDING_DIM)

        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
        )
        return np.array(embeddings)

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text string. Returns 1D array of shape (EMBEDDING_DIM,)."""
        return self.encode([text], normalize=normalize)[0]

    def similarity(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between a query embedding and corpus embeddings.

        Args:
            query_embedding: 1D array of shape (EMBEDDING_DIM,).
            corpus_embeddings: 2D array of shape (n, EMBEDDING_DIM).

        Returns:
            1D array of similarity scores of shape (n,).
        """
        if corpus_embeddings.size == 0:
            return np.array([])

        # Ensure 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if corpus_embeddings.ndim == 1:
            corpus_embeddings = corpus_embeddings.reshape(1, -1)

        # Cosine similarity (embeddings are already L2-normalized)
        scores = np.dot(query_embedding, corpus_embeddings.T).flatten()
        return scores

    def top_k_indices(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """
        Get top-k most similar corpus indices for a query.

        Args:
            query_embedding: 1D array of shape (EMBEDDING_DIM,).
            corpus_embeddings: 2D array of shape (n, EMBEDDING_DIM).
            top_k: Number of results to return.
            threshold: Minimum similarity score to include.

        Returns:
            List of (index, score) tuples, sorted by score descending.
        """
        if corpus_embeddings.size == 0:
            return []

        scores = self.similarity(query_embedding, corpus_embeddings)

        # Filter by threshold
        valid_mask = scores >= threshold
        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_mask]

        # Sort by score descending
        sorted_order = np.argsort(valid_scores)[::-1][:top_k]

        return [
            (int(valid_indices[i]), float(valid_scores[i]))
            for i in sorted_order
        ]


class RetrievalReranker:
    """
    Cross-encoder reranker for post-retrieval quality improvement.

    Uses cross-encoder/ms-marco-MiniLM-L6-v2:
    - Pairwise (query, passage) relevance scoring
    - ~23MB model size
    - Runs on CPU
    - No API key required

    In a typical retrieve-then-rerank pipeline:
    1. Bi-encoder (MemoryEmbedder) retrieves top ~50 candidates
    2. Cross-encoder (RetrievalReranker) re-scores top ~50 → returns top ~10

    Usage:
        reranker = RetrievalReranker()
        ranked = reranker.rerank("What does Alice like?", candidates, top_k=10)
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None

    def _load_model(self):
        """Load the cross-encoder model (lazy, first call only)."""
        if self._model is None:
            CrossEncoder = _lazy_import_cross_encoder()
            logger.info(f"Loading reranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            logger.info("Reranker model loaded")
        return self._model

    def rerank(
        self,
        query: str,
        passages: List[str],
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Rerank passages for a query using the cross-encoder.

        Args:
            query: The search query.
            passages: List of passage texts to rerank.
            top_k: Number of top results to return.

        Returns:
            List of (original_index, score) tuples, sorted by score descending.
        """
        if not passages:
            return []

        model = self._load_model()

        # Create query-passage pairs
        pairs = [(query, passage) for passage in passages]

        # Get cross-encoder scores
        scores = model.predict(pairs)

        # Sort by score descending
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return [(idx, float(score)) for idx, score in indexed_scores[:top_k]]

    def rerank_with_metadata(
        self,
        query: str,
        items: List[dict],
        text_key: str = "content",
        top_k: int = 10,
    ) -> List[dict]:
        """
        Rerank items (dicts with a text field) and return them with updated scores.

        Args:
            query: The search query.
            items: List of dicts, each containing a text field.
            text_key: Key in each dict that holds the passage text.
            top_k: Number of top results to return.

        Returns:
            List of dicts with added 'rerank_score' field, sorted by score descending.
        """
        if not items:
            return []

        passages = [item.get(text_key, "") for item in items]
        ranked = self.rerank(query, passages, top_k=top_k)

        result = []
        for idx, score in ranked:
            item = items[idx].copy()
            item["rerank_score"] = score
            result.append(item)

        return result


# Module-level singletons (lazy — no model loaded until first use)
_embedder_instance: Optional[MemoryEmbedder] = None
_reranker_instance: Optional[RetrievalReranker] = None


def get_embedder(model_name: str = None) -> MemoryEmbedder:
    """Get the shared embedder instance (singleton, lazy-loaded)."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = MemoryEmbedder(model_name=model_name)
    return _embedder_instance


def get_reranker(model_name: str = None) -> RetrievalReranker:
    """Get the shared reranker instance (singleton, lazy-loaded)."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = RetrievalReranker(model_name=model_name)
    return _reranker_instance
