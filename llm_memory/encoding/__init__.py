"""
Encoding module for memory transformation.

Provides:
- Embedding generation (Ollama, OpenAI, Sentence Transformers)
- Content summarization
- Feature extraction
"""

from llm_memory.encoding.embedder import (
    BaseEmbedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    MemoryEmbedder,
    EmbeddingResult,
    create_embedder,
)
from llm_memory.encoding.summarizer import (
    BaseSummarizer,
    OllamaSummarizer,
    OpenAISummarizer,
    AnthropicSummarizer,
    MemorySummarizer,
    SummaryResult,
    create_summarizer,
)

__all__ = [
    # Embedder base and implementations
    "BaseEmbedder",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    "MemoryEmbedder",
    "EmbeddingResult",
    "create_embedder",
    # Summarizer base and implementations
    "BaseSummarizer",
    "OllamaSummarizer",
    "OpenAISummarizer",
    "AnthropicSummarizer",
    "MemorySummarizer",
    "SummaryResult",
    "create_summarizer",
]
