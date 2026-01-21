"""
Retrieval module for intent-aware memory search.

Provides:
- Query intent classification
- Multi-tier memory search
- Result re-ranking with diversity
- Context-aware retrieval
- Production vector search (ChromaDB)
- Temporal scoring and filtering
- Multi-hop reasoning
- RAG pipeline
"""

from llm_memory.retrieval.intent import (
    QueryIntent,
    IntentSignal,
    ClassifiedIntent,
    IntentClassifier,
    LLMIntentClassifier,
)
from llm_memory.retrieval.searcher import (
    SearchResult,
    SearchResults,
    SearchConfig,
    MemorySearcher,
    KeywordSearcher,
)
from llm_memory.retrieval.ranker import (
    RankingConfig,
    RankedResult,
    RankedResults,
    ResultRanker,
    ContextualRanker,
)
from llm_memory.retrieval.vector_search import (
    VectorSearchConfig,
    VectorSearchResult,
    VectorSearchEngine,
    get_vector_engine,
)
from llm_memory.retrieval.temporal import (
    TemporalStrategy,
    TemporalConfig,
    TemporalScore,
    TemporalScorer,
    apply_temporal_scoring,
)
from llm_memory.retrieval.multi_hop import (
    HopType,
    ReasoningHop,
    ReasoningPath,
    MultiHopConfig,
    MultiHopReasoner,
    ChainOfThoughtRetriever,
    multi_hop_retrieve,
)
from llm_memory.retrieval.rag_pipeline import (
    AnswerQuality,
    RAGConfig,
    RAGResult,
    RAGPipeline,
    create_rag_pipeline,
)

__all__ = [
    # Intent
    "QueryIntent",
    "IntentSignal",
    "ClassifiedIntent",
    "IntentClassifier",
    "LLMIntentClassifier",
    # Searcher
    "SearchResult",
    "SearchResults",
    "SearchConfig",
    "MemorySearcher",
    "KeywordSearcher",
    # Ranker
    "RankingConfig",
    "RankedResult",
    "RankedResults",
    "ResultRanker",
    "ContextualRanker",
    # Vector Search
    "VectorSearchConfig",
    "VectorSearchResult",
    "VectorSearchEngine",
    "get_vector_engine",
    # Temporal
    "TemporalStrategy",
    "TemporalConfig",
    "TemporalScore",
    "TemporalScorer",
    "apply_temporal_scoring",
    # Multi-hop
    "HopType",
    "ReasoningHop",
    "ReasoningPath",
    "MultiHopConfig",
    "MultiHopReasoner",
    "ChainOfThoughtRetriever",
    "multi_hop_retrieve",
    # RAG Pipeline
    "AnswerQuality",
    "RAGConfig",
    "RAGResult",
    "RAGPipeline",
    "create_rag_pipeline",
]
