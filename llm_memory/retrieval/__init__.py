"""
Retrieval module for intent-aware memory search.

Provides:
- Query intent classification
- Multi-tier memory search
- Result re-ranking with diversity
- Context-aware retrieval
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
]
