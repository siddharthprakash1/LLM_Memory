"""
Memory System V3 - Advanced Implementation

Features:
1. Entity extraction and knowledge graph
2. Query decomposition for multi-hop reasoning
3. Re-ranking with cross-attention scoring
4. Iterative retrieval for complex questions
5. Temporal reasoning chain
6. Date normalization for better temporal accuracy
7. Answer extraction for improved QA
8. HyDE (Hypothetical Document Embeddings) for better retrieval
9. Evidence chain tracking for multi-hop reasoning
10. Advanced temporal reasoning with duration extraction
"""

from .entity_extractor import EntityExtractor
from .knowledge_graph import KnowledgeGraph, KGTriple
from .query_decomposer import QueryDecomposer
from .reranker import ReRanker
from .memory_v3 import HierarchicalMemoryV3, MemoryItemV3, create_memory_v3
from .embedder import CachedEmbedder
from .date_normalizer import DateNormalizer
from .answer_extractor import AnswerExtractor
from .hyde import HyDEGenerator, QueryExpander
from .temporal_reasoning import TemporalReasoner, DurationExtractor, TemporalEvent
from .evidence_chain import EvidenceChainBuilder, MultiHopReasoner, ReasoningChain

__all__ = [
    "EntityExtractor",
    "KnowledgeGraph",
    "KGTriple", 
    "QueryDecomposer",
    "ReRanker",
    "HierarchicalMemoryV3",
    "MemoryItemV3",
    "create_memory_v3",
    "CachedEmbedder",
    "DateNormalizer",
    "AnswerExtractor",
    "HyDEGenerator",
    "QueryExpander",
    "TemporalReasoner",
    "DurationExtractor",
    "TemporalEvent",
    "EvidenceChainBuilder",
    "MultiHopReasoner",
    "ReasoningChain",
]
