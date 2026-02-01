"""
Memory System V4 - CORE-Style Architecture

This is a fundamental redesign based on top-performing systems:
- CORE (88% on LOCOMO)
- Mem0 (26% better than OpenAI)
- MemGPT (virtual memory management)

Key differences from V3:
1. LLM-based fact extraction at INGEST time
2. Proper normalization (clean timestamps, resolve pronouns)
3. Conflict resolution (track changes over time)
4. Dual storage: FACTS + EPISODES
5. Temporal state tracking for duration questions
6. Multi-angle retrieval (keyword + semantic + graph)

Usage:
    from llm_memory.memory_v4 import create_memory_v4, create_retriever
    
    # Create memory
    memory = create_memory_v4(user_id="user123")
    
    # Add conversation turn (full pipeline)
    episode, facts = memory.add_conversation_turn(
        speaker="Caroline",
        text="I moved from Sweden 4 years ago and I love hiking!",
        date="7 May 2023"
    )
    
    # Retrieve with multi-angle search
    retriever = create_retriever(memory)
    context = retriever.build_context("What does Caroline like?")
    
    # Answer duration questions
    answer = memory.answer_duration_question("How long has Caroline been away from Sweden?")
"""

from .normalizer import TextNormalizer, NormalizationContext
from .llm_extractor import LLMFactExtractor, ExtractedFact, FactType
from .conflict_resolver import ConflictResolver, ConflictType, ConflictResult
from .temporal_state import TemporalStateTracker, TemporalState, TemporalType
from .memory_store import MemoryStoreV4, Episode, create_memory_v4
from .retrieval import MultiAngleRetriever, RetrievalResult, create_retriever
from .multi_hop import MultiHopReasoner, QueryDecomposer, ReasoningStep
from .advanced_retrieval import AdvancedRetriever

__all__ = [
    # Core classes
    "MemoryStoreV4",
    "Episode",
    "create_memory_v4",
    
    # Extraction
    "LLMFactExtractor",
    "ExtractedFact",
    "FactType",
    
    # Normalization
    "TextNormalizer",
    "NormalizationContext",
    
    # Conflict Resolution
    "ConflictResolver",
    "ConflictType",
    "ConflictResult",
    
    # Temporal
    "TemporalStateTracker",
    "TemporalState",
    "TemporalType",
    
    # Retrieval
    "MultiAngleRetriever",
    "RetrievalResult",
    "create_retriever",
    "AdvancedRetriever",
    
    # Reasoning
    "MultiHopReasoner",
    "QueryDecomposer",
    "ReasoningStep",
]
