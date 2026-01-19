"""
Consolidation module for memory promotion pipeline.

Provides:
- STM → Episodic promotion
- Episodic → Semantic abstraction
- Memory merging and deduplication
- Garbage collection
- Background consolidation scheduler
"""

from llm_memory.consolidation.promoter import (
    PromotionResult,
    PromotionCriteria,
    STMToEpisodicPromoter,
    EpisodicToSemanticPromoter,
    MemoryPromoter,
)
from llm_memory.consolidation.merger import (
    MergeResult,
    GarbageCollectionResult,
    MemoryMerger,
    GarbageCollector,
    MemoryDeduplicator,
)
from llm_memory.consolidation.scheduler import (
    ConsolidationPhase,
    ConsolidationRunResult,
    ConsolidationScheduler,
    ManualConsolidator,
)

__all__ = [
    # Promoter
    "PromotionResult",
    "PromotionCriteria",
    "STMToEpisodicPromoter",
    "EpisodicToSemanticPromoter",
    "MemoryPromoter",
    # Merger
    "MergeResult",
    "GarbageCollectionResult",
    "MemoryMerger",
    "GarbageCollector",
    "MemoryDeduplicator",
    # Scheduler
    "ConsolidationPhase",
    "ConsolidationRunResult",
    "ConsolidationScheduler",
    "ManualConsolidator",
]
