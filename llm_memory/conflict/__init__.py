"""
Conflict resolution module.

Provides:
- Contradiction detection between memories
- Multiple resolution strategies
- User-guided conflict resolution
- Batch conflict processing
"""

from llm_memory.conflict.detector import (
    ConflictType,
    ConflictSeverity,
    DetectedConflict,
    ConflictDetector,
    LLMConflictDetector,
)
from llm_memory.conflict.strategies import (
    ResolutionStrategy,
    ResolutionAction,
    ResolutionResult,
    BaseResolutionStrategy,
    RecencyStrategy,
    ConfidenceStrategy,
    SourceReliabilityStrategy,
    FrequencyStrategy,
    ImportanceStrategy,
    MergeStrategy,
    KeepBothStrategy,
    UserGuidedStrategy,
    get_strategy,
    get_default_strategy_for_conflict,
)
from llm_memory.conflict.resolver import (
    ConflictResolutionConfig,
    ConflictHistory,
    ConflictResolver,
    BatchConflictResolver,
)

__all__ = [
    # Detector
    "ConflictType",
    "ConflictSeverity",
    "DetectedConflict",
    "ConflictDetector",
    "LLMConflictDetector",
    # Strategies
    "ResolutionStrategy",
    "ResolutionAction",
    "ResolutionResult",
    "BaseResolutionStrategy",
    "RecencyStrategy",
    "ConfidenceStrategy",
    "SourceReliabilityStrategy",
    "FrequencyStrategy",
    "ImportanceStrategy",
    "MergeStrategy",
    "KeepBothStrategy",
    "UserGuidedStrategy",
    "get_strategy",
    "get_default_strategy_for_conflict",
    # Resolver
    "ConflictResolutionConfig",
    "ConflictHistory",
    "ConflictResolver",
    "BatchConflictResolver",
]
